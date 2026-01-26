import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from subprocess import CalledProcessError, run, Popen, PIPE

import os
from functools import lru_cache
from typing import Optional, Union, Tuple
try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

def exact_div(x, y):
    assert x % y == 0
    return x // y

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token



def get_T_after_cnn(L_in, dilation=1):
    for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out

def load_bytesio_audio(content, sr: int = SAMPLE_RATE):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", "pipe:",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "pipe:"
    ]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=-1)
    out, _ = p.communicate(input=content)
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

def trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)
    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "mel_filters.npz") # todo
        # os.path.join("assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # return super().forward(x.float()).type(x.dtype)
        return super().forward(x).type(x.dtype)




class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)



class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class WhisperEncoder(nn.Module):
    def __init__(
            self,
            n_mels: int,
            n_ctx: int,
            n_state: int,
            n_head: int,
            n_layer: int,
            output_dim: int = 512,
            avg_pool: bool = True,
            add_audio_bos_eos_token: bool = True,
            **kwargs
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

        if avg_pool:
            self.avg_pooler = nn.AvgPool1d(2, stride=2)
        else:
            self.avg_pooler = None
        self.proj = nn.Linear(n_state, output_dim)
        if add_audio_bos_eos_token:
            self.audio_bos_eos_token = nn.Embedding(2, output_dim)
        else:
            self.audio_bos_eos_token = None
        self.output_dim = output_dim
        self.n_head = n_head

    def forward(self, x: Tensor, padding_mask: Tensor=None, audio_lengths: Tensor=None, output_dict=True):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = x["waveform"]
        x = log_mel_spectrogram(
            x,
            n_mels=self.conv1.in_channels,
            padding=0,
            device=self.conv1.weight.device,
        )
        x = x.to(dtype=self.conv1.weight.dtype,
                 device=self.conv1.weight.device)
        if audio_lengths is not None:
            input_mel_len = audio_lengths[:,0] * 2
            max_mel_len_in_batch = input_mel_len.max()
            x = x[:, :, :max_mel_len_in_batch]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # B, L, D
        bsz = x.size(0)
        src_len = x.size(1)


        self.input_positional_embedding = self.positional_embedding[:src_len]
        assert x.shape[1:] == self.input_positional_embedding.shape, f"incorrect audio shape: {x.shape[1:], self.input_positional_embedding.shape}"
        x = (x + self.input_positional_embedding).to(x.dtype)
        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=self.conv1.weight.dtype,
                     device=self.conv1.weight.device)
            batch_src_len = padding_mask.size(1)
            x = x[:, :batch_src_len, :]
            padding_mask = padding_mask.view(
                bsz, -1, batch_src_len
            )
            padding_mask_ = padding_mask.all(1)
            x[padding_mask_] = 0
            key_padding_mask = padding_mask_.view(bsz, 1, 1, batch_src_len). \
                expand(-1, self.n_head, -1, -1).reshape(bsz, self.n_head, 1, batch_src_len)
            new_padding_mask = torch.zeros_like(key_padding_mask, dtype=x.dtype)
            padding_mask = new_padding_mask.masked_fill(key_padding_mask, float("-inf"))

        for block in self.blocks:
            x = block(x, mask=padding_mask)


        if self.avg_pooler:
            x = x.permute(0, 2, 1)
            x = self.avg_pooler(x)
            x = x.permute(0, 2, 1)


        x = self.ln_post(x)
        x = self.proj(x)

        if self.audio_bos_eos_token is not None:
            bos = self.audio_bos_eos_token.weight[0][None, :]
            eos = self.audio_bos_eos_token.weight[1][None, :]
        else:
            bos, eos = None, None
        if output_dict:
            return {
                "embedding": x,
                "audio_bos": bos,
                "audio_eos": eos,
            }
        return x, bos, eos

    def encode(self, input_audios: Tensor, input_audio_lengths: Tensor, audio_span_tokens: List):
        real_input_audio_lens = input_audio_lengths[:, 0].tolist()
        max_len_in_batch = max(real_input_audio_lens)
        padding_mask = torch.ones([input_audios.size(0), max_len_in_batch]).to(dtype=self.conv1.weight.dtype,
                                                                               device=self.conv1.weight.device)
        for index in range(len(input_audios)):
            padding_mask[index, :input_audio_lengths[index][0].item()] = 0
        x, bos, eos = self(input_audios, padding_mask,input_audio_lengths)
        output_audios = []
        for i in range(len(audio_span_tokens)):
            audio_span = audio_span_tokens[i]
            audio = x[i][:audio_span-2]
            if bos is not None:
                audio = torch.concat([bos, audio, eos])
            assert len(audio) == audio_span
            output_audios.append(audio)
        return output_audios

def create_whisper_model(audio_cfg, output_dim, **kwargs) -> WhisperEncoder:


    if audio_cfg.model_name == "tiny":
        n_layers = 4
        width = 384
        heads = 6
    elif audio_cfg.model_name == "base":
        n_layers = 6
        width = 512
        heads = 8
    elif audio_cfg.model_name == "small":
        n_layers = 6
        width = 768
        heads = 12
    elif audio_cfg.model_name == "medium":
        n_layers = 8
        width = 1024
        heads = 16
    elif audio_cfg.model_name == "large":
        n_layers = 12
        width = 1280
        heads = 20
    else:
        raise ValueError(f"Unknown whisper model name: {audio_cfg.model_name}")

    
    model = WhisperEncoder(
        n_mels=N_MELS,
        n_ctx=get_T_after_cnn(N_FRAMES),
        n_state=width,
        n_head=heads,
        n_layer=n_layers,
        output_dim=output_dim,
        avg_pool=True,
        add_audio_bos_eos_token=True,
        **kwargs,
    )
    if audio_cfg.pretrained:
        print(f"Loading pretrained Whisper weights for model {audio_cfg.model_name}")
        import whisper
        whisper_model = whisper.load_model(audio_cfg.model_name)
        model_state_dict = model.state_dict()
        whisper_state_dict = whisper_model.state_dict()
        for k in model_state_dict.keys():
            if k in whisper_state_dict:
                model_state_dict[k] = whisper_state_dict[k]
        model.load_state_dict(model_state_dict)
    return model

def interpolate_1d(x: torch.Tensor, upsample_factor: int) -> torch.Tensor:
    """
    x: [B, T, C]  -> returns [B, T*upsample_factor, C]
    Uses linear interpolation on time dimension.
    """
    if upsample_factor == 1:
        return x
    # F.interpolate expects [B, C, T]
    x = x.transpose(1, 2)  # [B, C, T]
    x = F.interpolate(x, scale_factor=upsample_factor, mode="linear", align_corners=False)
    return x.transpose(1, 2)  # [B, T', C]

class WhisperAsHTSAT(nn.Module):
    """
    Adapter that makes WhisperEncoder behave like HTSAT_Swin_Transformer
    from an interface/output perspective.

    Input expected: dict with at least {"waveform": Tensor[B, T] or [B, 1, T]}
    Output: dict with HTSAT-like keys
    """
    def __init__(
        self,
        whisper_encoder: nn.Module,
        fine_grained_upsample: int = 2,   # Whisper tokens are ~20ms; 2x -> ~10ms-ish grid
        return_logits: bool = False,      # if you want clipwise/framewise heads later
        num_classes: Optional[int] = None
    ):
        super().__init__()
        self.whisper = whisper_encoder
        self.fine_grained_upsample = fine_grained_upsample

        # Optional heads if you *really* want HTSAT-like outputs (sigmoid logits)
        self.return_logits = return_logits
        if return_logits:
            assert num_classes is not None, "num_classes required if return_logits=True"
            D = whisper_encoder.output_dim
            self.clip_head = nn.Linear(D, num_classes)
            self.frame_head = nn.Linear(D, num_classes)
        else:
            self.clip_head = None
            self.frame_head = None

    def forward(self, x: dict, mixup_lambda=None, infer_mode: bool = False, device=None):
        """
        HTSAT-like signature.
        - mixup_lambda ignored (Whisper pipeline here has no spec mixup)
        - infer_mode ignored (you can add chunking later if needed)
        """
        # Move waveform to device (HTSAT does this)
        if device is None:
            device = next(self.parameters()).device

        if "waveform" not in x:
            raise KeyError('Expected input dict with key "waveform" (like HTSAT).')

        wav = x["waveform"]
        if wav.dim() == 3 and wav.size(1) == 1:
            wav = wav[:, 0, :]
        wav = wav.to(device=device, non_blocking=True)
        x = dict(x)
        x["waveform"] = wav

        # Reuse your WhisperEncoder forward; it already expects dict-like input in your code.
        # Keep padding_mask/audio_lengths behavior if provided.
        out = self.whisper(
            x,
            padding_mask=x.get("padding_mask", None),
            audio_lengths=x.get("audio_lengths", None),
            output_dict=True,
        )

        # Whisper output "embedding" is sequence: [B, Twh, D]
        seq = out["embedding"]
        if seq.dim() != 3:
            raise RuntimeError(f"Whisper returned unexpected embedding shape: {seq.shape}")

        # HTSAT fine_grained_embedding is time-major [B, T, C]
        fine = interpolate_1d(seq, upsample_factor=self.fine_grained_upsample)

        # HTSAT embedding is clip-level [B, C] (avg pooled)
        clip = seq.mean(dim=1)

        output_dict = {
            "framewise_output": None,
            "clipwise_output": None,
            "fine_grained_embedding": fine,  # [B, T', D]
            "embedding": clip,               # [B, D]
        }

        # If you want real HTSAT-like logits:
        if self.return_logits:
            # framewise logits: [B, T', num_classes]
            frame_logits = self.frame_head(fine)
            # clipwise logits: [B, num_classes]
            clip_logits = self.clip_head(clip)
            output_dict["framewise_output"] = torch.sigmoid(frame_logits)
            output_dict["clipwise_output"] = torch.sigmoid(clip_logits)

        return output_dict