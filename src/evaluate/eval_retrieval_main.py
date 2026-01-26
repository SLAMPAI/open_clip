import os.path
import glob
import random
import numpy as np
import logging
import wandb
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from open_clip import create_model, create_model_and_transforms, get_tokenizer
from open_clip import tokenize
from training.logger import setup_logging
from training.data import get_data
from training.train import evaluate, evaluate_clotho_audiocaps
from open_clip.utils import get_tar_path_from_dataset_name, dataset_split
from training.params import parse_args
import sys
from contextlib import suppress


def find_params_value(file, key):
    # find value of params in params_file
    with open(file, 'r') as f:
        for line in f:
            if key + ': ' in line:
                return line.split(': ')[1].strip()
    return None

def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    

    for name, logit in logits.items():
        ground_truth = torch.arange(len(logit)).view(-1, 1)
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def evaluate_zeroshot(model, data, start_epoch, args, writer, tokenizer):
    dataloader = data["val"].dataloader
    metrics = {}
    device = torch.device(args.device)
    model.eval()
    # metrics.update({"epoch": start_epoch})
    model_cfg = model.model_cfg
    text_cfg = model_cfg["text_cfg"]

    all_audio_features = []
    all_text_features = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            keys = ["waveform", "longer"]
            audios = {k:batch[k].to(device) for k in keys}
            texts = batch['text']
            print(texts)

            texts = 

            texts = tokenizer(texts).to(device)
            

            out = model(audios, None)
            if isinstance(out, tuple):
                audio_features, _, _ = out
                _, text_features, logit_scale = model(None, texts)
            else: # Cocoa
                audio_features, _ = out.values()
                _, text_features, logit_scale = model(None, texts).values()

            

            all_text_features.append(text_features.detach().cpu())
            all_audio_features.append(audio_features.detach().cpu())

            
        all_audio_features = torch.cat(all_audio_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

        all_audio_features = F.normalize(all_audio_features, dim=-1)
        all_text_features = F.normalize(all_text_features, dim=-1)

        batch_size = args.batch_size
        metrics["num_samples"] = all_audio_features.shape[0]

  
        is_hf = text_cfg.get("hf_model_name", None)
        args.is_hf = is_hf


        logit_scale = logit_scale.cpu()

        logits_per_audio = (logit_scale * all_audio_features @ all_text_features.t()).detach().cpu()
        logits_per_text = logits_per_audio.t().detach().cpu()

        metrics = get_clip_metrics(
                image_features=all_audio_features,
                text_features=all_text_features,
                logit_scale=logit_scale.cpu(),
            )

        # autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
        # # metrics = evaluate_clotho_audiocaps(model, data, start_epoch, args, autocast, device, writer)

        # ground_truth = torch.arange(len(all_text_features)).view(-1, 1)
        # print("ground_truth", ground_truth.shape)
        # logit = logits_per_audio

        # print("logits_per_audio",logits_per_audio.shape)

        # labels = torch.arange(batch_size, device=device).long()

        # ranking = torch.argsort(logit, descending=True)
        # # # print("ranking", ranking.shape)
        # preds = torch.where(ranking == ground_truth)[1]  # (yusong) this line is slow because it uses single thread
        # preds = preds.detach().cpu().numpy()
        # metrics[f"{args.datasetnames[0]}_mean_rank"] = preds.mean() + 1
        # metrics[f"{args.datasetnames[0]}_median_rank"] = np.floor(np.median(preds)) + 1
        # for k in [1, 5, 10]:
        #     metrics[f"{args.datasetnames[0]}_R@{k}"] = np.mean(preds < k)
        # # map@10
        # metrics[f"{args.datasetnames[0]}_mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))
      
        # for key in metrics.keys():
        logging.info(
            f"Eval Epoch: {start_epoch} "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )

            # logging.info(key)


def main(args):
    # (yusong) repeated run might have different metric results.
    # This is because we randomly select crop 10s for each audio.
    args = parse_args(args)

    if os.path.isdir(args.pretrained):
        log_dir = os.path.dirname(args.pretrained)
    else:
        log_dir = os.path.dirname(os.path.dirname(args.pretrained))

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_path = os.path.join(log_dir, 'out.log')
    setup_logging(log_path, args.log_level)
    params_file = os.path.join(log_dir, 'params.txt')

    seed = 3407
    random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)

    # cudnn.benchmark = True
    # cudnn.deterministic = False
    pretrained = args.pretrained
    model_name = args.model
    # amodel = find_params_value(params_file, 'amodel')
    # tmodel = find_params_value(params_file, 'tmodel')

    # if amodel is None or tmodel is None:
    #     raise ValueError('model type not found in params file')

    # set up dummy values for args
    args.parallel_eval = False
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    args.val_frequency = 1
    args.epochs = 1
    args.precision = 'fp32'
    args.save_logs = True
    args.wandb = args.report_to == 'wandb'
    args.class_index_dict = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # if args.remotedata:
    #     for dataset_name in args.datasetnames:
    #         for split in dataset_split[dataset_name]:
    #             if not os.path.exists(f"./json_files/{dataset_name}/{split}"):
    #                 os.makedirs(f"./json_files/{dataset_name}/{split}")
    #             os.system(
    #                 f"aws s3 cp s3://s-laion-audio/webdataset_tar/{dataset_name}/{split}/sizes.json ./json_files/{dataset_name}/{split}/sizes.json"
    #             )

    if args.datasetinfos is None:
        args.datasetinfos = ["train", "unbalanced_train", "balanced_train"]
    if args.dataset_type == "webdataset-audio":
        args.train_data = get_tar_path_from_dataset_name(
            args.datasetnames,
            args.datasetinfos,
            proportion=args.dataset_proportion,
            dataset_path=args.datasetpath,
        )
        args.val_data = get_tar_path_from_dataset_name(
            args.datasetnames,
            ["valid", "test", "eval"],
            proportion=1,
            dataset_path=args.datasetpath,
        )

    model_kwargs = {}

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=False,
        **model_kwargs,
    )
    # model = create_model(
    #     model_name,
    #     pretrained,
    #     precision='fp32',
    #     device=device,
    #     jit=False,
    #     force_quick_gelu=False,
    #     openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
    #     skip_params=False,
    #     enable_fusion=args.enable_fusion,
    #     fusion_type=args.fusion_type
    # )  # a hack to get model_cfg

    args.model_cfg = model.model_cfg
    tokenizer = get_tokenizer(args.model)

    data = get_data(args, (preprocess_train, preprocess_val), tokenizer=None) # (yusong): hack: no model_cfg needed to get data

    writer = None  # if use tensorboard, initalize writer here

    if os.path.isdir(args.pretrained):
        all_model_checkpoints = sorted(glob.glob(os.path.join(log_dir, 'checkpoints', '*.pt')), key=os.path.getmtime)
    else:
        all_model_checkpoints = [args.pretrained]
    for model_path in all_model_checkpoints:
        args.checkpoint_path = os.path.dirname(model_path)
        model = create_model(
            model_name,
            pretrained,
            precision='fp32',
            device=device,
            jit=False,
            force_quick_gelu=False,
            output_dict=False,
            # openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
            # skip_params=False,
            # enable_fusion=args.enable_fusion,
            # fusion_type=args.fusion_type
        )

        # load model
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Loading checkpoint from {model_path}")
        if "epoch" in model_path:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            logging.info("state dict:", sd.keys())
            if next(iter(sd.items()))[0].startswith(
                    "module"
            ):
                sd = {k[len("module."):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            logging.info(
                f"=> resuming checkpoint '{model_path}' (epoch {start_epoch})"
            )
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            start_epoch = 0

        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        evaluate_zeroshot(model, data, start_epoch, args, writer, tokenizer=tokenizer)

if __name__ == '__main__':
    main(sys.argv[1:])