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
from training.train import evaluate
from open_clip.utils import get_tar_path_from_dataset_name, dataset_split
from training.params import parse_args
import sys


def find_params_value(file, key):
    # find value of params in params_file
    with open(file, 'r') as f:
        for line in f:
            if key + ': ' in line:
                return line.split(': ')[1].strip()
    return None


def evaluate_zeroshot(model, data, start_epoch, args, writer, tokenizer):
    dataloader = data["val"].dataloader
    metrics = {}
    device = torch.device(args.device)
    model.eval()
    metrics.update({"epoch": start_epoch})
    model_cfg = model.model_cfg
    text_cfg = model_cfg["text_cfg"]

    all_audio_features = []
    all_class_labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            keys = ["waveform", "longer"]
            audios = {k:batch[k].to(device) for k in keys}
            # audios = batch  # contains mel_spec, wavform, and longer list
            # audios = audios.cuda()
            audio_features, _, _ = model(audios, None)
            # audio_features = F.normalize(audio_features, dim=-1)
            all_audio_features.append(audio_features.detach().cpu())
            all_class_labels.append(torch.argmax(batch["class_label"], 1).long())
        all_audio_features = torch.cat(all_audio_features, dim=0)
        all_class_labels = torch.cat(all_class_labels, dim=0)
        metrics["num_samples"] = all_audio_features.shape[0]

        # get text features
        # if args.val_dataset_names == ['GTZAN']:
        #     all_texts = [f"This is a {t} song." for t in args.class_index_dict.keys()]
        # else:
        all_texts = [f"This is a sound of {t}." for t in args.class_index_dict.keys()]
        logging.info(f'class label prompts: {all_texts}')
        is_hf = text_cfg.get("hf_model_name", None)
        # (yusong): a hack, can make it better
        if not is_hf:
            from open_clip.tokenizer import tokenize
            all_texts = tokenize(all_texts)
        else:
            # from training.data import tokenizer
            all_texts = tokenizer(all_texts)

        all_texts = all_texts.cuda()
        _, all_text_features, _ = model(None, all_texts)
        # all_text_features = F.normalize(all_text_features, dim=-1)
        all_text_features = all_text_features.detach().cpu()

        # compute similarity
        _, _, logit_scale = model(None, None)
        if isinstance(logit_scale, tuple):
            logit_scale_a, logit_scale_t = logit_scale
            logit_scale_a = logit_scale_a.exp().detach().cpu()
            logit_scale_t = logit_scale_t.exp().detach().cpu()
        else:
            logit_scale_a = logit_scale_t = logit_scale.exp().detach().cpu()
        # logit_scale = logit_scale.cpu()

        logits_per_audio = (all_audio_features @ all_text_features.t()).detach().cpu()
        logits_per_text = (all_text_features @ all_audio_features.t()).detach().cpu()

        ground_truth = all_class_labels.view(-1, 1)
        logit = logits_per_audio

        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]  # (yusong) this line is slow because it uses single thread
        preds = preds.detach().cpu().numpy()
        metrics[f"{args.datasetnames[0]}_mean_rank"] = preds.mean() + 1
        metrics[f"{args.datasetnames[0]}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{args.datasetnames[0]}_R@{k}"] = np.mean(preds < k)
        # map@10
        metrics[f"{args.datasetnames[0]}_mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

        logging.info(
            f"Eval Epoch: {start_epoch} "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )

        # if args.wandb:
        #     assert wandb is not None, "Please install wandb."
        #     for name, val in metrics.items():
        #         wandb.log({f"val/{name}": val, "epoch": start_epoch})


def main(args):
    print("starting...")
    args = parse_args(args)
    print("pasred")

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
    # tmodel = find_params_value(params_file, 'tmodel')

    # if amodel is None or tmodel is None:
    #     raise ValueError('model type not found in params file')

    # set up dummy values for args
    args.parallel_eval = False
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    args.val_frequency = 1
    args.val_data = True
    args.epochs = 1
    args.precision = 'fp32'
    args.save_logs = True
    args.wandb = args.report_to == 'wandb'
    args.class_index_dict = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    print("setup")


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
    print("dataset")
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
        output_dict=True,
        **model_kwargs,
    )
    # model = create_model(
    #     model_name,
    #     pretrained,
    #     precision='fp32',
    #     device=device,
    #     jit=False,
    #     force_quick_gelu=False,
    #     # openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
    #     skip_params=False,
    #     # enable_fusion=args.enable_fusion,
    #     # fusion_type=args.fusion_type
    # )  # a hack to get model_cfg

    args.model_cfg = model.model_cfg

    tokenizer = get_tokenizer(args.model)

    data = get_data(args, (preprocess_train, preprocess_val), tokenizer=tokenizer)  # (yusong): hack: no model_cfg needed to get data

    writer = None  # if use tensorboard, initalize writer here

    if args.wandb:
        assert wandb is not None, "Please install wandb."

        # # find the line with "wandb_notes" and get the value
        # wandb_notes = find_params_value(params_file, 'wandb_notes')
        # if wandb_notes is None:
        #     print(f'wandb_notes not found in params file: {params_file}, set to timestamp.')
        #     wandb_notes = f'experiment_{time.strftime("%Y%m%d-%H%M%S")}'
        # wandb_notes = wandb_notes + '-eval-retrieval'
        wandb_notes = args.wandb_notes

        logging.debug("Starting wandb.")
        
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        else:
            args.train_sz = data["train"].dataloader.num_samples
        # # you will have to configure this for your project!
        # if args.wandb_id is not None:
        #     wandb.init(
        #         project="clap",
        #         id=args.wandb_id,
        #         resume=True
        #     )
        # else:
        #     wandb.init(
        #         project="clap",
        #         notes=wandb_notes,
        #         name=wandb_notes,
        #         tags=[],
        #         config=vars(args),
        #     )
        # logging.debug("Finished loading wandb.")

    if os.path.isdir(args.pretrained):
        all_model_checkpoints = sorted(glob.glob(os.path.join(log_dir, 'checkpoints', '*.pt')), key=os.path.getmtime)
    else:
        all_model_checkpoints = [args.pretrained]
    print(all_model_checkpoints)
    for model_path in all_model_checkpoints:
        args.checkpoint_path = os.path.dirname(model_path)
        model = create_model(
            model_name,
            pretrained,
            precision='fp32',
            device=device,
            jit=False,
            force_quick_gelu=False,
            # openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
            skip_params=False,
            # enable_fusion=args.enable_fusion,
            # fusion_type=args.fusion_type
        )

        # load model
        checkpoint = torch.load(model_path, map_location=device)
        if "epoch" in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
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

        evaluate_zeroshot(model, data, start_epoch, args, writer, tokenizer)

if __name__ == '__main__':
    main(sys.argv[1:])

