import argparse
from copy import deepcopy
import logging
import os
import pprint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm.models
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import yaml

from dataset.semi import SemiSmokeDataset, compute_transmission_map
from model.semseg.dpt import DPT
from supervised import evaluate_new
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--training-config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


# Fixed set of test images used for cross-model visual comparison.
# All are smoke-positive crops from expert_standard_test.txt.
VIS_IMAGE_STEMS = [
    "hoogovens_6_7__2024-11-16T12-36-20Z_frame_334_jpg.rf.972bf56bf713068def3ec973d3e202ae_crop_1",
    "hoogovens_6_7__2024-11-14T07-25-14Z_frame_524_jpg.rf.d02eb955950db5555fed641e1ca58a4d_crop_3",
    "kooks_1__2024-11-03T08-56-56Z_frame_3473_jpg.rf.c0778a5cfbbbd080e3f5895d26f67604_crop_4",
    "kooks_1__2024-11-01T11-27-25Z_frame_1294_jpg.rf.8578581c82de76b82a2b258dd0a5f3a0_crop_3",
    "kooks_1__2024-11-06T10-31-19Z_frame_370_jpg.rf.4ff1d0c903b7b03c040931ffc1ea8afb_crop_2",
    "kooks_1__2024-11-10T10-00-51Z_frame_563_jpg.rf.1f6bb53cc5b0e4ce17640ada772f7cea_crop_1",
    "kooks_2__2024-09-07T16-01-03Z_frame_70_jpg.rf.6552d5de498f385ead3d78672f55a44a_crop_1",
    "kooks_2__2024-09-23T14-29-35Z_frame_3555_jpg.rf.aabcb3465a66fba021cd4bb15fdc8db0_crop_1",
    "kooks_2__2024-10-03T10-02-12Z_frame_956_jpg.rf.ffed19010a6fb93fb34fc810a2b06151_crop_2",
    "kooks_2__2024-11-14T13-38-52Z_frame_1165_jpg.rf.18d47ba0217019ce67f7e9e3c06b76ce_crop_2",
]


def visualize_predictions(model, data_root, test_txt_path, save_path, multiplier, label, use_dcp=False):
    """Save side-by-side visualizations of fixed test images.

    For each image in VIS_IMAGE_STEMS, loads the raw image and its ground-truth
    mask, runs model inference, and saves a PNG with three columns:
    image | ground truth | prediction.

    Parameters
    ----------
    model : nn.Module
        Model in eval mode.
    data_root : str
        Root directory of the dataset.
    test_txt_path : str
        Absolute path to the test split txt file.
    save_path : str
        Directory where the visualization PNG files are written.
    multiplier : int
        Patch size used to pad images before inference.
    label : str
        Short label used as a suffix in the output filename (e.g. "model" or "ema").
    use_dcp : bool, optional
        Whether to append a transmission-map channel (channel 4) expected by models
        trained with ``use_dcp=True``. Default is False.
    """
    with open(test_txt_path, "r") as f:
        lines = f.read().splitlines()

    stem_to_line = {}
    for line in lines:
        parts = line.split(" ")
        img_rel = parts[0]
        stem = os.path.splitext(os.path.basename(img_rel))[0]
        stem_to_line[stem] = parts

    for stem in VIS_IMAGE_STEMS:
        if stem not in stem_to_line:
            continue

        parts = stem_to_line[stem]
        img_path = os.path.join(data_root, parts[0])
        has_mask = len(parts) >= 2 and parts[1] not in ("", "None")
        mask_path = os.path.join(data_root, parts[1]) if has_mask else None

        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)

        if has_mask:
            mask_pil = Image.open(mask_path)
            mask_np = np.array(mask_pil)
            if mask_np.ndim > 2:
                mask_np = np.max(mask_np, axis=-1)
            gt_binary = (mask_np > 0).astype(np.uint8)
        else:
            gt_binary = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

        # Inference
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        for c in range(3):
            img_tensor[c] = (img_tensor[c] - mean[c]) / std[c]
        if use_dcp:
            t = compute_transmission_map(img_pil)          # H×W float32
            t_tensor = torch.from_numpy(t).unsqueeze(0).float()
            t_tensor = (t_tensor - 0.5) / 0.5             # normalize to [-1, 1]
            img_tensor = torch.cat([img_tensor, t_tensor], dim=0)  # [4, H, W]
        img_tensor = img_tensor.unsqueeze(0).cuda()

        ori_h, ori_w = img_tensor.shape[-2:]
        new_h = int(ori_h / multiplier + 0.5) * multiplier
        new_w = int(ori_w / multiplier + 0.5) * multiplier
        max_dim = 400
        if max(new_h, new_w) > max_dim:
            scale = max_dim / max(new_h, new_w)
            new_h = max(int(ori_h * scale / multiplier + 0.5) * multiplier, multiplier)
            new_w = max(int(ori_w * scale / multiplier + 0.5) * multiplier, multiplier)
        img_resized = F.interpolate(img_tensor, (new_h, new_w), mode="bilinear", align_corners=True)

        with torch.no_grad():
            with torch.autocast("cuda"):
                pred = model(img_resized).float()
            pred = F.interpolate(pred, (ori_h, ori_w), mode="bilinear", align_corners=True)
            pred_mask = pred.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Plot: image | ground truth | prediction
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(gt_binary, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        fig.suptitle(stem, fontsize=8)
        plt.tight_layout()
        out_filename = os.path.join(save_path, f"vis_{label}_{stem}.png")
        plt.savefig(out_filename, dpi=100, bbox_inches="tight")
        plt.close(fig)


def main():
    torch.cuda.empty_cache()

    args = parser.parse_args()
    training_cfg = yaml.load(open(args.training_config, "r"), Loader=yaml.Loader)

    cfg = yaml.load(
        open(training_cfg['configuration'], "r"),
        Loader=yaml.Loader
    )

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}

        logger.info('Training config ({}):\n{}\n'.format(args.training_config, pprint.pformat(training_cfg)))
        logger.info('Model config ({}):\n{}\n'.format(training_cfg['configuration'], pprint.pformat(all_args)))

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    is_dinov3 = cfg['backbone'].startswith('dinov3_')

    model = DPT(**{
        **model_configs[cfg['backbone'].split('_')[-1]],
        'nclass': cfg['nclass'],
        'use_dcp': cfg.get('use_dcp', False),
        'backbone_type': 'dinov3' if is_dinov3 else 'dinov2',
    })

    # --- DINOv2-only: remove when dropping DINOv2 ---
    if not is_dinov3:
        state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
        model.backbone.load_state_dict(state_dict)
    # --- end DINOv2-only ---
    else:
        timm.models.load_checkpoint(
            model.backbone.model,
            f'./pretrained/{cfg["backbone"]}.safetensors',
            strict=False,
        )

    if cfg['lock_backbone']:
        model.lock_backbone()

    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda(local_rank)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True
    )

    model_ema = deepcopy(model.module)

    model_ema.eval()

    for param in model_ema.parameters():
        param.requires_grad = False

    if (test_model := training_cfg.get("test_model", None)) is None:
        raise Exception("test model not specified")

    test_model_path = os.path.join(args.save_path, test_model)

    if not os.path.exists(test_model_path):
        raise Exception(f"test model not found at `{test_model_path}`")

    checkpoint = torch.load(test_model_path, map_location='cpu', weights_only=False)
    model_checkpoint = checkpoint["checkpoint"]

    model.load_state_dict(model_checkpoint["model"])
    model.eval()

    logger.info("Loaded %s at epoch %s" % (test_model_path, model_checkpoint['epoch']))

    # Load EMA weights from companion file
    has_ema = False
    ema_filename = test_model.replace('best.pth', 'best_ema.pth')
    ema_path = os.path.join(args.save_path, ema_filename)
    if ema_filename != test_model and os.path.exists(ema_path):
        ema_checkpoint = torch.load(ema_path, map_location='cpu', weights_only=False)
        model_ema.load_state_dict(ema_checkpoint["checkpoint"]["model_ema"])
        has_ema = True
        logger.info("Loaded EMA from %s at epoch %s" % (ema_path, ema_checkpoint["checkpoint"]["epoch"]))
    else:
        logger.warning("No EMA checkpoint found at %s, skipping EMA evaluation" % ema_path)

    testset = SemiSmokeDataset(
        cfg['dataset'],
        cfg['data_root'],
        'test',
        None,
        id_path=training_cfg['test_dataset'],
        use_dcp=cfg.get('use_dcp', False),
    )

    testloader = DataLoader(
        testset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
        shuffle=False
    )

    if rank == 0:
        evaluation = evaluate_new(model, testloader, multiplier=model.module.patch_size)

        logger.info('***** Evaluation *****')

        for k, v in evaluation.items():
            logger.info(f"\t{k}: {v:.4f}")

        if has_ema:
            evaluation_ema = evaluate_new(model_ema, testloader, multiplier=model_ema.patch_size)

            print()

            logger.info('***** Evaluation EMA *****')

            for k, v in evaluation_ema.items():
                logger.info(f"\t{k}: {v:.4f}")

        bl_cfg = cfg.get('boundary_lenience', {})
        band_k = bl_cfg.get('band_kernel_size', 7)
        eval_bf = evaluate_new(model, testloader, multiplier=model.module.patch_size, band_kernel_size=band_k)

        print()

        logger.info('***** Evaluation (band-filtered) *****')

        for k, v in eval_bf.items():
            logger.info(f"\t{k}: {v:.4f}")

        if has_ema:
            eval_bf_ema = evaluate_new(model_ema, testloader, multiplier=model_ema.patch_size, band_kernel_size=band_k)

            print()

            logger.info('***** Evaluation EMA (band-filtered) *****')

            for k, v in eval_bf_ema.items():
                logger.info(f"\t{k}: {v:.4f}")

        # --- Visualize fixed test images ---
        test_txt_path = os.path.join(cfg['data_root'], training_cfg['test_dataset'])
        _use_dcp = cfg.get('use_dcp', False)
        visualize_predictions(model, cfg['data_root'], test_txt_path, args.save_path, model.module.patch_size, "model", use_dcp=_use_dcp)
        if has_ema:
            visualize_predictions(model_ema, cfg['data_root'], test_txt_path, args.save_path, model_ema.patch_size, "ema", use_dcp=_use_dcp)
        logger.info('Visualizations saved to %s' % args.save_path)


if __name__ == '__main__':
    main()
