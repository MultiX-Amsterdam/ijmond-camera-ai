"""Visualise DINOv3 CLS self-attention maps alongside predicted segmentation masks.

For each image the script saves a figure with columns:
  [original | pred mask | mean CLS attn | head 0 | head 1 | ... | head N-1]

Usage
-----
Single image file::

    python debug_cls_attention.py \
        --training-config splits/m-mix-100-awl.yaml \
        --checkpoint exp/dinov3_small_testrun/m-mix-100-awl/best.pth \
        --img-dir ../bbox_learn/dataset/ijmond_bbox/img \
        --out-dir attn_debug \
        --n-images 8

Directory of .png/.jpg images::

    python debug_cls_attention.py \
        --training-config splits/m-zeroshot.yaml \
        --checkpoint /path/to/exp/best.pth \
        --img-dir bbox_learn/dataset/ijmond_bbox/img \
        --out-dir /tmp/attn_debug

The script loads the full DPT segmentation model (no DDP), registers the CLS
attention hook on the DINOv3 backbone, then runs inference on each image.
"""
import argparse
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm.models
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

from model.semseg.dpt import DPT
from dataset.semi import compute_transmission_map

_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description="Debug CLS attention maps from a trained DINOv3 segmentation model")
parser.add_argument("--training-config", type=str, required=True,
                    help="Path to a split YAML file (e.g. splits/m-mix-20-awl.yaml)")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to best.pth checkpoint")
parser.add_argument("--img-dir", type=str, required=True,
                    help="Directory containing .png/.jpg images to visualise")
parser.add_argument("--out-dir", type=str, default="debug_attn",
                    help="Directory to save output figures")
parser.add_argument("--n-images", type=int, default=16,
                    help="Maximum number of images to process")
parser.add_argument("--size", type=int, default=352,
                    help="Crop/resize size (must match training crop_size, divisible by 16)")


def _load_model(cfg, checkpoint_path):
    """Load DPT model from checkpoint without DDP.

    Parameters
    ----------
    cfg : dict
        Model config dict (from the configuration YAML).
    checkpoint_path : str
        Path to best.pth checkpoint.

    Returns
    -------
    torch.nn.Module
        Model in eval mode on CPU (moved to CUDA if available).
    """
    model_configs = {
        "small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
        "base":  {"encoder_size": "base",  "features": 128, "out_channels": [96, 192, 384, 768]},
        "large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "giant": {"encoder_size": "giant", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    is_dinov3 = cfg["backbone"].startswith("dinov3_")
    size_key = cfg["backbone"].split("_")[-1]

    model = DPT(**{
        **model_configs[size_key],
        "nclass": cfg["nclass"],
        "use_dcp": cfg.get("use_dcp", False),
        "backbone_type": "dinov3" if is_dinov3 else "dinov2",
    })

    if is_dinov3:
        # Load backbone weights
        timm.models.load_checkpoint(
            model.backbone.model,
            f"./pretrained/{cfg['backbone']}.safetensors",
            strict=False,
        )

    # Load full checkpoint (strips module. prefix if saved under DDP)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("checkpoint", ckpt).get("model", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    return model


def _preprocess(img_pil, size, use_dcp=False):
    """Resize, normalise and optionally append DCP channel.

    Parameters
    ----------
    img_pil : PIL.Image
        RGB input image.
    size : int
        Target square size after resize.
    use_dcp : bool
        If True, append transmission map as a 4th channel.

    Returns
    -------
    torch.Tensor
        Shape (1, 3 or 4, size, size).
    """
    img_resized = img_pil.resize((size, size), Image.BILINEAR)
    t = transforms.ToTensor()(img_resized)  # (3, H, W)
    t = _NORMALIZE(t)
    if use_dcp:
        tm = compute_transmission_map(img_resized)
        tm_t = torch.from_numpy(tm).unsqueeze(0).float()
        tm_t = (tm_t - 0.5) / 0.5
        t = torch.cat([t, tm_t], dim=0)
    return t.unsqueeze(0)  # (1, C, H, W)


def _visualise(img_pil, pred_mask, mean_attn, per_head_attns, out_path, img_name):
    """Save a multi-panel attention visualisation figure.

    Parameters
    ----------
    img_pil : PIL.Image
        Original image (any size, shown as-is).
    pred_mask : np.ndarray
        Shape (H, W), int, predicted segmentation (0=bg, 1=smoke).
    mean_attn : np.ndarray
        Shape (H, W), float, mean CLS attention over all heads.
    per_head_attns : np.ndarray
        Shape (num_heads, H, W), float.
    out_path : str
        Full file path to write the PNG figure.
    img_name : str
        Title displayed on the figure.
    """
    num_heads = per_head_attns.shape[0]
    n_cols = 3 + num_heads  # original, pred, mean_attn, head_0..N
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))

    axes[0].imshow(img_pil)
    axes[0].set_title("Image", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Pred mask", fontsize=8)
    axes[1].axis("off")

    im = axes[2].imshow(mean_attn, cmap="inferno", vmin=mean_attn.min(), vmax=mean_attn.max())
    axes[2].set_title(
        f"Mean attn\n[{mean_attn.min():.3f}, {mean_attn.max():.3f}]", fontsize=7
    )
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for h in range(num_heads):
        a = per_head_attns[h]
        im_h = axes[3 + h].imshow(a, cmap="inferno", vmin=a.min(), vmax=a.max())
        axes[3 + h].set_title(f"Head {h}\n[{a.min():.3f}, {a.max():.3f}]", fontsize=7)
        axes[3 + h].axis("off")

    fig.suptitle(img_name, fontsize=9, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main():
    args = parser.parse_args()

    training_cfg = yaml.load(open(args.training_config, "r"), Loader=yaml.Loader)
    cfg = yaml.load(open(training_cfg["configuration"], "r"), Loader=yaml.Loader)

    use_dcp = cfg.get("use_dcp", False)
    size = args.size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = _load_model(cfg, args.checkpoint)
    model.to(device)

    # Register CLS attention hook
    model.backbone.enable_cls_attn_hook()
    print("CLS attention hook registered.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect image paths
    exts = ("*.png", "*.jpg", "*.jpeg")
    img_paths = []
    for ext in exts:
        img_paths.extend(sorted(glob.glob(os.path.join(args.img_dir, ext))))
    img_paths = img_paths[:args.n_images]

    if not img_paths:
        print(f"[error] No images found in {args.img_dir}")
        return

    print(f"Processing {len(img_paths)} images → {args.out_dir}")

    for img_path in img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_pil = Image.open(img_path).convert("RGB")
        inp = _preprocess(img_pil, size, use_dcp=use_dcp).to(device)

        with torch.no_grad():
            logits = model(inp)  # (1, 2, H, W); hook fires here

        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        mean_attn = model.backbone.get_last_cls_attn(size, size)
        if mean_attn is None:
            print(f"[warn] Hook did not fire for {img_name}, skipping.")
            continue
        mean_attn_np = mean_attn.squeeze(0).cpu().numpy()  # (H, W)

        per_head = model.backbone.get_last_cls_attn_per_head(size, size)
        per_head_np = per_head.squeeze(0).cpu().numpy()  # (num_heads, H, W)

        # Console summary
        print(
            f"  {img_name}: pred_smoke_px={pred_mask.sum()}, "
            f"attn mean={mean_attn_np.mean():.4f} "
            f"min={mean_attn_np.min():.4f} "
            f"max={mean_attn_np.max():.4f}"
        )

        out_path = os.path.join(args.out_dir, f"{img_name}_attn.png")
        _visualise(img_pil, pred_mask, mean_attn_np, per_head_np, out_path, img_name)

    print("Done.")


if __name__ == "__main__":
    main()
