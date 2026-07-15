"""
Debug script: visually verify SemiSmokeDataset data augmentation.

Passes the same sample through __getitem__ three times and plots the original
image alongside the augmented outputs for each run.

- train_l mode : original image | original mask | (image | mask) x3 runs
- train_u mode : original image | (weak | strong1 | strong2) x3 runs

Must be run from the unimatch_v2/ directory.

Example usage:

    # Labeled split (train_l) - with masks
    python debug_semi_augmentation.py \
        --id-path  experiment/expert_standard_train_100_with_masks.txt \
        --data-root ../bbox_learn/dataset \
        --mode train_l --crop-size 352 --base-size 704 --n 4 \
        --out debug_semi_train_l.png

    # Unlabeled split (train_u)
    python debug_semi_augmentation.py \
        --id-path  experiment/unlabeled.txt \
        --data-root ../bbox_learn/dataset \
        --mode train_u --crop-size 352 --base-size 704 --n 4 \
        --out debug_semi_train_u.png
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset.semi import SemiSmokeDataset

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Convert a normalised CHW float tensor to a HWC uint8 numpy array."""
    img = tensor[:3].permute(1, 2, 0).numpy()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def show_mask(ax, mask_tensor, title):
    """Display a binary/smoke mask."""
    m = mask_tensor.numpy()
    ax.imshow(m, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def show_img(ax, img_tensor, title):
    ax.imshow(denormalize(img_tensor))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Visualize SemiSmokeDataset augmentation")
    parser.add_argument("--id-path", type=str, required=True,
                        help="Path relative to data-root for the split txt file")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train_l",
                        choices=["train_l", "train_u"],
                        help="Dataset mode to visualize")
    parser.add_argument("--crop-size", type=int, default=352)
    parser.add_argument("--base-size", type=int, default=704)
    parser.add_argument("--n", type=int, default=4,
                        help="Number of samples to show")
    parser.add_argument("--out", type=str, default="debug_semi_augmentation.png")
    args = parser.parse_args()

    dataset = SemiSmokeDataset(
        name="ijmond",
        root=args.data_root,
        mode=args.mode,
        size=args.crop_size,
        base_size=args.base_size,
        id_path=args.id_path,
    )

    selected = random.sample(range(len(dataset)), min(args.n, len(dataset)))
    n_runs = 3

    if args.mode == "train_l":
        # orig img + orig mask, then (image + mask) per run
        n_cols = 2 + n_runs * 2
    else:
        # orig img, then (weak + strong1 + strong2) per run
        n_cols = 1 + n_runs * 3

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(selected):
        # Load original image (and mask) directly from disk, before any augmentation.
        item_id = dataset.ids[idx]
        split_item = item_id.split(" ")
        orig_img = Image.open(os.path.join(dataset.root, split_item[0])).convert("RGB")

        if args.mode == "train_l":
            axes[row, 0].imshow(orig_img)
            axes[row, 0].set_title(f"idx={idx}  original image", fontsize=7)
            axes[row, 0].axis("off")

            has_mask = len(split_item) > 1 and split_item[1] not in ("", "None")
            if has_mask:
                orig_mask = np.array(Image.open(os.path.join(dataset.root, split_item[1])))
                if orig_mask.ndim > 2:
                    orig_mask = orig_mask.max(axis=-1)
                axes[row, 1].imshow(orig_mask > 0, cmap="gray", vmin=0, vmax=1)
            else:
                axes[row, 1].imshow(np.zeros((orig_img.size[1], orig_img.size[0])), cmap="gray")
            axes[row, 1].set_title(f"idx={idx}  original mask", fontsize=7)
            axes[row, 1].axis("off")
        else:
            axes[row, 0].imshow(orig_img)
            axes[row, 0].set_title(f"idx={idx}  original image", fontsize=7)
            axes[row, 0].axis("off")

        for run in range(n_runs):
            sample = dataset[idx]

            if args.mode == "train_l":
                img_t, mask_t = sample
                base_col = 2 + run * 2
                show_img(axes[row, base_col], img_t, f"idx={idx}  run {run + 1}  image")
                show_mask(axes[row, base_col + 1], mask_t, f"idx={idx}  run {run + 1}  mask")

            else:  # train_u
                img_w_t, img_s1_t, img_s2_t, _ignore, _cb1, _cb2 = sample
                base_col = 1 + run * 3
                show_img(axes[row, base_col],     img_w_t,  f"idx={idx}  run {run + 1}  weak")
                show_img(axes[row, base_col + 1], img_s1_t, f"idx={idx}  run {run + 1}  strong1")
                show_img(axes[row, base_col + 2], img_s2_t, f"idx={idx}  run {run + 1}  strong2")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
