"""
Debug script: visually verify citizen bounding-box data augmentation.

Passes the same image through BoxSupDataset.__getitem__ three times and
plots the original image, weak view, and strong view with overlaid bounding
boxes for each run.

Must be run from the unimatch_v2/ directory.

Example usage:

    python debug_boxsup_augmentation.py \
        --json-path ../bbox_learn/dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json \
        --img-dir   ../bbox_learn/dataset/ijmond_bbox/img \
        --crop-size 352 \
        --base-size 704 \
        --n 4 \
        --out debug_boxsup_augmentation.png
"""

import argparse
import json
import os
import random
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset.boxsup import BoxSupDataset

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Convert a normalised CHW float tensor to a HWC uint8 numpy array."""
    img = tensor.permute(1, 2, 0).numpy()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def draw_bboxes(ax, bboxes_tensor, color, label):
    """Draw XYXY bounding boxes on a matplotlib axes."""
    for i, (x1, y1, x2, y2) in enumerate(bboxes_tensor.tolist()):
        if x1 < 0:
            continue
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
            label=label if i == 0 else None,
        )
        ax.add_patch(rect)


def main():
    parser = argparse.ArgumentParser(description="Visualize BoxSup augmentation")
    parser.add_argument("--json-path", type=str, required=True,
                        help="Path to filtered bbox JSON file")
    parser.add_argument("--img-dir", type=str, required=True,
                        help="Directory containing {id}.png images")
    parser.add_argument("--crop-size", type=int, default=352)
    parser.add_argument("--base-size", type=int, default=704)
    parser.add_argument("--n", type=int, default=4,
                        help="Number of images to show (must have at least one bbox)")
    parser.add_argument("--out", type=str, default="debug_boxsup_augmentation.png")
    args = parser.parse_args()

    dataset = BoxSupDataset(
        json_path=args.json_path,
        img_dir=args.img_dir,
        crop_size=args.crop_size,
        base_size=args.base_size,
    )

    # Select indices that have at least one bounding box.
    with open(args.json_path) as f:
        metadata = json.load(f)
    positive_indices = [i for i, s in enumerate(metadata) if s["bbox"]]
    assert positive_indices, "No samples with bounding boxes found."

    selected = random.sample(positive_indices, min(args.n, len(positive_indices)))

    n_runs = 3
    # Layout: orig column + 3 columns per run (weak + strong1 + strong2).
    n_cols = 1 + n_runs * 3
    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(selected):
        sample_id = metadata[idx]["id"]
        bbox_list = metadata[idx]["bbox"] or []

        # Original image with bboxes in original image coordinates.
        orig_img = Image.open(os.path.join(args.img_dir, f"{sample_id}.png")).convert("RGB")
        axes[row, 0].imshow(orig_img)
        orig_bboxes = torch.tensor(
            [[b["x_bbox"], b["y_bbox"], b["x_bbox"] + b["w_bbox"], b["y_bbox"] + b["h_bbox"]]
             for b in bbox_list],
            dtype=torch.float32,
        ) if bbox_list else torch.empty((0, 4))
        draw_bboxes(axes[row, 0], orig_bboxes, color="lime", label="bbox")
        axes[row, 0].set_title(f"id={sample_id}  original", fontsize=7)
        axes[row, 0].axis("off")

        for run in range(n_runs):
            img_w_t, img_s1_t, img_s2_t, _cb1, _cb2, bboxes = dataset[idx]

            col_w  = 1 + run * 3
            col_s1 = 1 + run * 3 + 1
            col_s2 = 1 + run * 3 + 2

            axes[row, col_w].imshow(denormalize(img_w_t[:3]))
            draw_bboxes(axes[row, col_w], bboxes, color="lime", label="bbox")
            axes[row, col_w].set_title(f"id={sample_id}  run {run + 1}  weak", fontsize=7)
            axes[row, col_w].axis("off")

            axes[row, col_s1].imshow(denormalize(img_s1_t[:3]))
            draw_bboxes(axes[row, col_s1], bboxes, color="lime", label="bbox")
            axes[row, col_s1].set_title(f"id={sample_id}  run {run + 1}  strong1", fontsize=7)
            axes[row, col_s1].axis("off")

            axes[row, col_s2].imshow(denormalize(img_s2_t[:3]))
            draw_bboxes(axes[row, col_s2], bboxes, color="lime", label="bbox")
            axes[row, col_s2].set_title(f"id={sample_id}  run {run + 1}  strong2", fontsize=7)
            axes[row, col_s2].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
