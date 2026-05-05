"""
Debug script: visually verify that the transmission map looks correct on sample images.

Must be run from the unimatch_v2/ directory.

Example usage:

    python debug_transmission_map.py \
        --id-path  ../bbox_learn/dataset/experiment/expert_standard_train_100_with_masks.txt \
        --data-root ../bbox_learn/dataset \
        --n 4 \
        --out debug_transmission_map.png
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Run from the unimatch_v2/ directory; dataset/ is a package there.
sys.path.insert(0, os.path.dirname(__file__))
from dataset.semi import compute_transmission_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualize transmission map on sample images')
    parser.add_argument('--id-path', type=str, required=True,
                        help='Path to a dataset split txt file (each line: img_path [mask_path])')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory for images')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of images to sample (default: 5)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show', action='store_true',
                        help='Display the figure interactively instead of saving')
    parser.add_argument('--out', type=str, default='debug_transmission_map.png',
                        help='Output filename when not using --show')
    args = parser.parse_args()

    with open(args.id_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    random.seed(args.seed)
    sampled = random.sample(lines, min(args.n, len(lines)))

    fig, axes = plt.subplots(len(sampled), 2, figsize=(10, 4 * len(sampled)))
    if len(sampled) == 1:
        axes = [axes]

    for row, line in enumerate(sampled):
        img_rel = line.split(' ')[0]
        img_path = os.path.join(args.data_root, img_rel)

        img_pil = Image.open(img_path).convert('RGB')
        t = compute_transmission_map(img_pil)

        ax_rgb, ax_t = axes[row]

        ax_rgb.imshow(img_pil)
        ax_rgb.set_title(os.path.basename(img_rel), fontsize=8)
        ax_rgb.axis('off')

        im = ax_t.imshow(t, cmap='gray', vmin=0, vmax=1)
        ax_t.set_title(f'Transmission map  min={t.min():.3f}  max={t.max():.3f}  mean={t.mean():.3f}', fontsize=8)
        ax_t.axis('off')
        fig.colorbar(im, ax=ax_t, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if args.show:
        plt.show()
    else:
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
