"""
This script creates metadata for the Smoke5k dataset.
"""
import sys
import os
import json
from pathlib import Path
from util.util import is_file_here


def create_smoke5k_metadata(img_root_dir, gt_root_dir, output_dir, output_name="train"):
    """
    Create metadata for the Smoke5k dataset in a single .txt file.

    Args:
        img_root_dir (str): The root directory containing the images.
        gt_root_dir (str): The root directory containing the ground truth segmentation masks.
        output_dir (str): The directory where the .txt file will be saved.
        output_name (str): The name prefix for the output file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare file path
    output_file = os.path.join(output_dir, f"{output_name}.txt")

    all_pairs = []

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in Path(img_root_dir).iterdir() if f.suffix.lower() in image_extensions]

    for image_file in image_files:
        if not is_file_here(image_file):
            continue

        # Extract image ID from filename
        image_id = image_file.stem
        img_ext = image_file.suffix

        # Find corresponding ground truth file (try common extensions)
        gt_file = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            candidate = Path(gt_root_dir) / f"{image_id}{ext}"
            if is_file_here(candidate):
                gt_file = candidate
                break

        if gt_file is None:
            print(f"Warning: Ground truth file for {image_id} not found in {gt_root_dir}. Skipping.")
            continue

        gt_ext = gt_file.suffix

        # Create relative paths
        img_relative_path = f"img/{image_id}{img_ext}"
        mask_relative_path = f"gt/{image_id}{gt_ext}"

        # Add all pairs to the single list
        all_pairs.append(f"{img_relative_path} {mask_relative_path}")

    # Write file
    with open(output_file, 'w') as f:
        for pair in all_pairs:
            f.write(f"{pair}\n")

    print(f"Created {output_name} file:")
    print(f"  {output_file} with {len(all_pairs)} image-mask pairs")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_smoke5k_metadata_and_npy.py <root_dir>")
        print("Example: python create_smoke5k_metadata_and_npy.py dataset/smoke5k/")
        sys.exit(1)

    root_dir = sys.argv[1]
    create_smoke5k_metadata(os.path.join(root_dir, "test/img"), os.path.join(root_dir, "test/gt"), os.path.join(root_dir, "test"), output_name="test")
    create_smoke5k_metadata(os.path.join(root_dir, "train/img"), os.path.join(root_dir, "train/gt"), os.path.join(root_dir, "train"), output_name="train")