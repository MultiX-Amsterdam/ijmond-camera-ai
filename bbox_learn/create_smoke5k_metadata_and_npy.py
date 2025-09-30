"""
This script does the following:
- Convert images in the dataset to npy files.
- Creates metadata for the Smoke5k dataset.
"""
import sys
import os
import numpy as np
import json
from PIL import Image
import numpy as np
from pathlib import Path
from util.util import (
    is_file_here,
    convert_images_to_npy
)


def convert_all_images_to_npy(input_dir, output_dir, gray_scale=False):
    """
    Convert all images in a directory to .npy files.

    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save .npy files
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Get all image files in the input directory
    image_files = []
    for filename in os.listdir(input_dir):
        if os.path.splitext(filename.lower())[1] in image_extensions:
            image_files.append(filename)

    if not image_files:
        print(f"No image files found in '{input_dir}'")
        return

    print(f"Found {len(image_files)} image files to convert...")

    # Convert each image
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        # Change extension to .npy
        npy_filename = os.path.splitext(filename)[0] + '.npy'
        npy_path = os.path.join(output_dir, npy_filename)
        convert_images_to_npy(image_path, npy_path, gray_scale=gray_scale)

    print(f"Conversion complete! Converted {len(image_files)} images to .npy format.")


def create_smoke5k_metadata(img_root_dir, gt_root_dir, output_dir, output_name="train"):
    """
    Create metadata for the Smoke5k dataset in a single .txt file.

    Args:
        img_root_dir (str): The root directory containing the images in .npy format.
        gt_root_dir (str): The root directory containing the ground truth segmentation masks in .npy format.
        output_dir (str): The directory where the .txt file will be saved.
        output_name (str): The name prefix for the output file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare file path
    output_file = os.path.join(output_dir, f"{output_name}.txt")

    all_pairs = []

    image_files = list(Path(img_root_dir).glob("*.npy"))

    for image_file in image_files:
        if not is_file_here(image_file):
            continue

        # Extract image ID from filename
        image_id = image_file.stem

        # Find corresponding ground truth file
        gt_file = Path(gt_root_dir) / f"{image_id}.npy"
        if not is_file_here(gt_file):
            print(f"Warning: Ground truth file for {image_id} not found in {gt_root_dir}. Skipping.")
            continue

        # Create relative paths (assuming the txt files will be in the same directory structure)
        img_relative_path = f"img_npy/{image_id}.npy"
        mask_relative_path = f"gt_npy/{image_id}.npy"

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
    convert_all_images_to_npy(os.path.join(root_dir, "test/img"), os.path.join(root_dir, "test/img_npy"))
    convert_all_images_to_npy(os.path.join(root_dir, "test/gt"), os.path.join(root_dir, "test/gt_npy"), gray_scale=True)
    convert_all_images_to_npy(os.path.join(root_dir, "train/img"), os.path.join(root_dir, "train/img_npy"))
    convert_all_images_to_npy(os.path.join(root_dir, "train/gt"), os.path.join(root_dir, "train/gt_npy"), gray_scale=True)
    create_smoke5k_metadata(os.path.join(root_dir, "test/img_npy"), os.path.join(root_dir, "test/gt_npy"), os.path.join(root_dir, "test"), output_name="test")
    create_smoke5k_metadata(os.path.join(root_dir, "train/img_npy"), os.path.join(root_dir, "train/gt_npy"), os.path.join(root_dir, "train"), output_name="train")