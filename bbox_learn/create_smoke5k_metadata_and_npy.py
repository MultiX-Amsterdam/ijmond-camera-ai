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


def is_file_here(file_path):
    return os.path.isfile(file_path)


def convert_images_to_npy(image_path, npy_path):
    """Convert a single jpg/png image to .npy file. Create output directory if needed."""
    output_dir = os.path.dirname(npy_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(npy_path):
        print(f"Skipping {npy_path} (already exists)")
        return
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    np.save(npy_path, arr)
    print(f"Converted {os.path.basename(image_path)} to {os.path.basename(npy_path)}")


def convert_all_images_to_npy(input_dir, output_dir):
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

        convert_images_to_npy(image_path, npy_path)

    print(f"Conversion complete! Converted {len(image_files)} images to .npy format.")


def create_smoke5k_metadata(img_root_dir, gt_root_dir, metadata_path):
    """
    Create metadata for the Smoke5k dataset.

    Args:
        img_root_dir (str): The root directory containing the images in .npy format.
        gt_root_dir (str): The root directory containing the ground truth segmentation masks in .npy format.
        metadata_path (str): The path where the metadata JSON file will be saved.
    """
    metadata = []
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

        # Create metadata entry
        entry = {
            "id": image_id
        }
        metadata.append(entry)

    # Save metadata to JSON file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata created and saved to {metadata_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_smoke5k_metadata_and_npy.py <root_dir>")
        print("Example: python create_smoke5k_metadata_and_npy.py dataset/smoke5k/")
        sys.exit(1)

    root_dir = sys.argv[1]
    convert_all_images_to_npy(os.path.join(root_dir, "test/img"), os.path.join(root_dir, "test/img_npy"))
    convert_all_images_to_npy(os.path.join(root_dir, "test/gt"), os.path.join(root_dir, "test/gt_npy"))
    convert_all_images_to_npy(os.path.join(root_dir, "train/img"), os.path.join(root_dir, "train/img_npy"))
    convert_all_images_to_npy(os.path.join(root_dir, "train/gt"), os.path.join(root_dir, "train/gt_npy"))
    create_smoke5k_metadata(os.path.join(root_dir, "test/img_npy"), os.path.join(root_dir, "test/gt_npy"), "smoke5k_metadata_test.json")
    create_smoke5k_metadata(os.path.join(root_dir, "train/img_npy"), os.path.join(root_dir, "train/gt_npy"), "smoke5k_metadata_train.json")