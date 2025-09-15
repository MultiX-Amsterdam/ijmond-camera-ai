"""
Crop segmentation masks for the IJmond Segmentation dataset and create npy files.
"""
import cv2
import numpy as np
import os
import sys
import json
from util.util import convert_images_to_npy


def crop_image(image_path, x, y, width, height):
    """
    Crop an image using OpenCV based on specified coordinates and dimensions.

    Args:
        image_path (str): Path to the input image file
        x (int): Top-left x-coordinate (horizontal axis)
        y (int): Top-left y-coordinate (vertical axis)
        width (int): Width of the crop region
        height (int): Height of the crop region

    Returns:
        numpy.ndarray: Cropped image as numpy array, or None if error occurs
    """
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'.")
        return None

    # Get image dimensions
    img_height, img_width, channels = image.shape

    # Calculate end coordinates
    x_end = x + width
    y_end = y + height

    # Validate coordinates
    if x < 0 or y < 0 or x_end > img_width or y_end > img_height:
        print(f"Error: Crop coordinates out of bounds.")
        print(f"Image dimensions: {img_width}x{img_height}")
        print(f"Requested crop: top-left({x}, {y}), size({width}x{height})")
        return None

    if width <= 0 or height <= 0:
        print(f"Error: Invalid crop dimensions. Width and height must be positive.")
        return None

    # Crop the image
    # Note: OpenCV uses [y:y_end, x:x_end] format for array slicing
    cropped_image = image[y:y_end, x:x_end]

    return cropped_image


def save_cropped_image(image_path, output_path, x, y, width, height):
    """
    Crop an image and save it to a specified output path.

    Args:
        image_path (str): Path to the input image file
        output_path (str): Path where the cropped image will be saved
        x (int): Top-left x-coordinate (horizontal axis)
        y (int): Top-left y-coordinate (vertical axis)
        width (int): Width of the crop region
        height (int): Height of the crop region

    Returns:
        bool: True if successful, False otherwise
    """
    cropped_image = crop_image(image_path, x, y, width, height)

    if cropped_image is None:
        return False

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the cropped image
    success = cv2.imwrite(output_path, cropped_image)
    if success:
        print(f"Cropped image saved to: {output_path}")
        return True
    else:
        print(f"Error: Failed to save cropped image to '{output_path}'.")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python crop_ijmond_seg_and_create_npy.py <input_folder> <coco_json_path> <masks_folder> <output_folder>")
        print("Example: python crop_ijmond_seg_and_create_npy.py dataset/ijmond_seg/test/images/ dataset/ijmond_seg/test/_annotations.coco.json dataset/ijmond_seg/test/masks/ dataset/ijmond_seg/test/cropped/")
        sys.exit(1)

    input_folder = sys.argv[1]
    coco_json_path = sys.argv[2]
    masks_folder = sys.argv[3]
    output_folder = sys.argv[4]

    # Validate input paths
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        sys.exit(1)

    if not os.path.exists(coco_json_path):
        print(f"Error: COCO annotation file '{coco_json_path}' not found.")
        sys.exit(1)

    if not os.path.exists(masks_folder):
        print(f"Error: Masks folder '{masks_folder}' not found.")
        sys.exit(1)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create images subfolder for cropped images
    output_images_folder = os.path.join(output_folder, "images")
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    # Create masks subfolder for cropped masks
    output_masks_folder = os.path.join(output_folder, "masks")
    if not os.path.exists(output_masks_folder):
        os.makedirs(output_masks_folder)

    # Create npy subfolders for converted images and masks
    output_images_npy_folder = os.path.join(output_folder, "images_npy")
    if not os.path.exists(output_images_npy_folder):
        os.makedirs(output_images_npy_folder)

    output_masks_npy_folder = os.path.join(output_folder, "masks_npy")
    if not os.path.exists(output_masks_npy_folder):
        os.makedirs(output_masks_npy_folder)

    # Load COCO annotations to get list of images
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract image filenames from COCO data
    coco_images = {img['id']: img['file_name'] for img in coco_data['images']}
    image_files = list(coco_images.values())

    print(f"Found {len(image_files)} images in COCO annotations")

    # Define crop parameters for different file types
    # Format: "pattern": [(x, y, width, height, view_id, camera_name), ...]
    crop_configs = {
        "kooks_2": [
            (128, 54, 900, 900, 1, "kooks_2"),
            (816, 88, 900, 900, 2, "kooks_2")
        ],
        "kooks_1": [
            (59, 344, 600, 600, 1, "kooks_1"),
            (500, 348, 600, 600, 2, "kooks_1"),
            (871, 340, 600, 600, 3, "kooks_1"),
            (1261, 362, 600, 600, 4, "kooks_1")
        ],
        "hoogovens_6_7": [
            (0, 263, 700, 700, 1, "hoogovens_6_7"),
            (417, 267, 700, 700, 2, "hoogovens_6_7"),
            (794, 166, 800, 800, 3, "hoogovens_6_7"),
            (1500, 500, 400, 400, 4, "hoogovens_6_7")
        ]
    }

    if not image_files:
        print(f"No image files found in COCO annotations.")
        sys.exit(1)

    # Metadata list to store crop information
    metadata = []

    # Lists to store image-mask pairs separated by mask content
    test_pairs_with_mask = []      # For pairs with smoke (non-zero mask values)
    test_pairs_without_mask = []   # For pairs without smoke (all-zero mask values)

    processed_count = 0
    total_crops = 0
    successful_crops = 0

    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        processed_count += 1
        print(f"Processing {processed_count}/{len(image_files)}: {filename}")

        # Check if corresponding mask exists
        mask_filename = os.path.splitext(filename)[0] + '.png'  # Assume masks are PNG
        mask_input_path = os.path.join(masks_folder, mask_filename)

        if not os.path.exists(mask_input_path):
            print(f"  Warning: Mask file not found: {mask_input_path}, skipping...")
            continue

        # Get file name without extension for creating crop names
        name_without_ext, file_ext = os.path.splitext(filename)

        # Check which crop configuration applies to this file
        crops_applied = False
        for pattern, crop_list in crop_configs.items():
            if pattern in filename:
                print(f"  Found pattern '{pattern}', applying {len(crop_list)} crops...")
                crops_applied = True

                for x, y, width, height, view_id, camera_name in crop_list:
                    total_crops += 1

                    # Create output filenames
                    cropped_filename = f"{name_without_ext}_crop_{view_id}{file_ext}"
                    cropped_mask_filename = f"{name_without_ext}_crop_{view_id}.png"
                    cropped_npy_filename = f"{name_without_ext}_crop_{view_id}.npy"
                    cropped_mask_npy_filename = f"{name_without_ext}_crop_{view_id}.npy"

                    output_image_path = os.path.join(output_images_folder, cropped_filename)
                    output_mask_path = os.path.join(output_masks_folder, cropped_mask_filename)
                    output_image_npy_path = os.path.join(output_images_npy_folder, cropped_npy_filename)
                    output_mask_npy_path = os.path.join(output_masks_npy_folder, cropped_mask_npy_filename)

                    print(f"    Crop {view_id}: x={x}, y={y}, width={width}, height={height}")

                    # Perform the crop on image
                    image_success = save_cropped_image(input_path, output_image_path, x, y, width, height)

                    # Perform the crop on mask
                    mask_success = save_cropped_image(mask_input_path, output_mask_path, x, y, width, height)

                    if image_success and mask_success:
                        # Convert cropped images to npy files
                        try:
                            convert_images_to_npy(output_image_path, output_image_npy_path, gray_scale=False)
                            convert_images_to_npy(output_mask_path, output_mask_npy_path, gray_scale=True)
                            successful_crops += 1
                        except Exception as e:
                            print(f"    Error converting to npy: {e}")
                            continue

                        # Add metadata entry
                        metadata_entry = {
                            "original_file_name": filename,
                            "cropped_file_name": cropped_filename,
                            "original_mask_name": mask_filename,
                            "cropped_mask_name": cropped_mask_filename,
                            "cropped_npy_name": cropped_npy_filename,
                            "cropped_mask_npy_name": cropped_mask_npy_filename,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "view_id": view_id,
                            "camera_name": camera_name
                        }
                        metadata.append(metadata_entry)

                        # Check mask content to determine if it has smoke or not
                        mask_npy = np.load(output_mask_npy_path)
                        has_smoke = np.any(mask_npy > 0)  # Check if mask has any non-zero values

                        pair_string = f"images_npy/{cropped_npy_filename} masks_npy/{cropped_mask_npy_filename}"

                        if has_smoke:
                            test_pairs_with_mask.append(pair_string)
                            print(f"    -> Has smoke: added to with_mask list")
                        else:
                            test_pairs_without_mask.append(pair_string)
                            print(f"    -> No smoke: added to without_mask list")
                    else:
                        if not image_success:
                            print(f"    Error: Failed to crop image")
                        if not mask_success:
                            print(f"    Error: Failed to crop mask")

                break  # Only apply the first matching pattern

        if not crops_applied:
            print(f"  No crop pattern matched for '{filename}', skipping...")

    # Save metadata to JSON file
    metadata_path = os.path.join(output_folder, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save test files with image-mask pairs separated by mask content
    test_with_mask_path = os.path.join(output_folder, "test_with_mask.txt")
    test_without_mask_path = os.path.join(output_folder, "test_without_mask.txt")

    # Save pairs with smoke (non-zero mask values)
    with open(test_with_mask_path, 'w') as f:
        for pair in test_pairs_with_mask:
            f.write(f"{pair}\n")

    # Save pairs without smoke (all-zero mask values)
    with open(test_without_mask_path, 'w') as f:
        for pair in test_pairs_without_mask:
            f.write(f"{pair}\n")

    print(f"\nProcessing complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Total crops attempted: {total_crops}")
    print(f"Successfully created crops: {successful_crops}")
    print(f"Crops with smoke: {len(test_pairs_with_mask)}")
    print(f"Crops without smoke: {len(test_pairs_without_mask)}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Pairs with smoke saved to: {test_with_mask_path}")
    print(f"Pairs without smoke saved to: {test_without_mask_path}")
    print(f"Output structure:")
    print(f"  {output_folder}/images/ - cropped images")
    print(f"  {output_folder}/masks/ - cropped masks")
    print(f"  {output_folder}/images_npy/ - cropped images as npy files")
    print(f"  {output_folder}/masks_npy/ - cropped masks as npy files")
    print(f"  {output_folder}/test_with_mask.txt - image-mask pairs with smoke (non-zero masks)")
    print(f"  {output_folder}/test_without_mask.txt - image-mask pairs without smoke (all-zero masks)")
    print(f"  {output_folder}/metadata.json - crop metadata")
