import cv2
import numpy as np
import os
import sys
import json


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
    if len(sys.argv) != 3:
        print("Usage: python crop_ijmond_seg_dataset.py <input_folder> <output_folder>")
        print("Example: python crop_ijmond_seg_dataset.py dataset/ijmond_seg/test/images/ dataset/ijmond_seg/test/cropped/")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        sys.exit(1)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = [f for f in os.listdir(input_folder)
                   if any(f.lower().endswith(ext) for ext in image_extensions)]

    if not image_files:
        print(f"No image files found in '{input_folder}'.")
        sys.exit(1)

    # Metadata list to store crop information
    metadata = []

    processed_count = 0
    total_crops = 0
    successful_crops = 0

    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        processed_count += 1
        print(f"Processing {processed_count}/{len(image_files)}: {filename}")

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

                    # Create output filename
                    cropped_filename = f"{name_without_ext}_crop_{view_id}{file_ext}"
                    output_path = os.path.join(output_folder, cropped_filename)

                    print(f"    Crop {view_id}: x={x}, y={y}, width={width}, height={height}")

                    # Perform the crop
                    success = save_cropped_image(input_path, output_path, x, y, width, height)

                    if success:
                        successful_crops += 1

                        # Add metadata entry
                        metadata_entry = {
                            "original_file_name": filename,
                            "cropped_file_name": cropped_filename,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "view_id": view_id,
                            "camera_name": camera_name
                        }
                        metadata.append(metadata_entry)

                break  # Only apply the first matching pattern

        if not crops_applied:
            print(f"  No crop pattern matched for '{filename}', skipping...")

    # Save metadata to JSON file
    metadata_path = os.path.join(output_folder, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Total crops attempted: {total_crops}")
    print(f"Successfully created crops: {successful_crops}")
    print(f"Metadata saved to: {metadata_path}")
