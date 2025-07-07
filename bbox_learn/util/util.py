"""
Helper functions
"""
import json
import os
from PIL import (
    Image,
    ImageDraw
)
import numpy as np


def load_json(fpath):
    with open(fpath, "r") as f:
        return json.load(f)


def save_json(content, fpath):
    with open(fpath, "w") as f:
        json.dump(content, f)


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


def draw_bbox_on_image(image_path, save_path, bboxes, color=(255, 0, 0, 0.5), width=2):
    """
    Draw a list of bounding boxes on the image and save to a file.
    If h_image/w_image do not match, resize first.
    Create save directory if needed.
    Each bbox can have a 'color' field, otherwise the default color is used.
    The color format is (R, G, B, A) where A is the alpha channel (0.0-1.0).
    Do nothing if save_path already exists.
    """
    if os.path.exists(save_path):
        print(f"Skipping {save_path} (already exists)")
        return
    if not bboxes:
        raise ValueError("bboxes list is empty")

    img = Image.open(image_path).convert("RGBA")
    img_w, img_h = img.size
    target_w = bboxes[0]["w_image"]
    target_h = bboxes[0]["h_image"]
    if (img_w, img_h) != (target_w, target_h):
        img = img.resize((target_w, target_h))

    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a transparent overlay for drawing
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for bbox in bboxes:
        x1 = bbox["x_bbox"]
        y1 = bbox["y_bbox"]
        x2 = x1 + bbox["w_bbox"]
        y2 = y1 + bbox["h_bbox"]
        box_color = bbox["color"] if "color" in bbox else color

        # Extract alpha value and convert to 0-255 range
        if len(box_color) == 4:
            r, g, b, alpha = box_color
            alpha_int = int(255 * alpha)  # Convert 0.0-1.0 to 0-255
        else:
            r, g, b = box_color[:3]
            alpha_int = int(255 * 0.5)  # Default alpha

        # Draw border with transparency (no fill)
        border_color = (r, g, b, alpha_int)
        for i in range(width):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=border_color)

    # Composite the overlay onto the original image
    result = Image.alpha_composite(img, overlay)

    # Convert back to RGB for saving
    result = result.convert("RGB")
    result.save(save_path)
    print(f"Saved image with transparent bbox(es) to {save_path}")