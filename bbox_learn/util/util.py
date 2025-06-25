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


def draw_bbox_on_image(image_path, save_path, bboxes, color=(255, 0, 0), width=2):
    """
    Draw a list of bounding boxes on the image and save to a file.
    If h_image/w_image do not match, resize first.
    Create save directory if needed.
    Each bbox can have a 'color' field, otherwise the default color is used.
    Do nothing if save_path already exists.
    """
    if os.path.exists(save_path):
        print(f"Skipping {save_path} (already exists)")
        return
    if not bboxes:
        raise ValueError("bboxes list is empty")
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    target_w = bboxes[0]["w_image"]
    target_h = bboxes[0]["h_image"]
    if (img_w, img_h) != (target_w, target_h):
        img = img.resize((target_w, target_h))
    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        x1 = bbox["x_bbox"]
        y1 = bbox["y_bbox"]
        x2 = x1 + bbox["w_bbox"]
        y2 = y1 + bbox["h_bbox"]
        box_color = tuple(bbox["color"]) if "color" in bbox else color
        for i in range(width):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=box_color)
    img.save(save_path)
    print(f"Saved image with bbox(es) to {save_path}")