"""
Helper functions
"""
import json
import os
from PIL import Image
import numpy as np


def load_json(fpath):
    with open(fpath, "r") as f:
        return json.load(f)


def is_file_here(file_path):
    return os.path.isfile(file_path)


def convert_images_to_npy(input_dir, output_dir):
    """Convert all jpg/png images in input_dir to .npy files in output_dir. Skip if .npy file exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_dir, fname)
            npy_name = os.path.splitext(fname)[0] + ".npy"
            npy_path = os.path.join(output_dir, npy_name)
            if os.path.exists(npy_path):
                print(f"Skipping {npy_name} (already exists)")
                continue
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            np.save(npy_path, arr)
            print(f"Converted {fname} to {npy_name}")