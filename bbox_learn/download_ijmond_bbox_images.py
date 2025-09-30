"""
This script downloads images from a JSON file containing bounding box labels.
"""
import json
import os
import requests
import sys
from collections import Counter


def load_bbox(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def build_video_url(record):
    """Build the video URL from record data."""
    camera_names = ["hoogovens", "kooksfabriek_1", "kooksfabriek_2"]
    video = record["video"]
    url_root = record["url_root"]

    src_url = (url_root +
               camera_names[video["camera_id"]] + "/" +
               video["url_part"] + "/" +
               video["file_name"] + ".mp4")
    return src_url


def build_image_url(record):
    """Build the image URL from record data."""
    url_root = record["url_root"]
    file_path = record["file_path"]
    image_file_name = record["image_file_name"]

    # Original image URL
    original_url = url_root + file_path + image_file_name

    # Image with bounding box URL
    bbox_image_file_name = image_file_name.replace("crop.png", "crop_with_bbox.png")
    bbox_url = url_root + file_path + bbox_image_file_name

    return original_url, bbox_url


def download_all_images(data, folder):
    """Download all original image URLs to the specified folder, naming by record id. Skip if file exists."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    for idx, record in enumerate(data['data']):
        image_url, _ = build_image_url(record)
        record_id = record.get('id')
        ext = os.path.splitext(record['image_file_name'])[1]
        image_file_name = f"{record_id}{ext}"
        save_path = os.path.join(folder, image_file_name)
        if os.path.exists(save_path):
            print(f"Skipping {image_file_name} (already exists)")
            continue
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {image_file_name} ({idx+1}/{len(data['data'])})")
        except Exception as e:
            print(f"Failed to download {image_url}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_ijmond_bbox_images.py <bbox_labels_file.json> <image_folder>")
        print("Example: python download_ijmond_bbox_images.py dataset/ijmond_bbox/bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img")
        sys.exit(1)

    bbox_file = sys.argv[1]
    image_folder = sys.argv[2]

    if not os.path.exists(image_folder):
        print(f"Creating folder: {image_folder}")
        os.makedirs(image_folder)

    if not os.path.exists(bbox_file):
        print(f"Error: File '{bbox_file}' not found.")
        sys.exit(1)

    data = load_bbox(bbox_file)

    # Example: Download all original images to the specified folder
    download_all_images(data, image_folder)

    # Print the last record in pretty format
    last_record = data['data'][-1]
    print("="*50)
    print("Last record (pretty format):")
    print(json.dumps(last_record, indent=2))

    # Build and print URLs for the last record
    print("\n" + "="*50)
    print("URLs for last record:")
    video_url = build_video_url(last_record)
    image_url, bbox_image_url = build_image_url(last_record)

    print(f"Video URL: {video_url}")
    print(f"Original Image URL: {image_url}")
    print(f"AI-Generated Bounding Box Image URL: {bbox_image_url}")

    # Get unique label_state_admin and label_state values
    label_state_admin_values = []
    label_state_values = []

    for record in data['data']:
        label_state_admin_values.append(record.get('label_state_admin'))
        label_state_values.append(record.get('label_state'))

    print("\n" + "="*50)
    print("Unique label_state_admin values:")
    print(sorted(set(label_state_admin_values)))

    # Print distribution of label_state_admin values with counts
    print("\nlabel_state_admin value distribution:")
    label_state_admin_counter = Counter(label_state_admin_values)
    for label, count in label_state_admin_counter.items():
        print(f"{label}: {count}")

    print("\nUnique label_state values:")
    print(sorted(set(label_state_values)))

    # Print distribution of label_state values with counts
    print("\nlabel_state value distribution:")
    label_state_counter = Counter(label_state_values)
    for label, count in label_state_counter.items():
        print(f"{label}: {count}")