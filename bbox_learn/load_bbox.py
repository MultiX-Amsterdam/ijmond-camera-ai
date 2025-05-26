import pandas as pd
import json


def load_bbox(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def build_video_url(record):
    """Build the video URL from record data"""
    camera_names = ["hoogovens", "kooksfabriek_1", "kooksfabriek_2"]
    video = record["video"]
    url_root = record["url_root"]

    src_url = (url_root +
               camera_names[video["camera_id"]] + "/" +
               video["url_part"] + "/" +
               video["file_name"] + ".mp4")
    return src_url


def build_image_url(record):
    """Build the image URL from record data"""
    url_root = record["url_root"]
    file_path = record["file_path"]
    image_file_name = record["image_file_name"]

    # Original image URL
    original_url = url_root + file_path + image_file_name

    # Image with bounding box URL
    bbox_image_file_name = image_file_name.replace("crop.png", "crop_with_bbox.png")
    bbox_url = url_root + file_path + bbox_image_file_name

    return original_url, bbox_url


if __name__ == "__main__":
    data = load_bbox("dataset/segmentation_labels_26_may_2025.json")

    # Print the last record in pretty format
    last_record = data['data'][-1]
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
    label_state_admin_values = set()
    label_state_values = set()

    for record in data['data']:
        label_state_admin_values.add(record.get('label_state_admin'))
        label_state_values.add(record.get('label_state'))

    print("\n" + "="*50)
    print("Unique label_state_admin values:")
    print(sorted(label_state_admin_values))

    print("\nUnique label_state values:")
    print(sorted(label_state_values))