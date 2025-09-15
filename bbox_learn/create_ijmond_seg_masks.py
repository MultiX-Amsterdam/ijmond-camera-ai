"""
Create segmentation masks for the IJmond Segmentation dataset using the provided annotation in MS COCO format.
"""
import json
import cv2
import numpy as np
import os
import sys
import argparse


def create_segmentation_masks(images_folder, coco_json_path, output_folder):
    """
    Create segmentation masks from MS COCO format annotations.

    Args:
        images_folder (str): Path to folder containing images
        coco_json_path (str): Path to MS COCO format annotation JSON file
        output_folder (str): Path to output folder for generated masks
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Load COCO annotations
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create mappings
    images = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}

    # Group annotations by image ID
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    print(f"Found {len(images)} images in annotations")
    print(f"Found {len(coco_data['annotations'])} annotations total")

    processed_count = 0
    masks_created = 0

    for img_id, img_info in images.items():
        image_filename = img_info['file_name']
        image_width = img_info['width']
        image_height = img_info['height']

        processed_count += 1
        print(f"Processing {processed_count}/{len(images)}: {image_filename}")

        # Check if image file exists
        image_path = os.path.join(images_folder, image_filename)
        if not os.path.exists(image_path):
            print(f"  Warning: Image file not found: {image_path}")
            continue

        # Get all annotations for this image
        annotations = annotations_by_image.get(img_id, [])

        # Create combined mask for all objects in the image (always create mask)
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        annotation_count = 0
        high_opacity_count = 0
        low_opacity_count = 0

        for ann in annotations:
            if 'segmentation' in ann and ann['segmentation']:
                annotation_count += 1
                category_id = ann.get('category_id', 1)  # Default to high opacity if not specified

                # Determine mask value based on category_id
                if category_id == 1:
                    mask_value = 255  # High opacity smoke
                    high_opacity_count += 1
                elif category_id == 2:
                    mask_value = 155  # Low opacity smoke
                    low_opacity_count += 1
                else:
                    mask_value = 255  # Default to high opacity for unknown categories
                    high_opacity_count += 1

                # Convert segmentation to mask
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    for polygon in ann['segmentation']:
                        if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                            # Reshape polygon to points format
                            polygon_points = np.array(polygon).reshape(-1, 2)
                            # Convert to integer coordinates
                            polygon_points = polygon_points.astype(np.int32)
                            # Fill polygon with appropriate mask value
                            cv2.fillPoly(mask, [polygon_points], mask_value)

                elif isinstance(ann['segmentation'], dict):
                    # RLE format - convert to polygon if needed
                    print(f"  Warning: RLE format not fully supported yet")
                    continue
                else:
                    print(f"  Warning: Unknown segmentation format")
                    continue

                # Add to combined mask (use max to prioritize higher values)
                combined_mask = np.maximum(combined_mask, mask)

        # Always save mask (even if no annotations)
        mask_filename = os.path.splitext(image_filename)[0] + '.png'
        mask_path = os.path.join(output_folder, mask_filename)

        success = cv2.imwrite(mask_path, combined_mask)
        if success:
            masks_created += 1
            if annotation_count > 0:
                opacity_info = []
                if high_opacity_count > 0:
                    opacity_info.append(f"{high_opacity_count} high opacity")
                if low_opacity_count > 0:
                    opacity_info.append(f"{low_opacity_count} low opacity")
                print(f"  Created mask with {annotation_count} annotations ({', '.join(opacity_info)}): {mask_path}")
            else:
                print(f"  Created empty mask (no annotations): {mask_path}")
        else:
            print(f"  Error: Failed to save mask: {mask_path}")

    print(f"\nProcessing complete!")
    print(f"Total images processed: {processed_count}")
    print(f"Masks successfully created: {masks_created}")
    print(f"Output folder: {output_folder}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python create_ijmond_seg_masks.py <images_folder> <coco_json_path> <output_folder>")
        print("Example: python create_ijmond_seg_masks.py dataset/ijmond_seg/test/images/ dataset/ijmond_seg/test/_annotations.coco.json dataset/ijmond_seg/test/masks/")
        sys.exit(1)

    images_folder = sys.argv[1]
    coco_json_path = sys.argv[2]
    output_folder = sys.argv[3]

    # Validate input paths
    if not os.path.exists(images_folder):
        print(f"Error: Images folder not found: {images_folder}")
        sys.exit(1)

    if not os.path.exists(coco_json_path):
        print(f"Error: COCO annotation file not found: {coco_json_path}")
        sys.exit(1)

    # Create segmentation masks
    create_segmentation_masks(images_folder, coco_json_path, output_folder)


if __name__ == "__main__":
    main()