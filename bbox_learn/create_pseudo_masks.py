"""
Create pseudo masks using SAM (Segment Anything Model) for the IJmond bounding boxes.
"""
import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
from segment_anything import (
    sam_model_registry,
    SamPredictor
)
from ijmond_bbox_dataset import IjmondBboxDataset
from util.util import plot
from tqdm import tqdm
from torch.utils.data import Subset


def get_model():
    """
    Get SAM (Segment Anything Model) for segmentation
    """
    # Load SAM model
    model_type = "vit_l" # "vit_b", "vit_l", "vit_h" for different model sizes
    sam = sam_model_registry[model_type](checkpoint="sam_vit_l_0b3195.pth")

    # Check if CUDA is available and move model to GPU
    if torch.cuda.is_available():
        sam = sam.cuda()
        print(f"SAM model moved to CUDA device")
    else:
        print(f"CUDA not available, using CPU")

    sam_predictor = SamPredictor(sam)

    return sam_predictor


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-sized bounding boxes
    """
    images = []
    targets = []

    for item in batch:
        img = item['img']
        images.append(img)

        # Prepare target dictionary
        target = {}
        if item['boxes'] is not None and len(item['boxes']) > 0:
            target['boxes'] = item['boxes']
            target['labels'] = torch.ones((len(item['boxes']),), dtype=torch.int64)
            # Check if there is a mask
            if 'masks' in item and item['masks'] is not None:
                target['masks'] = item['masks']
            else:
                target['masks'] = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
        else:
            # No objects in this image, which means no boxes or masks
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['masks'] = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)

        targets.append(target)

    return images, targets


def save_batch_masks(batch_masks, batch_idx, images, output_dir="dataset/ijmond_pseudo_masks"):
    """
    Save masks from a single batch to disk, including images, masks, and overlays in multiple formats
    """
    # Create all subdirectories
    img_npy_dir = os.path.join(output_dir, "img_npy")
    img_png_dir = os.path.join(output_dir, "img")
    mask_npy_dir = os.path.join(output_dir, "mask_npy")
    mask_png_dir = os.path.join(output_dir, "mask")
    overlay_dir = os.path.join(output_dir, "overlay")

    os.makedirs(img_npy_dir, exist_ok=True)
    os.makedirs(img_png_dir, exist_ok=True)
    os.makedirs(mask_npy_dir, exist_ok=True)
    os.makedirs(mask_png_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    for mask_info in batch_masks:
        image_idx = mask_info['image_idx']
        image_in_batch = mask_info['image_in_batch']

        # Get the original image
        image = images[image_in_batch]

        # Convert image tensor to numpy
        if image.max() <= 1.0:
            img_numpy = (image * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        else:
            img_numpy = image.to(torch.uint8).permute(1, 2, 0).cpu().numpy()

        # Save image as .npy
        img_npy_filename = f"img_{image_idx:04d}.npy"
        img_npy_filepath = os.path.join(img_npy_dir, img_npy_filename)
        np.save(img_npy_filepath, img_numpy)

        # Save image as .png
        img_png_filename = f"img_{image_idx:04d}.png"
        img_png_filepath = os.path.join(img_png_dir, img_png_filename)
        img_pil = Image.fromarray(img_numpy)
        img_pil.save(img_png_filepath)

        # Save mask files for each bounding box
        if len(mask_info['masks']) > 0:
            for mask_idx, mask in enumerate(mask_info['masks']):
                mask_numpy = mask.cpu().numpy().astype(np.uint8)

                # Save mask as .npy
                mask_npy_filename = f"mask_img_{image_idx:04d}_box_{mask_idx:02d}.npy"
                mask_npy_filepath = os.path.join(mask_npy_dir, mask_npy_filename)
                np.save(mask_npy_filepath, mask_numpy)

                # Save mask as .png (convert boolean mask to 0-255)
                mask_png_filename = f"mask_img_{image_idx:04d}_box_{mask_idx:02d}.png"
                mask_png_filepath = os.path.join(mask_png_dir, mask_png_filename)
                mask_png = (mask_numpy * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_png, mode='L')
                mask_pil.save(mask_png_filepath)

                # Create overlay image (mask overlaid on original image)
                overlay_filename = f"overlay_img_{image_idx:04d}_box_{mask_idx:02d}.png"
                overlay_filepath = os.path.join(overlay_dir, overlay_filename)

                # Create colored overlay while preserving original image colors and intensity
                overlay_img = img_numpy.copy().astype(np.float32)

                # Create green overlay only where mask is True
                mask_3d = np.stack([mask_numpy, mask_numpy, mask_numpy], axis=2)
                green_overlay = np.zeros_like(overlay_img)
                green_overlay[:, :, 1] = mask_numpy * 255  # Green channel only

                # Blend: keep original image where mask is False, blend with green where mask is True
                alpha = 0.2  # Transparency for the green overlay
                overlay_img = np.where(mask_3d,
                                     overlay_img * (1 - alpha) + green_overlay * alpha,
                                     overlay_img)

                overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
                overlay_pil = Image.fromarray(overlay_img)
                overlay_pil.save(overlay_filepath)


def main():
    metadata_path = "dataset/ijmond_bbox/filtered_bbox_labels_4_july_2025.json"
    root_dir = "dataset/ijmond_bbox/img_npy/"

    dataset = IjmondBboxDataset(metadata_path, root_dir)
    print(f"Dataset size: {len(dataset)}")

    debug = False # Set to True for debugging with limited samples
    n = 8 # Max number of samples to plot when debugging
    sam_model = get_model()
    print(f"Model loaded successfully!")
    print(f"SAM model type: {type(sam_model)}")

    # Filter dataset to only include samples with bounding boxes
    print("Filtering dataset to only include samples with bounding boxes...")
    sample_ids_with_boxes = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['boxes'] is not None and len(sample['boxes']) > 0:
            sample_ids_with_boxes.append(i)
    print(f"Found {len(sample_ids_with_boxes)} samples with bounding boxes out of {len(dataset)} total samples")

    # Create a subset dataset with only samples that have boxes
    filtered_dataset = Subset(dataset, sample_ids_with_boxes)

    # Create a DataLoader for testing with filtered dataset
    dataloader = DataLoader(
        filtered_dataset,
        batch_size=20,
        shuffle=False, # Do not shuffle for consistent processing
        collate_fn=collate_fn,
        num_workers=0
    )

    # Process all batches from the DataLoader
    plot_data = []
    all_metadata = []  # Only store metadata, not the actual masks
    total_processed = 0
    total_masks_generated = 0
    images_with_masks = 0

    # Create output directories
    output_dir = "dataset/ijmond_pseudo_masks"
    img_npy_dir = os.path.join(output_dir, "img_npy")
    img_png_dir = os.path.join(output_dir, "img")
    mask_npy_dir = os.path.join(output_dir, "mask_npy")
    mask_png_dir = os.path.join(output_dir, "mask")
    overlay_dir = os.path.join(output_dir, "overlay")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_npy_dir, exist_ok=True)
    os.makedirs(img_png_dir, exist_ok=True)
    os.makedirs(mask_npy_dir, exist_ok=True)
    os.makedirs(mask_png_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    print(f"Created/verified output directories:")
    print(f"  Main: {output_dir}")
    print(f"  Images (NPY): {img_npy_dir}")
    print(f"  Images (PNG): {img_png_dir}")
    print(f"  Masks (NPY): {mask_npy_dir}")
    print(f"  Masks (PNG): {mask_png_dir}")
    print(f"  Overlays: {overlay_dir}")

    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Processing batches")):
        print(f"\nProcessing batch {batch_idx + 1}")
        print(f"Number of images in batch: {len(images)}")
        print(f"Number of targets in batch: {len(targets)}")

        # Verify that all data points have bounding boxes
        for i, target in enumerate(targets):
            num_boxes = len(target['boxes'])
            print(f"Batch {batch_idx}, Target {i}: {num_boxes} boxes, {len(target['labels'])} labels")
            if num_boxes == 0:
                print(f"Error: Batch {batch_idx}, Target {i} has no bounding boxes!")
                continue

        # Process each data point with SAM in this batch
        batch_masks = []  # Store masks for this batch only
        batch_range = range(len(images)) if not debug else range(min(n - total_processed, len(images)))
        for i in batch_range:
            if debug and total_processed >= n:
                break
            print(f"Processing Image {total_processed + 1} (Batch {batch_idx + 1}, Image {i + 1}):")

            # Get the image and ground truth bounding box
            image = images[i]
            target = targets[i]
            gt_boxes = target['boxes']
            print(f"Image shape: {image.shape}")
            print(f"Ground truth boxes: {len(gt_boxes)}")

            # Process all bounding boxes for this image
            image_masks = []
            for box_idx, gt_box in enumerate(gt_boxes):
                print(f"Processing box {box_idx + 1}/{len(gt_boxes)}: [{gt_box[0]:.1f}, {gt_box[1]:.1f}, {gt_box[2]:.1f}, {gt_box[3]:.1f}]")

                # Prepare image for SAM (expects RGB in range [0, 255])
                if image.max() <= 1.0:
                    img_input = (image * 255).to(torch.uint8)
                else:
                    img_input = image.to(torch.uint8)

                # Set image for SAM - convert from CHW to HWC format and to numpy
                img_input_numpy = img_input.permute(1, 2, 0).cpu().numpy()

                # SAM expects HWC format as numpy array
                print(f"Image input numpy shape: {img_input_numpy.shape}")
                sam_model.set_image(img_input_numpy)

                # Use bounding box as prompt for SAM
                box_prompt = gt_box.cpu().numpy() # Ensure it's on CPU and convert to numpy
                print(f"Box prompt: {box_prompt}")

                # Generate mask using box prompt
                masks, scores, logits = sam_model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_prompt,
                    multimask_output=False,
                )

                pred_mask = None
                if len(masks) > 0:
                    # Use the first (and only) mask from SAM
                    pred_mask = masks[0]
                    pred_score = scores[0]
                    print(f"SAM prediction: Mask shape={pred_mask.shape}, Score={pred_score:.3f}")
                    print(f"Mask area: {pred_mask.sum()} pixels")
                    image_masks.append(torch.from_numpy(pred_mask).bool())
                else:
                    print(f"SAM prediction: No masks generated for box {box_idx + 1}")

            # Store results for this image (including masks for batch processing)
            mask_info = {
                'image_idx': total_processed,
                'batch_idx': batch_idx,
                'image_in_batch': i,
                'image_shape': image.shape,
                'num_boxes': len(gt_boxes),
                'masks': image_masks,  # Keep masks for this batch
                'boxes': gt_boxes
            }
            batch_masks.append(mask_info)

            # Store metadata only (no masks to save memory)
            metadata_info = {
                'image_idx': total_processed,
                'batch_idx': batch_idx,
                'image_in_batch': i,
                'image_shape': list(image.shape),
                'num_boxes': len(gt_boxes),
                'num_masks_generated': len(image_masks),
                'boxes': gt_boxes.tolist() if len(gt_boxes) > 0 else []
            }
            all_metadata.append(metadata_info)

            # Update statistics
            total_masks_generated += len(image_masks)
            if len(image_masks) > 0:
                images_with_masks += 1

            # Add to plot data if we haven't reached the limit
            if len(plot_data) < n:
                if len(image_masks) > 0:
                    # Combine all masks for this image into a single tensor
                    combined_masks = torch.stack(image_masks)
                    plot_dict = {
                        "boxes": gt_boxes,
                        "masks": combined_masks
                    }
                else:
                    # Still plot ground truth boxes even if no masks were generated
                    plot_dict = {"boxes": gt_boxes}
                plot_data.append((image, plot_dict))

            total_processed += 1

        # Save masks from this batch to disk and free memory
        if batch_masks:
            save_batch_masks(batch_masks, batch_idx, images, output_dir)
            total_files_saved = len(batch_masks) * 2  # img_npy + img_png per image
            total_mask_files = sum(len(info['masks']) for info in batch_masks) * 3  # npy + png + overlay per mask
            print(f"Saved {len(batch_masks)} images ({total_files_saved} files) and {sum(len(info['masks']) for info in batch_masks)} masks ({total_mask_files} files) from batch {batch_idx + 1}")

            # Clear batch masks from memory
            batch_masks.clear()
            if debug and total_processed >= n:
                break

    print(f"\nProcessing complete!")
    print(f"Total images processed: {total_processed}")
    print(f"Total masks generated: {total_masks_generated}")
    print(f"Images with at least one mask: {images_with_masks}")

    # Create train.txt file in MS COCO format
    train_txt_file = os.path.join(output_dir, "train.txt")
    with open(train_txt_file, 'w') as f:
        for metadata_info in all_metadata:
            if metadata_info['num_masks_generated'] > 0:
                image_idx = metadata_info['image_idx']
                # For each mask generated for this image
                for mask_idx in range(metadata_info['num_masks_generated']):
                    img_path = f"img_npy/img_{image_idx:04d}.npy"
                    mask_path = f"mask_npy/mask_img_{image_idx:04d}_box_{mask_idx:02d}.npy"
                    f.write(f"{img_path} {mask_path}\n")

    print(f"Created train.txt file with {total_masks_generated} image-mask pairs")

    print(f"\nFinal file structure:")
    print(f"  {output_dir}/train.txt - training pairs in MS COCO format")
    print(f"  {img_npy_dir}/ - original images as .npy files")
    print(f"  {img_png_dir}/ - original images as .png files")
    print(f"  {mask_npy_dir}/ - generated masks as .npy files")
    print(f"  {mask_png_dir}/ - generated masks as .png files")
    print(f"  {overlay_dir}/ - mask overlays on images as .png files")

    # Plot the results
    if len(plot_data) > 0:
        print(f"\nPlotting results for {len(plot_data)} data points...")
        plot_filename = "debug_plot_pseudo_masks.png"
        plot([plot_data], plot_filename, title=["Ground Truth Box (Yellow) + SAM Mask (Green)"])


if __name__ == "__main__":
    main()