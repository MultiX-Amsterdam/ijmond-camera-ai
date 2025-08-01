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
import matplotlib.pyplot as plt
import torch
from torchvision.utils import (
    draw_bounding_boxes,
    draw_segmentation_masks
)
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


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
            alpha_int = int(255 * alpha) # Convert 0.0-1.0 to 0-255
        else:
            r, g, b = box_color[:3]
            alpha_int = int(255 * 0.5) # Default alpha

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


def plot(imgs, save_path, title=None, **imshow_kwargs):
    """
    This function is downloaded and edited from the following:
    https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    # Flatten all images into a single list for grid layout
    all_images = []
    for row in imgs:
        all_images.extend(row)

    # Calculate grid dimensions (max 4 images per row)
    max_cols = 4
    num_images = len(all_images)
    num_rows = (num_images + max_cols - 1) // max_cols  # Ceiling division
    num_cols = min(max_cols, num_images)

    # Create figure with 600 pixel width per image
    fig_width = num_cols * 6 # 6 inches per image (roughly 600 pixels at 100 DPI)
    fig_height = num_rows * 6 # Keep aspect ratio square

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False,
                           figsize=(fig_width, fig_height))

    # Plot each image in the grid
    for idx, img in enumerate(all_images):
        row_idx = idx // max_cols
        col_idx = idx % max_cols

        boxes = None
        masks = None
        if isinstance(img, tuple):
            img, target = img
            if isinstance(target, dict):
                boxes = target.get("boxes")
                masks = target.get("masks")
            elif isinstance(target, tv_tensors.BoundingBoxes):
                boxes = target
            elif isinstance(target, torch.Tensor):
                # Handle PyTorch tensor as bounding boxes
                boxes = target
            elif target is not None:
                # Try to handle other tensor-like objects
                try:
                    boxes = torch.as_tensor(target)
                except:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            # If target is None, boxes and masks remain None (no detection case)
        img = F.to_image(img)
        if img.dtype.is_floating_point and img.min() < 0:
            # Poor man's re-normalization for the colors to be OK-ish. This
            # is useful for images coming out of Normalize()
            img -= img.min()
            img /= img.max()

        img = F.to_dtype(img, torch.uint8, scale=True)
        if boxes is not None:
            img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)

        ax = axs[row_idx, col_idx]
        ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)

        if masks is not None:
            # Handle soft masks (float) and hard masks (bool)
            if masks.dtype == torch.bool:
                # Hard mask - use existing behavior
                img = draw_segmentation_masks(img, masks, colors=["green"] * masks.shape[0], alpha=0.25)
                ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            else:
                # Soft mask - display as heatmap overlay
                for mask_idx in range(masks.shape[0]):
                    mask = masks[mask_idx].cpu().numpy()

                    # Create heatmap overlay
                    img_height, img_width = img.shape[1], img.shape[2]

                    # Create a custom colormap where 0 values are transparent
                    import matplotlib.colors as colors
                    cmap = plt.cm.viridis.copy()
                    cmap.set_under('none')  # Make values below vmin transparent

                    # Set values threshold to be transparent
                    mask_display = mask.copy()
                    mask_display[mask_display < 0.1] = np.nan  # NaN values will be transparent

                    # Overlay the soft mask as a heatmap
                    heatmap = ax.imshow(mask_display, cmap=cmap, alpha=0.3,
                                      extent=[0, img_width, img_height, 0],
                                      vmin=0, vmax=1)

                    # Add colorbar for this mask
                    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Confidence', rotation=270, labelpad=15, fontsize=8)

                print(f"Soft mask: max confidence={masks.max():.3f}, displayed as heatmap")
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        ax.set_title(f"Image {idx}")

    # Hide unused subplots
    for idx in range(num_images, num_rows * num_cols):
        row_idx = idx // max_cols
        col_idx = idx % max_cols
        axs[row_idx, col_idx].axis('off')

    # Add title at the top of the plot if provided
    if title is not None and len(title) > 0:
        fig.suptitle(title[0], fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()
    # Adjust layout to make room for the main title and ensure even spacing
    plt.subplots_adjust(top=0.90, hspace=0.1, wspace=0.05)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()