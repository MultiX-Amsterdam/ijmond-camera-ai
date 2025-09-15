import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import numpy as np
from util.util import (
    load_pair_txt,
    is_file_here
)
from torchvision.transforms import v2
from util.util import plot
import random


class SmokeDataset(Dataset):
    def __init__(self, metadata_path, root_dir, transform=None):
        """
        metadata_path (string): the full path to the metadata json file
        root_dir (string): the root directory that stores images and ground truth segmentation masks in .npy format
        transform (callable, optional): optional transform v2 (torchvision.transforms.v2) to be applied on an image and bounding boxes.
        """
        self.metadata = load_pair_txt(metadata_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        v = self.metadata[idx]

        # Construct file paths for image
        img_file_path = os.path.join(self.root_dir, f"{v[0]}")
        if not is_file_here(img_file_path):
            raise ValueError("Cannot find file: %s" % (img_file_path))

        # Load image from .npy file
        img = torch.from_numpy(np.load(img_file_path).astype(np.uint8))

        # Change dimensions fro (H, W, C) to (C, H, W)
        img = img.permute(2, 0, 1)

        if v[1] == "None":
            # For unlabeled data, return only the image
            if self.transform:
                img = self.transform(img)
            return {"img": img, "gt": None}
        else:
            # Construct file paths for ground truth mask
            gt_file_path = os.path.join(self.root_dir, f"{v[1]}")
            if not is_file_here(gt_file_path):
                raise ValueError("Cannot find file: %s" % (gt_file_path))
            # Load ground truth segmentation mask from .npy file
            gt = torch.from_numpy(np.load(gt_file_path).astype(np.uint8))
            # Convert to tv_tensors.Mask for proper transform handling
            gt = tv_tensors.Mask(gt)
            # Transform image
            if self.transform:
                img, gt = self.transform(img, gt)
            return {"img": img, "gt": gt}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python smoke_dataset.py <metadata_path> <root_dir> <dataset_name>")
        print("Example: python smoke_dataset.py dataset/smoke5k/test/test.txt dataset/smoke5k/test/ smoke5k_test")
        print("Example: python smoke_dataset.py dataset/smoke5k/train/train.txt dataset/smoke5k/train/ smoke5k_train")
        print("Example: python smoke_dataset.py dataset/ijmond_pseudo_masks/train_with_mask.txt dataset/ijmond_pseudo_masks/ ijmond_pseudo_mask_with_mask")
        print("Example: python smoke_dataset.py dataset/ijmond_pseudo_masks/train_without_mask.txt dataset/ijmond_pseudo_masks/ ijmond_pseudo_mask_without_mask")
        print("Example: python smoke_dataset.py dataset/ijmond_vid/unlabeled.txt dataset/ijmond_vid/ ijmond_vid_unlabeled")
        print("Example: python smoke_dataset.py dataset/ijmond_seg/test/cropped/test_with_mask.txt dataset/ijmond_seg/test/cropped/ ijmond_seg_cropped_with_mask")
        print("Example: python smoke_dataset.py dataset/ijmond_seg/test/cropped/test_without_mask.txt dataset/ijmond_seg/test/cropped/ ijmond_seg_cropped_without_mask")
        sys.exit(1)

    metadata_path = sys.argv[1]
    root_dir = sys.argv[2]
    dataset_name = sys.argv[3]

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file '{metadata_path}' not found.")
        sys.exit(1)

    if not os.path.exists(root_dir):
        print(f"Error: Root directory '{root_dir}' not found.")
        sys.exit(1)

    if dataset_name is None or dataset_name.strip() == "":
        print("Error: dataset_name cannot be empty.")
        sys.exit(1)

    D = SmokeDataset(metadata_path, root_dir)
    print(f"Dataset size: {len(D)}")

    # Randomly select 8 samples to plot
    random.seed(42)  # For reproducible results
    num_samples = min(8, len(D))  # Use 8 or dataset size if smaller
    sample_indices = random.sample(range(len(D)), num_samples)

    print(f"Randomly selected {num_samples} samples for plotting: {sample_indices}")

    # Collect the selected samples
    selected_samples = []
    for idx in sample_indices:
        sample = D[idx]
        selected_samples.append(sample)
        print(f"Sample {idx}: img shape: {sample['img'].shape}")
        if sample['gt'] is not None:
            print(f"Sample {idx}: gt shape: {sample['gt'].shape}")
            unique_values = torch.unique(sample['gt'])
            print(f"Sample {idx}: unique mask values: {unique_values}")

    # Process masks for plotting
    data_to_plot = []
    for sample in selected_samples:
        img = sample['img']
        gt = sample['gt']

        if gt is None:
            # No mask
            data_to_plot.append((img, {"masks": None}))
        else:
            unique_values = torch.unique(gt)
            print(f"Processing mask with unique values: {unique_values}")

            if len(unique_values) <= 2 and (0 in unique_values):
                # Binary mask case (0 and possibly 255 or 1)
                # Convert to boolean mask
                binary_mask = (gt > 0).to(torch.bool)
                data_to_plot.append((img, {"masks": binary_mask}))
                print("  -> Treated as single binary mask")
            else:
                # Multi-level mask case (e.g., 0, 155, 255)
                # Create separate binary masks for each non-zero value
                masks_list = []
                for value in unique_values:
                    if value > 0:  # Skip background (0)
                        binary_mask = (gt == value).to(torch.bool)
                        masks_list.append(binary_mask)

                if len(masks_list) == 1:
                    # Only one non-zero level, treat as single mask
                    data_to_plot.append((img, {"masks": masks_list[0]}))
                    print("  -> Treated as single binary mask")
                else:
                    # Multiple levels, create stack of masks
                    masks_tensor = torch.stack(masks_list)
                    data_to_plot.append((img, {"masks": masks_tensor}))
                    print(f"  -> Treated as {len(masks_list)} separate binary masks")
    plot(data_to_plot, f"debug_plot_{dataset_name}.png")

    # Keep the original sample checking for compatibility
    s = selected_samples[0] if selected_samples else None
    if s:
        print(f"First sample img values: {s['img'][0, 0:5, 0:5]}")
        if s['gt'] is not None:
            print(f"First sample gt values: {s['gt'][0:5, 0:5]}")

    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.01),
        v2.ToDtype({torch.uint8: torch.float32, tv_tensors.Mask: torch.uint8, "others": None}, scale=True),
        v2.Resize((400, 400), antialias=True)
    ])

    DT = SmokeDataset(metadata_path, root_dir, transform=transforms)

    # Apply transforms to the same selected samples
    transformed_samples = []
    for idx in sample_indices:
        sample = DT[idx]
        transformed_samples.append(sample)

    # Process transformed masks for plotting
    transformed_data_to_plot = []
    for sample in transformed_samples:
        img = sample['img']
        gt = sample['gt']

        if gt is None:
            # No mask
            transformed_data_to_plot.append((img, {"masks": None}))
        else:
            unique_values = torch.unique(gt)

            if len(unique_values) <= 2 and (0 in unique_values):
                # Binary mask case (0 and possibly 255 or 1)
                # Convert to boolean mask
                binary_mask = (gt > 0).to(torch.bool)
                transformed_data_to_plot.append((img, {"masks": binary_mask}))
            else:
                # Multi-level mask case (e.g., 0, 155, 255)
                # Create separate binary masks for each non-zero value
                masks_list = []
                for value in unique_values:
                    if value > 0:  # Skip background (0)
                        binary_mask = (gt == value).to(torch.bool)
                        masks_list.append(binary_mask)

                if len(masks_list) == 1:
                    # Only one non-zero level, treat as single mask
                    transformed_data_to_plot.append((img, {"masks": masks_list[0]}))
                else:
                    # Multiple levels, create stack of masks
                    masks_tensor = torch.stack(masks_list)
                    transformed_data_to_plot.append((img, {"masks": masks_tensor}))

    # Plot the images and masks
    plot(transformed_data_to_plot, f"debug_plot_{dataset_name}_transformed.png")

    # Show transform comparison for first sample
    st = transformed_samples[0]
    print(f"Sample img shape after transform: {st['img'].shape}")
    print(f"Sample img values after transform: {st['img'][0, 0:5, 0:5]}")
    if st['gt'] is not None:
        print(f"Sample gt shape after transform: {st['gt'].shape}")
        print(f"Sample gt values after transform: {st['gt'][0:5, 0:5]}")