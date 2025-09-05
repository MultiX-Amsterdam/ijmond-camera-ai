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
    s = None
    for d in D:
        print(f"Sample img shape: {d['img'].shape}")
        print(f"Sample img values: {d['img'][0, 0:5, 0:5]}")
        if d['gt'] is not None:
            print(f"Sample gt shape: {d['gt'].shape}")
            print(f"Sample gt values: {d['gt'][0:5, 0:5]}")
        s = d
        break

    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.01),
        v2.ToDtype({torch.uint8: torch.float32, tv_tensors.Mask: torch.uint8, "others": None}, scale=True),
        v2.Resize((400, 400), antialias=True)
    ])

    DT = SmokeDataset(metadata_path, root_dir, transform=transforms)
    for st in DT:
        print(f"Sample img shape after transform: {st['img'].shape}")
        print(f"Sample img values after transform: {st['img'][0, 0:5, 0:5]}")
        if st['gt'] is not None:
            print(f"Sample gt shape after transform: {st['gt'].shape}")
            print(f"Sample gt values after transform: {st['gt'][0:5, 0:5]}")
        plot([(s['img']), (st['img'])], f"debug_plot_{dataset_name}_img.png")
        if s['gt'] is not None and st['gt'] is not None:
            plot([(s['gt']), (st['gt'])], f"debug_plot_{dataset_name}_gt.png")
        break