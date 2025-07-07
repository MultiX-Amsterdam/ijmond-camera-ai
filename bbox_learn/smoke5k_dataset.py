import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import numpy as np
from util.util import (
    load_json,
    is_file_here
)
from torchvision.transforms import v2
from util.helpers import plot


class Smoke5kDataset(Dataset):
    def __init__(self, metadata_path, img_root_dir, gt_root_dir=None, transform=None):
        """
        metadata_path (string): the full path to the metadata json file
        img_root_dir (string): the root directory that stores images in .npy format
        gt_root_dir (string, optional): the root directory that stores ground truth segmentation masks in .npy format
        transform (callable, optional): optional transform v2 (torchvision.transforms.v2) to be applied on an image and bounding boxes.
        """
        self.metadata = load_json(metadata_path)
        self.img_root_dir = img_root_dir
        self.gt_root_dir = gt_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        v = self.metadata[idx]

        img_file_path = os.path.join(self.img_root_dir, f"{v["id"]}.npy")
        if not is_file_here(img_file_path):
            raise ValueError("Cannot find file: %s" % (img_file_path))

        # Load image from .npy file
        img = torch.from_numpy(np.load(img_file_path).astype(np.uint8))

        # Change dimensions fro (H, W, C) to (C, H, W)
        img = img.permute(2, 0, 1)

        gt = None
        if self.gt_root_dir is not None:
            gt_file_path = os.path.join(self.gt_root_dir, f"{v["id"]}.npy")
            if not is_file_here(gt_file_path):
                raise ValueError("Cannot find file: %s" % (gt_file_path))
            # Load ground truth segmentation mask from .npy file
            gt = torch.from_numpy(np.load(gt_file_path).astype(np.uint8))
            gt = gt.permute(2, 0, 1)
            # Convert to tv_tensors.Mask for proper transform handling
            gt = tv_tensors.Mask(gt)

        # Transform image
        if self.transform:
            if gt is None:
                img = self.transform(img)
            else:
                img, gt = self.transform(img, gt)

        return {"img": img, "gt": gt}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python smoke5k_dataset.py <metadata_path> <img_root_dir> <gt_root_dir>")
        print("Example: python smoke5k_dataset.py dataset/smoke5k/smoke5k_metadata_train.json dataset/smoke5k/train/img_npy/ dataset/smoke5k/train/gt_npy/")
        print("Example: python smoke5k_dataset.py dataset/smoke5k/smoke5k_metadata_test.json dataset/smoke5k/test/img_npy/ dataset/smoke5k/test/gt_npy/")
        sys.exit(1)

    metadata_path = sys.argv[1]
    img_root_dir = sys.argv[2]
    gt_root_dir = sys.argv[3]

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file '{metadata_path}' not found.")
        sys.exit(1)

    if not os.path.exists(img_root_dir):
        print(f"Error: Root directory '{img_root_dir}' not found.")
        sys.exit(1)

    D = Smoke5kDataset(metadata_path, img_root_dir, gt_root_dir=gt_root_dir)
    print(f"Dataset size: {len(D)}")
    s = None
    for d in D:
        print(f"Sample img shape: {d['img'].shape}")
        print(f"Sample img values: {d['img'][0, 0:5, 0:5]}")
        print(f"Sample gt shape: {d['gt'].shape}")
        print(f"Sample gt values: {d['gt'][0, 0:5, 0:5]}")
        s = d
        break

    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.01),
        v2.ToDtype({torch.uint8: torch.float32, tv_tensors.Mask: torch.uint8, "others": None}, scale=True),
        v2.Resize((400, 400), antialias=True)
    ])

    DT = Smoke5kDataset(metadata_path, img_root_dir, gt_root_dir=gt_root_dir, transform=transforms)
    for st in DT:
        print(f"Sample img shape after transform: {st['img'].shape}")
        print(f"Sample img values after transform: {st['img'][0, 0:5, 0:5]}")
        print(f"Sample gt shape after transform: {st['gt'].shape}")
        print(f"Sample gt values after transform: {st['gt'][0, 0:5, 0:5]}")
        plot([(s['img']), (st['img'])], "debug_plot_smoke5k_img.png")
        plot([(s['gt']), (st['gt'])], "debug_plot_smoke5k_gt.png")
        break