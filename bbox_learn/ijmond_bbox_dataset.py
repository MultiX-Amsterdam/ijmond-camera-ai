import os
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import numpy as np
from util.util import (
    load_json,
    is_file_here
)


class IjmondBboxDataset(Dataset):
    def __init__(self, metadata_path=None, root_dir=None, transform=None):
        """
        metadata_path (string): the full path to the metadata json file (after running the "filter_aggr_bbox.py" script)
        root_dir (string): the root directory that stores images in .npy format
        transform (callable, optional): optional transform to be applied on an image
        """
        self.metadata = load_json(metadata_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        v = self.metadata[idx]

        file_path = os.path.join(self.root_dir, f"{v["id"]}.npy")
        if not is_file_here(file_path):
            raise ValueError("Cannot find file: %s" % (file_path))
        img = np.load(file_path).astype(np.uint8)

        # Transform image
        if self.transform:
            img = self.transform(img)

        return {"img": img, "bbox": v["bbox"]}


if __name__ == "__main__":
    metadata_path = "dataset/ijmond_bbox/filtered_bbox_labels_26_may_2025.json"
    root_dir = "dataset/ijmond_bbox/img_npy/"
    dataset = IjmondBboxDataset(metadata_path=metadata_path, root_dir=root_dir)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[5]
    print(f"Sample img shape: {sample['img'].shape}")
    print(f"Sample bbox: {sample['bbox']}")