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


class IjmondBboxDataset(Dataset):
    def __init__(self, metadata_path=None, root_dir=None, transform=None):
        """
        metadata_path (string): the full path to the metadata json file (after running the "filter_aggr_bbox.py" script)
        root_dir (string): the root directory that stores images in .npy format
        transform (callable, optional): optional transform v2 (torchvision.transforms.v2) to be applied on an image and bounding boxes.
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

        # Load image from .npy file
        img = torch.from_numpy(np.load(file_path).astype(np.uint8))

        # Change dimensions fro (H, W, C) to (C, H, W)
        img = img.permute(2, 0, 1)

        # Load bounding boxes and convert to xyxy format
        b_list = v["bbox"]
        if b_list == None:
            boxes = None  # No bounding boxes available
        else:
            boxes = []
            for b in b_list:
                boxes.append([b["x_bbox"], b["y_bbox"], b["w_bbox"], b["h_bbox"]])
            boxes = tv_tensors.BoundingBoxes(
                boxes,
                format="XYWH",
                canvas_size=(b_list[0]["h_image"], b_list[0]["w_image"])
            )
            xxwh_to_xyxy = v2.ConvertBoundingBoxFormat("XYXY")
            boxes = xxwh_to_xyxy(boxes)

        # Transform image
        if self.transform:
            if boxes is None:
                img = self.transform(img)
            else:
                img, boxes = self.transform(img, boxes)

        return {"img": img, "boxes": boxes}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ijmond_bbox_dataset.py <metadata_path> <root_dir>")
        print("Example: python ijmond_bbox_dataset.py dataset/ijmond_bbox/filtered_bbox_labels_4_july_2025.json dataset/ijmond_bbox/img_npy/")
        sys.exit(1)

    metadata_path = sys.argv[1]
    root_dir = sys.argv[2]

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file '{metadata_path}' not found.")
        sys.exit(1)

    if not os.path.exists(root_dir):
        print(f"Error: Root directory '{root_dir}' not found.")
        sys.exit(1)

    D = IjmondBboxDataset(metadata_path=metadata_path, root_dir=root_dir)
    print(f"Dataset size: {len(D)}")
    s = None
    for d in D:
        if d['boxes'] is not None:
            print(f"Sample img shape: {d['img'].shape}")
            print(f"Sample img values: {d['img'][0, 0:5, 0:5]}")
            print(f"Sample bbox: {d['boxes']}")
            s = d
            break

    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.01),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((400, 400), antialias=True)
    ])

    DT = IjmondBboxDataset(metadata_path=metadata_path, root_dir=root_dir, transform=transforms)
    for st in DT:
        if st['boxes'] is not None:
            print(f"Sample img shape after transform: {st['img'].shape}")
            print(f"Sample img values after transform: {st['img'][0, 0:5, 0:5]}")
            print(f"Sample bbox after transform: {st['boxes']}")
            plot([(s['img'], s['boxes']), (st['img'], st['boxes'])], "debug_plot.png")
            break