from .transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiSmokeDataset(Dataset):
    def __init__(self, name, root, mode, size=None, base_size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.base_size = base_size
        self.data = os.path.join(root, id_path)

        with open(self.data, 'r') as f:
            self.ids = f.read().splitlines()

        if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
            self.ids *= math.ceil(nsample / len(self.ids))
            self.ids = self.ids[:nsample]

    def __getitem__(self, item):
        item_id = self.ids[item]
        split_item = item_id.split(' ')
        img_filename = os.path.join(self.root, split_item[0])

        img = Image.open(img_filename).convert('RGB')

        # Each line in the txt file can have one of three formats:
        #   "img_path"            – image with no smoke (no mask, zero mask used)
        #   "img_path None"       – unlabeled image (no ground truth, zero mask used)
        #   "img_path mask_path"  – image with a smoke mask
        if self.mode == 'train_u' or len(split_item) < 2 or split_item[1] == 'None':
            mask_npy = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        else:
            mask_filename = os.path.join(self.root, split_item[1])
            mask_npy = np.array(Image.open(mask_filename))

            if mask_npy.ndim > 2:
                mask_npy = np.max(mask_npy, axis=-1)

            max_value = np.max(mask_npy)

            if self.mode == 'val_hi' or self.mode == 'test_hi' or self.mode == 'train_l_hi':
                mask_npy = mask_npy == max_value
            elif self.mode == 'train_l_hi' or self.mode == 'train_l_lo':
                mask_npy = mask_npy > 0

        mask_npy = mask_npy.astype(np.uint8)
        mask = Image.fromarray((mask_npy > 0).astype(np.uint8))
        max_mask = np.max(mask)
        assert (max_mask in [0, 1])

        if self.mode.startswith('val') or self.mode.startswith('test'):
            img, mask = normalize(img, mask)
            return img, mask, item_id

        if self.base_size is not None:
            img, mask = rescale(img, mask, self.base_size)
        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size, 255)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)

        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)

        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)
        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 255] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)