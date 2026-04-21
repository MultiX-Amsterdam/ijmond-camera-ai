from .transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from pathlib import Path
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Transmission map helpers (ported from bvm_training/transmission_map.py)
# ---------------------------------------------------------------------------

def _dc_dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    return cv2.erode(dc, kernel)


def _dc_atmospheric_light(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)
    indices = darkvec.argsort()
    indices = indices[imsz - numpx:]
    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]
    return atmsum / numpx


def _dc_transmission_estimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(3):
        im3[:, :, ind] = im[:, :, ind] / np.maximum(A[0, ind], 1e-6)
    return 1 - omega * _dc_dark_channel(im3, sz)


def _dc_guided_filter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    return mean_a * im + mean_b


def compute_transmission_map(img_pil, sz=15):
    """Compute the transmission map from a PIL RGB image.

    Args:
        img_pil: PIL Image in RGB mode.
        sz: patch size for dark channel prior (default 15).

    Returns:
        np.ndarray, shape H×W, float32, values approximately in [0, 1].
    """
    img_bgr = np.array(img_pil)[:, :, ::-1].copy()   # RGB → BGR uint8
    I = img_bgr.astype('float64') / 255
    dark = _dc_dark_channel(I, sz)
    A = _dc_atmospheric_light(I, dark)
    te = _dc_transmission_estimate(I, A, sz)
    # guided filter refinement uses the original uint8 image for grayscale guide
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    t = _dc_guided_filter(gray, te, r=60, eps=0.0001)
    return t.astype(np.float32)


class SemiSmokeDataset(Dataset):
    def __init__(self, name, root, mode, size=None, base_size=None, id_path=None, nsample=None, use_dcp=False):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.base_size = base_size
        self.use_dcp = use_dcp
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
            img_pil = img
            img, mask = normalize(img, mask)
            img = self._cat_dcp(img_pil, img)
            return img, mask, item_id

        if self.base_size is not None:
            img, mask = rescale(img, mask, self.base_size)
        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size, 255)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            img_pil = img
            img_t, mask_t = normalize(img, mask)
            img_t = self._cat_dcp(img_pil, img_t)
            return img_t, mask_t

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
        img_s1_pil, img_s2_pil, img_w_pil = img_s1, img_s2, img_w
        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s1 = self._cat_dcp(img_s1_pil, img_s1)
        img_s2 = normalize(img_s2)
        img_s2 = self._cat_dcp(img_s2_pil, img_s2)
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 255] = 255

        img_w_t = normalize(img_w)
        img_w_t = self._cat_dcp(img_w_pil, img_w_t)
        return img_w_t, img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def _cat_dcp(self, img_pil, img_tensor):
        """Append transmission map as a 4th channel to img_tensor.

        Args:
            img_pil: PIL Image (RGB, uint8) corresponding to img_tensor.
            img_tensor: float Tensor of shape [3, H, W] (already normalized RGB).

        Returns:
            Tensor of shape [3, H, W] if use_dcp is False, else [4, H, W].
        """
        if not self.use_dcp:
            return img_tensor
        t = compute_transmission_map(img_pil)              # H×W float32 in [0,1]
        t_tensor = torch.from_numpy(t).unsqueeze(0).float()  # [1, H, W]
        t_tensor = (t_tensor - 0.5) / 0.5                  # normalize to [-1, 1]
        return torch.cat([img_tensor, t_tensor], dim=0)    # [4, H, W]

    def __len__(self):
        return len(self.ids)