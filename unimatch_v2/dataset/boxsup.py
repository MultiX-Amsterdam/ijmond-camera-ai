"""
BoxSup dataset and loss utilities.

Provides:
- BoxSupDataset          : loads ALL citizen images (positive + negative), returns weak/strong aug pair
- boxsup_collate_fn      : pads variable-length bbox tensors in a batch
- make_boxsup_mask       : EMA-constrained bbox pseudo-mask (outside all boxes → background)
- make_box_corrected_mask: zeroes EMA predictions outside GT boxes (used for citizen consistency)
- mil_box_loss           : differentiable MIL inside-box loss (≥20% box area must be smoke)
"""
import json
import os
import random

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.semi import compute_transmission_map


_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_TO_TENSOR = transforms.ToTensor()


def _rescale_pil(img, base_size):
    """Rescale a PIL image so its longest side equals base_size.

    Parameters
    ----------
    img : PIL.Image
    base_size : int

    Returns
    -------
    PIL.Image
    """
    w, h = img.size
    if h >= w:
        oh = base_size
        ow = max(1, int(w * base_size / h + 0.5))
    else:
        ow = base_size
        oh = max(1, int(h * base_size / w + 0.5))
    return img.resize((ow, oh), Image.BILINEAR)


def _rescale_bboxes(bboxes_xywh, orig_w, orig_h, new_w, new_h):
    """Scale XYWH bboxes to a new image size and convert to XYXY.

    Parameters
    ----------
    bboxes_xywh : list of [x, y, w, h]
    orig_w, orig_h : int
        Original image dimensions.
    new_w, new_h : int
        Target image dimensions after rescaling.

    Returns
    -------
    list of [x1, y1, x2, y2]
    """
    sx = new_w / orig_w
    sy = new_h / orig_h
    result = []
    for x, y, w, h in bboxes_xywh:
        result.append([x * sx, y * sy, (x + w) * sx, (y + h) * sy])
    return result


def _hflip_bboxes_xyxy(bboxes_xyxy, img_w):
    """Flip XYXY bboxes horizontally.

    Parameters
    ----------
    bboxes_xyxy : list of [x1, y1, x2, y2]
    img_w : int
        Image width after rescaling.

    Returns
    -------
    list of [x1, y1, x2, y2]
    """
    return [[img_w - x2, y1, img_w - x1, y2] for x1, y1, x2, y2 in bboxes_xyxy]


def _cat_dcp(img_pil, img_tensor, use_dcp):
    """Optionally append a transmission-map channel to img_tensor.

    Parameters
    ----------
    img_pil : PIL.Image
        RGB image used to compute the DCP transmission map.
    img_tensor : torch.Tensor
        Shape (3, H, W), float32, already normalized.
    use_dcp : bool

    Returns
    -------
    torch.Tensor
        Shape (3, H, W) if use_dcp is False, else (4, H, W).
    """
    if not use_dcp:
        return img_tensor
    t = compute_transmission_map(img_pil)
    t_tensor = torch.from_numpy(t).unsqueeze(0).float()
    t_tensor = (t_tensor - 0.5) / 0.5
    return torch.cat([img_tensor, t_tensor], dim=0)


class BoxSupDataset(Dataset):
    """Dataset of citizen images paired with their annotated bounding boxes.

    Each sample returns a weak-augmented and a strong-augmented view of the
    same image, together with the corresponding XYXY bounding boxes in pixel
    coordinates of the rescaled image.  Both views share the same random
    horizontal flip so bboxes remain consistent.

    Only images that have at least one non-null bounding box are included.

    Parameters
    ----------
    json_path : str
        Path to the filtered bbox JSON file.  Each entry is a dict with keys
        ``id`` (int, used as ``{id}.png`` filename) and ``bbox`` (list of
        dicts with ``x_bbox``, ``y_bbox``, ``w_bbox``, ``h_bbox``,
        ``w_image``, ``h_image``, or ``None``).
    img_dir : str
        Directory containing ``{id}.png`` image files.
    base_size : int
        Target longest-side length for rescaling before augmentation.
    use_dcp : bool, optional
        If True, append a 4th DCP transmission-map channel (default False).
    """

    def __init__(self, json_path, img_dir, base_size, use_dcp=False):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        # Include ALL images: positive (has bboxes) and negative (bbox is null/empty).
        # Negative images return an empty bbox tensor so make_box_corrected_mask treats
        # them as all-background targets.
        self.samples = metadata
        self.img_dir = img_dir
        self.base_size = base_size
        self.use_dcp = use_dcp

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.img_dir, f"{item['id']}.png")
        img = Image.open(img_path).convert("RGB")

        img = _rescale_pil(img, self.base_size)
        new_w, new_h = img.size

        bbox_list = item["bbox"] or []
        if bbox_list:
            bboxes_xywh = [
                [b["x_bbox"], b["y_bbox"], b["w_bbox"], b["h_bbox"]]
                for b in bbox_list
            ]
            orig_w = bbox_list[0]["w_image"]
            orig_h = bbox_list[0]["h_image"]
            bboxes_xyxy = _rescale_bboxes(bboxes_xywh, orig_w, orig_h, new_w, new_h)
        else:
            bboxes_xyxy = []

        do_flip = random.random() < 0.5
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if bboxes_xyxy:
                bboxes_xyxy = _hflip_bboxes_xyxy(bboxes_xyxy, new_w)

        img_s = img.copy()
        if random.random() < 0.8:
            img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
        img_s = transforms.RandomGrayscale(p=0.2)(img_s)
        if random.random() < 0.5:
            sigma = np.random.uniform(0.1, 2.0)
            img_s = img_s.filter(ImageFilter.GaussianBlur(radius=sigma))

        img_w_t = _NORMALIZE(_TO_TENSOR(img))
        img_s_t = _NORMALIZE(_TO_TENSOR(img_s))

        img_w_t = _cat_dcp(img, img_w_t, self.use_dcp)
        img_s_t = _cat_dcp(img_s, img_s_t, self.use_dcp)

        if bboxes_xyxy:
            bboxes_tensor = torch.tensor(bboxes_xyxy, dtype=torch.float32)
        else:
            bboxes_tensor = torch.empty((0, 4), dtype=torch.float32)
        return img_w_t, img_s_t, bboxes_tensor


def boxsup_collate_fn(batch):
    """Collate BoxSupDataset items, padding bboxes to a uniform count.

    Parameters
    ----------
    batch : list of (img_w, img_s, bboxes)

    Returns
    -------
    tuple of (imgs_w, imgs_s, bboxes_padded)
        ``imgs_w`` and ``imgs_s`` are stacked tensors of shape (B, C, H, W).
        ``bboxes_padded`` is shape (B, N_max, 4) with ``-1`` for absent entries.
    """
    imgs_w, imgs_s, bboxes_list = zip(*batch)
    # Guard against all-negative batches where every item has 0 boxes.
    n_max = max(1, max(b.shape[0] for b in bboxes_list))
    padded = []
    for bboxes in bboxes_list:
        n = bboxes.shape[0]
        if n < n_max:
            pad = torch.full((n_max - n, 4), -1.0)
            bboxes = torch.cat([bboxes, pad], dim=0)
        padded.append(bboxes)
    return torch.stack(imgs_w), torch.stack(imgs_s), torch.stack(padded)


def make_boxsup_mask(pred_argmax, bboxes_padded, conf=None, conf_thresh=0.0):
    """Generate a BoxSup pseudo-mask from EMA teacher predictions.

    Pixels outside every valid bounding box are forced to background (0).
    Pixels inside at least one valid box keep the EMA argmax value, unless
    a confidence map is provided and the pixel confidence is below
    ``conf_thresh`` — those pixels are set to 255 (CE ignore index) so they
    do not contribute a gradient.

    Parameters
    ----------
    pred_argmax : torch.Tensor
        Shape (B, H, W), dtype long.  EMA teacher argmax predictions.
    bboxes_padded : torch.Tensor
        Shape (B, N, 4), dtype float32.  XYXY pixel-coordinate bboxes in the
        coordinate frame of the rescaled image.  Entries with x1 == -1 are
        padding and are ignored.
    conf : torch.Tensor or None, optional
        Shape (B, H, W), float32.  Per-pixel max softmax confidence from the
        EMA teacher.  If provided, inside-box pixels with confidence below
        ``conf_thresh`` are masked out (set to 255).
    conf_thresh : float, optional
        Confidence threshold (default 0.0 = no filtering).

    Returns
    -------
    torch.Tensor
        Shape (B, H, W), dtype long.  Valid CE target.
    """
    B, H, W = pred_argmax.shape
    device = pred_argmax.device
    N = bboxes_padded.shape[1]

    xs = torch.arange(W, device=device).float().view(1, W)
    ys = torch.arange(H, device=device).float().view(H, 1)

    x1 = bboxes_padded[:, :, 0].view(B, N, 1, 1)
    y1 = bboxes_padded[:, :, 1].view(B, N, 1, 1)
    x2 = bboxes_padded[:, :, 2].view(B, N, 1, 1)
    y2 = bboxes_padded[:, :, 3].view(B, N, 1, 1)
    valid = (x1 >= 0)

    in_x = (xs >= x1) & (xs < x2)
    in_y = (ys >= y1) & (ys < y2)
    in_box = in_x & in_y & valid

    any_box = in_box.any(dim=1)

    mask = torch.where(any_box, pred_argmax, torch.zeros_like(pred_argmax))
    if conf is not None and conf_thresh > 0.0:
        # Inside-box pixels with low teacher confidence → ignore (255)
        mask = torch.where(
            any_box & (conf < conf_thresh),
            torch.full_like(mask, 255),
            mask,
        )
    return mask


def make_fixed_box_mask(bboxes_padded, H, W, device=None):
    """Generate a fixed box mask using bbox annotations as direct supervision.

    Inside all valid bounding boxes → smoke (1).
    Outside all bounding boxes → background (0).
    Does not depend on any model predictions.

    Parameters
    ----------
    bboxes_padded : torch.Tensor
        Shape (B, N, 4), dtype float32.  XYXY pixel-coordinate bboxes.
        Entries with x1 == -1 are padding and are ignored.
    H : int
        Mask height in pixels.
    W : int
        Mask width in pixels.
    device : torch.device or None
        Target device.  Defaults to ``bboxes_padded.device``.

    Returns
    -------
    torch.Tensor
        Shape (B, H, W), dtype long.  Values in {0, 1}.
    """
    if device is None:
        device = bboxes_padded.device
    B, N, _ = bboxes_padded.shape
    smoke = torch.ones(B, H, W, dtype=torch.long, device=device)
    return make_boxsup_mask(smoke, bboxes_padded)


def make_box_corrected_mask(ema_argmax, bboxes_padded):
    """Zero EMA teacher predictions outside GT bounding boxes.

    For each image, smoke predictions that fall outside every GT box are
    replaced with background (0).  Pixels inside at least one valid GT box
    keep the original EMA argmax value.

    For negative images (no valid boxes), the entire mask becomes background.

    Parameters
    ----------
    ema_argmax : torch.Tensor
        Shape (B, H, W), dtype long.  EMA teacher argmax predictions.
    bboxes_padded : torch.Tensor
        Shape (B, N, 4), dtype float32.  XYXY bboxes; x1 == -1 means padding.

    Returns
    -------
    torch.Tensor
        Shape (B, H, W), dtype long.  Corrected pseudo-labels.
    """
    return make_boxsup_mask(ema_argmax, bboxes_padded)


def mil_box_loss(pred_logits, bboxes_padded, area_ratio=0.2):
    """Differentiable MIL inside-box loss for smoke detection.

    For each valid bounding box, the top-``area_ratio`` fraction of pixels
    (by smoke probability) inside the box must have high confidence.  The
    loss is the mean negative log-probability of those top pixels:

        loss = -mean(log(top_k_probs + eps))

    where k = max(1, floor(area_ratio * M)) and M is the number of pixels
    inside the box.  This forces the model to assign high smoke probability
    to at least ``area_ratio`` of each annotated region (not just one pixel).

    Parameters
    ----------
    pred_logits : torch.Tensor
        Shape (B, nclass, H, W). Raw student logits.
    bboxes_padded : torch.Tensor
        Shape (B, N, 4), dtype float32. XYXY bboxes; x1 == -1 means padding.
    area_ratio : float, optional
        Fraction of box pixels that must be smoke (default 0.2 = 20%).

    Returns
    -------
    torch.Tensor
        Scalar loss (0 if no valid boxes in the batch).
    """
    smoke_probs = pred_logits.softmax(dim=1)[:, 1, :, :]
    B, H, W = smoke_probs.shape
    N = bboxes_padded.shape[1]

    box_losses = []
    for b in range(B):
        for n in range(N):
            x1, y1, x2, y2 = bboxes_padded[b, n]
            if x1 < 0:
                continue
            ix1 = max(0, int(x1.item()))
            iy1 = max(0, int(y1.item()))
            ix2 = min(W, int(x2.item()))
            iy2 = min(H, int(y2.item()))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            region = smoke_probs[b, iy1:iy2, ix1:ix2].flatten()
            k = max(1, int(area_ratio * region.numel()))
            top_k_probs, _ = torch.topk(region, k)
            box_losses.append(-torch.log(top_k_probs + 1e-6).mean())

    if not box_losses:
        return pred_logits.sum() * 0.0
    return torch.stack(box_losses).mean()
