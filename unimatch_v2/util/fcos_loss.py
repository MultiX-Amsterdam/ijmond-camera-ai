"""
FCOS classification focal loss for box-supervised binary segmentation.

Derives pixel-level foreground/background labels purely from bounding boxes
(no pixel masks required) and applies sigmoid focal loss, analogous to the
classification branch in FCOS (Tian et al., ICCV 2019).

Key design choices for this full-resolution segmentation setting:
- Operates directly on the full-resolution model output (B, 2, H, W), using the
  class-1 (smoke) logit for binary focal loss — no architectural changes needed.
- Negative citizen images (all box rows are padding) contribute all-background
  labels, directly suppressing false alarms (FAR) via focal loss.
- Proportional center sampling (`radius_ratio`) keeps only pixels within a dynamic
  percentage of the box size. This reduces noise from box edges (which may contain
  background) without using a microscopic, fixed-pixel radius that fails at full resolution.
- Loss is normalized by the total number of valid pixels in the image (segmentation
  standard) rather than just the positive pixels (detection standard). This prevents
  the gradient magnitude from exploding due to extreme class imbalance.

Usage example::

    from util.fcos_loss import fcos_cls_loss

    # pred:   (B, 2, H, W) raw logits from the student model
    # bboxes: (B, N_max, 4) XYXY pixel coords in crop space; padding rows have x1=-1

    # Default usage
    loss = fcos_cls_loss(pred, bboxes)

    # Custom center sampling (e.g., tight 25% core of the box)
    loss = fcos_cls_loss(pred, bboxes, alpha=0.25, gamma=2.0,
                         center_sample=True, radius_ratio=0.25)
"""

import torch
import torch.nn.functional as F


def _build_fg_bg_labels(bboxes, H, W, center_sample=True, radius_ratio=0.5):
    """
    Build per-pixel foreground/background label maps from bounding boxes.

    Pixels inside a GT box are foreground (1); pixels outside all boxes are
    background (0). Images with no valid boxes (all rows padding with x1=-1)
    receive an all-zero (all-background) map.

    When `center_sample=True`, only pixels whose distance from the box
    centroid is within a dynamic radius (based on the box dimensions and
    `radius_ratio`) are marked as foreground; the rest remain background.
    This reduces noise from box edges.

    Parameters
    ----------
    bboxes : torch.Tensor
        Shape (B, N_max, 4), float32. Each row is [x1, y1, x2, y2] in pixel
        coordinates in the crop space. Padding rows have x1 = -1.
    H : int
        Label map height (== model output height).
    W : int
        Label map width (== model output width).
    center_sample : bool
        If True, restrict positive pixels to a circular radius around the
        box centroid to avoid noisy bounding box edges.
    radius_ratio : float
        The multiplier applied to the shortest side of the bounding box to
        determine the radius for center sampling. For example, 0.5 means the
        radius is half the length of the shortest box dimension.

    Returns
    -------
    torch.Tensor
        Shape (B, H, W), float32. 1 for foreground, 0 for background.
    """
    B = bboxes.shape[0]
    labels = torch.zeros(B, H, W, dtype=torch.float32, device=bboxes.device)

    # Pixel coordinate grids — shape (H, W)
    ys = torch.arange(H, dtype=torch.float32, device=bboxes.device).view(H, 1).expand(H, W)
    xs = torch.arange(W, dtype=torch.float32, device=bboxes.device).view(1, W).expand(H, W)

    for b in range(B):
        for box in bboxes[b]:
            x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            if x1 < 0:
                continue  # padding row

            # All pixels inside the box
            inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)

            if center_sample:
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                box_w = x2 - x1
                box_h = y2 - y1

                # Dynamic radius based on box size
                dynamic_radius = min(box_w, box_h) * radius_ratio
                dist = ((xs - cx) ** 2 + (ys - cy) ** 2).sqrt()
                inside = inside & (dist <= dynamic_radius)

            labels[b] = labels[b].masked_fill(inside, 1.0)

    return labels


def fcos_cls_loss(pred, bboxes, alpha=0.25, gamma=2.0, center_sample=True, radius_ratio=0.5):
    """
    Sigmoid focal loss derived from bounding-box labels (no pixel masks needed).

    Computes binary sigmoid focal loss on the class-1 (smoke) logit channel.
    Negative images (all bboxes padding) contribute all-background supervision,
    which directly reduces the false alarm rate.

    Loss is averaged over all pixels in the batch to ensure stable gradients
    during full-resolution dense prediction, preventing the background loss
    from exploding when positives are sparse.

    Parameters
    ----------
    pred : torch.Tensor
        Shape (B, 2, H, W), raw (pre-softmax / pre-sigmoid) logits from the
        model. The class-1 channel (smoke) is used for the binary focal loss.
    bboxes : torch.Tensor
        Shape (B, N_max, 4), float32 XYXY bounding boxes in crop-space pixel
        coordinates. Padding rows have x1 = -1.
    alpha : float
        Focal loss alpha (foreground weighting). Default 0.25.
    gamma : float
        Focal loss gamma (hard-example focusing). Default 2.0.
    center_sample : bool
        Whether to restrict positives to a dynamic radius around each box centroid.
        Default True.
    radius_ratio : float
        The ratio of the bounding box's shortest dimension used to define the
        center-sampling radius. Default 0.5.

    Returns
    -------
    torch.Tensor
        Scalar focal loss, averaged over all spatial dimensions and batch elements.
    """
    B, _, H, W = pred.shape

    # Use the smoke (class-1) logit for binary focal loss
    smoke_logit = pred[:, 1]  # (B, H, W)

    labels = _build_fg_bg_labels(
        bboxes, H, W,
        center_sample=center_sample,
        radius_ratio=radius_ratio,
    )  # (B, H, W), values in {0, 1}

    # Sigmoid focal loss, computed in a numerically stable way.
    # For each pixel:
    #   p = sigmoid(logit)
    #   FL = -alpha * (1-p)^gamma * log(p)        [fg]
    #      = -(1-alpha) * p^gamma * log(1-p)      [bg]
    p = torch.sigmoid(smoke_logit)

    # log(p) and log(1-p) computed from log-sigmoid for stability
    log_p = F.logsigmoid(smoke_logit)          # log(sigmoid(x))
    log_1mp = F.logsigmoid(-smoke_logit)       # log(1 - sigmoid(x))

    fg_loss = -alpha * ((1.0 - p) ** gamma) * log_p     # (B, H, W)
    bg_loss = -(1.0 - alpha) * (p ** gamma) * log_1mp   # (B, H, W)

    loss_map = labels * fg_loss + (1.0 - labels) * bg_loss  # (B, H, W)

    # Average over all pixels to prevent exploding loss in semantic segmentation
    return loss_map.mean()