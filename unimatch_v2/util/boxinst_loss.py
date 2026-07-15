"""
BoxInst loss terms for box-supervised binary semantic segmentation.

Adapts the projection loss and pairwise affinity loss from the BoxInst paper
(Tian et al., CVPR 2021) to the UniMatch V2 binary segmentation setting where
the model outputs (B, 2, H, W) logits (smoke vs. background).

Key differences from the original AdelaiDet implementation:
- Binary semantic segmentation (not instance segmentation): one shared mask per
  image formed by taking the union of all GT bounding boxes.
- Model output is 2-class softmax logits, so log_softmax is used instead of
  logsigmoid.
- LAB color conversion is done in pure PyTorch on GPU to avoid CPU round-trips.
- Images with no valid bounding boxes (negative citizen images) receive a
  background suppression loss (CE toward all-background) to prevent the model
  from hallucinating smoke on all-clear images (reduces false alarm rate).

Usage example::

    from util.boxinst_loss import boxinst_loss

    # pred:   (B, 2, H, W) raw logits from the student model
    # img_c:  (B, C, H, W) ImageNet-normalised image (first 3 channels are RGB)
    # bboxes: (B, N_max, 4) XYXY pixel coords in crop space; padded rows have x1=-1
    loss = boxinst_loss(
        pred, img_c, bboxes,
        tau=0.1, kernel_size=3, dilation=2,
    )
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ImageNet normalisation constants (used to undo _NORMALIZE in the dataset)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unfold_wo_center(x, kernel_size, dilation):
    """Unfold a feature map into K*K-1 neighbour slices, excluding the center.

    Parameters
    ----------
    x : torch.Tensor
        Shape (B, C, H, W).
    kernel_size : int
        Odd neighbourhood size (e.g. 3 for a 3×3 patch).
    dilation : int
        Dilation rate for enlarging the receptive field without extra cost.

    Returns
    -------
    torch.Tensor
        Shape (B, C, K*K-1, H, W), where K = kernel_size.
    """
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded = F.unfold(
        x, kernel_size=kernel_size, padding=padding, dilation=dilation
    )
    # unfolded: (B, C*K*K, H*W) → reshape to (B, C, K*K, H, W)
    B, C, H, W = x.shape
    K2 = kernel_size ** 2
    unfolded = unfolded.view(B, C, K2, H, W)
    # Remove the center pixel (index K*K//2)
    center = K2 // 2
    unfolded = torch.cat([unfolded[:, :, :center], unfolded[:, :, center + 1:]], dim=2)
    return unfolded  # (B, C, K*K-1, H, W)


def _rgb_to_lab(img_normalized):
    """Convert a batch of ImageNet-normalised RGB images to CIE LAB colour space.

    Uses a pure-PyTorch, GPU-compatible implementation (no skimage / CPU round-trip).
    The conversion follows the standard sRGB → linear RGB → XYZ (D65) → LAB pipeline.

    Parameters
    ----------
    img_normalized : torch.Tensor
        Shape (B, C, H, W).  The first three channels must be ImageNet-normalised
        RGB.  Additional channels (e.g. DCP) are ignored.

    Returns
    -------
    torch.Tensor
        Shape (B, 3, H, W), dtype float32.  L in [0, 100], a and b in ~[-128, 127].
    """
    device = img_normalized.device
    mean = _IMAGENET_MEAN.to(device, dtype=torch.float32).view(1, 3, 1, 1)
    std = _IMAGENET_STD.to(device, dtype=torch.float32).view(1, 3, 1, 1)

    # Undo ImageNet normalisation → sRGB in [0, 1]
    rgb = img_normalized[:, :3].float() * std + mean
    rgb = rgb.clamp(0.0, 1.0)

    # sRGB → linear RGB (inverse gamma approximation)
    mask = rgb > 0.04045
    linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # Linear RGB → XYZ (D65 illuminant, sRGB primaries)
    # Matrix from IEC 61966-2-1:2003
    M = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        device=device, dtype=torch.float32
    )
    # linear: (B, 3, H, W) → (B, H*W, 3) for matmul
    B, _, H, W = linear.shape
    lin_flat = linear.permute(0, 2, 3, 1).reshape(B * H * W, 3)
    xyz_flat = lin_flat @ M.T
    xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)

    # XYZ → Lab (D65 white point: Xn=0.95047, Yn=1.00000, Zn=1.08883)
    white = torch.tensor([0.95047, 1.00000, 1.08883], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    t = xyz / white

    eps = 216.0 / 24389.0  # (6/29)^3
    kappa = 24389.0 / 27.0  # (29/3)^3
    f = torch.where(t > eps, t.clamp(min=1e-10) ** (1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    L = 116.0 * f[:, 1] - 16.0           # (B, H, W)
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])

    return torch.stack([L, a, b], dim=1)  # (B, 3, H, W)


def _compute_color_similarity(images_lab, kernel_size, dilation):
    """Compute pairwise colour similarity in LAB space for each pixel's neighbourhood.

    Similarity: S_e = exp(-||c_i - c_j|| / theta), theta=2 (per BoxInst paper).

    Parameters
    ----------
    images_lab : torch.Tensor
        Shape (B, 3, H, W). CIE LAB image.
    kernel_size : int
        Neighbourhood kernel size (odd).
    dilation : int
        Dilation rate.

    Returns
    -------
    torch.Tensor
        Shape (B, K*K-1, H, W), float32.  Values in (0, 1].
    """
    # neighbours: (B, 3, K*K-1, H, W)
    neighbours = _unfold_wo_center(images_lab, kernel_size, dilation)
    # diff between center and each neighbour
    diff = images_lab.unsqueeze(2) - neighbours          # (B, 3, K*K-1, H, W)
    dist = diff.norm(dim=1)                              # (B, K*K-1, H, W)
    return torch.exp(-dist * 0.5)                        # theta = 2


def _build_box_bitmask(bboxes, H, W):
    """Build a union binary bitmask from padded XYXY bounding boxes.

    Parameters
    ----------
    bboxes : torch.Tensor
        Shape (B, N_max, 4), float32.  Each row is [x1, y1, x2, y2] in pixel
        coordinates clipped to [0, crop_size].  Padding rows have x1 = -1.
    H : int
        Mask height.
    W : int
        Mask width.

    Returns
    -------
    torch.Tensor
        Shape (B, H, W), float32.  1 inside any valid box, 0 outside.
    torch.Tensor
        Shape (B,), bool.  True for images that have at least one valid box.
    """
    B = bboxes.shape[0]
    bitmask = torch.zeros(B, H, W, device=bboxes.device, dtype=torch.float32)

    ys = torch.arange(H, device=bboxes.device).view(1, H, 1)
    xs = torch.arange(W, device=bboxes.device).view(1, 1, W)

    for b in range(B):
        for box in bboxes[b]:
            x1, y1, x2, y2 = box
            if x1 < 0:
                continue  # padding
            # Clamp to valid image range and use inclusive integer bounds
            x1i = int(x1.item())
            y1i = int(y1.item())
            x2i = min(int(x2.item()) + 1, W)
            y2i = min(int(y2.item()) + 1, H)
            if x2i <= x1i or y2i <= y1i:
                continue
            bitmask[b, y1i:y2i, x1i:x2i] = 1.0

    has_box = bitmask.view(B, -1).any(dim=1)  # (B,)
    return bitmask, has_box


# ---------------------------------------------------------------------------
# Public loss functions
# ---------------------------------------------------------------------------

def _dice_1d(pred_proj, gt_proj):
    """Dice loss between 1-D projection vectors.

    Parameters
    ----------
    pred_proj : torch.Tensor
        Shape (N, L), float32. Predicted projection (probability values).
    gt_proj : torch.Tensor
        Shape (N, L), float32. GT projection (0/1 values).

    Returns
    -------
    torch.Tensor
        Shape (N,), float32. Per-instance Dice loss.
    """
    intersection = (pred_proj * gt_proj).sum(dim=1)
    return 1.0 - (2.0 * intersection + 1.0) / (pred_proj.sum(dim=1) + gt_proj.sum(dim=1) + 1.0)


def boxinst_projection_loss(pred, bboxes):
    """Projection loss: Dice on horizontal and vertical max-projections.

    Ensures the tightest bounding box covering the predicted smoke mask matches
    the GT box extent (per BoxInst Section 2.1, Eq. 3).

    Parameters
    ----------
    pred : torch.Tensor
        Shape (B, 2, H, W). Raw logits from the student model.
    bboxes : torch.Tensor
        Shape (B, N_max, 4). XYXY pixel coords, padded with -1.

    Returns
    -------
    torch.Tensor
        Scalar projection loss, averaged over images that have at least one
        valid box.  Returns zero if no image in the batch has a valid box.
    """
    B, _, H, W = pred.shape
    smoke_prob = pred.softmax(dim=1)[:, 1]  # (B, H, W)

    bitmask, has_box = _build_box_bitmask(bboxes, H, W)

    if not has_box.any():
        return pred.sum() * 0.0

    # Max-projection along H axis → shape (B, W)
    pred_proj_x = smoke_prob.max(dim=1)[0]      # (B, W)
    gt_proj_x   = bitmask.max(dim=1)[0]          # (B, W)

    # Max-projection along W axis → shape (B, H)
    pred_proj_y = smoke_prob.max(dim=2)[0]       # (B, H)
    gt_proj_y   = bitmask.max(dim=2)[0]           # (B, H)

    loss_x = _dice_1d(pred_proj_x[has_box], gt_proj_x[has_box])
    loss_y = _dice_1d(pred_proj_y[has_box], gt_proj_y[has_box])

    return (loss_x + loss_y).mean()


def boxinst_pairwise_loss(pred, img_normalized, bboxes, tau=0.1, kernel_size=3, dilation=2):
    """Pairwise affinity loss: BCE for colour-consistent neighbouring pixel pairs.

    For each pixel inside a GT box, computes -log P(same_label) for all K*K-1
    neighbours whose colour similarity >= tau.  Only edges where the *center*
    pixel is inside a GT box are included (E_in, per BoxInst Eq. 5/8).

    Parameters
    ----------
    pred : torch.Tensor
        Shape (B, 2, H, W). Raw logits.
    img_normalized : torch.Tensor
        Shape (B, C, H, W). ImageNet-normalised image; first 3 channels are RGB.
    bboxes : torch.Tensor
        Shape (B, N_max, 4). XYXY pixel coords, padded with -1.
    tau : float, optional
        Colour similarity threshold (default 0.1, per BoxInst ablation Table 1a).
    kernel_size : int, optional
        Neighbourhood size (default 3).
    dilation : int, optional
        Dilation rate (default 2 to capture long-range while keeping cost low).

    Returns
    -------
    torch.Tensor
        Scalar pairwise loss, averaged over images with at least one valid box.
        Returns zero if no image in the batch has a valid box.
    """
    B, _, H, W = pred.shape

    bitmask, has_box = _build_box_bitmask(bboxes, H, W)

    if not has_box.any():
        return pred.sum() * 0.0

    # --- Colour similarity ---
    images_lab = _rgb_to_lab(img_normalized)         # (B, 3, H, W)
    color_sim = _compute_color_similarity(images_lab, kernel_size, dilation)  # (B, K²-1, H, W)

    # --- Pairwise same-label log-probability ---
    log_p = F.log_softmax(pred, dim=1)
    log_fg = log_p[:, 1:2]      # (B, 1, H, W)
    log_bg = log_p[:, 0:1]      # (B, 1, H, W)

    log_fg_unfold = _unfold_wo_center(log_fg, kernel_size, dilation)  # (B, 1, K²-1, H, W)
    log_bg_unfold = _unfold_wo_center(log_bg, kernel_size, dilation)  # (B, 1, K²-1, H, W)

    # log P(same) = log(exp(log_fg_i + log_fg_j) + exp(log_bg_i + log_bg_j))
    log_same_fg = log_fg.unsqueeze(2) + log_fg_unfold   # (B, 1, K²-1, H, W)
    log_same_bg = log_bg.unsqueeze(2) + log_bg_unfold   # (B, 1, K²-1, H, W)

    # Numerically stable log-sum-exp
    max_ = torch.max(log_same_fg, log_same_bg)
    log_same_prob = torch.log(
        torch.exp(log_same_fg - max_) + torch.exp(log_same_bg - max_)
    ) + max_  # (B, 1, K²-1, H, W)
    log_same_prob = log_same_prob[:, 0]  # (B, K²-1, H, W)

    # Weights: (color_sim >= tau) AND center pixel is inside a box (E_in)
    color_mask = (color_sim >= tau).float()             # (B, K²-1, H, W)
    box_mask = bitmask.unsqueeze(1)                     # (B, 1, H, W)
    weights = color_mask * box_mask                     # (B, K²-1, H, W)

    # Per-image pairwise loss
    loss_per_image = -(weights * log_same_prob).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp(min=1.0)

    return loss_per_image[has_box].mean()


def boxinst_negative_loss(pred, bboxes):
    """Background suppression loss for negative citizen images (no valid boxes).

    The projection and pairwise losses only fire on positive images (those with
    at least one valid bounding box), leaving negative (all-clear) images with
    zero gradient.  This causes the model to hallucinate smoke on clear images,
    driving up the false alarm rate.

    This loss penalises any smoke prediction on negative images by computing
    the mean cross-entropy toward the background class across all pixels.

    Parameters
    ----------
    pred : torch.Tensor
        Shape (B, 2, H, W). Raw logits from the student model.
    bboxes : torch.Tensor
        Shape (B, N_max, 4). XYXY pixel coords, padded with -1.

    Returns
    -------
    torch.Tensor
        Scalar background CE loss, averaged over negative images and all pixels.
        Returns zero if every image in the batch has at least one valid box.
    """
    _, has_box = _build_box_bitmask(bboxes, pred.shape[2], pred.shape[3])
    neg_mask = ~has_box  # (B,)

    if not neg_mask.any():
        return pred.sum() * 0.0

    # log P(background) for each pixel of each negative image
    log_p_bg = F.log_softmax(pred[neg_mask], dim=1)[:, 0]  # (N_neg, H, W)
    return -log_p_bg.mean()


def boxinst_loss(pred, img_normalized, bboxes, tau=0.1, kernel_size=3, dilation=2, neg_weight=1.0):
    """Combined BoxInst loss: projection + pairwise affinity + background suppression.

    Parameters
    ----------
    pred : torch.Tensor
        Shape (B, 2, H, W). Raw logits from the student model.
    img_normalized : torch.Tensor
        Shape (B, C, H, W). ImageNet-normalised image; first 3 channels are RGB.
    bboxes : torch.Tensor
        Shape (B, N_max, 4). XYXY pixel coords, padded with -1.
    tau : float, optional
        Colour similarity threshold (default 0.1).
    kernel_size : int, optional
        Neighbourhood kernel size (default 3).
    dilation : int, optional
        Dilation rate (default 2).
    neg_weight : float, optional
        Weight for the background suppression loss on negative images (default
        1.0).  Set to 0 to disable background suppression.

    Returns
    -------
    torch.Tensor
        Scalar combined loss = L_proj + L_pairwise + neg_weight * L_neg.
    """
    l_proj = boxinst_projection_loss(pred, bboxes)
    l_pairwise = boxinst_pairwise_loss(pred, img_normalized, bboxes, tau, kernel_size, dilation)
    l_neg = boxinst_negative_loss(pred, bboxes)
    return l_proj + l_pairwise + neg_weight * l_neg
