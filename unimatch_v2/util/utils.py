import json
import numpy as np
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLenienceCELoss(nn.Module):
    """Cross-entropy loss with boundary leniency for amorphous objects like smoke.

    For each pixel, an average pooling window over the binary ground-truth mask
    produces a local smoke density P. Pixels deep inside smoke or background
    (P≈1 or P≈0) receive full penalty. Edge pixels (P≈0.5) receive a reduced
    penalty controlled by alpha:

        W = 1 - alpha * 4 * P * (1 - P)

    When alpha=0 this reduces to standard cross-entropy. When alpha=1 a perfect
    half-and-half boundary pixel gets zero weight.

    Args:
        ignore_index: label value to ignore (default 255).
        alpha: leniency strength in [0, 1] (default 0.5).
        window_size: AvgPool2d kernel size for local density estimation (default 7).
        reduction: 'mean' returns a scalar; 'none' returns per-pixel weighted
            losses of shape (B, H, W) for use with external masking (e.g. pseudo-
            label confidence thresholding).
    """

    def __init__(self, ignore_index=255, alpha=0.5, window_size=7, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.window_size = window_size
        self.reduction = reduction

    def forward(self, logits, target):
        # Per-pixel CE, shape (B, H, W)
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction='none')

        # Binary smoke map; treat ignore pixels as background (0) so they don't
        # bias the density estimate near image borders.
        gt_smoke = (target == 1).float().unsqueeze(1)  # (B, 1, H, W)

        # Local smoke density via average pooling, output shape (B, H, W)
        padding = self.window_size // 2
        P = F.avg_pool2d(gt_smoke, kernel_size=self.window_size, stride=1, padding=padding)
        P = P.squeeze(1)

        # Weight: 1 at pure smoke/background, dips toward (1-alpha) at edges
        W = 1.0 - self.alpha * 4.0 * P * (1.0 - P)

        weighted = ce * W

        if self.reduction == 'none':
            return weighted

        # 'mean': average over valid (non-ignore) pixels only
        valid = target != self.ignore_index
        return (weighted * valid).sum() / valid.sum().clamp(min=1)


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def eval_score(evaluation):
    """Harmonic mean of gF1 and (1 - FAR) used for model selection."""
    gf1 = evaluation["gF1"]
    inv_far = 1.0 - evaluation["FAR"]
    denom = gf1 + inv_far
    if denom < 1e-12:
        return 0.0
    return 2.0 * gf1 * inv_far / denom


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


class BoundaryLenienceDiceLoss(nn.Module):
    """Soft dice loss on the smoke class with boundary leniency.

    Each pixel's contribution to the dice numerator and denominator is weighted
    by W = 1 - alpha * 4 * P * (1 - P), where P is the local smoke density
    estimated via average pooling. Boundary pixels (P ≈ 0.5) receive reduced
    weight, making the model less penalised for prediction errors near smoke
    edges.

    Args:
        ignore_index: label value to ignore (default 255).
        alpha: leniency strength in [0, 1] (default 0.5).
        window_size: AvgPool2d kernel size for local density estimation (default 7).
    """

    def __init__(self, ignore_index=255, alpha=0.5, window_size=7):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.window_size = window_size

    def forward(self, logits, target):
        prob = logits.softmax(dim=1)[:, 1]          # (B, H, W)
        valid = (target != self.ignore_index).float()

        gt_smoke = (target == 1).float().unsqueeze(1)  # (B, 1, H, W)
        padding = self.window_size // 2
        P = F.avg_pool2d(
            gt_smoke, kernel_size=self.window_size, stride=1, padding=padding
        ).squeeze(1)                                 # (B, H, W)

        W = 1.0 - self.alpha * 4.0 * P * (1.0 - P)

        p = prob * valid * W
        t = (target == 1).float() * valid * W
        intersection = (p * t).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (p.sum() + t.sum() + 1.0)


class AttentionWeightedLoss(nn.Module):
    """Box2Seg-style Attention Weighted Loss for bounding-box-supervised training.

    Bounding boxes are treated as noisy foreground labels: pixels inside a box
    are *likely* smoke but not guaranteed; pixels outside all boxes are definite
    background.

    The per-pixel attention map alpha (B, H, W) down-weights unreliable
    inside-box pixels.  It is typically the backbone CLS self-attention map so
    the weighting is independent of the segmentation head.  If alpha is None,
    the model's own detached smoke softmax probability is used as a fallback.

    Loss is computed per-image then averaged over the batch so that large boxes
    in one image cannot dominate the gradient.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, box_mask, alpha=None):
        """Compute AWL.

        Parameters
        ----------
        logits : torch.Tensor
            Shape (B, 2, H, W). Raw segmentation logits.
        box_mask : torch.Tensor
            Shape (B, H, W), float32. 1 inside a GT bounding box, 0 outside.
        alpha : torch.Tensor or None
            Shape (B, H, W), float32, non-negative. Per-pixel attention weights.
            If None, the detached smoke softmax probability is used as a fallback.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        log_p = F.log_softmax(logits, dim=1)  # (B, 2, H, W)
        log_p_smoke = log_p[:, 1]             # (B, H, W)
        log_p_bg = log_p[:, 0]               # (B, H, W)

        if alpha is None:
            alpha = logits.detach().softmax(dim=1)[:, 1]

        eps = 1e-8
        fg = box_mask          # 1 inside box
        bg = 1.0 - box_mask   # 1 outside box

        # Per-image denominators — shape (B,)
        fg_denom = fg.sum(dim=(1, 2)).clamp(min=eps)
        bg_denom = bg.sum(dim=(1, 2)).clamp(min=eps)

        # Per-image foreground loss: attention-weighted CE inside boxes
        loss_fg = -(alpha * fg * log_p_smoke).sum(dim=(1, 2)) / fg_denom

        # Per-image background loss: standard CE outside boxes (definite background)
        loss_bg = -(bg * log_p_bg).sum(dim=(1, 2)) / bg_denom

        return loss_fg.mean() + loss_bg.mean()


def dice_loss(pred_logits, target, ignore_index=255):
    """Soft dice loss on the smoke class (class 1).

    Computes 1 - (2 * sum(p * t) + 1) / (sum(p) + sum(t) + 1) where the sum
    is over all non-ignored pixels across the batch.  Laplace smoothing avoids
    division by zero on all-background batches.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Shape (B, nclass, H, W). Raw logits.
    target : torch.Tensor
        Shape (B, H, W), dtype long. Ground-truth labels.
    ignore_index : int, optional
        Label value to exclude from the computation (default 255).

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    prob = pred_logits.softmax(dim=1)[:, 1]
    valid = target != ignore_index
    t = (target == 1).float()
    p = prob * valid.float()
    t = t * valid.float()
    intersection = (p * t).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (p.sum() + t.sum() + 1.0)


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255, 0, 0])
        cmap[13] = np.array([0, 0, 142])
        cmap[14] = np.array([0, 0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0, 0, 230])
        cmap[18] = np.array([119, 11, 32])

    elif dataset == 'ade20k':
        cmap[0] = np.array([120, 120, 120])
        cmap[1] = np.array([180, 120, 120])
        cmap[2] = np.array([6, 230, 230])
        cmap[3] = np.array([80, 50, 50])
        cmap[4] = np.array([4, 200, 3])
        cmap[5] = np.array([120, 120, 80])
        cmap[6] = np.array([140, 140, 140])
        cmap[7] = np.array([204, 5, 255])
        cmap[8] = np.array([230, 230, 230])
        cmap[9] = np.array([4, 250, 7])
        cmap[10] = np.array([224, 5, 255])
        cmap[11] = np.array([235, 255, 7])
        cmap[12] = np.array([150, 5, 61])
        cmap[13] = np.array([120, 120, 70])
        cmap[14] = np.array([8, 255, 51])
        cmap[15] = np.array([255, 6, 82])
        cmap[16] = np.array([143, 255, 140])
        cmap[17] = np.array([204, 255, 4])
        cmap[18] = np.array([255, 51, 7])
        cmap[19] = np.array([204, 70, 3])
        cmap[20] = np.array([0, 102, 200])
        cmap[21] = np.array([61, 230, 250])
        cmap[22] = np.array([255, 6, 51])
        cmap[23] = np.array([11, 102, 255])
        cmap[24] = np.array([255, 7, 71])
        cmap[25] = np.array([255, 9, 224])
        cmap[26] = np.array([9, 7, 230])
        cmap[27] = np.array([220, 220, 220])
        cmap[28] = np.array([255, 9, 92])
        cmap[29] = np.array([112, 9, 255])
        cmap[30] = np.array([8, 255, 214])
        cmap[31] = np.array([7, 255, 224])
        cmap[32] = np.array([255, 184, 6])
        cmap[33] = np.array([10, 255, 71])
        cmap[34] = np.array([255, 41, 10])
        cmap[35] = np.array([7, 255, 255])
        cmap[36] = np.array([224, 255, 8])
        cmap[37] = np.array([102, 8, 255])
        cmap[38] = np.array([255, 61, 6])
        cmap[39] = np.array([255, 194, 7])
        cmap[40] = np.array([255, 122, 8])
        cmap[41] = np.array([0, 255, 20])
        cmap[42] = np.array([255, 8, 41])
        cmap[43] = np.array([255, 5, 153])
        cmap[44] = np.array([6, 51, 255])
        cmap[45] = np.array([235, 12, 255])
        cmap[46] = np.array([160, 150, 20])
        cmap[47] = np.array([0, 163, 255])
        cmap[48] = np.array([140, 140, 140])
        cmap[49] = np.array([250, 10, 15])
        cmap[50] = np.array([20, 255, 0])
        cmap[51] = np.array([31, 255, 0])
        cmap[52] = np.array([255, 31, 0])
        cmap[53] = np.array([255, 224, 0])
        cmap[54] = np.array([153, 255, 0])
        cmap[55] = np.array([0, 0, 255])
        cmap[56] = np.array([255, 71, 0])
        cmap[57] = np.array([0, 235, 255])
        cmap[58] = np.array([0, 173, 255])
        cmap[59] = np.array([31, 0, 255])
        cmap[60] = np.array([11, 200, 200])
        cmap[61] = np.array([255, 82, 0])
        cmap[62] = np.array([0, 255, 245])
        cmap[63] = np.array([0, 61, 255])
        cmap[64] = np.array([0, 255, 112])
        cmap[65] = np.array([0, 255, 133])
        cmap[66] = np.array([255, 0, 0])
        cmap[67] = np.array([255, 163, 0])
        cmap[68] = np.array([255, 102, 0])
        cmap[69] = np.array([194, 255, 0])
        cmap[70] = np.array([0, 143, 255])
        cmap[71] = np.array([51, 255, 0])
        cmap[72] = np.array([0, 82, 255])
        cmap[73] = np.array([0, 255, 41])
        cmap[74] = np.array([0, 255, 173])
        cmap[75] = np.array([10, 0, 255])
        cmap[76] = np.array([173, 255, 0])
        cmap[77] = np.array([0, 255, 153])
        cmap[78] = np.array([255, 92, 0])
        cmap[79] = np.array([255, 0, 255])
        cmap[80] = np.array([255, 0, 245])
        cmap[81] = np.array([255, 0, 102])
        cmap[82] = np.array([255, 173, 0])
        cmap[83] = np.array([255, 0, 20])
        cmap[84] = np.array([255, 184, 184])
        cmap[85] = np.array([0, 31, 255])
        cmap[86] = np.array([0, 255, 61])
        cmap[87] = np.array([0, 71, 255])
        cmap[88] = np.array([255, 0, 204])
        cmap[89] = np.array([0, 255, 194])
        cmap[90] = np.array([0, 255, 82])
        cmap[91] = np.array([0, 10, 255])
        cmap[92] = np.array([0, 112, 255])
        cmap[93] = np.array([51, 0, 255])
        cmap[94] = np.array([0, 194, 255])
        cmap[95] = np.array([0, 122, 255])
        cmap[96] = np.array([0, 255, 163])
        cmap[97] = np.array([255, 153, 0])
        cmap[98] = np.array([0, 255, 10])
        cmap[99] = np.array([255, 112, 0])
        cmap[100] = np.array([143, 255, 0])
        cmap[101] = np.array([82, 0, 255])
        cmap[102] = np.array([163, 255, 0])
        cmap[103] = np.array([255, 235, 0])
        cmap[104] = np.array([8, 184, 170])
        cmap[105] = np.array([133, 0, 255])
        cmap[106] = np.array([0, 255, 92])
        cmap[107] = np.array([184, 0, 255])
        cmap[108] = np.array([255, 0, 31])
        cmap[109] = np.array([0, 184, 255])
        cmap[110] = np.array([0, 214, 255])
        cmap[111] = np.array([255, 0, 112])
        cmap[112] = np.array([92, 255, 0])
        cmap[113] = np.array([0, 224, 255])
        cmap[114] = np.array([112, 224, 255])
        cmap[115] = np.array([70, 184, 160])
        cmap[116] = np.array([163, 0, 255])
        cmap[117] = np.array([153, 0, 255])
        cmap[118] = np.array([71, 255, 0])
        cmap[119] = np.array([255, 0, 163])
        cmap[120] = np.array([255, 204, 0])
        cmap[121] = np.array([255, 0, 143])
        cmap[122] = np.array([0, 255, 235])
        cmap[123] = np.array([133, 255, 0])
        cmap[124] = np.array([255, 0, 235])
        cmap[125] = np.array([245, 0, 255])
        cmap[126] = np.array([255, 0, 122])
        cmap[127] = np.array([255, 245, 0])
        cmap[128] = np.array([10, 190, 212])
        cmap[129] = np.array([214, 255, 0])
        cmap[130] = np.array([0, 204, 255])
        cmap[131] = np.array([20, 0, 255])
        cmap[132] = np.array([255, 255, 0])
        cmap[133] = np.array([0, 153, 255])
        cmap[134] = np.array([0, 41, 255])
        cmap[135] = np.array([0, 255, 204])
        cmap[136] = np.array([41, 0, 255])
        cmap[137] = np.array([41, 255, 0])
        cmap[138] = np.array([173, 0, 255])
        cmap[139] = np.array([0, 245, 255])
        cmap[140] = np.array([71, 0, 255])
        cmap[141] = np.array([122, 0, 255])
        cmap[142] = np.array([0, 255, 184])
        cmap[143] = np.array([0, 92, 255])
        cmap[144] = np.array([184, 255, 0])
        cmap[145] = np.array([0, 133, 255])
        cmap[146] = np.array([255, 214, 0])
        cmap[147] = np.array([25, 194, 194])
        cmap[148] = np.array([102, 255, 0])
        cmap[149] = np.array([92, 0, 255])

    return cmap


def update_loss_history(save_path, epoch, train_loss, val_loss):
    """Append one epoch's losses to loss_history.json in save_path.

    Loads the existing JSON history (or starts a fresh list), removes any
    pre-existing entry for this epoch (handles resume-and-overwrite), appends
    the new entry, sorts by epoch, and writes back.  Safe to call on every
    epoch so that interrupting and resuming training preserves the full history.

    Parameters
    ----------
    save_path : str
        Directory where the model checkpoints live.
    epoch : int
        Zero-based epoch index.
    train_loss : float
        Average training loss for this epoch.
    val_loss : float
        Validation loss for this epoch.
    """
    history_path = os.path.join(save_path, "loss_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = []

    history = [e for e in history if e["epoch"] != epoch]
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    history.sort(key=lambda e: e["epoch"])

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def compute_lr(iters, total_iters, warmup_iters, max_lr, min_lr):
    """Compute learning rate with linear warmup then polynomial decay.

    Linear ramp from ``min_lr`` to ``max_lr`` over ``warmup_iters`` steps,
    then polynomial decay from ``max_lr`` back down to ``min_lr`` for the
    remaining steps using exponent 0.9:

        lr = min_lr + (max_lr - min_lr) * (1 - progress) ** 0.9

    Using ``min_lr`` as the warmup starting point (rather than 0) is
    recommended for AdamW: a cold start at exactly 0 stalls Adam's second
    moment estimate; ``min_lr`` provides a safe non-zero floor while still
    giving a meaningful ramp.

    Parameters
    ----------
    iters : int
        Current global training iteration (0-based).
    total_iters : int
        Total number of training iterations.
    warmup_iters : int
        Number of linear warmup iterations.
    max_lr : float
        Peak learning rate (reached at end of warmup).
    min_lr : float
        Floor learning rate (warmup start and decay end).

    Returns
    -------
    float
        Learning rate for iteration ``iters``.
    """
    if iters < warmup_iters:
        return min_lr + (max_lr - min_lr) * iters / max(warmup_iters, 1)
    progress = (iters - warmup_iters) / max(total_iters - warmup_iters, 1)
    return min_lr + (max_lr - min_lr) * (1 - progress) ** 0.9
