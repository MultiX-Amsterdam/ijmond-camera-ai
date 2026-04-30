import argparse
import logging
import glob
import os
import pprint

import timm.models
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiSmokeDataset
from model.semseg.dpt import DPT
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log, BoundaryLenienceCELoss
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--training-config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--unlabeled-sample-size', type=int, required=False)
parser.add_argument('--unlabeled-sample-seed', type=int, required=False)

"""
def evaluate(model, loader, cfg, multiplier=None):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda()
            ori_h, ori_w = img.shape[-2:]

            if multiplier is not None:
                if multiplier == 512:
                    new_h, new_w = 512, 512
                else:
                    new_h = int(ori_h / multiplier + 0.5) * multiplier
                    new_w = int(ori_w / multiplier + 0.5) * multiplier
                    max_dim = 400
                    if max(new_h, new_w) > max_dim:
                        scale = max_dim / max(new_h, new_w)
                        new_h = max(int(ori_h * scale / multiplier + 0.5) * multiplier, multiplier)
                        new_w = max(int(ori_w * scale / multiplier + 0.5) * multiplier, multiplier)
                    logger.info(f"Original size: ({ori_h}, {ori_w}), Resized size: ({new_h}, {new_w})")
                img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)

            pred = model(img)

            if multiplier is not None:
                pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)

            pred = pred.argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            correct_pixel.update((pred.cpu() == mask).sum().item())
            total_pixel.update(pred.numel())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    overall_acc = correct_pixel.sum / (total_pixel.sum + 1e-10) * 100.0

    return mIOU, iou_class, overall_acc
"""

def _compute_boundary_band(masks, kernel_size):
    """Return a bool tensor marking ambiguous boundary pixels.

    Uses morphological dilation and erosion (via max-pooling) on the binary
    smoke mask. Pixels where the two operations disagree lie on the boundary
    between guaranteed smoke and guaranteed background.

    Args:
        masks: integer label tensor of shape (B, H, W).
        kernel_size: square kernel side length for dilation/erosion.

    Returns:
        Boolean tensor of shape (B, H, W); True where pixels are in the band.
    """
    m = (masks == 1).float().unsqueeze(1)  # (B, 1, H, W)
    padding = kernel_size // 2
    dilated = F.max_pool2d(m, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-m, kernel_size, stride=1, padding=padding)
    band = (dilated - eroded).squeeze(1)  # (B, H, W)
    return band > 0.5


def evaluate_new(model, dataloader, multiplier=None, band_kernel_size=None):
    """
    Calculates smoke-class evaluation metrics.

    Global metrics (gIoU, gF1, gPre, gRec, gAccu) are computed by
    accumulating TP/FP/FN across all images.

    Per-image metrics (mIoU, mF1) are averaged over positive images only
    (images where ground truth contains smoke). Negative images are excluded
    from mIoU/mF1 to avoid degenerate computation.

    False Alarm Rate (FAR) is computed as the fraction of negative images
    where the model incorrectly predicts any smoke pixels.

    When band_kernel_size is set, ambiguous boundary pixels (computed via
    morphological dilation/erosion with that kernel) are excluded from all
    TP/FP/FN accumulation before computing metrics.
    """
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_correct = 0
    total_pixels = 0

    per_image_ious = []
    per_image_f1s = []
    num_negative = 0
    num_false_alarm = 0

    smooth = 1e-7

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images, masks = images.cuda(), masks.cuda()

            if multiplier is not None:
                ori_h, ori_w = images.shape[-2:]
                new_h = int(ori_h / multiplier + 0.5) * multiplier
                new_w = int(ori_w / multiplier + 0.5) * multiplier
                max_dim = 400
                if max(new_h, new_w) > max_dim:
                    scale = max_dim / max(new_h, new_w)
                    new_h = max(int(ori_h * scale / multiplier + 0.5) * multiplier, multiplier)
                    new_w = max(int(ori_w * scale / multiplier + 0.5) * multiplier, multiplier)
                images = F.interpolate(images, (new_h, new_w), mode='bilinear', align_corners=True)

            with torch.autocast("cuda"):
                outputs = model(images)
            outputs = outputs.float()

            if multiplier is not None:
                outputs = F.interpolate(outputs, (ori_h, ori_w), mode='bilinear', align_corners=True)

            preds = outputs.argmax(dim=1)

            smoke_pred = preds == 1
            smoke_gt = masks == 1

            # valid_mask excludes boundary-band pixels when band_kernel_size is set
            if band_kernel_size is not None:
                valid_mask = ~_compute_boundary_band(masks, band_kernel_size)
            else:
                valid_mask = torch.ones_like(masks, dtype=torch.bool)

            total_tp += (smoke_pred & smoke_gt & valid_mask).sum().item()
            total_fp += (smoke_pred & ~smoke_gt & valid_mask).sum().item()
            total_fn += (~smoke_pred & smoke_gt & valid_mask).sum().item()
            total_correct += ((preds == masks) & valid_mask).sum().item()
            total_pixels += valid_mask.sum().item()

            for b in range(masks.shape[0]):
                sp = smoke_pred[b]
                sg = smoke_gt[b]
                vm = valid_mask[b]
                if sg.sum().item() > 0:
                    tp = (sp & sg & vm).sum().item()
                    fp = (sp & ~sg & vm).sum().item()
                    fn = (~sp & sg & vm).sum().item()
                    img_iou = (tp + smooth) / (tp + fp + fn + smooth)
                    img_pre = (tp + smooth) / (tp + fp + smooth)
                    img_rec = (tp + smooth) / (tp + fn + smooth)
                    img_f1 = (2 * img_pre * img_rec) / (img_pre + img_rec + smooth)
                    per_image_ious.append(img_iou)
                    per_image_f1s.append(img_f1)
                else:
                    num_negative += 1
                    # Only trigger an alarm if a cluster of pixels is predicted
                    noise_threshold = 10
                    if sp.sum().item() > noise_threshold:
                        num_false_alarm += 1

    iou = (total_tp + smooth) / (total_tp + total_fp + total_fn + smooth)
    precision = (total_tp + smooth) / (total_tp + total_fp + smooth)
    recall = (total_tp + smooth) / (total_tp + total_fn + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    accuracy = total_correct / max(total_pixels, 1)

    miou = np.mean(per_image_ious) if per_image_ious else 0.0
    mf1 = np.mean(per_image_f1s) if per_image_f1s else 0.0
    far = num_false_alarm / max(num_negative, 1)

    return {
        "gIoU": iou,
        "gF1": f1,
        "gPre": precision,
        "gRec": recall,
        "gAccu": accuracy,
        "mIoU": miou,
        "mF1": mf1,
        "FAR": far,
    }


def main():
    args = parser.parse_args()
    training_cfg = yaml.load(open(args.training_config, "r"), Loader=yaml.Loader)

    cfg = yaml.load(
        open(training_cfg['configuration'], "r"),
        Loader=yaml.Loader
    )

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}

        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    is_dinov3 = cfg['backbone'].startswith('dinov3_')

    model = DPT(**{
        **model_configs[cfg['backbone'].split('_')[-1]],
        'nclass': cfg['nclass'],
        'use_dcp': cfg.get('use_dcp', False),
        'backbone_type': 'dinov3' if is_dinov3 else 'dinov2',
    })

    # --- DINOv2-only: remove when dropping DINOv2 ---
    if not is_dinov3:
        state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
        model.backbone.load_state_dict(state_dict)
    # --- end DINOv2-only ---
    else:
        timm.models.load_checkpoint(
            model.backbone.model,
            f'./pretrained/{cfg["backbone"]}.safetensors',
            strict=False,
        )

    if cfg['lock_backbone']:
        model.lock_backbone()

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)

    optimizer = AdamW(
        [
            {
                'params': [p for p in model.backbone.parameters() if p.requires_grad],
                'lr': cfg['lr']
            },
            {
                'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                'lr': cfg['lr'] * cfg['lr_multi']
            }
        ],
        lr=cfg['lr'],
        betas=(0.9, 0.999),
        weight_decay=cfg['weight_decay']
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True
    )

    bl_cfg = cfg.get('boundary_lenience', {})
    if cfg['criterion']['name'] == 'CELoss':
        if bl_cfg.get('enabled', False):
            criterion = BoundaryLenienceCELoss(
                ignore_index=cfg['criterion']['kwargs'].get('ignore_index', 255),
                alpha=bl_cfg.get('alpha', 0.5),
                window_size=bl_cfg.get('window_size', 7),
                reduction='mean',
            ).cuda(local_rank)
        else:
            criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiSmokeDataset(
        cfg['dataset'],
        cfg['data_root'],
        'train_l',
        cfg['crop_size'],
        base_size=cfg.get('base_size'),
        id_path=training_cfg['smoke_dataset'],
        nsample=None,
        use_dcp=cfg.get('use_dcp', False),
    )

    valset = SemiSmokeDataset(
        cfg['dataset'],
        cfg['data_root'],
        'val',
        id_path=training_cfg['validation_dataset'],
        use_dcp=cfg.get('use_dcp', False),
    )

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)

    trainloader = DataLoader(
        trainset,
        batch_size=cfg['batch_size'],
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler
    )

    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
        shuffle=False
    )

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    best_epoch = -1
    best_eval = None
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_eval = checkpoint['best_eval']

        if rank == 0:
            logger.info('************ Resumed from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            if best_eval is not None:
                logger.info(
                    'Current Epoch: {}, LR: {:.7f} | '
                    'Best Epoch: {}, '
                    'gIoU: {:.4f}, '
                    'gF1: {:.4f}, '
                    'gAccu: {:.4f}, '
                    'mIoU: {:.4f}, '
                    'mF1: {:.4f}, '
                    'FAR: {:.4f}'.format(
                        epoch,
                        optimizer.param_groups[0]['lr'],
                        best_epoch,
                        best_eval["gIoU"],
                        best_eval["gF1"],
                        best_eval["gAccu"],
                        best_eval["mIoU"],
                        best_eval["mF1"],
                        best_eval["FAR"]
                    )
                )
            else:
                logger.info(
                    'Current Epoch: {}, LR: {:.7f} | '
                    'Best Epoch: N/A'.format(
                        epoch,
                        optimizer.param_groups[0]['lr']
                    )
                )

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img = img.cuda(local_rank, non_blocking=True)
            mask = mask.cuda(local_rank, non_blocking=True)
            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i

            lr = max(cfg['lr'] * (1 - iters / total_iters) ** 0.9, cfg['min_lr'])
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/lr', lr, iters)

            if (i % max(len(trainloader) // 8, 1) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg))

        if rank == 0:
            evaluation = evaluate_new(model, valloader, multiplier=model.module.patch_size)

            logger.info(
                '***** Evaluation ***** >>>> gIoU: {:.4f}, gF1: {:.4f}, gAccu: {:.4f}'.format(
                    evaluation["gIoU"], evaluation["gF1"], evaluation["gAccu"]
                ))

            logger.info(
                '***** Evaluation ***** >>>> gPre: {:.4f}, gRec: {:.4f}'.format(
                    evaluation["gPre"], evaluation["gRec"]
                ))

            logger.info(
                '***** Evaluation ***** >>>> mIoU: {:.4f}, mF1: {:.4f}, FAR: {:.4f}'.format(
                    evaluation["mIoU"], evaluation["mF1"], evaluation["FAR"]
                ))

            writer.add_scalar('eval/gIoU', evaluation["gIoU"], epoch)
            writer.add_scalar('eval/gF1', evaluation["gF1"], epoch)
            writer.add_scalar('eval/gPre', evaluation["gPre"], epoch)
            writer.add_scalar('eval/gRec', evaluation["gRec"], epoch)
            writer.add_scalar('eval/gAccu', evaluation["gAccu"], epoch)
            writer.add_scalar('eval/mIoU', evaluation["mIoU"], epoch)
            writer.add_scalar('eval/mF1', evaluation["mF1"], epoch)
            writer.add_scalar('eval/FAR', evaluation["FAR"], epoch)

            if best_eval is None or evaluation["gF1"] > best_eval["gF1"]:
                best_epoch = epoch
                best_eval = {k: v for k, v in evaluation.items()}
                save_dict = {**best_eval, "checkpoint": {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }}
                torch.save(save_dict, os.path.join(args.save_path, "best.pth"))

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_eval': best_eval,
            }, os.path.join(args.save_path, 'latest.pth'))

        dist.barrier()


if __name__ == '__main__':
    main()
