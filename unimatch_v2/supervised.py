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
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log, BoundaryLenienceCELoss, BoundaryLenienceDiceLoss, dice_loss, eval_score, update_loss_history, compute_lr, is_pareto_optimal, update_pareto_front, calculate_angle_score
from plot_training_curves import plot_loss_curve
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--training-config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

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


def evaluate_new(model, dataloader, multiplier=None, band_kernel_size=None, criterion=None):
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
    val_loss_meter = AverageMeter()

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

            if criterion is not None:
                val_loss = criterion(outputs, masks)
                val_loss_meter.update(val_loss.item(), images.shape[0])

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

    result = {
        "gIoU": iou,
        "gF1": f1,
        "gPre": precision,
        "gRec": recall,
        "gAccu": accuracy,
        "mIoU": miou,
        "mF1": mf1,
        "FAR": far,
    }
    if criterion is not None:
        result["val_loss"] = val_loss_meter.avg
    return result


def main():
    torch.cuda.empty_cache()

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

        logger.info('Training config ({}):\n{}\n'.format(args.training_config, pprint.pformat(training_cfg)))
        logger.info('Model config ({}):\n{}\n'.format(training_cfg['configuration'], pprint.pformat(all_args)))

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
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))

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

    if pretrained_model_name := training_cfg.get("pretrained_model", "").strip():
        pretrained_best_pth = os.path.join(os.path.dirname(args.save_path), pretrained_model_name, "best.pth")

        if os.path.exists(pretrained_best_pth):
            checkpoint = torch.load(
                pretrained_best_pth,
                map_location='cpu',
                weights_only=False
            )
            model.load_state_dict(checkpoint["checkpoint"]["model"])
            if rank == 0:
                logger.info("Loaded %s at epoch %s" % (pretrained_best_pth, checkpoint["checkpoint"]["epoch"]))

    bl_cfg = cfg.get('boundary_lenience', {})
    if cfg['criterion']['name'] == 'CELossAndDiceLoss':
        if bl_cfg.get('enabled', False):
            criterion = BoundaryLenienceCELoss(
                ignore_index=cfg['criterion']['kwargs'].get('ignore_index', 255),
                alpha=bl_cfg.get('alpha', 0.5),
                window_size=bl_cfg.get('window_size', 7),
                reduction='mean',
            ).cuda(local_rank)
            dice_loss_fn = BoundaryLenienceDiceLoss(
                ignore_index=cfg['criterion']['kwargs'].get('ignore_index', 255),
                alpha=bl_cfg.get('alpha', 0.5),
                window_size=bl_cfg.get('window_size', 7),
            ).cuda(local_rank)
        else:
            criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
            dice_loss_fn = dice_loss
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

    val_robust_set = SemiSmokeDataset(
        cfg['dataset'],
        cfg['data_root'],
        'val_robust',
        id_path=training_cfg['val_robust_dataset'],
        use_dcp=cfg.get('use_dcp', False),
    )

    val_robust_loader = DataLoader(
        val_robust_set,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
        shuffle=False
    )

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    warmup_iters = cfg.get('warmup_epochs', 5) * len(trainloader)
    best_epoch = -1
    best_eval = None
    best_robust_eval = None
    best_score = None
    pareto_history = []
    prev_angle_score = None
    epoch = -1

    initial_weights_path = os.path.join(args.save_path, 'initial_weights.pth')

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_eval = checkpoint['best_eval']
        best_score = checkpoint.get('best_score')
        pareto_history = checkpoint.get('pareto_history', [])
        prev_angle_score = checkpoint.get('prev_angle_score', None)
        best_robust_eval = checkpoint.get('best_robust_eval', None)

        if rank == 0:
            logger.info('************ Resumed from checkpoint at epoch %i\n' % epoch)

    if rank == 0 and not os.path.exists(initial_weights_path):
        torch.save(
            {k: v.cpu().clone() for k, v in model.module.state_dict().items()},
            initial_weights_path
        )

    dist.barrier()

    initial_state_dict = torch.load(initial_weights_path, map_location='cpu', weights_only=True)

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
        total_loss_ce = AverageMeter()
        total_loss_dice = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img = img.cuda(local_rank, non_blocking=True)
            mask = mask.cuda(local_rank, non_blocking=True)
            pred = model(img)
            loss_ce_w = cfg['criterion'].get('loss_ce_weight', 0.5)
            loss_dice_w = cfg['criterion'].get('loss_dice_weight', 0.5)
            loss_ce = criterion(pred, mask) if loss_ce_w > 0 else torch.zeros(1, device=img.device)
            loss_dice = dice_loss_fn(pred, mask) if loss_dice_w > 0 else torch.zeros(1, device=img.device)
            loss = loss_ce_w * loss_ce + loss_dice_w * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_ce.update(loss_ce.item())
            total_loss_dice.update(loss_dice.item())

            iters = epoch * len(trainloader) + i

            lr = compute_lr(iters, total_iters, warmup_iters, cfg['lr'], cfg['min_lr'])
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
                writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
                writer.add_scalar('train/lr', lr, iters)

            if (i % max(len(trainloader) // 8, 1) == 0) and (rank == 0):
                logger.info('Iters: {:04d}, LR: {:.7f}, Total loss: {:.3f}, Loss CE: {:.3f}, Loss dice: {:.3f}'.format(
                    i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_ce.avg, total_loss_dice.avg))

        if rank == 0:
            logger.info('***** Epoch {:04d} ***** >>>> Train Loss: {:.4f}'.format(epoch, total_loss.avg))

            evaluation = evaluate_new(model, valloader, multiplier=model.module.patch_size, criterion=criterion)

            logger.info(
                '***** Evaluation ***** >>>> Val Loss: {:.4f}'.format(evaluation["val_loss"])
            )
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

            robust_evaluation = evaluate_new(model, val_robust_loader, multiplier=model.module.patch_size, criterion=criterion)

            logger.info(
                '***** Robust Evaluation ***** >>>> Val Loss: {:.4f}'.format(robust_evaluation["val_loss"])
            )
            logger.info(
                '***** Robust Evaluation ***** >>>> gIoU: {:.4f}, gF1: {:.4f}, gAccu: {:.4f}'.format(
                    robust_evaluation["gIoU"], robust_evaluation["gF1"], robust_evaluation["gAccu"]
                ))
            logger.info(
                '***** Robust Evaluation ***** >>>> gPre: {:.4f}, gRec: {:.4f}'.format(
                    robust_evaluation["gPre"], robust_evaluation["gRec"]
                ))
            logger.info(
                '***** Robust Evaluation ***** >>>> mIoU: {:.4f}, mF1: {:.4f}, FAR: {:.4f}'.format(
                    robust_evaluation["mIoU"], robust_evaluation["mF1"], robust_evaluation["FAR"]
                ))

            gF1 = evaluation["gF1"]
            rF1 = robust_evaluation["gF1"]

            writer.add_scalar('train/loss_epoch', total_loss.avg, epoch)
            writer.add_scalar('eval/val_loss', evaluation["val_loss"], epoch)
            writer.add_scalar('eval/gIoU', evaluation["gIoU"], epoch)
            writer.add_scalar('eval/gF1', evaluation["gF1"], epoch)
            writer.add_scalar('eval/gPre', evaluation["gPre"], epoch)
            writer.add_scalar('eval/gRec', evaluation["gRec"], epoch)
            writer.add_scalar('eval/gAccu', evaluation["gAccu"], epoch)
            writer.add_scalar('eval/mIoU', evaluation["mIoU"], epoch)
            writer.add_scalar('eval/mF1', evaluation["mF1"], epoch)
            writer.add_scalar('eval/FAR', evaluation["FAR"], epoch)
            writer.add_scalar('eval/robust_val_loss', robust_evaluation["val_loss"], epoch)
            writer.add_scalar('eval/robust_gIoU', robust_evaluation["gIoU"], epoch)
            writer.add_scalar('eval/robust_gF1', robust_evaluation["gF1"], epoch)
            writer.add_scalar('eval/robust_gPre', robust_evaluation["gPre"], epoch)
            writer.add_scalar('eval/robust_gRec', robust_evaluation["gRec"], epoch)
            writer.add_scalar('eval/robust_gAccu', robust_evaluation["gAccu"], epoch)
            writer.add_scalar('eval/robust_mIoU', robust_evaluation["mIoU"], epoch)
            writer.add_scalar('eval/robust_mF1', robust_evaluation["mF1"], epoch)
            writer.add_scalar('eval/robust_FAR', robust_evaluation["FAR"], epoch)

            model_selection_warmup = cfg.get('model_selection_warmup_epochs', 0)
            if epoch <= model_selection_warmup:
                logger.info('***** Pareto Optimal ***** >>>> Skipping model selection at epoch {:04d} (warmup until epoch {:04d})'.format(epoch, model_selection_warmup))
            elif is_pareto_optimal(gF1, rF1, pareto_history):
                best_epoch = epoch
                best_eval = {k: v for k, v in evaluation.items()}
                best_robust_eval = {k: v for k, v in robust_evaluation.items()}
                best_score = eval_score(evaluation)
                pareto_history = update_pareto_front(epoch, gF1, rF1, pareto_history)
                save_dict = {
                    **{"eval_" + k: v for k, v in best_eval.items()},
                    **{"robust_" + k: v for k, v in best_robust_eval.items()},
                    "checkpoint": {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch
                    }
                }
                torch.save(save_dict, os.path.join(args.save_path, "best.pth"))
                logger.info('***** Pareto Optimal ***** >>>> Saved best.pth at epoch {:04d} (gF1: {:.4f}, rF1: {:.4f})'.format(epoch, gF1, rF1))
            else:
                logger.info('***** Pareto Optimal ***** >>>> Epoch {:04d} is NOT Pareto-optimal (gF1: {:.4f}, rF1: {:.4f})'.format(epoch, gF1, rF1))

            angle_score = calculate_angle_score(model.module, initial_state_dict)
            if prev_angle_score is not None:
                delta_angle = abs(angle_score - prev_angle_score)
                converged = delta_angle < 1e-4
                logger.info(
                    '***** Convergence Check ***** >>>> Angle Score: {:.6f}, Delta Angle: {:.6f}'.format(
                        angle_score, delta_angle
                    )
                )
                logger.info(
                    '***** Convergence Check ***** >>>> Mathematical Condition: {} (Delta={:.6f} vs epsilon=1e-4)'.format(
                        "CONVERGED" if converged else "NOT CONVERGED", delta_angle
                    )
                )
            else:
                logger.info(
                    '***** Convergence Check ***** >>>> Angle Score: {:.6f}, Delta Angle: N/A (first epoch)'.format(
                        angle_score
                    )
                )
            prev_angle_score = angle_score

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_eval': best_eval,
                'best_robust_eval': best_robust_eval,
                'best_score': best_score,
                'pareto_history': pareto_history,
                'prev_angle_score': prev_angle_score,
            }, os.path.join(args.save_path, 'latest.pth'))

            update_loss_history(args.save_path, epoch, total_loss.avg, evaluation["val_loss"])

        dist.barrier()

    if rank == 0:
        plot_loss_curve(args.save_path)


if __name__ == '__main__':
    main()
