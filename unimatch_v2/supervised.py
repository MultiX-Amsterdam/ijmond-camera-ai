import argparse
import logging
import glob
import os
import pprint

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
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
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

def evaluate_new(model, dataloader, w_pos=0.8, w_neg=0.2, threshold=0.5, multiplier=None):
    """
    Calculates weighted mIoU, mF2, mRecall, and mPrecision.
    """
    model.eval()

    # Grouped storage for per-image metrics
    smoke_f2s, smoke_ious, smoke_recalls, smoke_precisions = [], [], [], []
    clear_f2s, clear_ious, clear_recalls, clear_precisions = [], [], [], []
    smoke_accu, clear_accu = [], []

    smooth = 1e-7

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images, masks = images.cuda(), masks.cuda()

            if multiplier is not None:
                ori_h, ori_w = images.shape[-2:]
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

                images = F.interpolate(images, (new_h, new_w), mode='bilinear', align_corners=True)

            outputs = model(images)

            if multiplier is not None:
                outputs = F.interpolate(outputs, (ori_h, ori_w), mode='bilinear', align_corners=True)

            preds = (outputs > threshold).float()
            preds = preds.argmax(dim=1)

            for p, m in zip(preds, masks):
                p_np = p.cpu().numpy()
                m_np = m.cpu().numpy()

                intersection, union_arr, _ = intersectionAndUnion(p_np[None], m_np[None], 2, 255)

                # Pixel-level components
                tp = (p * m).sum().item()
                fp = (p * (1 - m)).sum().item()
                fn = ((1 - p) * m).sum().item()
                union = p.sum().item() + m.sum().item() - tp

                precision = (tp + smooth) / (tp + fp + smooth)
                recall = (tp + smooth) / (tp + fn + smooth)

                # --- CASE 1: NEGATIVE SAMPLE (No Smoke Present) ---
                if m.sum() == 0:
                    clear_iou = (intersection[0].sum() + smooth) / (union_arr[0].sum() + smooth)
                    clear_ious.append(clear_iou)
                    clear_accu.append((p_np == m_np).sum() / p_np.size)

                    score = 1.0 if p.sum() == 0 else 0.0
                    clear_f2s.append(score)
                    clear_recalls.append(score)
                    clear_precisions.append(score)

                # --- CASE 2: POSITIVE SAMPLE (Smoke Present) ---
                else:
                    smoke_iou = (intersection[1].sum() + smooth) / (union_arr[1].sum() + smooth)
                    smoke_ious.append(smoke_iou)
                    smoke_accu.append((p_np == m_np).sum() / p_np.size)

                    # F2 Score
                    f2 = (5 * precision * recall) / (4 * precision + recall + smooth)
                    smoke_f2s.append(f2)
                    smoke_recalls.append(recall)
                    smoke_precisions.append(precision)

    # 1. Calculate the raw means for both groups
    mF2_smoke = np.mean(smoke_f2s) if smoke_f2s else 0.0
    mIoU_smoke = np.mean(smoke_ious) if smoke_ious else 0.0
    mRec_smoke = np.mean(smoke_recalls) if smoke_recalls else 0.0
    mPre_smoke = np.mean(smoke_precisions) if smoke_precisions else 0.0
    mAccu_smoke = np.mean(smoke_accu) if smoke_accu else 0.0

    mF2_clear = np.mean(clear_f2s) if clear_f2s else 0.0
    mIoU_clear = np.mean(clear_ious) if clear_ious else 0.0
    mRec_clear = np.mean(clear_recalls) if clear_recalls else 0.0
    mPre_clear = np.mean(clear_precisions) if clear_precisions else 0.0
    mAccu_clear = np.mean(clear_accu) if clear_accu else 0.0

    # 2. Compute Weighted Final Metrics (The ones used for ranking)
    weight_sum = w_pos + w_neg

    results = {
        "mIoU": (w_pos * mIoU_smoke + w_neg * mIoU_clear) / weight_sum,
        "mF2": (w_pos * mF2_smoke + w_neg * mF2_clear) / weight_sum,
        "mRec": (w_pos * mRec_smoke + w_neg * mRec_clear) / weight_sum,
        "mPre": (w_pos * mPre_smoke + w_neg * mPre_clear) / weight_sum,
        "mAccu": (w_pos * mAccu_smoke + w_neg * mAccu_clear) / weight_sum,
        "mF2_smoke": mF2_smoke,
        "mIoU_smoke": mIoU_smoke,
        "mRec_smoke": mRec_smoke,
        "mPre_smoke": mPre_smoke,
        "mAccu_smoke": mAccu_smoke,
        "mF2_clear": mF2_clear,
        "mIoU_clear": mIoU_clear,
        "mRec_clear": mRec_clear,
        "mPre_clear": mPre_clear,
        "mAccu_clear": mAccu_clear
    }

    return results


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

    model = DPT(**{
        **model_configs[cfg['backbone'].split('_')[-1]],
        'nclass': cfg['nclass']
    })

    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')

    model.backbone.load_state_dict(state_dict)

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
        weight_decay=0.01
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True
    )

    if cfg['criterion']['name'] == 'CELoss':
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
        nsample=None
    )

    valset = SemiSmokeDataset(
        cfg['dataset'],
        cfg['data_root'],
        'val',
        id_path=training_cfg['validation_dataset']
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
        num_workers=1,
        drop_last=False,
        shuffle=False
    )

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    best_evaluations = {}
    best_epoch = -1
    best_iou = 0.0
    best_f2 = 0.0
    best_accu = 0.0
    epoch = -1

    # if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
    #     checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu', weights_only=False)
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     epoch = checkpoint['epoch']
    #     previous_best = checkpoint['previous_best']

    #     if rank == 0:
    #         logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info(
                'Current Epoch: {}, LR: {:.7f} | '
                'Best Epoch: {}, '
                'mIoU: {:.4f}, '
                'mF2: {:.4f}, '
                'mAccu {:.4f}'.format(
                    epoch,
                    optimizer.param_groups[0]['lr'],
                    best_epoch,
                    best_iou,
                    best_f2,
                    best_accu
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
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)

            if (i % (len(trainloader) // 8) == 0) and (rank == 0):
                logger.info('Rank {}: Iters: {:}, Total loss: {:.3f}'.format(rank, i, total_loss.avg))

        if rank == 0:
            evaluation = evaluate_new(model, valloader, multiplier=14)

            logger.info('***** Evaluation ***** >>>> mIoU: {:.4f}[{:.4f}, {:.4f}]'.format(
                evaluation["mIoU"], evaluation["mIoU_clear"], evaluation["mIoU_smoke"]
            ))

            logger.info('***** Evaluation ***** >>>> mF2: {:.4f}[{:.4f}, {:.4f}]'.format(
                evaluation["mF2"], evaluation["mF2_clear"], evaluation["mF2_smoke"]
            ))

            logger.info('***** Evaluation ***** >>>> mAccu: {:.4f}[{:.4f}, {:.4f}]'.format(
                evaluation["mAccu"], evaluation["mAccu_clear"], evaluation["mAccu_smoke"]
            ))

            writer.add_scalar('eval/mIoU', evaluation["mIoU"], epoch)
            writer.add_scalar('eval/mF2', evaluation["mF2"], epoch)
            writer.add_scalar('eval/mRec', evaluation["mRec"], epoch)
            writer.add_scalar('eval/mPre', evaluation["mPre"], epoch)
            writer.add_scalar('eval/mAccu', evaluation["mAccu"], epoch)
            writer.add_scalar('eval/mIoU_smoke', evaluation["mIoU_smoke"], epoch)
            writer.add_scalar('eval/mF2_smoke', evaluation["mF2_smoke"], epoch)
            writer.add_scalar('eval/mRec_smoke', evaluation["mRec_smoke"], epoch)
            writer.add_scalar('eval/mPre_smoke', evaluation["mPre_smoke"], epoch)
            writer.add_scalar('eval/mAccu_smoke', evaluation["mAccu_smoke"], epoch)
            writer.add_scalar('eval/mIoU_clear', evaluation["mIoU_clear"], epoch)
            writer.add_scalar('eval/mF2_clear', evaluation["mF2_clear"], epoch)
            writer.add_scalar('eval/mRec_clear', evaluation["mRec_clear"], epoch)
            writer.add_scalar('eval/mPre_clear', evaluation["mPre_clear"], epoch)
            writer.add_scalar('eval/mAccu_clear', evaluation["mAccu_clear"], epoch)

            evaluation["checkpoint"] = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }

            miou, mf2 = evaluation["mIoU"], evaluation["mF2"]
            deleted = -1

            if len(best_evaluations) < 10:
                best_evaluations[epoch] = evaluation
            else:
                lowest_epoch, lowest_evaluation = sorted(
                    best_evaluations.items(),
                    key=lambda kv: kv[1]["mIoU"]
                )[0]
                if miou > lowest_evaluation['mIoU']:
                    del best_evaluations[lowest_epoch]
                    best_evaluations[epoch] = evaluation

            best_epoch, best_evaluation = sorted(
                best_evaluations.items(),
                key=lambda kv: kv[1]["mF2"],
                reverse=True
            )[0]

            best_iou = best_evaluation["mIoU"]
            best_f2 = best_evaluation["mF2"]
            best_accu = best_evaluation["mAccu"]

        dist.barrier()

    if rank == 0:
        for ep, eval_ in best_evaluations.items():
            torch.save(eval_, os.path.join(args.save_path, f"checkpoint_{ep}.pth"))

        best_epoch, best_evaluation = sorted(
            best_evaluations.items(),
            key=lambda kv: kv[1]["mF2"],
            reverse=True
        )[0]

        torch.save(best_evaluation, os.path.join(args.save_path, f"best.pth"))


if __name__ == '__main__':
    main()
