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

def evaluate_new(model, dataloader, multiplier=None):
    """
    Calculates smoke-class global IoU, F1 score, and pixel accuracy.

    Metrics are computed globally by accumulating TP/FP/FN across all images,
    focusing solely on smoke pixel detection (class 1).
    """
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_correct = 0
    total_pixels = 0

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

            outputs = model(images)

            if multiplier is not None:
                outputs = F.interpolate(outputs, (ori_h, ori_w), mode='bilinear', align_corners=True)

            preds = outputs.argmax(dim=1)

            smoke_pred = preds == 1
            smoke_gt = masks == 1

            total_tp += (smoke_pred & smoke_gt).sum().item()
            total_fp += (smoke_pred & ~smoke_gt).sum().item()
            total_fn += (~smoke_pred & smoke_gt).sum().item()
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

    iou = (total_tp + smooth) / (total_tp + total_fp + total_fn + smooth)
    precision = (total_tp + smooth) / (total_tp + total_fp + smooth)
    recall = (total_tp + smooth) / (total_tp + total_fn + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    accuracy = total_correct / max(total_pixels, 1)

    return {
        "gIoU": iou,
        "gF1": f1,
        "gPre": precision,
        "gRec": recall,
        "gAccu": accuracy,
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
    best_f1 = 0.0
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
                'gIoU: {:.4f}, '
                'gF1: {:.4f}, '
                'gAccu {:.4f}'.format(
                    epoch,
                    optimizer.param_groups[0]['lr'],
                    best_epoch,
                    best_iou,
                    best_f1,
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
                writer.add_scalar('train/lr', lr, iters)

            if (i % (len(trainloader) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg))

        if rank == 0:
            evaluation = evaluate_new(model, valloader, multiplier=14)

            logger.info('***** Evaluation ***** >>>> gIoU: {:.4f}, gF1: {:.4f}, gAccu: {:.4f}'.format(
                evaluation["gIoU"], evaluation["gF1"], evaluation["gAccu"]
            ))

            logger.info('***** Evaluation ***** >>>> gPre: {:.4f}, gRec: {:.4f}'.format(
                evaluation["gPre"], evaluation["gRec"]
            ))

            writer.add_scalar('eval/gIoU', evaluation["gIoU"], epoch)
            writer.add_scalar('eval/gF1', evaluation["gF1"], epoch)
            writer.add_scalar('eval/gPre', evaluation["gPre"], epoch)
            writer.add_scalar('eval/gRec', evaluation["gRec"], epoch)
            writer.add_scalar('eval/gAccu', evaluation["gAccu"], epoch)

            evaluation["checkpoint"] = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }

            miou, mf1 = evaluation["gIoU"], evaluation["gF1"]

            if len(best_evaluations) < 10:
                best_evaluations[epoch] = evaluation
            else:
                lowest_epoch, lowest_evaluation = sorted(
                    best_evaluations.items(),
                    key=lambda kv: kv[1]["gIoU"]
                )[0]
                if miou > lowest_evaluation['gIoU']:
                    del best_evaluations[lowest_epoch]
                    best_evaluations[epoch] = evaluation

            best_epoch, best_evaluation = sorted(
                best_evaluations.items(),
                key=lambda kv: kv[1]["gF1"],
                reverse=True
            )[0]

            best_iou = best_evaluation["gIoU"]
            best_f1 = best_evaluation["gF1"]
            best_accu = best_evaluation["gAccu"]

        dist.barrier()

    if rank == 0:
        for ep, eval_ in best_evaluations.items():
            torch.save(eval_, os.path.join(args.save_path, f"checkpoint_{ep}.pth"))

        best_epoch, best_evaluation = sorted(
            best_evaluations.items(),
            key=lambda kv: kv[1]["gF1"],
            reverse=True
        )[0]

        torch.save(best_evaluation, os.path.join(args.save_path, f"best.pth"))


if __name__ == '__main__':
    main()
