import argparse
from copy import deepcopy
import logging
import os
import pprint

from itertools import chain, cycle
import timm.models
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.boxsup import BoxSupDataset, boxsup_collate_fn, make_box_corrected_mask, mil_box_loss
from dataset.semi import SemiSmokeDataset
from model.semseg.dpt import DPT
from supervised import evaluate_new
from util.utils import count_params, init_log, AverageMeter, BoundaryLenienceCELoss, dice_loss
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--training-config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--unlabeled-sample-size', type=int, required=False)
parser.add_argument('--unlabeled-sample-seed', type=int, required=False)


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

    model_ema = deepcopy(model.module)

    model_ema.eval()

    if pretrained_model_name := training_cfg.get("pretrained_model", "").strip():
        pretrained_best_pth = os.path.join("exp", pretrained_model_name, "best.pth")

        if os.path.exists(pretrained_best_pth):
            checkpoint = torch.load(
                pretrained_best_pth,
                map_location='cpu',
                weights_only=False
            )

            model.load_state_dict(checkpoint["checkpoint"]["model"])

            # Load pretrained weights into model_ema (strip module. prefix for non-DDP model_ema)
            ema_state = {k.replace('module.', '', 1): v for k, v in checkpoint["checkpoint"]["model"].items()}
            model_ema.load_state_dict(ema_state)

            logger.info("Loaded %s at epoch %s" % (pretrained_best_pth, checkpoint["checkpoint"]["epoch"]))

    for param in model_ema.parameters():
        param.requires_grad = False

    bl_cfg = cfg.get('boundary_lenience', {})
    if cfg['criterion']['name'] == 'CELoss':
        if bl_cfg.get('enabled', False):
            criterion_l = BoundaryLenienceCELoss(
                ignore_index=cfg['criterion']['kwargs'].get('ignore_index', 255),
                alpha=bl_cfg.get('alpha', 0.5),
                window_size=bl_cfg.get('window_size', 7),
                reduction='mean',
            ).cuda(local_rank)
            criterion_u = BoundaryLenienceCELoss(
                ignore_index=cfg['criterion']['kwargs'].get('ignore_index', 255),
                alpha=bl_cfg.get('alpha', 0.5),
                window_size=bl_cfg.get('window_size', 7),
                reduction='none',
            ).cuda(local_rank)
        else:
            criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
            criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    current_dataset = cfg['dataset']

    trainset_u = SemiSmokeDataset(
        current_dataset,
        cfg['data_root'],
        'train_u',
        cfg['crop_size'],
        base_size=cfg.get('base_size'),
        id_path=training_cfg['unlabeled_dataset'],
        use_dcp=cfg.get('use_dcp', False),
    )

    current_nsample = len(trainset_u)

    if args.unlabeled_sample_size is not None and args.unlabeled_sample_size > 0 and args.unlabeled_sample_size < current_nsample:
        current_nsample = args.unlabeled_sample_size

    if current_nsample < len(trainset_u):
        logger.info(
            "Randomly sampling %s of %s unlabeled images each epoch" % (current_nsample, len(trainset_u))
        )

        seed = args.unlabeled_sample_seed or 1
        rand_gen = torch.Generator().manual_seed(seed)

        trainsampler_u = torch.utils.data.RandomSampler(
            trainset_u,
            replacement=True,
            num_samples=current_nsample,
            generator=rand_gen
        )
    else:
        logger.info("Sampling %s unlabeled images each epoch" % current_nsample)
        trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)

    smoke_batch_size = cfg['batch_size'];
    clear_batch_size = max(smoke_batch_size // 10, 1)
    batch_size = smoke_batch_size + clear_batch_size

    trainloader_u = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_u
    )

    # --- Citizen consistency loader ---
    citizen_data = training_cfg.get('citizen_data', '').strip()
    citizen_img = training_cfg.get('citizen_img', '').strip()
    citizen_correction = bool(training_cfg.get('citizen_correction', True))
    citizen_mil_weight = float(training_cfg.get('citizen_mil_weight', 0.0))
    if citizen_data and citizen_img:
        citizen_json_path = os.path.join(cfg['data_root'], citizen_data)
        citizen_img_dir = os.path.join(cfg['data_root'], citizen_img)
        trainset_citizen = BoxSupDataset(
            citizen_json_path,
            citizen_img_dir,
            base_size=cfg.get('base_size', cfg['crop_size']),
            use_dcp=cfg.get('use_dcp', False),
        )
        citizen_num_samples = len(trainloader_u) * smoke_batch_size
        trainloader_citizen = DataLoader(
            trainset_citizen,
            batch_size=smoke_batch_size,
            pin_memory=True,
            num_workers=4,
            drop_last=True,
            collate_fn=boxsup_collate_fn,
            sampler=torch.utils.data.RandomSampler(
                trainset_citizen,
                replacement=True,
                num_samples=citizen_num_samples,
                generator=torch.Generator().manual_seed(4),
            ),
        )
        if rank == 0:
            logger.info(
                'Citizen dataset: %s  (%d total images, correction=%s, mil_weight=%.3f, batch_size=%d)\n' %
                (citizen_data, len(trainset_citizen), citizen_correction, citizen_mil_weight, smoke_batch_size)
            )
    else:
        trainloader_citizen = None

    trainset_smoke = SemiSmokeDataset(
        current_dataset,
        cfg['data_root'],
        'train_l',
        cfg['crop_size'],
        base_size=cfg.get('base_size'),
        id_path=training_cfg['smoke_dataset'],
        nsample=current_nsample,
        use_dcp=cfg.get('use_dcp', False),
    )

    trainsampler_smoke = torch.utils.data.distributed.DistributedSampler(trainset_smoke)

    trainloader_smoke = DataLoader(
        trainset_smoke,
        batch_size=smoke_batch_size,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_smoke
    )

    trainset_clear = SemiSmokeDataset(
        current_dataset,
        cfg['data_root'],
        'train_l',
        cfg['crop_size'],
        base_size=cfg.get('base_size'),
        id_path=training_cfg['clear_dataset'],
        nsample=current_nsample,
        use_dcp=cfg.get('use_dcp', False),
    )

    trainsampler_clear = torch.utils.data.RandomSampler(
        trainset_clear,
        replacement=True,
        generator=torch.Generator().manual_seed(1)
    )

    trainloader_clear = DataLoader(
        trainset_clear,
        batch_size=clear_batch_size,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_clear
    )

    valset = SemiSmokeDataset(
        cfg['dataset'],
        cfg['data_root'],
        'val',
        id_path=training_cfg['validation_dataset'],
        use_dcp=cfg.get('use_dcp', False),
    )

    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
        shuffle=False
    )

    total_iters = len(trainloader_u) * cfg['epochs']
    best_epoch = -1
    best_eval = None
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
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

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_dice = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_c = AverageMeter()
        total_loss_mil = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_smoke.sampler.set_epoch(epoch)
        if isinstance(trainsampler_u, torch.utils.data.distributed.DistributedSampler):
            trainsampler_u.set_epoch(epoch)

        combined_loader = (
            tuple(torch.cat(pair) for pair in zip(*batch))
            for batch in zip(trainloader_smoke, cycle(iter(trainloader_clear)))
        )

        loader = zip(combined_loader, trainloader_u)

        citizen_iter = cycle(iter(trainloader_citizen)) if trainloader_citizen is not None else None

        model.train()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):

            img_x = img_x.cuda(local_rank, non_blocking=True)
            mask_x = mask_x.cuda(local_rank, non_blocking=True)
            img_u_w = img_u_w.cuda(local_rank, non_blocking=True)
            img_u_s1 = img_u_s1.cuda(local_rank, non_blocking=True)
            img_u_s2 = img_u_s2.cuda(local_rank, non_blocking=True)
            ignore_mask = ignore_mask.cuda(local_rank, non_blocking=True)
            cutmix_box1 = cutmix_box1.cuda(local_rank, non_blocking=True)
            cutmix_box2 = cutmix_box2.cuda(local_rank, non_blocking=True)

            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            cutmix_mask1 = cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1
            img_u_s1 = torch.where(cutmix_mask1, img_u_s1.flip(0), img_u_s1)
            cutmix_mask2 = cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1
            img_u_s2 = torch.where(cutmix_mask2, img_u_s2.flip(0), img_u_s2)

            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / max((ignore_mask_cutmixed1 != 255).sum().item(), 1)

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / max((ignore_mask_cutmixed2 != 255).sum().item(), 1)

            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0

            if citizen_iter is not None:
                img_c_w, img_c_s, bboxes_c = next(citizen_iter)
                img_c_w = img_c_w.cuda(local_rank, non_blocking=True)
                img_c_s = img_c_s.cuda(local_rank, non_blocking=True)
                bboxes_c = bboxes_c.cuda(local_rank, non_blocking=True)

                with torch.no_grad():
                    pred_c_w = model_ema(img_c_w).detach()
                    conf_c_w = pred_c_w.softmax(dim=1).max(dim=1)[0]
                    mask_c_w = pred_c_w.argmax(dim=1)

                if citizen_correction:
                    mask_c_w = make_box_corrected_mask(mask_c_w, bboxes_c)

                pred_c_s = model(img_c_s)
                loss_c_s = criterion_u(pred_c_s, mask_c_w)
                loss_c_s = loss_c_s * (conf_c_w >= cfg['conf_thresh'])
                loss_c_s = loss_c_s.sum() / max(loss_c_s.numel(), 1)

                loss_u_s = (loss_u_s1 + loss_u_s2 + loss_c_s) / 3.0

                if citizen_mil_weight > 0:
                    loss_mil = mil_box_loss(pred_c_s, bboxes_c)
                else:
                    loss_mil = torch.zeros(1).cuda(local_rank)
            else:
                loss_c_s = torch.zeros(1).cuda(local_rank)
                loss_mil = torch.zeros(1).cuda(local_rank)

            loss_x_ce = criterion_l(pred_x, mask_x)
            loss_x_dice = dice_loss(pred_x, mask_x)
            loss = 0.25 * loss_x_ce + 0.25 * loss_x_dice + 0.5 * loss_u_s + citizen_mil_weight * loss_mil

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x_ce.item())
            total_loss_dice.update(loss_x_dice.item())
            total_loss_s.update(loss_u_s.item())
            total_loss_c.update(loss_c_s.item())
            total_loss_mil.update(loss_mil.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / max(
                    (ignore_mask != 255).sum().item(), 1)
            total_mask_ratio.update(mask_ratio)

            iters = epoch * len(trainloader_u) + i
            lr = max(cfg['lr'] * (1 - iters / total_iters) ** 0.9, cfg['min_lr'])
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            for param, param_ema in zip(model.module.parameters(), model_ema.parameters()):
                param_ema.data.lerp_(param.data, 1.0 - ema_ratio)
            for buffer, buffer_ema in zip(model.module.buffers(), model_ema.buffers()):
                if buffer.dtype.is_floating_point:
                    buffer_ema.data.lerp_(buffer.data, 1.0 - ema_ratio)
                else:
                    buffer_ema.data.copy_(buffer.data)

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x_ce.item(), iters)
                writer.add_scalar('train/loss_dice', loss_x_dice.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/loss_c', loss_c_s.item(), iters)
                writer.add_scalar('train/loss_mil', loss_mil.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % max(len(trainloader_u) // 8, 1) == 0) and (rank == 0):
                logger.info(
                    'Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss dice: {:.3f}, '
                    'Loss s: {:.3f}, Loss c: {:.3f}, Loss mil: {:.3f}, Mask ratio: {:.3f}'.format(
                        i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg,
                        total_loss_dice.avg, total_loss_s.avg, total_loss_c.avg,
                        total_loss_mil.avg, total_mask_ratio.avg
                    )
                )

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
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_eval': best_eval,
            }, os.path.join(args.save_path, 'latest.pth'))

        dist.barrier()


if __name__ == '__main__':
    main()
