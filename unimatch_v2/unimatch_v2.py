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

from dataset.boxsup import BoxSupDataset, boxsup_collate_fn, make_box_corrected_mask, make_fixed_box_mask, compute_box_confidence_weights
from dataset.semi import SemiSmokeDataset
from model.semseg.dpt import DPT
from supervised import evaluate_new
from util.utils import count_params, init_log, AverageMeter, BoundaryLenienceCELoss, BoundaryLenienceDiceLoss, dice_loss, dice_loss_per_image, eval_score, update_loss_history, compute_lr, AttentionWeightedLoss, is_pareto_optimal, update_pareto_front, calculate_angle_score
from plot_training_curves import plot_loss_curve
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--training-config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


class DistributedFixedSizeSampler(torch.utils.data.Sampler):
    """DDP-aware sampler that draws a fixed total number of samples per epoch.

    Each rank receives num_samples // world_size unique indices. Indices are
    re-randomized every epoch via set_epoch, ensuring different samples are
    seen each epoch across the full dataset.
    """

    def __init__(self, dataset, num_samples, rank, world_size, seed=0):
        self.dataset_len = len(dataset)
        self.num_samples_total = num_samples
        self.rank = rank
        self.world_size = world_size
        self.per_rank = num_samples // world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_len, generator=g)[:self.num_samples_total]
        local_indices = indices[self.rank::self.world_size]
        return iter(local_indices.tolist())

    def __len__(self):
        return self.per_rank


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

    model_ema = deepcopy(model.module)

    model_ema.eval()

    if pretrained_model_name := training_cfg.get("pretrained_model", "").strip():
        pretrained_best_pth = os.path.join(os.path.dirname(args.save_path), pretrained_model_name, "best.pth")

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

    use_awl = bool(training_cfg.get('awl', False))
    awl_weight = float(training_cfg.get('awl_weight', 1.0))
    awl_citizen_batch_ratio = float(training_cfg.get('awl_citizen_batch_ratio', 0.25))

    if cfg['criterion']['name'] == 'CELossAndDiceLoss':
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
            dice_loss_fn = BoundaryLenienceDiceLoss(
                ignore_index=cfg['criterion']['kwargs'].get('ignore_index', 255),
                alpha=bl_cfg.get('alpha', 0.5),
                window_size=bl_cfg.get('window_size', 7),
            ).cuda(local_rank)
        else:
            criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
            criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
            dice_loss_fn = dice_loss
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    if use_awl:
        awl_criterion = AttentionWeightedLoss().cuda(local_rank)
        model.module.backbone.enable_cls_attn_hook()
        if rank == 0:
            logger.info('AWL enabled: weight %.2f, citizen batch ratio %.2f\n' %
                        (awl_weight, awl_citizen_batch_ratio))
    else:
        awl_criterion = None

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

    # --- Citizen consistency loader ---
    citizen_data = training_cfg.get('citizen_data', '').strip()
    citizen_img = training_cfg.get('citizen_img', '').strip()
    citizen_correction = bool(training_cfg.get('citizen_correction', True))
    citizen_is_boxsup = False
    if citizen_img and citizen_data:
        # BoxSupDataset: citizen_data is the bbox JSON, citizen_img is the image directory.
        # citizen_correction=True  → apply box correction to teacher pseudo-labels.
        # citizen_correction=False → use EMA pseudo-labels as-is (treat as unlabeled).
        citizen_json_path = os.path.join(cfg['data_root'], citizen_data)
        citizen_img_dir = os.path.join(cfg['data_root'], citizen_img)
        trainset_citizen = BoxSupDataset(
            citizen_json_path,
            citizen_img_dir,
            crop_size=cfg['crop_size'],
            base_size=cfg.get('base_size'),
            use_dcp=cfg.get('use_dcp', False),
        )
        citizen_is_boxsup = True
        if rank == 0:
            logger.info(
                'Citizen dataset: %s  (%d total images, correction=%s)\n' %
                (citizen_data, len(trainset_citizen), citizen_correction)
            )
    else:
        pass  # no citizen data

    current_nsample = len(trainset_u)

    cfg_nsample = cfg.get("unlabeled_sample_size", 1500)
    if 0 < cfg_nsample < current_nsample:
        current_nsample = cfg_nsample

    trainsampler_u = DistributedFixedSizeSampler(
        trainset_u, current_nsample, rank, world_size
    )

    if rank == 0:
        logger.info(
            "Unlabeled dataset: %s total images, sampling %s per epoch\n" %
            (len(trainset_u), current_nsample)
        )

    smoke_batch_size = cfg['batch_size']
    clear_batch_size = max(smoke_batch_size // 10, 1)
    has_clear_dataset = bool(training_cfg.get('clear_dataset'))
    batch_size = smoke_batch_size + (clear_batch_size if has_clear_dataset else 0)
    citizen_batch_size = max(1, int(smoke_batch_size * awl_citizen_batch_ratio)) if use_awl else smoke_batch_size

    trainloader_u = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_u
    )

    if citizen_is_boxsup:
        trainloader_citizen = DataLoader(
            trainset_citizen,
            batch_size=citizen_batch_size,
            pin_memory=True,
            num_workers=4,
            drop_last=True,
            collate_fn=boxsup_collate_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(trainset_citizen),
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

    if has_clear_dataset:
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
        trainsampler_clear = torch.utils.data.distributed.DistributedSampler(trainset_clear)
        trainloader_clear = DataLoader(
            trainset_clear,
            batch_size=clear_batch_size,
            pin_memory=True,
            num_workers=4,
            drop_last=True,
            sampler=trainsampler_clear
        )
    else:
        trainloader_clear = None

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

    total_iters = len(trainloader_u) * cfg['epochs']
    warmup_iters = cfg.get('warmup_epochs', 5) * len(trainloader_u)
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
        model_ema.load_state_dict(checkpoint['model_ema'])
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

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_dice = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_c = AverageMeter()
        total_loss_awl = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_smoke.sampler.set_epoch(epoch)
        trainsampler_u.set_epoch(epoch)
        if trainloader_clear is not None:
            trainloader_clear.sampler.set_epoch(epoch)
        if trainloader_citizen is not None:
            trainloader_citizen.sampler.set_epoch(epoch)

        if trainloader_clear is not None:
            combined_loader = (
                tuple(torch.cat(pair) for pair in zip(*batch))
                for batch in zip(cycle(iter(trainloader_smoke)), cycle(iter(trainloader_clear)))
            )
        else:
            combined_loader = cycle(iter(trainloader_smoke))

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

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / max((ignore_mask_cutmixed1 != 255).sum().item(), 1)

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / max((ignore_mask_cutmixed2 != 255).sum().item(), 1)

            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0

            if citizen_iter is not None:
                img_c_w, img_c_s1, img_c_s2, cutmix_box1_c, cutmix_box2_c, bboxes_c = next(citizen_iter)
                img_c_w       = img_c_w.cuda(local_rank, non_blocking=True)
                bboxes_c      = bboxes_c.cuda(local_rank, non_blocking=True)

                if not use_awl:
                    # Use citizen data in the unsupervised pseudo-labeling branch
                    img_c_s1      = img_c_s1.cuda(local_rank, non_blocking=True)
                    img_c_s2      = img_c_s2.cuda(local_rank, non_blocking=True)
                    cutmix_box1_c = cutmix_box1_c.cuda(local_rank, non_blocking=True)
                    cutmix_box2_c = cutmix_box2_c.cuda(local_rank, non_blocking=True)

                    with torch.no_grad():
                        pred_c_w = model_ema(img_c_w).detach()
                        mask_c_w = pred_c_w.argmax(dim=1)

                    if citizen_correction:
                        mask_c_w = make_box_corrected_mask(mask_c_w, bboxes_c)

                    cutmix_mask1_c = cutmix_box1_c.unsqueeze(1).expand(img_c_s1.shape) == 1
                    img_c_s1 = torch.where(cutmix_mask1_c, img_c_s1.flip(0), img_c_s1)
                    cutmix_mask2_c = cutmix_box2_c.unsqueeze(1).expand(img_c_s2.shape) == 1
                    img_c_s2 = torch.where(cutmix_mask2_c, img_c_s2.flip(0), img_c_s2)

                    pred_c_s1, pred_c_s2 = model(torch.cat((img_c_s1, img_c_s2)), comp_drop=True).chunk(2)

                    mask_c_w1 = mask_c_w.clone()
                    mask_c_w2 = mask_c_w.clone()
                    # Compute per-image confidence weight (Mask-Aware Confidence Score)
                    # before CutMix mixing, based on the box-corrected EMA prediction.
                    smoke_prob_c_w = pred_c_w.softmax(dim=1)[:, 1]   # (B, H, W)
                    box_weights = compute_box_confidence_weights(smoke_prob_c_w, mask_c_w, bboxes_c)  # (B,)

                    mask_c_w1[cutmix_box1_c == 1] = mask_c_w.flip(0)[cutmix_box1_c == 1]
                    mask_c_w2[cutmix_box2_c == 1] = mask_c_w.flip(0)[cutmix_box2_c == 1]

                    # Mix per-image weights through CutMix: each blended image draws
                    # (1-frac) weight from itself and frac from its CutMix partner.
                    cutmix_frac1 = cutmix_box1_c.float().mean(dim=(1, 2))   # (B,)
                    cutmix_frac2 = cutmix_box2_c.float().mean(dim=(1, 2))   # (B,)
                    box_weights_1 = (1.0 - cutmix_frac1) * box_weights + cutmix_frac1 * box_weights.flip(0)
                    box_weights_2 = (1.0 - cutmix_frac2) * box_weights + cutmix_frac2 * box_weights.flip(0)

                    # Confidence-weighted per-image Dice loss (replaces CE + hard threshold)
                    loss_c_s1 = (dice_loss_per_image(pred_c_s1, mask_c_w1) * box_weights_1).mean()
                    loss_c_s2 = (dice_loss_per_image(pred_c_s2, mask_c_w2) * box_weights_2).mean()

                    loss_c_s = (loss_c_s1 + loss_c_s2) / 2.0
                    loss_u_s = (loss_u_s + loss_c_s) / 2.0
                else:
                    loss_c_s = torch.zeros(1).cuda(local_rank)

                if awl_criterion is not None:
                    # AWL on the weak citizen view using student model (hook already fired
                    # during the student's forward on the strong views above; fire again
                    # on the weak view to get the correct attention for these pixels)
                    pred_c_w_student = model(img_c_w)
                    H_c, W_c = img_c_w.shape[-2:]
                    alpha_c = model.module.backbone.get_last_cls_attn(H_c, W_c)
                    box_mask_c = make_fixed_box_mask(bboxes_c, H_c, W_c).float()
                    loss_awl = awl_criterion(pred_c_w_student, box_mask_c, alpha=alpha_c)
                else:
                    loss_awl = torch.zeros(1, device=img_x.device)
            else:
                loss_c_s = torch.zeros(1).cuda(local_rank)
                loss_awl = torch.zeros(1).cuda(local_rank)

            loss_ce_w = cfg['criterion'].get('loss_ce_weight', 0.25)
            loss_dice_w = cfg['criterion'].get('loss_dice_weight', 0.25)
            loss_u_w = cfg['criterion'].get('loss_u_weight', 0.5)
            loss_x_ce = criterion_l(pred_x, mask_x) if loss_ce_w > 0 else torch.zeros(1).cuda(local_rank)
            loss_x_dice = dice_loss_fn(pred_x, mask_x) if loss_dice_w > 0 else torch.zeros(1).cuda(local_rank)
            loss = loss_ce_w * loss_x_ce + loss_dice_w * loss_x_dice + loss_u_w * loss_u_s + awl_weight * loss_awl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x_ce.item())
            total_loss_dice.update(loss_x_dice.item())
            total_loss_s.update(loss_u_s.item())
            total_loss_c.update(loss_c_s.item())
            total_loss_awl.update(loss_awl.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / max(
                    (ignore_mask != 255).sum().item(), 1)
            total_mask_ratio.update(mask_ratio)

            iters = epoch * len(trainloader_u) + i
            lr = compute_lr(iters, total_iters, warmup_iters, cfg['lr'], cfg['min_lr'])
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
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                if awl_criterion is not None:
                    writer.add_scalar('train/loss_awl', loss_awl.item(), iters)

            if (i % max(len(trainloader_u) // 8, 1) == 0) and (rank == 0):
                if awl_criterion is not None:
                    logger.info(
                        'Iters: {:04d}, LR: {:.7f}, Total loss: {:.3f}, Loss CE: {:.3f}, Loss dice: {:.3f}, '
                        'Loss u: {:.3f}, Loss c: {:.3f}, Loss AWL: {:.3f}, Mask ratio: {:.3f}'.format(
                            i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg,
                            total_loss_dice.avg, total_loss_s.avg, total_loss_c.avg,
                            total_loss_awl.avg, total_mask_ratio.avg
                        )
                    )
                else:
                    logger.info(
                        'Iters: {:04d}, LR: {:.7f}, Total loss: {:.3f}, Loss CE: {:.3f}, Loss dice: {:.3f}, '
                        'Loss u: {:.3f}, Loss c: {:.3f}, Mask ratio: {:.3f}'.format(
                            i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg,
                            total_loss_dice.avg, total_loss_s.avg, total_loss_c.avg,
                            total_mask_ratio.avg
                        )
                    )

        if rank == 0:
            logger.info('***** Epoch {:04d} ***** >>>> Train Loss: {:.4f}'.format(epoch, total_loss.avg))

            evaluation = evaluate_new(model, valloader, multiplier=model.module.patch_size, criterion=criterion_l)

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

            robust_evaluation = evaluate_new(model, val_robust_loader, multiplier=model.module.patch_size, criterion=criterion_l)

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
                'model_ema': model_ema.state_dict(),
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
