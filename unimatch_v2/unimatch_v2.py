import argparse
from copy import deepcopy
import logging
import os
import pprint

from itertools import chain, cycle
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiSmokeDataset
from model.semseg.dpt import DPT
from supervised import evaluate_new
from util.utils import count_params, init_log, AverageMeter
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

    model = DPT(**{
        **model_configs[cfg['backbone'].split('_')[-1]],
        'nclass': cfg['nclass']
    })

    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')

    model.backbone.load_state_dict(state_dict)

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
        weight_decay=0.01
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True
    )

    model_ema = deepcopy(model)

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
            # model_ema.load_state_dict(checkpoint['model_ema'])

            logger.info("Loaded %s at epoch %s" % (pretrained_best_pth, checkpoint["checkpoint"]["epoch"]))

    for param in model_ema.parameters():
        param.requires_grad = False

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    current_dataset = cfg['dataset']

    trainset_u = SemiSmokeDataset(
        current_dataset,
        cfg['data_root'],
        'train_u',
        cfg['crop_size'],
        base_size=cfg.get('base_size'),
        id_path=training_cfg['unlabeled_dataset']
    )

    current_nsample = len(trainset_u)

    if args.unlabeled_sample_size > 0 and args.unlabeled_sample_size < current_nsample:
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

    trainset_smoke = SemiSmokeDataset(
        current_dataset,
        cfg['data_root'],
        'train_l',
        cfg['crop_size'],
        base_size=cfg.get('base_size'),
        id_path=training_cfg['smoke_dataset'],
        nsample=current_nsample
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
        nsample=current_nsample
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
        id_path=training_cfg['validation_dataset']
    )

    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        drop_last=False,
        shuffle=False
    )

    total_iters = (current_nsample or len(trainloader_u)) * cfg['epochs']
    best_evaluations = {}
    best_epoch = -1
    best_iou = 0.0
    best_f1 = 0.0
    best_accu = 0.0
    best_miou = 0.0
    best_mf1 = 0.0
    best_far = 0.0
    epoch = -1

    # if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
    #     checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu', weights_only=False)

    #     model.load_state_dict(checkpoint['model'])
    #     model_ema.load_state_dict(checkpoint['model_ema'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    #     epoch = checkpoint['epoch']
    #     previous_best_iou = checkpoint['previous_best_iou']
    #     previous_best_acc = checkpoint['previous_best_acc']
    #     previous_best_ema_iou = checkpoint['previous_best_ema_iou']
    #     previous_best_ema_acc = checkpoint['previous_best_ema_acc']
    #     best_epoch = checkpoint['best_epoch']
    #     best_epoch_ema = checkpoint['best_epoch_ema']

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
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
                        best_iou,
                        best_f1,
                        best_accu,
                        best_miou,
                        best_mf1,
                        best_far
                    )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_smoke.sampler.set_epoch(epoch)

        combined_loader = map(
            lambda batch: map(lambda pair: torch.cat(pair), zip(*batch)),
            zip(trainloader_smoke, cycle(iter(trainloader_clear)))
        )

        loader = zip(combined_loader, trainloader_u)

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

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[
                cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[
                cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

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
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0

            loss = (loss_x + loss_u_s) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (
                    ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg,
                                            total_loss_s.avg, total_mask_ratio.avg))

        if rank == 0:
            evaluation = evaluate_new(model, valloader, multiplier=14)
            evaluation_ema = evaluate_new(model_ema, valloader, multiplier=14)

            logger.info(
                '***** Evaluation ***** >>>> gIoU: {:.4f}, gF1: {:.4f}, gAccu: {:.4f} | '
                'EMA: gIoU: {:.4f}, gF1: {:.4f}, gAccu: {:.4f}'.format(
                    evaluation["gIoU"], evaluation["gF1"], evaluation["gAccu"],
                    evaluation_ema["gIoU"], evaluation_ema["gF1"], evaluation_ema["gAccu"]
                ))

            logger.info(
                '***** Evaluation ***** >>>> gPre: {:.4f}, gRec: {:.4f} | '
                'EMA: gPre: {:.4f}, gRec: {:.4f}'.format(
                    evaluation["gPre"], evaluation["gRec"],
                    evaluation_ema["gPre"], evaluation_ema["gRec"]
                ))

            logger.info(
                '***** Evaluation ***** >>>> mIoU: {:.4f}, mF1: {:.4f}, FAR: {:.4f} | '
                'EMA: mIoU: {:.4f}, mF1: {:.4f}, FAR: {:.4f}'.format(
                    evaluation["mIoU"], evaluation["mF1"], evaluation["FAR"],
                    evaluation_ema["mIoU"], evaluation_ema["mF1"], evaluation_ema["FAR"]
                ))

            writer.add_scalar('eval/gIoU', evaluation["gIoU"], epoch)
            writer.add_scalar('eval/gF1', evaluation["gF1"], epoch)
            writer.add_scalar('eval/gPre', evaluation["gPre"], epoch)
            writer.add_scalar('eval/gRec', evaluation["gRec"], epoch)
            writer.add_scalar('eval/gAccu', evaluation["gAccu"], epoch)
            writer.add_scalar('eval/mIoU', evaluation["mIoU"], epoch)
            writer.add_scalar('eval/mF1', evaluation["mF1"], epoch)
            writer.add_scalar('eval/FAR', evaluation["FAR"], epoch)
            writer.add_scalar('eval/gIoU_EMA', evaluation_ema["gIoU"], epoch)
            writer.add_scalar('eval/gF1_EMA', evaluation_ema["gF1"], epoch)
            writer.add_scalar('eval/gPre_EMA', evaluation_ema["gPre"], epoch)
            writer.add_scalar('eval/gRec_EMA', evaluation_ema["gRec"], epoch)
            writer.add_scalar('eval/gAccu_EMA', evaluation_ema["gAccu"], epoch)
            writer.add_scalar('eval/mIoU_EMA', evaluation_ema["mIoU"], epoch)
            writer.add_scalar('eval/mF1_EMA', evaluation_ema["mF1"], epoch)
            writer.add_scalar('eval/FAR_EMA', evaluation_ema["FAR"], epoch)

            evaluation["checkpoint"] = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
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
            best_miou = best_evaluation["mIoU"]
            best_mf1 = best_evaluation["mF1"]
            best_far = best_evaluation["FAR"]

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
