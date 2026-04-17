import argparse
from copy import deepcopy
import logging
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import yaml

from dataset.semi import SemiSmokeDataset
from model.semseg.dpt import DPT
from supervised import evaluate_new
from util.utils import count_params, init_log
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

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda(local_rank)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True
    )

    model_ema = deepcopy(model.module)

    model_ema.eval()

    for param in model_ema.parameters():
        param.requires_grad = False

    if (test_model := training_cfg.get("test_model", None)) is None:
        raise Exception("test model not specified")

    test_model_path = os.path.join(args.save_path, test_model)

    if not os.path.exists(test_model_path):
        raise Exception(f"test model not found at `{test_model_path}`")

    checkpoint = torch.load(test_model_path, map_location='cpu', weights_only=False)
    model_checkpoint = checkpoint["checkpoint"]

    model.load_state_dict(model_checkpoint["model"])
    model.eval()

    logger.info("Loaded %s at epoch %s" % (test_model_path, model_checkpoint['epoch']))

    # Load EMA weights from companion file
    has_ema = False
    ema_filename = test_model.replace('best.pth', 'best_ema.pth')
    ema_path = os.path.join(args.save_path, ema_filename)
    if ema_filename != test_model and os.path.exists(ema_path):
        ema_checkpoint = torch.load(ema_path, map_location='cpu', weights_only=False)
        model_ema.load_state_dict(ema_checkpoint["checkpoint"]["model_ema"])
        has_ema = True
        logger.info("Loaded EMA from %s at epoch %s" % (ema_path, ema_checkpoint["checkpoint"]["epoch"]))
    else:
        logger.warning("No EMA checkpoint found at %s, skipping EMA evaluation" % ema_path)

    testset = SemiSmokeDataset(
        cfg['dataset'],
        cfg['data_root'],
        'test',
        None,
        id_path=training_cfg['test_dataset']
    )

    testloader = DataLoader(
        testset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        drop_last=False,
        shuffle=False
    )

    if rank == 0:
        evaluation = evaluate_new(model, testloader, multiplier=14)

        logger.info('***** Evaluation *****')

        for k, v in evaluation.items():
            logger.info(f"\t{k}: {v:.4f}")

        if has_ema:
            evaluation_ema = evaluate_new(model_ema, testloader, multiplier=14)

            print()

            logger.info('***** Evaluation EMA *****')

            for k, v in evaluation_ema.items():
                logger.info(f"\t{k}: {v:.4f}")


if __name__ == '__main__':
    main()
