import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import os

from dataloader.flow.datasets import build_train_dataset
from unimatch.unimatch import UniMatch
from loss.vo_loss import vo_loss_func

from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='chairs', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                        help='validation datasets')
    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding or resizing')
    
    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # model: learnable parameters
    parser.add_argument('--task', default='flow', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')

    # model: parameter-free
    parser.add_argument('--attn_type', default='swin', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='exponential weighting')
    parser.add_argument("--tau", default=0.5, type=float,
                        help="translational relative weight in loss.")

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # misc
    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time')

    parser.add_argument('--debug', action='store_true')

    return parser

def main(args):
    print_info = not args.eval and not args.submission and args.inference_dir is None and args.inference_video is None

    if print_info and args.local_rank == 0:
        print(args)
        misc.save_args(args)
        misc.check_path(args.checkpoint_dir)
        misc.save_command(args.checkpoint_dir)

    misc.check_path(args.output_path)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)

    # model
    model = UniMatch(feature_channels=args.feature_channels,
                     num_scales=args.num_scales,
                     upsample_factor=args.upsample_factor,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_layers=args.num_transformer_layers,
                     reg_refine=args.reg_refine,
                     task=args.task).to(device)

    if print_info:
        print(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

            model_without_ddp = model.module
        else:
            model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    if print_info:
        print('Number of params:', num_params)
    if not args.eval and not args.submission and args.inference_dir is None and args.inference_video is None:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    start_step = 0
    # resume checkpoints
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)

        model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        if print_info:
            print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

    train_dataset = build_train_dataset(args)
    print('Number of training images:', len(train_dataset))

    # multi-processing
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank)
    else:
        train_sampler = None

    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

    last_epoch = start_step if args.resume and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr,
        args.num_steps + 10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos',
        last_epoch=last_epoch,
    )

    if args.local_rank == 0:
        summary_writer = SummaryWriter(args.checkpoint_dir)
        logger = Logger(lr_scheduler, summary_writer, args.summary_freq,
                        start_step=start_step)

    total_steps = start_step
    epoch = start_epoch
    print('Start training')

    while total_steps < args.num_steps:
        model.train()

        # mannually change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i, sample in enumerate(train_loader):
            img1, img2, rot_gt, trans_gt = [x.to(device) for x in sample]

            results_dict = model(img1, img2,
                                 attn_type=args.attn_type,
                                 attn_splits_list=args.attn_splits_list,
                                 corr_radius_list=args.corr_radius_list,
                                 prop_radius_list=args.prop_radius_list,
                                 num_reg_refine=args.num_reg_refine,
                                 task='flow',
                                 )

            flow_preds = results_dict['flow_preds']

            loss, metrics = vo_loss_func(flow_preds, rot_gt, trans_gt,
                                         batch_size=args.batch_size,
                                         tau=args.tau,
                                         gamma=args.gamma,
                                        )

            if isinstance(loss, float):
                continue

            if torch.isnan(loss):
                continue

            metrics.update({'total_loss': loss.item()})

            # more efficient zero_grad
            for param in model_without_ddp.parameters():
                param.grad = None

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            lr_scheduler.step()

            if args.local_rank == 0:
                logger.push(metrics)

                logger.add_image_summary(img1, img2, flow_preds, flow_gt)

            total_steps += 1

            if total_steps % args.save_ckpt_freq == 0 or total_steps == args.num_steps:
                if args.local_rank == 0:
                    print('Save checkpoint at step: %d' % total_steps)
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_without_ddp.state_dict()
                    }, checkpoint_path)

            if total_steps % args.save_latest_ckpt_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

                if args.local_rank == 0:
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }, checkpoint_path)

            if total_steps >= args.num_steps:
                print('Training done')

                return

        epoch += 1


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
