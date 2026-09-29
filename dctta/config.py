import argparse


def get_parser(adapt=True):
    p = argparse.ArgumentParser(description="DCTTA -- test-time adaptation for all-in-one image restoration")
    # data
    p.add_argument('--lq_dir', type=str, 
                   default='testdata/Rain100H/LQ', 
                   help='degraded target images')
    p.add_argument('--gt_dir', type=str, 
                   default='testdata/Rain100H/GT', 
                   help='ground truth (evaluation only)')
    p.add_argument('--ckpt', type=str, default='pretrain/model.ckpt',
                   help='PromptIR source checkpoint: model.ckpt (3-task) or epoch=80.ckpt (5-task)')
    p.add_argument('--results_dir', type=str, default='results/Rain100H_DCTTA', 
                   help='where restored images are written')
    p.add_argument('--no_save_images', action='store_true', help='do not write restored PNGs')
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--gpu', type=str, default='1', help='CUDA_VISIBLE_DEVICES value')
    p.add_argument('--seed', type=int, default=23)
    p.add_argument('--crop_base', type=int, default=16, help='centre-crop evaluation images to a multiple of this')
    if adapt:
        # adaptation loop
        p.add_argument('--patch_size', type=int, default=320)
        p.add_argument('--batch_size', type=int, default=1)
        p.add_argument('--iterations', type=int, default=1, help='adaptation steps per patch')
        p.add_argument('--lr', type=float, default=2e-4)
        p.add_argument('--beta1', type=float, default=0.9)
        p.add_argument('--beta2', type=float, default=0.999)
        p.add_argument('--train_prefixes', type=str, nargs='+', default=['prompt', 'encoder', 'decoder'],
                       help="top-level module name substrings that are trainable, or 'all'")
        p.add_argument('--dataset_seed', type=int, default=42,
                       help='the original re-seeds all RNGs with 42 inside the dataset constructor; -1 disables')
        # loss
        p.add_argument('--teacher_weight', type=float, default=5.0, help='weight of the pseudo-label loss (alpha)')
        p.add_argument('--pixel_weight', type=float, default=1.0)
        p.add_argument('--perceptual_weight', type=float, default=0.01)
        p.add_argument('--lowpass_weight', type=float, default=0.1)
        # teacher / TIPS
        p.add_argument('--ema_decay', type=float, default=0.45, help='teacher EMA decay (eta)')
        p.add_argument('--fisher_ratio', type=float, default=0.6, help='per-tensor fraction of entries frozen by TIPS')
        p.add_argument('--no_fisher', action='store_true', help='disable TIPS')
        p.add_argument('--save_fisher', type=str, default=None,
                       help='save the TIPS {name: [fisher, mask]} dict here (for tools/param_groups.py)')
        # generator
        p.add_argument('--gen_train_steps', type=int, default=5, help='RDDM optimiser steps per test image')
        p.add_argument('--gen_sampling_steps', type=int, default=5, help='DDIM steps')
        p.add_argument('--gen_lr', type=float, default=2e-4)
        p.add_argument('--gen_sum_scale', type=float, default=0.01)
        p.add_argument('--gen_debug_dir', type=str, default=None, help='dump every x_sd here (off by default)')
        # outputs
        p.add_argument('--save_ckpt', type=str, default=None, help='save the adapted student to this path')
    return p


def get_args(adapt=True):
    return get_parser(adapt).parse_args()
