"""DCTTA: adapt a source PromptIR on a target test set, then evaluate (was ``train_test_promptir.py``).

    python adapt.py --lq_dir testdata/Rain100H/LQ --gt_dir testdata/Rain100H/GT --ckpt pretrain/model.ckpt

Regime: transductive, one pass over the target set (batch 1, one adaptation step per random
320x320 patch), followed by full-image inference with the adapted student.
"""
import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dctta.config import get_args
from dctta.data import AdaptationPatchDataset, PairedImageDataset
from dctta.models.promptir import PromptIR
from dctta.rddm.generator import ResidualDegradationGenerator
from dctta.tta import setup_dctta
from dctta.utils import load_promptir_checkpoint, set_seed
from evaluate import evaluate


def main():
    args = get_args(adapt=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.results_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.results_dir, 'adapt.log'), mode='w'),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(vars(args))

    # --- construction order is kept identical to the original for RNG reproducibility ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = AdaptationPatchDataset(args.lq_dir, args.gt_dir, patch_size=args.patch_size,
                                      seed=None if args.dataset_seed < 0 else args.dataset_seed)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    logger.info(f"target set: {len(train_loader)} patches")

    model = PromptIR(decoder=True)
    load_promptir_checkpoint(model, args.ckpt, logger)
    model = model.to(device)

    tta = setup_dctta(model, lr=args.lr, betas=(args.beta1, args.beta2), train_prefixes=args.train_prefixes,
                      teacher_weight=args.teacher_weight, ema_decay=args.ema_decay, fisher_ratio=args.fisher_ratio,
                      use_fisher=not args.no_fisher, pixel_weight=args.pixel_weight,
                      perceptual_weight=args.perceptual_weight, lowpass_weight=args.lowpass_weight, device=device)

    generator = ResidualDegradationGenerator(device, train_steps=args.gen_train_steps,
                                             sampling_timesteps=args.gen_sampling_steps, lr=args.gen_lr,
                                             sum_scale=args.gen_sum_scale, debug_dir=args.gen_debug_dir)

    # --- TIPS: Fisher mask on the target set, computed once before adaptation ---
    t0 = time.time()
    if not args.no_fisher:
        logger.info("computing TIPS Fisher mask")
        tta.compute_fisher(train_loader)
        if args.save_fisher:
            torch.save({k: [f.cpu(), m.cpu()] for k, (f, m) in tta.fishers.items()}, args.save_fisher)

    # --- adaptation ---
    for batch_idx, (names, degrad_patch, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Adapting"):
        loss = tta(degrad_patch.to(device), generator, name=names[0], iterations=args.iterations)
        logger.info(f"[{batch_idx}] {names[0]} loss {loss:.5f}")
    adapt_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.
    logger.info(f"adaptation done in {adapt_time:.1f}s (incl. Fisher), peak memory {peak_mem:.2f} GB")

    if args.save_ckpt:
        tta.save(args.save_ckpt, extra={'args': vars(args)})

    # --- evaluation with the adapted student ---
    testset = PairedImageDataset(args.lq_dir, args.gt_dir, base=args.crop_base)
    test_loader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=args.num_workers)
    psnr, ssim = evaluate(tta.model, test_loader, device, None if args.no_save_images else args.results_dir, logger)
    logger.info(f"DCTTA  PSNR {psnr:.3f}  SSIM {ssim:.4f}  ({args.lq_dir}, {args.ckpt})")
    print(f"PSNR {psnr:.3f}  SSIM {ssim:.4f}")


if __name__ == '__main__':
    main()
