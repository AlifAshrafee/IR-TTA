"""Evaluate the un-adapted source PromptIR on a target set (was ``test_promptir.py``).

    python test.py --lq_dir testdata/Rain100H/LQ --gt_dir testdata/Rain100H/GT --ckpt pretrain/model.ckpt

``--self_ensemble`` reports the "*" rows of the paper (8-way geometric self-ensemble at inference,
no adaptation).
"""
import logging
import os

import torch
from torch.utils.data import DataLoader

from dctta.config import get_parser
from dctta.data import PairedImageDataset
from dctta.models.promptir import PromptIR
from dctta.utils import SelfEnsemble, load_promptir_checkpoint, set_seed
from evaluate import evaluate


def main():
    parser = get_parser(adapt=False)
    parser.add_argument('--self_ensemble', action='store_true', help='8-way flip/rot self-ensemble at inference')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.results_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.results_dir, 'test.log'), mode='w'),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(vars(args))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PromptIR(decoder=True)
    load_promptir_checkpoint(model, args.ckpt, logger)
    model = model.to(device)

    testset = PairedImageDataset(args.lq_dir, args.gt_dir, base=args.crop_base)
    test_loader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=args.num_workers)

    net = SelfEnsemble(model) if args.self_ensemble else model
    psnr, ssim = evaluate(net, test_loader, device, None if args.no_save_images else args.results_dir, logger)
    tag = "PromptIR*" if args.self_ensemble else "PromptIR"
    logger.info(f"{tag}  PSNR {psnr:.3f}  SSIM {ssim:.4f}  ({args.lq_dir}, {args.ckpt})")
    print(f"PSNR {psnr:.3f}  SSIM {ssim:.4f}")


if __name__ == '__main__':
    main()
