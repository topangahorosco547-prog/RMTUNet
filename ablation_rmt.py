"""
RMT Ablation Study Script
Supports two ablation dimensions:
  1. Spatial decay (use_decay): with/without Manhattan-distance spatial decay
  2. ELA position (ela_position): after_fusion / encoder / both / none

Usage:
    # Spatial decay ablation (no decay = standard SA)
    python ablation_rmt.py --ablation decay --use-decay 0

    # ELA position ablation (encoder-side)
    python ablation_rmt.py --ablation ela --ela-position encoder

    # Custom configuration
    python ablation_rmt.py --ablation decay --use-decay 0 --device cuda:0 --batch-size 32 --max_epochs 300
"""
import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RMT_UNet import VisRetNet
from trainer import trainer_synapse


def get_args():
    parser = argparse.ArgumentParser('RMT Ablation Study')
    parser.add_argument('--batch-size', default=50, type=int)
    parser.add_argument('--base_lr', type=float, default=5e-4)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--max_epochs', default=300, type=int)
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--deterministic', type=int, default=0)

    parser.add_argument('--ablation', type=str, default='decay',
                        choices=['decay', 'ela', 'both'],
                        help='decay: spatial decay ablation | ela: ELA position ablation | both: both')

    parser.add_argument('--use-decay', type=int, default=0,
                        choices=[0, 1], help='0=no spatial decay 1=with spatial decay')

    parser.add_argument('--ela-position', type=str, default='encoder',
                        choices=['after_fusion', 'encoder', 'both', 'none'],
                        help="'after_fusion': after concat | 'encoder': before concat | 'both': both sides | 'none': no ELA")

    parser.add_argument('--pretrain', type=str, default=None,
                        help='load pretrained weights for backbone (strict=False)')

    parser.add_argument('--output_dir', default='./output_ablation', type=str)

    return parser.parse_args()


def build_rmt_unet(use_decay: bool, ela_position: str = 'encoder') -> VisRetNet:
    """Build RMT-UNet with same parameters as RMT_T."""
    return VisRetNet(
        in_chans=3,
        num_classes=1,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, True, False, False],
        layerscales=[False, False, False, False],
        use_decay=use_decay,
        ela_position=ela_position,
    )


def main():
    args = get_args()
    print(args)
    print("=" * 70)

    device = torch.device(args.device)

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    jobs = []  # (tag, use_decay, ela_position)

    if args.ablation == 'both':
        desc = f"decay_{args.use_decay}_ela_{args.ela_position}"
        jobs.append((desc, bool(args.use_decay), args.ela_position))
    else:
        if args.ablation == 'decay':
            for use_d in [False, True]:
                desc = 'spatial_decay' if use_d else 'no_decay'
                jobs.append((desc, use_d, 'encoder'))
        if args.ablation == 'ela':
            jobs.append((f'ela_{args.ela_position}', True, args.ela_position))

    print(f"\nWill train {len(jobs)} variant(s):")
    for tag, use_d, ela_p in jobs:
        print(f"  [{tag}] use_decay={use_d}, ela_position={ela_p}")

    for tag, use_d, ela_p in jobs:
        ela_desc = {'after_fusion': 'ELA after concat',
                    'encoder': 'ELA on encoder',
                    'both': 'ELA on encoder + after concat',
                    'none': 'No ELA'}[ela_p]
        decay_desc = 'With spatial decay' if use_d else 'No spatial decay (standard SA)'

        print(f"\n{'='*70}")
        print(f"Training variant: [{tag}]")
        print(f"  {decay_desc}")
        print(f"  {ela_desc}")
        print(f"{'='*70}")

        net = build_rmt_unet(use_decay=use_d, ela_position=ela_p)

        if args.pretrain:
            state = torch.load(args.pretrain, map_location='cpu')
            missing, unexpected = net.load_state_dict(state, strict=False)
            print(f"Loaded pretrained weights: {args.pretrain}")
            print(f"  Missing keys: {len(missing)} (decay/ELA related)")
            print(f"  Unexpected keys: {len(unexpected)}")

        net.to(device)

        params_m = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
        print(f"Trainable parameters: {params_m:.2f} M")

        output_dir = os.path.join(args.output_dir, tag)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        trainer_synapse(args, net, output_dir)
        print(f"\nVariant '{tag}' training completed. Best model: {output_dir}/epoch_best.pth")

    print(f"\n{'='*70}")
    print("All ablation experiments completed.")
    print(f"Models saved in: {args.output_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
