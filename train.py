import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from fvcore.nn import FlopCountAnalysis, flop_count_table

from RMT_UNet import RMT_T, RMT_S, RMT_B, RMT_L
from trainer import trainer_synapse

archs = {
    'RMT_T': RMT_T,
    'RMT_S': RMT_S,
    'RMT_B': RMT_B,
    'RMT_L': RMT_L
}

def get_args_parser():
    parser = argparse.ArgumentParser('RMT-UNet training script')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--base_lr', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='output channel of network')
    parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs')
    parser.add_argument('--max_epochs', default=300, type=int)
    parser.add_argument('--model', default='RMT_T', type=str,
                        help='model variant: RMT_T, RMT_S, RMT_B, RMT_L')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input image size')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device for training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', default='./output',
                        help='output directory for checkpoints and logs')
    parser.add_argument('--deterministic', type=int, default=0,
                        help='whether to use deterministic training')
    parser.add_argument('--num_workers', default=8, type=int)
    return parser


def main(args):
    print(args)
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

    net = archs[args.model](args)
    print(net)
    flops = FlopCountAnalysis(net, torch.rand(1, 3, args.input_size, args.input_size))
    print(flop_count_table(flops))

    net.to(device)
    trainer.trainer_synapse(args, net, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RMT-UNet training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
