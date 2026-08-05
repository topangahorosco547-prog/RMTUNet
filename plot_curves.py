"""
Plot training convergence curves: Loss (per iteration) + Validation mIoU & HD95 (per epoch)

Usage:
    python plot_curves.py --output_dir ./output/RMT_T_ELA-sentinel --save convergence.pdf
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def smooth(data, window=21):
    """Moving average smoothing (larger window = smoother)."""
    return uniform_filter1d(np.array(data, dtype=float), size=window)


def main():
    parser = argparse.ArgumentParser('Plot training convergence curves')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='training output directory (snapshot_path from trainer.py)')
    parser.add_argument('--save', type=str, default='convergence.pdf',
                        help='output filename (supports .pdf .png)')
    parser.add_argument('--smooth', type=int, default=51,
                        help='Loss smoothing window size')
    parser.add_argument('--max_iter', type=int, default=None,
                        help='limit x-axis max iteration for loss plot (None=all)')
    parser.add_argument('--max_epoch', type=int, default=None,
                        help='limit x-axis max epoch for validation plot (None=all)')
    args = parser.parse_args()

    loss_file = os.path.join(args.output_dir, 'training_log.json')
    val_file = os.path.join(args.output_dir, 'validation_log.json')

    with open(loss_file) as f:
        loss_data = np.array(json.load(f))
    with open(val_file) as f:
        val_data = np.array(json.load(f))

    iters = loss_data[:, 0]
    losses = loss_data[:, 1]
    epochs = val_data[:, 0]
    mious = val_data[:, 1]
    hd95s = val_data[:, 3]

    print(f"Loss records: {len(iters)} steps, range [{min(iters)}, {max(iters)}]")

    if args.max_epoch is not None:
        mask_val = epochs <= args.max_epoch
        epochs = epochs[mask_val]
        mious = mious[mask_val]
        hd95s = hd95s[mask_val]

    print(f"Validation records: {len(epochs)} epochs, best mIoU @ epoch {int(epochs[np.argmax(mious)])} = {max(mious):.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # --- Left: Training Loss ---
    ax = axes[0]
    if args.max_iter is not None and max(iters) > args.max_iter:
        mask = iters <= args.max_iter
        iters_plot = iters[mask]
        losses_plot = losses[mask]
    else:
        iters_plot = iters
        losses_plot = losses

    ax.plot(iters_plot, losses_plot, color='#4C72B0', alpha=0.15, linewidth=0.5, label='Raw loss')
    if len(losses_plot) > args.smooth:
        losses_smooth = smooth(losses_plot, window=args.smooth)
        ax.plot(iters_plot, losses_smooth, color='#C44E52', linewidth=1.2, label='Smoothed loss')

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('(a) Training Loss Curve', fontsize=11.5)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # --- Right: Validation mIoU + HD95 ---
    ax = axes[1]
    color1 = '#4C72B0'
    ax.plot(epochs, mious, color=color1, marker='o', markersize=3, linewidth=1.2, label='mIoU')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('mIoU (%)', fontsize=11, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)

    ax2 = ax.twinx()
    color2 = '#55A868'
    ax2.plot(epochs, hd95s, color=color2, marker='s', markersize=3, linewidth=1.0, label='HD95')
    ax2.set_ylabel('HD95', fontsize=11, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax.set_title('(b) Validation Curve', fontsize=11.5)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, args.save)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
