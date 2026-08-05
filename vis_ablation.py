"""
Ablation variant segmentation comparison figure.
Displays predictions from different variants side by side: SAR | GT | Full | w/o Decay | w/o ELA

Usage:
    python vis_ablation.py --full ./output/.../epoch_best.pth \
                           --no-decay ./output_ablation/no_decay/epoch_best.pth \
                           --no-ela ./output_ablation/ela_none/epoch_best.pth \
                           --test-dir ./data/test/images \
                           --gt-dir ./data/test/labels \
                           --output-dir ./vis_ablation
"""
import argparse
import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RMT_UNet import VisRetNet

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def build_model(use_decay=True, ela_position='encoder'):
    return VisRetNet(
        in_chans=3, num_classes=1,
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


def load_and_preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    img_np = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    return torch.Tensor(img_np).to(DEVICE)


def predict(model, img_tensor):
    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor))
    output = output.cpu().numpy().squeeze()
    return (output >= 0.54).astype(np.uint8)


def overlay_boundary(img_rgb, mask, color=(255, 0, 0)):
    """Overlay segmentation boundary on RGB image."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img_rgb.copy()
    cv2.drawContours(overlay, contours, -1, color, 1)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', type=str, required=True, help='full model checkpoint')
    parser.add_argument('--no-decay', type=str, required=True, help='w/o spatial decay checkpoint')
    parser.add_argument('--no-ela', type=str, required=True, help='w/o ELA checkpoint')
    parser.add_argument('--test-dir', type=str, default='./data/test/images')
    parser.add_argument('--gt-dir', type=str, default='./data/test/labels')
    parser.add_argument('--output-dir', type=str, default='./vis_ablation')
    parser.add_argument('--samples', type=str, default=None,
                        help='comma-separated filenames; auto-select if not specified')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    models = {
        'RMT-UNet (full)': build_model(True, 'encoder'),
        'RMT-UNet w/o Decay': build_model(False, 'encoder'),
        'RMT-UNet w/o ELA': build_model(True, 'none'),
    }
    paths = {
        'RMT-UNet (full)': args.full,
        'RMT-UNet w/o Decay': args.no_decay,
        'RMT-UNet w/o ELA': args.no_ela,
    }
    for name, model in models.items():
        model.load_state_dict(torch.load(paths[name], map_location=DEVICE), strict=False)
        model.to(DEVICE)
        model.eval()

    if args.samples:
        im_names = [n.strip() for n in args.samples.split(',')]
    else:
        im_names = sorted(os.listdir(args.test_dir))[:5]

    for name in im_names:
        img_path = os.path.join(args.test_dir, name)
        gt_path = os.path.join(args.gt_dir, name)

        img_tensor = load_and_preprocess(img_path)
        img_show = cv2.imread(img_path)
        img_show = cv2.resize(img_show, (224, 224))
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
        if gt is not None:
            gt = cv2.resize(gt, (224, 224))

        preds = {}
        for name_m, model in models.items():
            preds[name_m] = predict(model, img_tensor)

        fig, axes = plt.subplots(2, 3, figsize=(13, 8))

        panels = [
            (0, 0, img_show, 'SAR Image'),
            (0, 1, gt, 'Ground Truth'),
            (0, 2, preds['RMT-UNet (full)'], 'RMT-UNet (full)'),
            (1, 0, preds['RMT-UNet w/o Decay'], 'w/o Spatial Decay'),
            (1, 1, preds['RMT-UNet w/o ELA'], 'w/o ELA'),
        ]

        for row, col, data, title in panels:
            ax = axes[row, col]
            if data is not None:
                if data.ndim == 3:
                    ax.imshow(data)
                elif data.max() > 1:
                    ax.imshow(data, cmap='gray')
                else:
                    ax.imshow(data, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        axes[1, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f'ablation_vis_{os.path.splitext(name)[0]}.pdf')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[OK] {save_path}")

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
