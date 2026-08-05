"""
Feature response heatmap visualization — before/after ELA comparison.
Compares encoder feature responses with and without ELA in skip connections.

Usage:
    python vis_features.py --model-path ./output/epoch_best-RMT_T_ELA-sentinel.pth \
                           --test-dir ./data/test/images --output-dir ./vis_features
"""
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--no-ela-model-path', type=str, default=None,
                        help='path to w/o ELA variant (optional)')
    parser.add_argument('--test-dir', type=str, default='./data/test/images')
    parser.add_argument('--gt-dir', type=str, default='./data/test/labels')
    parser.add_argument('--output-dir', type=str, default='./vis_features')
    parser.add_argument('--num-samples', type=int, default=3)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(use_decay=True, ela_position='encoder').to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE), strict=False)
    model.eval()

    if args.no_ela_model_path:
        model_no_ela = build_model(use_decay=True, ela_position='none').to(DEVICE)
        model_no_ela.load_state_dict(torch.load(args.no_ela_model_path, map_location=DEVICE))
        model_no_ela.eval()
    else:
        model_no_ela = None

    features = {}

    def make_hook(name):
        def hook(module, input, output):
            features[name] = {
                'input': input[0].detach().cpu(),
                'output': output.detach().cpu(),
            }
        return hook

    ela_module = model.ela_modules[0]
    handle = ela_module.register_forward_hook(make_hook('ela_0'))

    im_names = sorted(os.listdir(args.test_dir))[:args.num_samples * 2]

    for idx, name in enumerate(im_names):
        if idx >= args.num_samples:
            break

        img_tensor = load_and_preprocess(os.path.join(args.test_dir, name))

        features.clear()
        with torch.no_grad():
            model(img_tensor)

        if 'ela_0' not in features:
            print(f"[SKIP] {name}: hook not triggered")
            continue

        feat_in = features['ela_0']['input']
        feat_out = features['ela_0']['output']

        if model_no_ela is not None:
            features.clear()
            with torch.no_grad():
                model_no_ela(img_tensor)

        img_show = cv2.imread(os.path.join(args.test_dir, name))
        img_show = cv2.resize(img_show, (224, 224))
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        gt_path = os.path.join(args.gt_dir, name)
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (224, 224))
        else:
            gt = np.zeros((224, 224), dtype=np.uint8)

        H = W = int(np.sqrt(feat_in.shape[1]))
        zoom_factor = 224.0 / H

        def to_smooth_heatmap(feat_2d):
            up = zoom(feat_2d, zoom_factor, order=1)
            return gaussian_filter(up, sigma=zoom_factor * 0.8)

        feat_in_2d = feat_in[0].mean(dim=-1).view(H, W).numpy()
        feat_out_2d = feat_out[0].mean(dim=-1).view(H, W).numpy()
        heat_in = to_smooth_heatmap(feat_in_2d)
        heat_out = to_smooth_heatmap(feat_out_2d)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        axes[0, 0].imshow(img_show)
        axes[0, 0].set_title('SAR Image', fontsize=11)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gt, cmap='gray')
        axes[0, 1].set_title('Ground Truth', fontsize=11)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(heat_in, cmap='jet')
        axes[0, 2].set_title('Before ELA (skip feature)', fontsize=11)
        axes[0, 2].axis('off')

        axes[1, 0].imshow(heat_out, cmap='jet')
        axes[1, 0].set_title('After ELA (gated feature)', fontsize=11)
        axes[1, 0].axis('off')

        heat_in_norm = (heat_in - heat_in.min()) / (heat_in.max() - heat_in.min() + 1e-8)
        axes[1, 1].imshow(gt, cmap='gray')
        axes[1, 1].imshow(heat_in_norm, cmap='jet', alpha=0.4)
        axes[1, 1].set_title('Before ELA on GT', fontsize=11)
        axes[1, 1].axis('off')

        heat_out_norm = (heat_out - heat_out.min()) / (heat_out.max() - heat_out.min() + 1e-8)
        axes[1, 2].imshow(gt, cmap='gray')
        axes[1, 2].imshow(heat_out_norm, cmap='jet', alpha=0.4)
        axes[1, 2].set_title('After ELA on GT', fontsize=11)
        axes[1, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f'feature_vis_{name}')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[OK] {save_path}")

    handle.remove()
    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
