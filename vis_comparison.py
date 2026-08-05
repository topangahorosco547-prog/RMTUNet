"""
Figure 2: Multi-model qualitative comparison figure.
Layout: 3 samples (rows) x 6 columns (SAR | GT | RMT-UNet | Swin-Unet | TransUNet | DeepLabv3+)
Each row can optionally highlight a region of interest (red box).

Directory structure:
    vis_comparision/
    ├── Images/
    │   ├── Sentinel-1A/   (SAR images)
    │   └── PALSAR/
    ├── Labels/
    │   ├── Sentinel-1A/   (GT masks)
    │   └── PALSAR/
    ├── RMT-UNet/
    │   ├── Sentinel-1A/   (predictions)
    │   └── PALSAR/
    ├── Swin-Unet/
    ├── TransUNet/
    └── DeepLabv3+/

Usage:
    python vis_comparison.py \
        --pred-root ./vis_comparision \
        --dataset Sentinel-1A \
        --samples sample_A.png,sample_B.png,sample_C.png \
        --output fig2_comparison.pdf
"""
import argparse
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def load_image(path, size=(224, 224), is_mask=False):
    """Load and resize image or mask."""
    if is_mask:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    return img


def overlay_boundary(img_rgb, mask, color=(255, 0, 0), thickness=1):
    """Overlay segmentation boundary on RGB image."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img_rgb.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def draw_prediction(mask, cmap='gray'):
    """Convert prediction mask to grayscale display."""
    return mask * 255


# -- ROI parameters (per-sample region [x, y, w, h] in pixel coordinates) --
# If not specified, default region is used; set to None to skip zoom-in.
DEFAULT_ROIS = {
    # 'sample_A.png': (80, 60, 64, 64),
}


def main():
    parser = argparse.ArgumentParser(description='Generate streamlined comparison figure')
    parser.add_argument('--pred-root', type=str, required=True,
                        help='Root dir (e.g. ./vis_comparision), expects Images/dataset/, Labels/dataset/, model/dataset/')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset subdir name (e.g. Sentinel-1A, PALSAR)')
    parser.add_argument('--samples', type=str, required=True,
                        help='3 sample filenames, comma-separated, e.g.: a.png,b.png,c.png')
    parser.add_argument('--output', type=str, default='fig2_comparison.pdf')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--rois', type=str, default=None,
                        help='ROI regions, format: name:x,y,w,h;name2:x,y,w,h (optional)')
    args = parser.parse_args()

    sample_names = [s.strip() for s in args.samples.split(',')]
    assert len(sample_names) == 3, "Exactly 3 samples required"

    rois = {}
    if args.rois:
        for item in args.rois.split(';'):
            parts = item.strip().split(':')
            if len(parts) == 2:
                name = parts[0].strip()
                coords = [int(x) for x in parts[1].split(',')]
                rois[name] = tuple(coords)

    test_dir = os.path.join(args.pred_root, 'Images', args.dataset)
    gt_dir   = os.path.join(args.pred_root, 'Labels', args.dataset)

    methods = {
        'RMT-UNet':    os.path.join(args.pred_root, 'RMT-UNet', args.dataset),
        'Swin-Unet':   os.path.join(args.pred_root, 'Swin-Unet', args.dataset),
        'TransUNet':   os.path.join(args.pred_root, 'TransUNet', args.dataset),
        'DeepLabv3+':  os.path.join(args.pred_root, 'DeepLabv3+', args.dataset),
    }

    n_rows, n_cols = 3, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 4.0))

    col_titles = ['SAR Image', 'Ground Truth', 'RMT-UNet', 'Swin-Unet', 'TransUNet',
                  'DeepLabv3+\n(ResNet101)']

    method_keys = ['RMT-UNet', 'Swin-Unet', 'TransUNet', 'DeepLabv3+']

    for row_idx, sname in enumerate(sample_names):
        img = load_image(os.path.join(test_dir, sname))
        gt = load_image(os.path.join(gt_dir, sname), is_mask=True)

        roi = rois.get(sname, None)

        for col_idx, title in enumerate(col_titles):
            ax = axes[row_idx, col_idx]

            if col_idx == 0:
                ax.imshow(img)
                if roi is not None:
                    x, y, w, h = roi
                    rect = Rectangle((x, y), w, h, linewidth=0.8,
                                     edgecolor='red', facecolor='none', linestyle='-')
                    ax.add_patch(rect)
            elif col_idx == 1:
                ax.imshow(gt, cmap='gray', vmin=0, vmax=255)
            else:
                pred_dir = methods[method_keys[col_idx - 2]]
                pred_path = os.path.join(pred_dir, sname)
                pred = load_image(pred_path, is_mask=True)
                img_with_boundary = overlay_boundary(img.copy(), pred, color=(0, 255, 0), thickness=1)
                ax.imshow(img_with_boundary)

            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=7.5, pad=2)

    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel(f'Sample {row_idx+1}', fontsize=7.5, labelpad=2)

    plt.tight_layout(pad=0.5, w_pad=0.3, h_pad=0.5)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    png_path = args.output.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {args.output}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
