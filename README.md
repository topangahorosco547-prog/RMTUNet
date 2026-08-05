# RMT-UNet

Implementation of RMT-UNet for SAR oil spill segmentation, combining **Manhattan-distance spatial decay attention** with **ELA-gated skip connections**.

## Setup

```bash
pip install -r requirements.txt
```

Dataset: [Deep-SAR Oil Spill (SOS)](https://github.com/zhu-xlab/OilSpill-SAR-Dataset). Organize as:

```
data/
├── train/images/     # SAR patches
├── train/labels/     # binary masks
├── test/images/
└── test/labels/
```

## Usage

```bash
# Training
bash train.sh

# Evaluation
python test.py --model-path ./output/epoch_best.pth

# Ablation
python ablation_rmt.py --ablation decay
```
