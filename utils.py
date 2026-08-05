"""Loss functions for RMT-UNet segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """Dice loss for binary segmentation.

    Args:
        smooth: smoothing factor to avoid division by zero
        p: exponent for denominator
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception(f'Unexpected reduction {self.reduction}')


class DiceLoss(nn.Module):
    """Multi-class Dice loss."""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss for binary segmentation."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets.float()
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    Args:
        alpha: weighting factor for the rare class
        gamma: focusing parameter (0 = standard cross-entropy)
        size_average: if True, average over batch; otherwise sum
        ignore_index: target value to ignore in loss computation
    """

    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


# ---------------------------------------------------------------------------
#  General-purpose utilities
# ---------------------------------------------------------------------------


class AverageMeter:
    """Track running average, sum, and current value of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: nn.Module, trainable_only: bool = True) -> float:
    """Return the number of (trainable) parameters in millions.

    Args:
        model: PyTorch module.
        trainable_only: if True, count only parameters with requires_grad=True.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return sum(p.numel() for p in model.parameters()) / 1e6


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility across numpy, torch, and cuda.

    Args:
        seed: integer seed value.
        deterministic: if True, enable cudnn deterministic mode (slower).
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def save_checkpoint(state: dict, path: str, is_best: bool = False):
    """Save model checkpoint with optional best-model copy.

    Args:
        state: dict containing 'epoch', 'state_dict', 'optimizer', etc.
        path: file path for the checkpoint (e.g. './output/checkpoint.pth').
        is_best: if True, also save a copy as 'model_best.pth' next to *path*.
    """
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), 'model_best.pth')
        torch.save(state, best_path)


def calculate_confusion_matrix(pred, target, num_classes):
    """Compute per-class TP, FP, FN from integer labels.

    Args:
        pred: (B, H, W) tensor of predicted class indices.
        target: (B, H, W) tensor of ground-truth class indices.
        num_classes: total number of classes.

    Returns:
        tp, fp, fn: tensors of shape (num_classes,).
    """
    pred = pred.view(-1)
    target = target.view(-1)
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    for c in range(num_classes):
        tp[c] = ((pred == c) & (target == c)).sum()
        fp[c] = ((pred == c) & (target != c)).sum()
        fn[c] = ((pred != c) & (target == c)).sum()

    return tp, fp, fn


def poly_lr_scheduler(optimizer, init_lr, iteration, max_iteration, power=0.9):
    """Polynomial learning rate decay (as used in the paper).

    Sets each param group LR to  init_lr * (1 - iter/max_iter)^power.

    Args:
        optimizer: torch optimizer.
        init_lr: base learning rate.
        iteration: current iteration (0-indexed).
        max_iteration: total number of iterations.
        power: polynomial decay exponent (default 0.9).
    """
    lr = init_lr * (1.0 - iteration / max_iteration) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
