import argparse
import os
import cv2
import numpy as np
import torch
from medpy.metric import binary
from sklearn.metrics import f1_score, precision_score, accuracy_score, jaccard_score

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def inference_single(model, model_path, test_path, save_path):
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    im_names = os.listdir(test_path)
    for name in im_names:
        image = _preprocess_image(os.path.join(test_path, name))
        with torch.no_grad():
            output = torch.sigmoid(model(image))
        output = output.cpu().numpy().squeeze()
        output = (output >= 0.5).astype(np.uint8)
        save_full = os.path.join(save_path, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_full, output * 255)


def _preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    image = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
    image = np.array(image, np.float32).transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return torch.Tensor(image).to(DEVICE)


def _run_evaluation(model, test_path, gt_path, verbose=True):
    im_names = os.listdir(test_path)
    hd95_list, f1_list, precision_list, accuracy_list, dice_list, iou_list = [], [], [], [], [], []
    fp_rate_list, fn_rate_list = [], []
    tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0

    for name in im_names:
        image = _preprocess_image(os.path.join(test_path, name))
        with torch.no_grad():
            output = torch.sigmoid(model(image))
        output = output.cpu().numpy().squeeze()
        output = (output >= 0.5).astype(np.uint8)

        gt_full = os.path.join(gt_path, name)
        gt = cv2.imread(gt_full, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (224, 224), interpolation=cv2.INTER_NEAREST)
        gt = np.array(gt, np.float32) / 255.0

        hd95 = binary_hd95(output, gt)
        f1, precision, accuracy, dice, iou, tp, fp, tn, fn, fp_rate, fn_rate = calculate_metrics(output, gt)

        iou_list.append(iou)
        f1_list.append(f1)
        dice_list.append(dice)
        hd95_list.append(hd95)
        precision_list.append(precision)
        accuracy_list.append(accuracy)
        fp_rate_list.append(fp_rate)
        fn_rate_list.append(fn_rate)
        tp_total += tp
        fp_total += fp
        tn_total += tn
        fn_total += fn

    mean_iou = np.mean(iou_list)
    mean_dice = np.mean(dice_list)
    mean_f1 = np.mean(f1_list)
    mean_hd95 = np.mean(hd95_list)
    mean_precision = np.mean(precision_list)
    mean_accuracy = np.mean(accuracy_list)
    mean_fpr = np.mean(fp_rate_list)
    mean_fnr = np.mean(fn_rate_list)
    overall_fpr = fp_total / (fp_total + tn_total + 1e-7)
    overall_fnr = fn_total / (fn_total + tp_total + 1e-7)

    if verbose:
        print(f'Average IoU: {mean_iou:.4f}')
        print(f'Average F1 Score: {mean_f1:.4f}')
        print(f'Average Dice: {mean_dice:.4f}')
        print(f'Average HD95: {mean_hd95:.4f}')
        print(f'Average Precision: {mean_precision:.4f}')
        print(f'Average Accuracy: {mean_accuracy:.4f}')
        print(f'Average FP Rate (FPR): {mean_fpr * 100:.4f}%')
        print(f'Average FN Rate (FNR): {mean_fnr * 100:.4f}%')
        print(f'Overall FP Rate: {overall_fpr * 100:.4f}%')
        print(f'Overall FN Rate: {overall_fnr * 100:.4f}%')

    return [mean_iou, mean_f1, mean_dice, mean_hd95, mean_precision, mean_accuracy, mean_fpr, mean_fnr]


def evaluate_model(model, model_path, test_path, gt_path):
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path), False)
    model.eval()
    _run_evaluation(model, test_path, gt_path, verbose=True)


def evaluate_model_training(model, test_path, gt_path):
    was_training = model.training
    model.eval()
    results = _run_evaluation(model, test_path, gt_path, verbose=True)
    if was_training:
        model.train()
    return results


def calculate_metrics(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5

    smooth = 1e-7

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    f1 = f1_score(gt.flatten(), pred.flatten(), zero_division=1)
    precision = precision_score(gt.flatten(), pred.flatten(), zero_division=1)
    accuracy = accuracy_score(gt.flatten(), pred.flatten())
    dice = (2 * tp + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = jaccard_score(gt.flatten(), pred.flatten(), zero_division=1)

    fp_rate = fp / (fp + tn + smooth)
    fn_rate = fn / (fn + tp + smooth)

    return f1, precision, accuracy, dice, iou, tp, fp, tn, fn, fp_rate, fn_rate


def binary_iou(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    else:
        return intersection / union


def binary_dice(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()
    if denominator == 0:
        return 1.0
    else:
        return 2 * intersection / denominator


def binary_hd95(pred, gt):

    pred = pred > 0.5
    gt = gt > 0.5

    pred_empty = np.all(pred == 0)
    gt_empty = np.all(gt == 0)

    if pred_empty and gt_empty:
        return 0.0
    if pred_empty != gt_empty:
        return 100.0

    hd95_value = binary.hd95(pred, gt)

    return hd95_value


def binary_precision(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    if tp + fp == 0:
        return 1.0
    else:
        return tp / (tp + fp)


def binary_accuracy(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    if tp + tn + fp + fn == 0:
        return 1.0
    else:
        return (tp + tn) / (tp + tn + fp + fn)


if __name__ == "__main__":
    from RMT_UNet import VisRetNet

    parser = argparse.ArgumentParser()
    parser.add_argument('--use-decay', type=int, default=1, choices=[0, 1])
    parser.add_argument('--ela-position', type=str, default='encoder',
                        choices=['after_fusion', 'encoder', 'both', 'none'])
    parser.add_argument('--model-path', type=str,
                        default='./output/epoch_best-RMT_T_ELA-sentinel.pth',
                        help='model checkpoint path')
    args = parser.parse_args()

    net = VisRetNet(
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
        use_decay=bool(args.use_decay),
        ela_position=args.ela_position,
    ).to(DEVICE)

    test_root = './data/test/images'
    test_save_path = './predictions/'
    gt_root = './data/test/labels'
    model_path = args.model_path
    inference_single(net, model_path, test_root, test_save_path)
    evaluate_model(net, model_path, test_root, gt_root)