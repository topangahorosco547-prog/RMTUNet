import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from OilSpillDatasets import ImageFolder
from utils import BCEDiceLoss


def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=os.path.join(snapshot_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu

    db_train = ImageFolder("./data", mode='train')

    print(f"The length of train set is: {len(db_train)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)
        torch.manual_seed(args.seed + worker_id)

    weights = []
    for r in db_train.fg_ratios:
        if r <= 0:
            weights.append(0.5)
        elif r < 0.005:
            weights.append(2.0)
        elif r < 0.02:
            weights.append(3.0)
        else:
            weights.append(1.5)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-3)

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

    best_loss = float('inf')
    log_loss = []

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for image_batch, label_batch in trainloader:
            image_batch = image_batch.to(args.device)
            label_batch = label_batch.to(args.device).float()

            seg_logits = model(image_batch)

            loss = loss_fn(seg_logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            log_loss.append([iter_num, loss.item()])

        logging.info(f'iteration {iter_num} : loss : {loss.item():f}')

        if loss.item() < best_loss:
            best_loss = loss.item()
            save_mode_path = os.path.join(snapshot_path, 'epoch_best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved best model (loss {best_loss:.4f}) to {save_mode_path}")

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"save model to {save_mode_path}")

    loss_file = os.path.join(snapshot_path, 'training_log.json')
    with open(loss_file, 'w') as f:
        json.dump(log_loss, f)
    logging.info(f"Loss log saved to {loss_file} ({len(log_loss)} iterations)")
    iterator.close()

    writer.close()
    return "Training Finished!"
