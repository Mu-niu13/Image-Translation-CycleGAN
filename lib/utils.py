# lib/utils.py

import torch
from torchvision.utils import save_image


def adjust_learning_rate(optimizer, initial_lr, epoch, decay_epoch):
    lr = initial_lr * (0.1 ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
