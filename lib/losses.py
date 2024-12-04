# lib/losses.py

import torch
import torch.nn as nn

def discriminator_loss(real_output, fake_output):
    criterion = nn.MSELoss()
    real_loss = criterion(real_output, torch.ones_like(real_output))
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
    total_loss = (real_loss + fake_loss) * 0.5
    return total_loss

def generator_loss(fake_output):
    criterion = nn.MSELoss()
    loss = criterion(fake_output, torch.ones_like(fake_output))
    return loss

def cycle_consistency_loss(real_image, reconstructed_image, lambda_cycle):
    criterion = nn.L1Loss()
    loss = criterion(reconstructed_image, real_image) * lambda_cycle
    return loss

def identity_loss(real_image, same_image, lambda_identity):
    criterion = nn.L1Loss()
    loss = criterion(same_image, real_image) * lambda_identity
    return loss
