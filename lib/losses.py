import torch


def cycle_consistency_loss(x, reconstructed_x, lambda_cycle=10.0):
    return lambda_cycle * torch.mean(torch.abs(x - reconstructed_x))


def identity_loss(x, identity_x, lambda_identity=5.0):
    return lambda_identity * torch.mean(torch.abs(x - identity_x))


def gan_loss(pd, target_is_real=True):
    target = torch.ones_like(pd) if target_is_real else torch.zeros_like(pd)
    return torch.mean((pd - target) ** 2)
