import torch


def weighted_mse_loss(output, target, target_weights):
    return torch.sum(target_weights * (output - target) ** 2)