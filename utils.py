import torch
import logging

logger = logging.getLogger(__name__)

def weighted_mse_loss(output, target, target_weights):
    # logging.info(f"target_weights.shape = {target_weights.shape}")
    # logging.info(f"output.shape = {output.shape}")
    # logging.info(f"target.shape = {target.shape}")

    return torch.sum(target_weights * (output.squeeze(2) - target) ** 2)