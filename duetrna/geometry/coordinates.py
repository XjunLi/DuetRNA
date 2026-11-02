"""Coordinate-unit conversion shared by model and evaluation code."""

import torch

NM_TO_ANG_SCALE = 10.0

ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

def to_numpy(value: torch.Tensor):
    """Detach a tensor and expose its CPU NumPy array."""
    return value.detach().cpu().numpy()
