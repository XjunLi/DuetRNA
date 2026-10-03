from __future__ import annotations

import numpy as np
import torch

from duetrna_shared_core.chemistry import nc


def compute_rmsd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    error = (pred - target) ** 2
    if mask is not None:
        error = error * mask[..., None]
        denom = mask.sum(dim=-1).clamp(min=1.0) * pred.shape[-1]
    else:
        denom = torch.tensor(pred.shape[-2] * pred.shape[-1], device=pred.device, dtype=pred.dtype)
    return torch.sqrt(error.sum(dim=(-1, -2)) / denom)


def calc_rna_c4_c4_metrics(c4_pos: np.ndarray, bond_tol: float = 0.1, clash_tol: float = 1.0) -> dict[str, float]:
    coords = np.asarray(c4_pos, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"Expected C4 coordinates with shape [N, 3], got {coords.shape}")
    if coords.shape[0] < 2:
        return {
            "avg_c4_bond_dists": 0.0,
            "c4_c4_deviation": 0.0,
            "c4_c4_valid_percent": 0.0,
            "num_c4_c4_clashes": 0.0,
            "radius_of_gyration": 0.0,
        }

    c4_bond_dists = np.linalg.norm(coords[1:] - coords[:-1], axis=-1)
    avg_c4_bond_dist = float(c4_bond_dists.mean())
    c4_c4_dev = float(np.mean(np.abs(c4_bond_dists - nc.c4_c4)))
    c4_c4_valid = float(np.mean(c4_bond_dists < (nc.c4_c4 + bond_tol)))

    c4_c4_dists2d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    inter_dists = c4_c4_dists2d[np.where(np.triu(c4_c4_dists2d, k=1) > 0)]
    clashes = inter_dists < clash_tol

    center = coords.mean(axis=0, keepdims=True)
    dists_sq = np.sum((coords - center) ** 2, axis=-1)
    rad_gyr = float(np.sqrt(dists_sq.mean()))

    return {
        "avg_c4_bond_dists": avg_c4_bond_dist,
        "c4_c4_deviation": c4_c4_dev,
        "c4_c4_valid_percent": c4_c4_valid,
        "num_c4_c4_clashes": float(np.sum(clashes)),
        "radius_of_gyration": rad_gyr,
    }
