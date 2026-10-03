"""Chi angle (glycosidic torsion) extraction and comparison.

Evaluates the standard glycosidic chi torsion:
- A/G: ["O4'", "C1'", "N9", "C4"]
- U/C: ["O4'", "C1'", "N1", "C2"]
"""

from __future__ import annotations

import math
import numpy as np


def _dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute dihedral angle in degrees from 4 points."""
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-10:
        return float("nan")
    b1 = b1 / b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) < 1e-10 or np.linalg.norm(w) < 1e-10:
        return float("nan")
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def extract_chi_angle_deg(
    atom_coords: np.ndarray,  # (N_res, N_atoms, 3) or (N_res, 23, 3)
    atom_mask: np.ndarray,    # (N_res, N_atoms) bool
    atom_names: list[list[str]],  # (N_res, N_atoms) atom names
    aatype4: np.ndarray,      # (N_res,) 0=A, 1=U, 2=G, 3=C
    res_mask: np.ndarray,     # (N_res,) bool
    chi_index: int = 9,       # retained for compatibility with the old caller
) -> np.ndarray:
    """
    Extract the standard glycosidic chi angle for each residue.

    Returns: (N_res,) array of angles in degrees, NaN for missing atoms.
    """
    num_res = atom_coords.shape[0]
    angles = np.full(num_res, np.nan)

    if chi_index != 9:
        raise ValueError("Only the glycosidic chi torsion is supported")

    # Standard IUPAC atom quartets for each base type.
    chi_atoms = {
        0: ["O4'", "C1'", "N9", "C4"],  # A
        1: ["O4'", "C1'", "N1", "C2"],  # U
        2: ["O4'", "C1'", "N9", "C4"],  # G
        3: ["O4'", "C1'", "N1", "C2"],  # C
    }

    for i in range(num_res):
        if not res_mask[i]:
            continue

        base_type = aatype4[i]
        atom_quartet = chi_atoms[base_type]

        # Find atom indices
        coords = []
        valid = True
        for atom_name in atom_quartet:
            found = False
            for j, name in enumerate(atom_names[i]):
                if name == atom_name and atom_mask[i, j]:
                    coords.append(atom_coords[i, j])
                    found = True
                    break
            if not found:
                valid = False
                break

        if valid and len(coords) == 4:
            angles[i] = _dihedral_angle(coords[0], coords[1], coords[2], coords[3])

    return angles


def compute_chi_metrics(
    pred_angles: np.ndarray,  # (N_res,) predicted chi angles
    gt_angles: np.ndarray,    # (N_res,) ground truth chi angles
    mask: np.ndarray | None = None,  # (N_res,) valid mask
) -> dict:
    """
    Compute chi angle comparison metrics.

    Returns:
        {
            "chi_mae": float,           # Mean absolute error (degrees)
            "chi_within_5deg": float,    # Ratio within ±5°
            "chi_within_10deg": float,   # Ratio within ±10°
            "chi_median_error": float,  # Median absolute error
            "n_valid": int,             # Number of valid residues compared
        }
    """
    if mask is None:
        # Both pred and gt must be valid (not NaN)
        mask = ~(np.isnan(pred_angles) | np.isnan(gt_angles))
    else:
        mask = mask & ~(np.isnan(pred_angles) | np.isnan(gt_angles))

    valid_pred = pred_angles[mask]
    valid_gt = gt_angles[mask]

    n_valid = len(valid_pred)
    if n_valid == 0:
        return {
            "chi_mae": float("nan"),
            "chi_within_5deg": float("nan"),
            "chi_within_10deg": float("nan"),
            "chi_median_error": float("nan"),
            "n_valid": 0,
        }

    # Compute absolute angular difference (handle wrap-around)
    diff = np.abs(valid_pred - valid_gt)
    # Handle angles > 180: wrap to [-180, 180]
    diff = np.where(diff > 180, 360 - diff, diff)

    mae = np.mean(diff)
    median_error = np.median(diff)
    within_5 = np.mean(diff <= 5.0)
    within_10 = np.mean(diff <= 10.0)

    return {
        "chi_mae": float(mae),
        "chi_within_5deg": float(within_5),
        "chi_within_10deg": float(within_10),
        "chi_median_error": float(median_error),
        "n_valid": n_valid,
    }
