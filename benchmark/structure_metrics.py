"""RNA heavy-atom steric clashes and glycosidic chi-angle metrics.

Steric thresholds follow RNA-FrameFlow Appendix C.4. Chi angles use the
standard O4'-C1'-N9-C4 (purine) or O4'-C1'-N1-C2 (pyrimidine) quartet.
"""

from __future__ import annotations

import math
import numpy as np

VDW_RADIUS = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "P": 1.8,
}
VDW_TOLERANCE = 0.6  # Half of hydrogen's VdW radius (paper value)

def _get_element(atom_name: str) -> str:
    """Get element symbol from atom name (first non-digit character)."""
    for c in atom_name:
        if c.isalpha():
            return c
    return "C"  # Default fallback


def _get_vdw_radius(atom_name: str) -> float:
    """Get VdW radius for an atom based on its element."""
    element = _get_element(atom_name)
    return VDW_RADIUS.get(element, 1.7)  # Default to carbon


def count_steric_clashes(
    atom_coords: np.ndarray,  # (N_res, N_atoms, 3)
    atom_mask: np.ndarray,    # (N_res, N_atoms) bool
    atom_names: list[list[str]],  # (N_res, N_atoms) atom names per residue
    aatype4: np.ndarray,      # (N_res,) nucleotide type 0-3
    res_mask: np.ndarray,     # (N_res,) bool
) -> dict:
    """
    Count steric clashes between inter-residue atom pairs using VdW thresholds.

    Returns:
        {
            "num_clashes": int,
            "clashes_per_100_atoms": float,
            "num_heavy_atoms": int,
            "num_valid_pairs": int,
        }
    """
    num_res = atom_coords.shape[0]
    num_atoms = atom_coords.shape[1]

    # Count total heavy atoms for normalization
    num_heavy_atoms = 0
    for i in range(num_res):
        if not res_mask[i]:
            continue
        for j in range(num_atoms):
            if atom_mask[i, j] and atom_names[i][j]:
                num_heavy_atoms += 1

    if num_heavy_atoms == 0:
        return {
            "num_clashes": 0,
            "clashes_per_100_atoms": 0.0,
            "num_heavy_atoms": 0,
            "num_valid_pairs": 0,
        }

    # Collect valid atoms with their metadata
    valid_atoms = []
    for i in range(num_res):
        if not res_mask[i]:
            continue
        for j in range(num_atoms):
            if atom_mask[i, j] and atom_names[i][j]:
                name = atom_names[i][j]
                valid_atoms.append({
                    "res_idx": i,
                    "atom_idx": j,
                    "coord": atom_coords[i, j],
                    "name": name,
                    "vdw": _get_vdw_radius(name),
                })

    num_clashes = 0
    num_valid_pairs = 0

    # Check all inter-residue pairs
    for ia, atom_a in enumerate(valid_atoms):
        for ib, atom_b in enumerate(valid_atoms):
            if ia >= ib:
                continue

            # Only check inter-residue pairs (same residue atoms are bonded)
            if atom_a["res_idx"] == atom_b["res_idx"]:
                continue

            # Exclude the covalent O3'(res_i)-P(res_i+1) linkage by atom name.
            # Compact atom indices are residue-type dependent, whereas the PDB
            # atom names are the invariant chemical identity needed here.
            res_diff = atom_b["res_idx"] - atom_a["res_idx"]
            if res_diff == 1 and atom_a["name"] == "O3'" and atom_b["name"] == "P":
                continue
            if res_diff == -1 and atom_a["name"] == "P" and atom_b["name"] == "O3'":
                continue

            # Compute distance
            dist = np.linalg.norm(atom_a["coord"] - atom_b["coord"])

            # VdW threshold
            threshold = atom_a["vdw"] + atom_b["vdw"] - VDW_TOLERANCE

            num_valid_pairs += 1
            if dist <= threshold:
                num_clashes += 1

    clashes_per_100 = num_clashes / num_heavy_atoms * 100.0 if num_heavy_atoms > 0 else 0.0

    return {
        "num_clashes": num_clashes,
        "clashes_per_100_atoms": clashes_per_100,
        "num_heavy_atoms": num_heavy_atoms,
        "num_valid_pairs": num_valid_pairs,
    }


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
