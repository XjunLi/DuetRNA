"""Steric clash computation using VdW-radius-based thresholds.

Implementation follows RNA-FrameFlow paper Appendix C.4:
- d_steric = vdw_radius_i + vdw_radius_j - 0.6 Å
- Count inter-residue atom pairs where distance <= d_steric
- Exclude phosphodiester bonds (O3'-P between consecutive residues)
"""

from __future__ import annotations

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
