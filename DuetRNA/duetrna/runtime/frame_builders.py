from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from duetrna_shared_core.chemistry import nucleotide_constants as nc
from duetrna_shared_core.chemistry import vocabulary
from duetrna_shared_core.geometry import rigid_from_3_points_np
from duetrna_shared_core.geometry.rigid_utils import Rigid, Rotation


_ELEMENT_MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974}
_BACKBONE_FRAME_ATOMS = ("O4'", "C4'", "C3'")
_FRAME_CONVENTION = np.diag([-1.0, 1.0, -1.0])
_FAMILY_TO_NLINK = {"purine": "N9", "pyrimidine": "N1"}


def _atom_mass(atom_name: str) -> float:
    return _ELEMENT_MASS.get(atom_name.rstrip("'")[0], 12.0)


def _restype_from_encoded_aatype(aatype: int, is_deoxy: bool) -> Optional[str]:
    if 0 <= aatype < len(nc.restypes):
        return nc.restypes[aatype]
    if aatype <= vocabulary.protein_restype_num:
        return None

    na_code = aatype - (vocabulary.protein_restype_num + 1)
    if na_code == 0:
        return "DA" if is_deoxy else "A"
    if na_code == 1:
        return "DC" if is_deoxy else "C"
    if na_code == 2:
        return "DG" if is_deoxy else "G"
    if na_code == 3:
        return "DT"
    if na_code == 4:
        return "DT" if is_deoxy else "U"
    return None


def _family(restype: str) -> str:
    return "purine" if restype in {"A", "G", "DA", "DG"} else "pyrimidine"


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def _project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = _normalize(normal)
    return v - np.dot(v, normal) * normal


def _reorthogonalize(rot: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rot)
    out = u @ vh
    if np.linalg.det(out) < 0.0:
        vh[-1, :] *= -1.0
        out = u @ vh
    return out


def _ensure_right_handed(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    rot = np.stack([_normalize(x_axis), _normalize(y_axis), _normalize(z_axis)], axis=-1)
    rot[:, 0] = _normalize(rot[:, 0])
    rot[:, 1] = _normalize(_project_to_plane(rot[:, 1], rot[:, 0]))
    rot[:, 2] = _normalize(np.cross(rot[:, 0], rot[:, 1]))
    rot[:, 1] = _normalize(np.cross(rot[:, 2], rot[:, 0]))
    rot = _reorthogonalize(rot)
    if np.linalg.det(rot) < 0.0:
        rot[:, 2] *= -1.0
        rot[:, 1] = _normalize(np.cross(rot[:, 2], rot[:, 0]))
        rot = _reorthogonalize(rot)
    return rot


def _pca_basis(points: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    centered = points - np.average(points, axis=0, weights=weights)
    if weights is None:
        cov = centered.T @ centered / max(points.shape[0], 1)
    else:
        w = weights / np.sum(weights)
        cov = (np.sqrt(w)[:, None] * centered).T @ (np.sqrt(w)[:, None] * centered)
    eigvals_asc, eigvecs_asc = np.linalg.eigh(cov)
    order = np.argsort(eigvals_asc)[::-1]
    return eigvals_asc[order], eigvecs_asc[:, order]


def _base_full_atom_names(restype: str) -> Tuple[str, ...]:
    out = []
    for idx, atom_name in enumerate(nc.restype_name_to_compact_atom_names[restype]):
        if idx < 11 or not atom_name or atom_name == "O2'":
            continue
        out.append(atom_name)
    return tuple(out)


def _base_chemical_axis(restype: str, coords_row: np.ndarray) -> Optional[np.ndarray]:
    needed = ("C4", "C8") if _family(restype) == "purine" else ("C4", "C2")
    if not all(name in nc.atom_order for name in needed):
        return None
    return coords_row[nc.atom_order[needed[0]]] - coords_row[nc.atom_order[needed[1]]]


def _base_normal_reference(restype: str, coords_row: np.ndarray) -> Optional[np.ndarray]:
    if _family(restype) == "purine":
        return np.cross(
            coords_row[nc.atom_order["C4"]] - coords_row[nc.atom_order["C8"]],
            coords_row[nc.atom_order["C2"]] - coords_row[nc.atom_order["N9"]],
        )
    return np.cross(
        coords_row[nc.atom_order["C4"]] - coords_row[nc.atom_order["C2"]],
        coords_row[nc.atom_order["C6"]] - coords_row[nc.atom_order["N1"]],
    )


def _compute_origin(coords_row: np.ndarray, restype: str, atom_names: Sequence[str], origin_mode: str) -> np.ndarray:
    if origin_mode == "c4":
        return coords_row[nc.atom_order["C4'"]]
    if origin_mode == "c1":
        return coords_row[nc.atom_order["C1'"]]
    if origin_mode == "connection_atom":
        if atom_names and atom_names[0].startswith("N"):
            link = _FAMILY_TO_NLINK[_family(restype)]
            return coords_row[nc.atom_order[link]]
        return coords_row[nc.atom_order["C4'"]]
    if origin_mode == "mass_centroid":
        weights = np.asarray([_atom_mass(name) for name in atom_names], dtype=np.float64)
        weights = weights / np.sum(weights)
        points = np.stack([coords_row[nc.atom_order[name]] for name in atom_names], axis=0)
        return np.sum(points * weights[:, None], axis=0)
    raise ValueError(f"Unsupported origin_mode: {origin_mode}")


def _atoms_exist(mask_row: np.ndarray, atom_names: Sequence[str]) -> bool:
    return all(mask_row[nc.atom_order[name]] > 0.5 for name in atom_names)


def _build_sugar_gs_frame(coords_row: np.ndarray, origin_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    # GT-closure frame: this is the same three anchor definition used by the
    # torsion decoder constants, so C4'/O4'/C3' close under reconstruction.
    rot, _ = rigid_from_3_points_np(
        p_neg_x_axis=coords_row[nc.atom_order["O4'"]],
        origin=coords_row[nc.atom_order["C4'"]],
        p_xy_plane=coords_row[nc.atom_order["C3'"]],
    )
    origin = _compute_origin(coords_row, "A", _BACKBONE_FRAME_ATOMS, origin_mode)
    return rot @ _FRAME_CONVENTION, origin


def _build_base_plane_anchor_frame(coords_row: np.ndarray, restype: str, origin_mode: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    atom_names = _base_full_atom_names(restype)
    points = np.stack([coords_row[nc.atom_order[name]] for name in atom_names], axis=0)
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _normalize(vh[-1])
    chem_axis = _base_chemical_axis(restype, coords_row)
    normal_ref = _normalize(_base_normal_reference(restype, coords_row))
    if chem_axis is None or not np.any(normal_ref):
        return None
    chem_proj = _project_to_plane(chem_axis, normal)
    if np.linalg.norm(chem_proj) < 1e-8:
        return None
    x_axis = _normalize(chem_proj)
    if np.dot(normal, normal_ref) < 0.0:
        normal *= -1.0
    y_axis = _normalize(np.cross(normal, x_axis))
    if not np.any(y_axis):
        return None
    rot = _ensure_right_handed(x_axis, y_axis, normal)
    origin = _compute_origin(coords_row, restype, atom_names, origin_mode)
    return rot @ _FRAME_CONVENTION, origin


def build_dual_frames(
    na,
    *,
    base_origin_mode: str = "connection_atom",
    sugar_origin_mode: str = "c4",
) -> Dict[str, Tuple[Rigid, torch.Tensor]]:
    atom27_pos = na["all_atom_positions"]
    atom27_mask = na["all_atom_mask"]
    aatype = na["aatype"].to(torch.long)
    atom_deoxy = na["atom_deoxy"].to(torch.bool)
    num_res = atom27_pos.shape[0]

    base_rot = torch.eye(3, dtype=atom27_pos.dtype, device=atom27_pos.device).repeat(num_res, 1, 1)
    base_trans = torch.zeros(num_res, 3, dtype=atom27_pos.dtype, device=atom27_pos.device)
    base_exists = torch.zeros(num_res, dtype=atom27_mask.dtype, device=atom27_mask.device)

    sugar_rot = torch.eye(3, dtype=atom27_pos.dtype, device=atom27_pos.device).repeat(num_res, 1, 1)
    sugar_trans = torch.zeros(num_res, 3, dtype=atom27_pos.dtype, device=atom27_pos.device)
    sugar_exists = torch.zeros(num_res, dtype=atom27_mask.dtype, device=atom27_mask.device)

    for idx in range(num_res):
        restype = _restype_from_encoded_aatype(
            int(aatype[idx].item()),
            bool(atom_deoxy[idx].item()),
        )
        if restype is None or restype == "X":
            continue
        coords_row = atom27_pos[idx].detach().cpu().numpy().astype(np.float64)
        mask_row = atom27_mask[idx].detach().cpu().numpy().astype(np.float64)

        atom_names = _base_full_atom_names(restype)
        if atom_names and _atoms_exist(mask_row, atom_names):
            result = _build_base_plane_anchor_frame(coords_row, restype, base_origin_mode)
            if result is not None:
                rot_i, trans_i = result
                base_rot[idx] = torch.tensor(rot_i, dtype=atom27_pos.dtype, device=atom27_pos.device)
                base_trans[idx] = torch.tensor(trans_i, dtype=atom27_pos.dtype, device=atom27_pos.device)
                base_exists[idx] = 1.0

        if _atoms_exist(mask_row, _BACKBONE_FRAME_ATOMS):
            result = _build_sugar_gs_frame(coords_row, sugar_origin_mode)
            if result is not None:
                rot_i, trans_i = result
                sugar_rot[idx] = torch.tensor(rot_i, dtype=atom27_pos.dtype, device=atom27_pos.device)
                sugar_trans[idx] = torch.tensor(trans_i, dtype=atom27_pos.dtype, device=atom27_pos.device)
                sugar_exists[idx] = 1.0

    return {
        "base_frame": (Rigid(rots=Rotation(rot_mats=base_rot), trans=base_trans), base_exists),
        "sugar_frame": (Rigid(rots=Rotation(rot_mats=sugar_rot), trans=sugar_trans), sugar_exists),
    }
