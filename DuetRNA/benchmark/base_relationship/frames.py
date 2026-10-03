from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from duetrna_shared_core.geometry.frame_primitives import (
    FRAME_CONVENTION,
    ensure_right_handed,
    normalize,
    project_to_plane,
    rigid_from_3_points_np,
)

from .pdb_parser import Residue


PURINES = {"A", "G"}
PYRIMIDINES = {"C", "U"}
BASE_ATOMS = {
    "A": ("N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "G": ("N9", "C8", "N7", "C5", "C6", "O6", "N1", "N2", "C2", "N3", "C4"),
    "C": ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "U": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
}


@dataclass(frozen=True)
class Frame:
    rot: np.ndarray
    trans: np.ndarray

    def relative_to(self, other: "Frame") -> tuple[np.ndarray, np.ndarray]:
        rel_rot = self.rot.T @ other.rot
        rel_trans = self.rot.T @ (other.trans - self.trans)
        return rel_rot, rel_trans


@dataclass(frozen=True)
class ResidueFrames:
    residue: Residue
    sugar_gs: Frame | None
    base_plane: Frame | None


def _has_atoms(residue: Residue, names: Iterable[str]) -> bool:
    return all(name in residue.atoms for name in names)


def build_sugar_gs_frame(residue: Residue) -> Frame | None:
    """Standard O4'-C4'-C3' Gram-Schmidt sugar frame.

    This matches the DuetRNA Sugar-GS convention: C4' is the origin,
    +x points to O4', +y is the C3' direction after x-projection removal, and
    +z is the sugar-plane normal after applying FRAME_CONVENTION.
    """
    if not _has_atoms(residue, ("O4'", "C4'", "C3'")):
        return None
    rot, _ = rigid_from_3_points_np(
        p_neg_x_axis=residue.atoms["O4'"],
        origin=residue.atoms["C4'"],
        p_xy_plane=residue.atoms["C3'"],
    )
    return Frame(rot=rot @ FRAME_CONVENTION, trans=residue.atoms["C4'"])


def _base_atom_points(residue: Residue) -> tuple[list[str], np.ndarray] | None:
    names = [name for name in BASE_ATOMS.get(residue.resname, ()) if name in residue.atoms]
    if len(names) < 5:
        return None
    return names, np.stack([residue.atoms[name] for name in names], axis=0)


def _base_origin(residue: Residue) -> np.ndarray | None:
    if residue.resname in PURINES and "N9" in residue.atoms:
        return residue.atoms["N9"]
    if residue.resname in PYRIMIDINES and "N1" in residue.atoms:
        return residue.atoms["N1"]
    return None


def _base_chemical_axis(residue: Residue) -> np.ndarray | None:
    if residue.resname in PURINES and _has_atoms(residue, ("C4", "C8")):
        return residue.atoms["C4"] - residue.atoms["C8"]
    if residue.resname in PYRIMIDINES and _has_atoms(residue, ("C4", "C2")):
        return residue.atoms["C4"] - residue.atoms["C2"]
    return None


def _base_normal_reference(residue: Residue) -> np.ndarray | None:
    if residue.resname in PURINES and _has_atoms(residue, ("N9", "C4", "C8")):
        return np.cross(residue.atoms["C4"] - residue.atoms["N9"], residue.atoms["C8"] - residue.atoms["N9"])
    if residue.resname in PYRIMIDINES and _has_atoms(residue, ("N1", "C4", "C2")):
        return np.cross(residue.atoms["C4"] - residue.atoms["N1"], residue.atoms["C2"] - residue.atoms["N1"])
    return None


def build_base_plane_frame(residue: Residue) -> Frame | None:
    """Base-plane frame anchored at N9/N1 and oriented by base chemistry."""
    atom_points = _base_atom_points(residue)
    origin = _base_origin(residue)
    chem_axis = _base_chemical_axis(residue)
    normal_ref = _base_normal_reference(residue)
    if atom_points is None or origin is None or chem_axis is None or normal_ref is None:
        return None
    _, points = atom_points
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1])
    normal_ref = normalize(normal_ref)
    if np.dot(normal, normal_ref) < 0.0:
        normal = -normal
    chem_proj = project_to_plane(chem_axis, normal)
    if np.linalg.norm(chem_proj) < 1e-8:
        return None
    x_axis = normalize(chem_proj)
    y_axis = normalize(np.cross(normal, x_axis))
    if np.linalg.norm(y_axis) < 1e-8:
        return None
    rot = ensure_right_handed(x_axis, y_axis, normal)
    return Frame(rot=rot @ FRAME_CONVENTION, trans=origin)


def build_residue_frames(residues: list[Residue]) -> list[ResidueFrames]:
    return [
        ResidueFrames(
            residue=residue,
            sugar_gs=build_sugar_gs_frame(residue),
            base_plane=build_base_plane_frame(residue),
        )
        for residue in residues
    ]


def rotmat_to_rotvec(rot: np.ndarray) -> np.ndarray:
    cos_angle = (float(np.trace(rot)) - 1.0) * 0.5
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    denom = 2.0 * np.sin(angle)
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    ) / denom
    return axis * angle


def relative_pose_vector(src: Frame, dst: Frame) -> np.ndarray:
    rel_rot, rel_trans = src.relative_to(dst)
    return np.concatenate([rotmat_to_rotvec(rel_rot), rel_trans], axis=0)
