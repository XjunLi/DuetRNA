from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


FRAME_CONVENTION = np.diag([-1.0, 1.0, -1.0])
ELEMENT_MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974}


def atom_mass(atom_name: str) -> float:
    return ELEMENT_MASS.get(atom_name.rstrip("'")[0], 12.0)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = normalize(normal)
    return v - np.dot(v, normal) * normal


def reorthogonalize(rot: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rot)
    out = u @ vh
    if np.linalg.det(out) < 0.0:
        vh[-1, :] *= -1.0
        out = u @ vh
    return out


def ensure_right_handed(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    rot = np.stack([normalize(x_axis), normalize(y_axis), normalize(z_axis)], axis=-1)
    rot[:, 0] = normalize(rot[:, 0])
    rot[:, 1] = normalize(project_to_plane(rot[:, 1], rot[:, 0]))
    rot[:, 2] = normalize(np.cross(rot[:, 0], rot[:, 1]))
    rot[:, 1] = normalize(np.cross(rot[:, 2], rot[:, 0]))
    rot = reorthogonalize(rot)
    if np.linalg.det(rot) < 0.0:
        rot[:, 2] *= -1.0
        rot[:, 1] = normalize(np.cross(rot[:, 2], rot[:, 0]))
        rot = reorthogonalize(rot)
    return rot


def rigid_from_3_points_np(
    p_neg_x_axis: np.ndarray,
    origin: np.ndarray,
    p_xy_plane: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    e0 = normalize(origin - p_neg_x_axis)
    e1 = p_xy_plane - origin
    e1 = e1 - np.dot(e1, e0) * e0
    e1 = normalize(e1)
    e2 = normalize(np.cross(e0, e1))
    e1 = normalize(np.cross(e2, e0))
    rot = np.stack([e0, e1, e2], axis=-1)
    return reorthogonalize(rot), origin


def pca_basis(points: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    centered = points - np.average(points, axis=0, weights=weights)
    if weights is None:
        cov = centered.T @ centered / max(points.shape[0], 1)
    else:
        w = weights / np.sum(weights)
        cov = (np.sqrt(w)[:, None] * centered).T @ (np.sqrt(w)[:, None] * centered)
    eigvals_asc, eigvecs_asc = np.linalg.eigh(cov)
    order = np.argsort(eigvals_asc)[::-1]
    return eigvals_asc[order], eigvecs_asc[:, order]


__all__ = [
    "FRAME_CONVENTION",
    "ELEMENT_MASS",
    "atom_mass",
    "ensure_right_handed",
    "normalize",
    "pca_basis",
    "project_to_plane",
    "reorthogonalize",
    "rigid_from_3_points_np",
]
