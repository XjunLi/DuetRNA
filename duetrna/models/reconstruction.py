from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Tuple

import numpy as np
import torch

from duetrna.chemistry.residues import (
    AATYPE4_TO_RESTYPE,
    aatype4_to_aatype9,
    compact_atom_index,
    nc,
)
from duetrna.geometry.frames import FRAME_CONVENTION, ensure_right_handed, normalize, project_to_plane
from duetrna.geometry.rigid import create_rigid
from duetrna.geometry import all_atom as all_atom_reconstruction


def _base_atom_names(restype: str) -> tuple[str, ...]:
    return tuple(
        atom_name
        for idx, atom_name in enumerate(nc.restype_name_to_compact_atom_names[restype])
        if idx >= 11 and atom_name and atom_name != "O2'"
    )


def _family(restype: str) -> str:
    return "purine" if restype in {"A", "G", "DA", "DG"} else "pyrimidine"


def _base_chemical_axis(restype: str, coords_row: torch.Tensor) -> torch.Tensor:
    needed = ("C4", "C8") if _family(restype) == "purine" else ("C4", "C2")
    return coords_row[compact_atom_index(restype, needed[0])] - coords_row[compact_atom_index(restype, needed[1])]


def _base_normal_reference(restype: str, coords_row: torch.Tensor) -> torch.Tensor:
    if _family(restype) == "purine":
        return torch.cross(
            coords_row[compact_atom_index(restype, "C4")] - coords_row[compact_atom_index(restype, "C8")],
            coords_row[compact_atom_index(restype, "C2")] - coords_row[compact_atom_index(restype, "N9")],
            dim=-1,
        )
    return torch.cross(
        coords_row[compact_atom_index(restype, "C4")] - coords_row[compact_atom_index(restype, "C2")],
        coords_row[compact_atom_index(restype, "C6")] - coords_row[compact_atom_index(restype, "N1")],
        dim=-1,
    )


def _build_base_plane_anchor_frame_from_atom23(coords_row: torch.Tensor, restype: str) -> tuple[torch.Tensor, torch.Tensor]:
    atom_names = _base_atom_names(restype)
    points = torch.stack([coords_row[compact_atom_index(restype, name)] for name in atom_names], dim=0)
    centered = points - points.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1].detach().cpu().numpy())
    chem_axis = _base_chemical_axis(restype, coords_row).detach().cpu().numpy()
    normal_ref = normalize(_base_normal_reference(restype, coords_row).detach().cpu().numpy())
    chem_proj = project_to_plane(chem_axis, normal)
    x_axis = normalize(chem_proj)
    if np.dot(normal, normal_ref) < 0.0:
        normal *= -1.0
    y_axis = normalize(np.cross(normal, x_axis))
    rot = ensure_right_handed(x_axis, y_axis, normal) @ FRAME_CONVENTION
    link = "N9" if _family(restype) == "purine" else "N1"
    origin = coords_row[compact_atom_index(restype, link)].detach().cpu().numpy()
    return torch.tensor(rot, dtype=coords_row.dtype), torch.tensor(origin, dtype=coords_row.dtype)


@lru_cache(maxsize=None)
def _base_local_template_aatype4(aatype4_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if int(aatype4_index) not in AATYPE4_TO_RESTYPE:
        raise ValueError(f"Unsupported aatype4 code `{aatype4_index}` for base template construction.")
    aatype9_code = int(aatype4_to_aatype9(torch.tensor([aatype4_index], dtype=torch.long))[0].item())
    rot = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
    trans = torch.zeros(1, 1, 3, dtype=torch.float32)
    # Torsions are stored as [sin, cos] pairs; zero angle is [0, 1], not [0, 0].
    torsions = torch.zeros(1, 1, 10, 2, dtype=torch.float32)
    torsions[..., 1] = 1.0
    torsions = torsions.reshape(1, 1, 20)
    aatype9 = torch.tensor([[aatype9_code]], dtype=torch.long)
    atom23 = all_atom_reconstruction.compute_backbone(
        bb_rigids=create_rigid(rot, trans),
        torsions=torsions,
        is_na_residue_mask=torch.ones(1, 1, dtype=torch.bool),
        aatype=aatype9,
    )[4][0]
    restype = AATYPE4_TO_RESTYPE[int(aatype4_index)]
    base_rot, base_trans = _build_base_plane_anchor_frame_from_atom23(atom23[0], restype)
    base_rigid = create_rigid(base_rot.view(1, 1, 3, 3), base_trans.view(1, 1, 3))[0]
    atom_indices = torch.tensor(
        [compact_atom_index(restype, name) for name in _base_atom_names(restype)],
        dtype=torch.long,
    )
    coords = atom23[0].index_select(0, atom_indices)
    local = base_rigid[0].invert_apply(coords.unsqueeze(0))[0]
    return atom_indices, local


@lru_cache(maxsize=None)
def _base_template_tables_cpu() -> tuple[torch.Tensor, torch.Tensor]:
    num_classes = len(AATYPE4_TO_RESTYPE)
    local_templates = torch.zeros(num_classes, 23, 3, dtype=torch.float32)
    atom_masks = torch.zeros(num_classes, 23, dtype=torch.bool)
    for class_idx in sorted(AATYPE4_TO_RESTYPE):
        atom_indices, local = _base_local_template_aatype4(int(class_idx))
        local_templates[int(class_idx), atom_indices] = local.to(dtype=torch.float32)
        atom_masks[int(class_idx), atom_indices] = True
    return local_templates, atom_masks


def _base_template_tables(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    local_templates, atom_masks = _base_template_tables_cpu()
    return (
        local_templates.to(device=device, dtype=dtype),
        atom_masks.to(device=device),
    )


def base_atom_class_mask(device: torch.device) -> torch.Tensor:
    """Return a [4, 23] mask for atoms produced by each base template."""
    _, atom_masks = _base_template_tables(device=device, dtype=torch.float32)
    return atom_masks


def _place_base_atoms_hard(base_trans: torch.Tensor, base_rots: torch.Tensor, aatype4: torch.Tensor) -> torch.Tensor:
    batch, num_res = aatype4.shape
    local_templates, atom_masks = _base_template_tables(base_trans.device, base_trans.dtype)
    flat_class = aatype4.long().reshape(batch * num_res)
    local = local_templates.index_select(0, flat_class).reshape(batch, num_res, 23, 3)
    mask = atom_masks.index_select(0, flat_class).reshape(batch, num_res, 23)
    placed = torch.einsum("bnij,bnaj->bnai", base_rots, local) + base_trans[..., None, :]
    return placed * mask[..., None].to(dtype=placed.dtype)


def reconstruct_soft_base_atoms(
    base_trans: torch.Tensor,
    base_rots: torch.Tensor,
    base_probs: torch.Tensor,
) -> torch.Tensor:
    batch, num_res, num_classes = base_probs.shape
    if num_classes != len(AATYPE4_TO_RESTYPE):
        raise ValueError(f"Expected {len(AATYPE4_TO_RESTYPE)} base classes, got {num_classes}.")

    local_templates, atom_masks = _base_template_tables(base_trans.device, base_trans.dtype)
    stacked = torch.einsum("bnij,caj->bncai", base_rots, local_templates)
    stacked = stacked + base_trans[:, :, None, None, :]
    stacked = stacked * atom_masks.view(1, 1, num_classes, 23, 1).to(dtype=stacked.dtype)
    probs = base_probs.unsqueeze(-1).unsqueeze(-1)
    return torch.sum(stacked * probs, dim=2)


def reconstruct_atom23(
    base_trans: torch.Tensor,
    base_rots: torch.Tensor,
    sugar_trans: torch.Tensor,
    sugar_rots: torch.Tensor,
    torsions: torch.Tensor,
    is_na_residue_mask: torch.Tensor,
    aatype4: torch.Tensor,
) -> torch.Tensor:
    """Decode atom23 from the Base-Plane and Sugar-GS frames.

    Backbone path: sugar_rigids is treated as the O4'-C4'-C3' Gram-Schmidt
    sugar frame and fed to compute_backbone, which
    uses the canonical default frame table (DEFAULT_NA_RESIDUE_FRAMES) to grow
    non-anchor atoms via torsions. Base atoms (idx >= 11, except O2') are
    placed via the base frame and the aatype4 base template.
    """
    sugar_rigids = create_rigid(sugar_rots, sugar_trans)
    aatype9 = aatype4_to_aatype9(aatype4)
    sugar_backbone = all_atom_reconstruction.compute_backbone(
        bb_rigids=sugar_rigids,
        torsions=torsions,
        is_na_residue_mask=is_na_residue_mask.bool(),
        aatype=aatype9.long(),
    )[4]
    out = torch.zeros_like(sugar_backbone)
    keep_indices = torch.tensor(list(range(0, 11)) + [12], dtype=torch.long, device=sugar_backbone.device)
    out.index_copy_(-2, keep_indices, sugar_backbone.index_select(-2, keep_indices))
    out = out + _place_base_atoms_hard(base_trans, base_rots, aatype4)
    return out


def reconstruct_atom23_trajectory(
    base_traj: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    sugar_traj: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    torsions: torch.Tensor,
    is_na_residue_mask: torch.Tensor,
    aatype4: torch.Tensor,
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for (base_trans, base_rots), (sugar_trans, sugar_rots) in zip(base_traj, sugar_traj):
        out.append(
            reconstruct_atom23(
                base_trans,
                base_rots,
                sugar_trans,
                sugar_rots,
                torsions,
                is_na_residue_mask,
                aatype4,
            ).detach().cpu()
        )
    return out


__all__ = [
    "base_atom_class_mask",
    "reconstruct_atom23",
    "reconstruct_atom23_trajectory",
    "reconstruct_soft_base_atoms",
]
