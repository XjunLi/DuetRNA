from __future__ import annotations

import math

import torch

from duetrna_shared_core.chemistry import AATYPE4_TO_RESTYPE, nc


def _base_atom_mask_lookup() -> torch.Tensor:
    lookup = torch.zeros(len(AATYPE4_TO_RESTYPE), 23, dtype=torch.bool)
    for aatype4, restype in AATYPE4_TO_RESTYPE.items():
        atom_names = nc.restype_name_to_compact_atom_names[restype]
        for atom_idx, atom_name in enumerate(atom_names[:23]):
            if atom_idx >= 11 and atom_name and atom_name != "O2'":
                lookup[int(aatype4), atom_idx] = True
    return lookup


BASE_ATOM_MASK_LOOKUP = _base_atom_mask_lookup()
MAX_BASE_ATOMS = int(BASE_ATOM_MASK_LOOKUP.sum(dim=-1).max().item())


def _base_atom_index_lookup() -> tuple[torch.Tensor, torch.Tensor]:
    index_lookup = torch.zeros(len(AATYPE4_TO_RESTYPE), MAX_BASE_ATOMS, dtype=torch.long)
    mask_lookup = torch.zeros(len(AATYPE4_TO_RESTYPE), MAX_BASE_ATOMS, dtype=torch.bool)
    for aatype4 in range(len(AATYPE4_TO_RESTYPE)):
        atom_indices = torch.nonzero(BASE_ATOM_MASK_LOOKUP[aatype4], as_tuple=False).flatten()
        if atom_indices.numel() == 0:
            continue
        index_lookup[aatype4, : atom_indices.numel()] = atom_indices
        mask_lookup[aatype4, : atom_indices.numel()] = True
    return index_lookup, mask_lookup


BASE_ATOM_INDEX_LOOKUP, BASE_ATOM_SLOT_MASK_LOOKUP = _base_atom_index_lookup()


def pairwise_residue_mask(res_mask: torch.Tensor) -> torch.Tensor:
    pair_mask = res_mask[:, :, None] * res_mask[:, None, :]
    diag = torch.eye(res_mask.shape[1], device=res_mask.device, dtype=torch.bool).unsqueeze(0)
    return (pair_mask > 0.5) & (~diag)


def base_origin_distance_map(base_trans: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(base_trans[:, :, None, :] - base_trans[:, None, :, :], dim=-1)


def base_normal_alignment_map(base_rotmats: torch.Tensor) -> torch.Tensor:
    normals = base_rotmats[..., 2]
    return torch.einsum("b i c, b j c -> b i j", normals, normals)


def parallel_separation_map(base_trans: torch.Tensor, base_rotmats: torch.Tensor) -> torch.Tensor:
    normals = base_rotmats[..., 2]
    offsets = base_trans[:, None, :, :] - base_trans[:, :, None, :]
    proj_i = torch.abs(torch.sum(offsets * normals[:, :, None, :], dim=-1))
    proj_j = torch.abs(torch.sum(offsets * normals[:, None, :, :], dim=-1))
    return 0.5 * (proj_i + proj_j)


def base_lateral_offset_map(
    base_trans: torch.Tensor,
    base_rotmats: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    center_dist = base_origin_distance_map(base_trans)
    plane_sep = parallel_separation_map(base_trans, base_rotmats)
    lateral_sq = torch.clamp(center_dist**2 - plane_sep**2, min=0.0)
    # Clamp away from zero so the backward pass does not hit the singular
    # derivative of sqrt at exactly 0 for near-coplanar residue pairs.
    return torch.sqrt(torch.clamp(lateral_sq, min=eps))


def base_atom_mask(aatype4: torch.Tensor, atom23_mask: torch.Tensor | None = None) -> torch.Tensor:
    lookup = BASE_ATOM_MASK_LOOKUP.to(device=aatype4.device)
    mask = lookup.index_select(0, aatype4.reshape(-1).long()).reshape(*aatype4.shape, 23)
    if atom23_mask is not None:
        mask = mask & (atom23_mask > 0.5)
    return mask


def min_base_atom_distance_map(
    atom23: torch.Tensor,
    aatype4: torch.Tensor,
    atom23_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_res, _, _ = atom23.shape
    index_lookup = BASE_ATOM_INDEX_LOOKUP.to(device=aatype4.device)
    slot_mask_lookup = BASE_ATOM_SLOT_MASK_LOOKUP.to(device=aatype4.device)

    gather_indices = index_lookup.index_select(0, aatype4.reshape(-1).long()).reshape(batch_size, num_res, MAX_BASE_ATOMS)
    slot_mask = slot_mask_lookup.index_select(0, aatype4.reshape(-1).long()).reshape(batch_size, num_res, MAX_BASE_ATOMS)
    packed_coords = torch.gather(
        atom23,
        dim=2,
        index=gather_indices[..., None].expand(-1, -1, -1, 3),
    )
    packed_mask = slot_mask & torch.gather(
        atom23_mask > 0.5,
        dim=2,
        index=gather_indices,
    )

    min_dists = []
    for batch_idx in range(batch_size):
        coords_b = packed_coords[batch_idx]
        mask_b = packed_mask[batch_idx]
        dists = torch.linalg.norm(
            coords_b[:, None, :, None, :] - coords_b[None, :, None, :, :],
            dim=-1,
        )
        valid = mask_b[:, None, :, None] & mask_b[None, :, None, :]
        dists = torch.where(valid, dists, torch.full_like(dists, float("inf")))
        min_dist = dists.amin(dim=(-1, -2))
        pair_has_atoms = (mask_b.any(dim=-1)[:, None] & mask_b.any(dim=-1)[None, :])
        min_dist = torch.where(pair_has_atoms, min_dist, torch.zeros_like(min_dist))
        min_dists.append(min_dist)
    return torch.stack(min_dists, dim=0)


def _sigmoid_score(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def pair_affinity_from_components(
    center_dist: torch.Tensor,
    normal_align: torch.Tensor,
    plane_sep: torch.Tensor,
    lateral_offset: torch.Tensor,
) -> torch.Tensor:
    near_contact = _sigmoid_score((8.0 - center_dist) / 1.0)
    coplanar = torch.exp(-(plane_sep**2) / (2.0 * (0.8**2)))
    aligned = torch.exp(-((1.0 - normal_align) ** 2) / (2.0 * (0.20**2)))
    lateral_band = torch.exp(-((lateral_offset - 4.5) ** 2) / (2.0 * (1.5**2)))
    return (
        near_contact
        * coplanar
        * aligned
        * lateral_band
    )


def stack_affinity_from_components(
    center_dist: torch.Tensor,
    normal_align: torch.Tensor,
    plane_sep: torch.Tensor,
    lateral_offset: torch.Tensor,
) -> torch.Tensor:
    near_contact = _sigmoid_score((8.5 - center_dist) / 1.0)
    aligned = torch.exp(-((1.0 - normal_align) ** 2) / (2.0 * (0.12**2)))
    stack_sep = torch.exp(-((plane_sep - 3.4) ** 2) / (2.0 * (0.7**2)))
    lateral_small = torch.exp(-(lateral_offset**2) / (2.0 * (2.0**2)))
    return near_contact * aligned * stack_sep * lateral_small


def pair_affinity_map(
    atom23: torch.Tensor,
    atom23_mask: torch.Tensor,
    aatype4: torch.Tensor,
    base_trans: torch.Tensor,
    base_rotmats: torch.Tensor,
) -> torch.Tensor:
    del atom23, atom23_mask, aatype4
    center_dist = base_origin_distance_map(base_trans)
    normal_align = torch.abs(base_normal_alignment_map(base_rotmats))
    plane_sep = parallel_separation_map(base_trans, base_rotmats)
    lateral_offset = base_lateral_offset_map(base_trans, base_rotmats)
    return pair_affinity_from_components(center_dist, normal_align, plane_sep, lateral_offset)


def stack_affinity_map(
    atom23: torch.Tensor,
    atom23_mask: torch.Tensor,
    aatype4: torch.Tensor,
    base_trans: torch.Tensor,
    base_rotmats: torch.Tensor,
) -> torch.Tensor:
    del atom23, atom23_mask, aatype4
    center_dist = base_origin_distance_map(base_trans)
    normal_align = torch.abs(base_normal_alignment_map(base_rotmats))
    plane_sep = parallel_separation_map(base_trans, base_rotmats)
    lateral_offset = base_lateral_offset_map(base_trans, base_rotmats)
    return stack_affinity_from_components(center_dist, normal_align, plane_sep, lateral_offset)


def soft_sequence_pair_compatibility(base_probs: torch.Tensor) -> torch.Tensor:
    p_a = base_probs[..., 0]
    p_u = base_probs[..., 1]
    p_g = base_probs[..., 2]
    p_c = base_probs[..., 3]
    compat = (
        p_a[:, :, None] * p_u[:, None, :]
        + p_u[:, :, None] * p_a[:, None, :]
        + p_g[:, :, None] * p_c[:, None, :]
        + p_c[:, :, None] * p_g[:, None, :]
        + 0.5 * (p_g[:, :, None] * p_u[:, None, :] + p_u[:, :, None] * p_g[:, None, :])
    )
    return compat


def inter_residue_clash_loss(
    atom23: torch.Tensor,
    atom23_mask: torch.Tensor,
    res_mask: torch.Tensor,
    o3_index: int,
    p_index: int,
    clash_distance: float = 1.6,
    normalization: str = "pairs",
    exclude_neighbor_residues: int = 0,
) -> torch.Tensor:
    """Penalize inter-residue atom clashes.

    Args:
        normalization: ``pairs`` preserves the historical behavior by averaging
            over all valid inter-residue pairs. ``residue`` divides the summed
            penalty by the number of valid residues, which gives sparse clashes
            a useful gradient signal for generation-quality experiments.
        exclude_neighbor_residues: Number of sequence-neighbor residues to
            exclude from clash accounting. Use ``1`` for nonlocal steric checks
            that ignore adjacent backbone contacts.
    """
    batch_size, num_res, num_atoms, _ = atom23.shape
    normalization_key = str(normalization).lower()
    if normalization_key not in {"pairs", "residue"}:
        raise ValueError(f"Unsupported clash normalization `{normalization}`.")
    exclude_neighbor_residues = max(0, int(exclude_neighbor_residues))
    losses = []
    for batch_idx in range(batch_size):
        coords = atom23[batch_idx]
        atom_mask_b = atom23_mask[batch_idx] > 0.5
        res_mask_b = res_mask[batch_idx] > 0.5
        flat_coords = coords.reshape(num_res * num_atoms, 3)
        flat_mask = atom_mask_b.reshape(num_res * num_atoms)
        flat_res_idx = torch.arange(num_res, device=coords.device).repeat_interleave(num_atoms)
        flat_atom_idx = torch.arange(num_atoms, device=coords.device).repeat(num_res)
        valid_res = res_mask_b.index_select(0, flat_res_idx)
        valid_atoms = flat_mask & valid_res

        dists = torch.cdist(flat_coords.unsqueeze(0), flat_coords.unsqueeze(0))[0]
        valid_pairs = valid_atoms[:, None] & valid_atoms[None, :]
        valid_pairs &= flat_res_idx[:, None] < flat_res_idx[None, :]
        valid_pairs &= flat_res_idx[:, None] != flat_res_idx[None, :]
        if exclude_neighbor_residues > 0:
            valid_pairs &= torch.abs(flat_res_idx[:, None] - flat_res_idx[None, :]) > exclude_neighbor_residues

        adjacent_o3_p = (
            (flat_res_idx[:, None] + 1 == flat_res_idx[None, :])
            & (flat_atom_idx[:, None] == o3_index)
            & (flat_atom_idx[None, :] == p_index)
        )
        valid_pairs &= ~adjacent_o3_p

        valid_dists = dists[valid_pairs]
        if valid_dists.numel() == 0:
            losses.append(coords.new_tensor(0.0))
            continue
        penalties = torch.relu(valid_dists.new_tensor(clash_distance) - valid_dists) ** 2
        if normalization_key == "pairs":
            losses.append(penalties.mean())
        else:
            denom = res_mask_b.float().sum().clamp(min=1.0)
            losses.append(penalties.sum() / denom)
    return torch.stack(losses, dim=0)


def chi_angle_loss(
    pred_chi: torch.Tensor,
    gt_chi: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(
        torch.linalg.norm(pred_chi - gt_chi, dim=-1) ** 2 * loss_mask,
        dim=-1,
    ) / torch.sum(loss_mask, dim=-1).clamp(min=1.0)


def pairwise_mse(
    pred_map: torch.Tensor,
    gt_map: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    pair_mask = pair_mask.float()
    return torch.sum((pred_map - gt_map) ** 2 * pair_mask, dim=(-1, -2)) / pair_mask.sum(dim=(-1, -2)).clamp(min=1.0)


def pairwise_weighted_mse(
    pred_map: torch.Tensor,
    gt_map: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor,
) -> torch.Tensor:
    weights = pair_mask.float() * pair_weights
    return torch.sum((pred_map - gt_map) ** 2 * weights, dim=(-1, -2)) / weights.sum(dim=(-1, -2)).clamp(min=1.0)


__all__ = [
    "base_origin_distance_map",
    "base_normal_alignment_map",
    "parallel_separation_map",
    "base_lateral_offset_map",
    "base_atom_mask",
    "min_base_atom_distance_map",
    "pair_affinity_map",
    "pair_affinity_from_components",
    "stack_affinity_map",
    "stack_affinity_from_components",
    "soft_sequence_pair_compatibility",
    "inter_residue_clash_loss",
    "chi_angle_loss",
    "pairwise_mse",
    "pairwise_weighted_mse",
    "pairwise_residue_mask",
]
