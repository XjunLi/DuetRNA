"""Coordinate alignment and scGDT for the RNA benchmark."""

from collections import namedtuple
from typing import Optional

import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Float
from torch import Tensor


def exists(x: object) -> bool:
    return x is not None


def safe_divide(a: Tensor, b: Tensor, eps: float = 1e-10) -> Tensor:
    return a / (b + eps)


def masked_mean(x: Tensor, mask: Tensor, dim: int = None, keepdim: bool = False) -> Tensor:
    assert x.shape == mask.shape
    x_masked = x * mask
    num_valid = mask.sum(dim=dim, keepdim=keepdim)
    return safe_divide(x_masked.sum(dim=dim, keepdim=keepdim), num_valid)


def center_of_mass(x: Tensor, mask: Optional[Tensor] = None, keepdim: bool = True) -> Float[Tensor, "B 1 3"]:
    if not exists(mask):
        mask = torch.ones_like(x[..., 0], dtype=torch.bool)
    # ... compute center of mass over unmasked atoms in each sample
    x_masked = x * rearrange(mask, "B N -> B N 1")
    x_mean = reduce(x_masked, "B N C -> B 1 C", "sum") / reduce(mask, "B N -> B 1 1", "sum")
    return x_mean if keepdim else x_mean.squeeze(1)


def masked_cov(x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> Float[Tensor, "B 3 3"]:
    if not exists(mask):
        mask = torch.ones_like(x[..., 0], dtype=torch.bool)
    # Mask the input (invalid values are set to 0)
    x_masked = x * rearrange(mask, "B N -> B N 1")
    y_masked = y * rearrange(mask, "B N -> B N 1")
    cov = einsum(x_masked, y_masked, "B N i, B N j -> B i j")
    return cov


def kabsch(
    fixed: Tensor, mobile: Tensor, mask: Optional[Tensor] = None
) -> tuple[Float[Tensor, "B 3 3"], Float[Tensor, "B 3"]]:
    """
    Applies the Kabsch algorithm to align two batches (possibly with masks) of 3D point clouds.

    Args:
        fixed (Tensor): A tensor of shape (B, N, 3) representing the first set of points.
        mobile (Tensor): A tensor of shape (B, N, 3) representing the second set of points.
        mask (Tensor): A tensor of shape (B, N) representing the mask of valid points.

    Returns:
        A tuple containing two tensors:
        - rot_mats (Float[Tensor, "B 3 3"]): A tensor of shape (B, 3, 3) representing the rotation matrices that
            align `mobile` with `fixed`.
        - trans_vecs (Float[Tensor, "B 3"]): A tensor of shape (B, 3) representing the translation vectors that
            align `mobile` with `fixed`.
    """
    B, N, _ = fixed.shape  # [B, N, 3]
    device = fixed.device
    assert fixed.shape == mobile.shape
    if not exists(mask):
        mask = torch.ones_like(fixed[..., 0], dtype=torch.bool)
    assert torch.all(mask.sum(dim=1) >= 3), "At least 3 points are required to compute the Kabsch algorithm."

    # Compute centers of mass
    fixed_com = center_of_mass(fixed, mask)  # [B, 1, 3]
    mobile_com = center_of_mass(mobile, mask)  # [B, 1, 3]

    # Compute (spatial) covariance matrix
    cov = masked_cov(fixed - fixed_com, mobile - mobile_com, mask)  # [B, 3, 3]
    assert not torch.isnan(cov).any(), "Covariance matrix contains NaN values."

    # Compute SVD
    U, S, Vt = torch.linalg.svd(cov, full_matrices=True)  # [B, 3, 3], [B, 3], [B, 3, 3]

    # Planar/collinear inputs still have a least-squares rigid solution. Adding
    # random noise changes that solution and makes a coordinate metric stochastic.

    # Compute rotation matrices and translation vectors
    flip_mat = torch.eye(3, device=device, dtype=cov.dtype).repeat(B, 1, 1)  # [B, 3, 3]
    flip_mat[:, -1, -1] = torch.linalg.det(U @ Vt).sign()  # Proper rotations, including rank-deficient covariance
    rot_mats = U @ flip_mat @ Vt  # [B, 3, 3]
    trans_vecs = fixed_com.squeeze(1) - einsum(rot_mats, mobile_com.squeeze(1), "B i j, B j -> B i")  # [B, 3]

    return namedtuple("KabschResult", ["rot_mats", "trans_vecs"])(rot_mats, trans_vecs)


def rototranslate(
    p: Tensor,
    rot_mat: Float[Tensor, "*B 3 3"],
    trans_vec: Float[Tensor, "*B 3"],
    inverse: bool = False,
    validate: bool = False,
) -> torch.Tensor:
    """
    Apply batched rototranslation to a batch of 3D point clouds.
        p' = R @ p + t
    (First rotate, then translate.)

    Args:
        p (Tensor): A tensor of shape (B, N, 3) representing the batch of point clouds.
        rot_mat (Tensor): A tensor of shape (B, 3, 3) representing the rotation matrices.
        trans_vec (Tensor): A tensor of shape (B, 3) representing the translation vectors.
        validate (bool): Whether to validate the input.

    Returns:
        A tensor of shape (B, N, 3) representing the set of points after rototranslation.
    """
    if validate:
        p.device
        B, N, C = p.shape
        assert rot_mat.shape == (B, C, C)
        assert trans_vec.shape == (B, C)
    trans_vec = rearrange(trans_vec, "B i -> B 1 i")  # [B, 1, 3]
    if inverse:
        return einsum((p - trans_vec), rot_mat, "B N i, B i j -> B N j")
    return einsum(p, rot_mat, "B N i, B j i -> B N j") + trans_vec  # [B, N, 3]


def superimpose(fixed: Tensor, mobile: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Superimpose two sets of coordinates using the Kabsch algorithm.

    Args:
        fixed (Tensor): A tensor of shape (B, N, 3) representing the fixed set of points.
        mobile (Tensor): A tensor of shape (B, N, 3) representing the mobile set of points.
        mask (Tensor): A tensor of shape (B, N) representing the mask of valid points.

    Returns:
        A tensor of shape (B, N, 3) representing the mobile set of points after superimposition.
    """
    rots, trans = kabsch(fixed, mobile, mask)
    return rototranslate(mobile, rots, trans)


def gdt_score(
    x: Tensor,
    y: Tensor,
    mask: Optional[Tensor] = None,
    thresholds: list[float] = [1.0, 2.0, 4.0, 8.0],
    apply_superimpose: bool = True,
) -> Tensor:
    """
    Average distance-threshold coverage after one global least-squares fit.

    This retains RNA-FrameFlow's scGDT convention. A full GDT search would
    optimize separate subset alignments for each threshold.
    """
    # Ensure the mask exists; if not, create a mask of ones (include all residues)
    if mask is None:
        mask = torch.ones_like(x[..., 0], dtype=torch.bool)

    if apply_superimpose:
        y = superimpose(x, y, mask)

    # Calculate squared distances
    squared_distances = (x - y).pow(2).sum(dim=-1)  # Shape: [B, N]

    # Calculate scores for each threshold
    scores = []
    for threshold in thresholds:
        within_threshold = (squared_distances <= threshold**2).float()  # Shape: [B, N]
        score = masked_mean(within_threshold, mask, dim=-1)  # Average over residues
        scores.append(score)

    # Stack scores across thresholds and average them
    scores = torch.stack(scores, dim=-1)
    return scores.mean(dim=-1)  # Average over thresholds
