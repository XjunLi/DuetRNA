from __future__ import annotations

from collections.abc import Mapping

import torch

from duetrna.chemistry import torsions as atom_torsion_core
from duetrna.chemistry.torsions import convert_na_aatype6_to_aatype9
from duetrna.chemistry import nucleotide_constants as nc, vocabulary
from duetrna.geometry.rigid import Rigid
from duetrna.geometry import all_atom as all_atom_reconstruction
from duetrna.geometry.frames import build_dual_frames


DEFAULT_DUAL_FRAME_CONFIG = {
    "mode": "sugar_gs",
    "base_origin_mode": "connection_atom",
    "sugar_origin_mode": "c4",
}


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_dual_frame_cfg(frame_cfg):
    cfg = dict(DEFAULT_DUAL_FRAME_CONFIG)
    if frame_cfg is None:
        return cfg
    requested_mode = str(_cfg_get(frame_cfg, "mode", cfg["mode"])).lower()
    if requested_mode != "sugar_gs":
        raise ValueError(
            f"DuetRNA only supports `sugar_gs`, got `{requested_mode}`."
        )
    cfg["mode"] = "sugar_gs"
    cfg["base_origin_mode"] = str(
        _cfg_get(frame_cfg, "base_origin_mode", cfg["base_origin_mode"])
    ).lower()
    cfg["sugar_origin_mode"] = str(
        _cfg_get(frame_cfg, "sugar_origin_mode", cfg["sugar_origin_mode"])
    ).lower()
    return cfg


def _resolve_aatype9(aatype: torch.Tensor, atom_deoxy: torch.Tensor) -> torch.Tensor:
    aatype = aatype.long().clone()
    if aatype.numel() == 0:
        return aatype
    if aatype.min() > vocabulary.protein_restype_num:
        shifted = aatype - (vocabulary.protein_restype_num + 1)
        if shifted.max() <= nc.NA_AATYPE9_MASK_RESIDUE_INDEX:
            return shifted
    if aatype.max() <= nc.NA_AATYPE9_MASK_RESIDUE_INDEX:
        return aatype
    return convert_na_aatype6_to_aatype9(aatype, deoxy_offset_mask=atom_deoxy.bool())


def atom27_to_dual_frames(na, frame_cfg=None):
    cfg = _resolve_dual_frame_cfg(frame_cfg)
    dual = build_dual_frames(
        na,
        base_origin_mode=cfg["base_origin_mode"],
        sugar_origin_mode=cfg["sugar_origin_mode"],
    )
    base_rigid, base_exists = dual["base_frame"]
    sugar_rigid, sugar_exists = dual["sugar_frame"]

    rel_exists = ((base_exists > 0.5) & (sugar_exists > 0.5)).to(base_exists.dtype)
    sugar_rel = base_rigid.invert().compose(sugar_rigid)

    na["base_rigidgroups_gt_frames"] = base_rigid.to_tensor_4x4()
    na["base_rigidgroups_gt_exists"] = base_exists
    na["sugar_rigidgroups_gt_frames"] = sugar_rigid.to_tensor_4x4()
    na["sugar_rigidgroups_gt_exists"] = sugar_exists
    na["base_to_sugar_gt_frames"] = sugar_rel.to_tensor_4x4()
    na["base_to_sugar_gt_exists"] = rel_exists
    na["dual_frame_cfg"] = cfg
    return na


def prepare_oracle_dual_frame_batch(na, frame_cfg=None):
    feats = {k: v.clone() if torch.is_tensor(v) else v for k, v in na.items()}
    feats = atom_torsion_core.make_atom23_masks(feats)
    atom_torsion_core.atom23_list_to_atom27_list(
        feats,
        ["all_atom_positions", "all_atom_mask"],
        inplace=True,
    )
    feats = atom27_to_dual_frames(feats, frame_cfg=frame_cfg)
    feats = atom_torsion_core.atom27_to_torsion_angles()(feats)
    return feats


def reconstruct_atom37_from_sugar_frame(na):
    sugar_rigid = Rigid.from_tensor_4x4(na["sugar_rigidgroups_gt_frames"])
    aatype9 = _resolve_aatype9(na["aatype"], na["atom_deoxy"])
    torsions = na["torsion_angles_sin_cos"][:, :10, :].reshape(1, -1, 20)
    is_na_residue_mask = torch.ones(
        (1, sugar_rigid.get_trans().shape[0]),
        dtype=torch.bool,
        device=sugar_rigid.get_trans().device,
    )
    return all_atom_reconstruction.to_atom37_rna(
        sugar_rigid.get_trans().unsqueeze(0),
        sugar_rigid.get_rots().get_rot_mats().unsqueeze(0),
        is_na_residue_mask=is_na_residue_mask,
        torsions=torsions,
        aatype=aatype9.unsqueeze(0),
    )
