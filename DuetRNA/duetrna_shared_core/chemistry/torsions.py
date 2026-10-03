from __future__ import annotations

import torch

from . import atom_torsion_core


def convert_na_aatype6_to_aatype9(aatype: torch.Tensor, deoxy_offset_mask: torch.Tensor) -> torch.Tensor:
    return atom_torsion_core.convert_na_aatype6_to_aatype9(
        aatype,
        deoxy_offset_mask=deoxy_offset_mask,
    )


def extract_torsion_angles_atom23(
    aatype: torch.Tensor,
    atom23_positions: torch.Tensor,
    atom23_mask: torch.Tensor,
    atom_deoxy: torch.Tensor,
) -> torch.Tensor:
    feats = {
        "aatype": aatype,
        "all_atom_positions": atom23_positions,
        "all_atom_mask": atom23_mask,
        "atom_deoxy": atom_deoxy,
    }
    feats = atom_torsion_core.make_atom23_masks(feats)
    atom_torsion_core.atom23_list_to_atom27_list(feats, ["all_atom_positions", "all_atom_mask"], inplace=True)
    feats = atom_torsion_core.atom27_to_torsion_angles()(feats)
    return feats["torsion_angles_sin_cos"]


__all__ = ["convert_na_aatype6_to_aatype9", "extract_torsion_angles_atom23"]
