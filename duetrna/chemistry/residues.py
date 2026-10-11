"""RNA residue encodings and compact atom23 indexing.

Keep the model's A/U/G/C classes distinct from the parser and atom-table encodings.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from . import nucleotide_constants as nc

if TYPE_CHECKING:
    import torch

RNA_BASE_TOKENS = ("A", "U", "G", "C")
RESTYPE_TO_AATYPE4 = {
    "A": 0,
    "DA": 0,
    "U": 1,
    "DT": 1,
    "G": 2,
    "DG": 2,
    "C": 3,
    "DC": 3,
}
AATYPE4_TO_RESTYPE = {
    0: "A",
    1: "U",
    2: "G",
    3: "C",
}
AATYPE4_TO_AATYPE9 = {
    0: 4,  # A
    1: 7,  # U
    2: 6,  # G
    3: 5,  # C
}
AATYPE9_TO_AATYPE4 = {
    0: 0,  # DA -> A
    1: 3,  # DC -> C
    2: 2,  # DG -> G
    3: 1,  # DT -> U
    4: 0,  # A
    5: 3,  # C
    6: 2,  # G
    7: 1,  # U
}


def aatype9_to_aatype4(aatype9: torch.Tensor) -> torch.Tensor:
    import torch

    out = torch.full_like(aatype9, fill_value=-1)
    for src, dst in AATYPE9_TO_AATYPE4.items():
        out = torch.where(aatype9 == int(src), torch.full_like(out, int(dst)), out)
    if (out < 0).any():
        bad = torch.unique(aatype9[out < 0]).detach().cpu().tolist()
        raise ValueError(f"Unsupported aatype9 codes for 4-class RNA mapping: {bad}")
    return out.long()


def aatype4_to_aatype9(aatype4: torch.Tensor) -> torch.Tensor:
    import torch

    out = torch.full_like(aatype4, fill_value=-1)
    for src, dst in AATYPE4_TO_AATYPE9.items():
        out = torch.where(aatype4 == int(src), torch.full_like(out, int(dst)), out)
    if (out < 0).any():
        bad = torch.unique(aatype4[out < 0]).detach().cpu().tolist()
        raise ValueError(f"Unsupported aatype4 codes for RNA mapping: {bad}")
    return out.long()


def aatype4_to_sequence(aatype4: torch.Tensor) -> str:
    seq = []
    for idx, code in enumerate(aatype4.detach().cpu().tolist()):
        if int(code) not in AATYPE4_TO_RESTYPE:
            raise ValueError(f"Unsupported aatype4 code `{code}` at residue {idx + 1}.")
        seq.append(AATYPE4_TO_RESTYPE[int(code)])
    return "".join(seq)


@lru_cache(maxsize=None)
def compact_atom_order(restype: str) -> dict[str, int]:
    return {
        atom_name: idx
        for idx, atom_name in enumerate(nc.restype_name_to_compact_atom_names[restype])
        if atom_name
    }


def compact_atom_index(restype: str, atom_name: str) -> int:
    order = compact_atom_order(restype)
    if atom_name not in order:
        raise KeyError(f"Atom `{atom_name}` not present in compact atom23 layout for restype `{restype}`.")
    return order[atom_name]


COMMON_ATOM_ORDER = compact_atom_order("A")
