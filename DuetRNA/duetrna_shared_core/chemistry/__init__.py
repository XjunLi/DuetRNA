from .base_types import (
    AATYPE4_TO_AATYPE9,
    AATYPE4_TO_RESTYPE,
    AATYPE9_TO_AATYPE4,
    RESTYPE_TO_AATYPE4,
    RNA_BASE_TOKENS,
    aatype4_to_aatype9,
    aatype4_to_sequence,
    aatype9_to_aatype4,
)
from .rna_constants import COMMON_ATOM_ORDER, compact_atom_index, compact_atom_order, nc, vocabulary


def convert_na_aatype6_to_aatype9(*args, **kwargs):
    from .torsions import convert_na_aatype6_to_aatype9 as _impl

    return _impl(*args, **kwargs)


def extract_torsion_angles_atom23(*args, **kwargs):
    from .torsions import extract_torsion_angles_atom23 as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "AATYPE4_TO_AATYPE9",
    "AATYPE4_TO_RESTYPE",
    "AATYPE9_TO_AATYPE4",
    "COMMON_ATOM_ORDER",
    "RESTYPE_TO_AATYPE4",
    "RNA_BASE_TOKENS",
    "aatype4_to_aatype9",
    "aatype4_to_sequence",
    "aatype9_to_aatype4",
    "compact_atom_index",
    "compact_atom_order",
    "nc",
    "vocabulary",
    "convert_na_aatype6_to_aatype9",
    "extract_torsion_angles_atom23",
]
