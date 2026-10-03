from functools import lru_cache

from . import nucleotide_constants as nc
from . import vocabulary


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


__all__ = ["COMMON_ATOM_ORDER", "compact_atom_index", "compact_atom_order", "nc", "vocabulary"]
