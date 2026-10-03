from __future__ import annotations

import functools as fn
from pathlib import Path
from typing import Iterable

import pandas as pd


def _resolve_split_dir(split_dir: str | Path | None) -> Path | None:
    if split_dir in (None, "", "null"):
        return None
    return Path(split_dir).expanduser().resolve()


def _split_file(split_dir: str | Path, split_name: str) -> Path:
    resolved = _resolve_split_dir(split_dir)
    if resolved is None:
        raise ValueError("split_dir must be set to resolve a split manifest.")
    return resolved / f"{split_name}.txt"


@fn.lru_cache(maxsize=32)
def load_split_index(split_dir: str | Path, split_name: str) -> frozenset[str]:
    split_file = _split_file(split_dir, split_name)
    if not split_file.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_file}")
    with split_file.open("r", encoding="utf-8") as handle:
        paths = [line.strip() for line in handle if line.strip()]
    return frozenset(paths)


@fn.lru_cache(maxsize=8)
def load_split_assignments(split_dir: str | Path) -> pd.DataFrame:
    resolved = _resolve_split_dir(split_dir)
    if resolved is None:
        raise ValueError("split_dir must be set to load split assignments.")
    assignments_path = resolved / "split_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Split assignments not found: {assignments_path}")
    return pd.read_csv(assignments_path)


def filter_metadata_by_split(
    pdb_csv: pd.DataFrame,
    split_dir: str | Path | None,
    split_name: str | None,
    key: str = "processed_path",
) -> pd.DataFrame:
    if split_dir in (None, "", "null") or split_name in (None, "", "null"):
        return pdb_csv
    allowed = load_split_index(split_dir, str(split_name))
    return pdb_csv[pdb_csv[key].isin(allowed)].copy()


def split_overlap(a: Iterable[str], b: Iterable[str]) -> list[str]:
    return sorted(set(a).intersection(set(b)))


__all__ = [
    "filter_metadata_by_split",
    "load_split_assignments",
    "load_split_index",
    "split_overlap",
]
