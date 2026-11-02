"""RNA sequence/PDB I/O and dataset split manifests.

Pandas is imported only for type checking.
"""
from __future__ import annotations

import functools as fn
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from duetrna.chemistry.residues import AATYPE4_TO_RESTYPE
from duetrna.chemistry import nucleotide_constants as nc

if TYPE_CHECKING:
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


def sample_output_paths(
    output_dir: str, num_res: int, sample_id: int, *, overwrite: bool = False,
) -> dict[str, str]:
    """Resolve a sample's three outputs and reject accidental replacement."""
    prefix = Path(output_dir) / f"length_{num_res}" / f"sample_{sample_id}"
    paths = {
        "sample_path": str(prefix) + ".pdb",
        "traj_path": str(prefix) + "_traj.pdb",
        "sequence_path": str(prefix) + ".fasta",
    }
    existing = [path for path in paths.values() if os.path.lexists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            "Sample output already exists: " + ", ".join(existing)
            + ". Set inference.samples.overwrite=true to replace it."
        )
    return paths


PDB_CHAIN_ID = "A"


def _normalize_models(atom23: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(atom23)
    if arr.ndim == 3:
        return arr[None, ...], False
    if arr.ndim == 4:
        return arr, True
    raise ValueError(f"Expected atom tensor with ndim 3 or 4, got {arr.shape}")


def _resolved_path(output_filepath: str, is_traj: bool) -> str:
    out = output_filepath
    if not out.endswith(".pdb"):
        out = out + ("_traj.pdb" if is_traj else ".pdb")
    return str(Path(out))


def _residue_names_from_aatype4(aatype4: np.ndarray) -> list[str]:
    names = []
    for idx, code in enumerate(np.asarray(aatype4).tolist()):
        key = int(code)
        if key not in AATYPE4_TO_RESTYPE:
            raise ValueError(f"Unsupported aatype4 code `{code}` at residue {idx + 1} for DuetRNA export.")
        names.append(AATYPE4_TO_RESTYPE[key])
    return names


def write_atom23_to_pdb(atom23: np.ndarray, aatype4: np.ndarray, output_filepath: str) -> str:
    models, is_traj = _normalize_models(atom23)
    save_path = _resolved_path(output_filepath, is_traj=is_traj)
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    residue_names = _residue_names_from_aatype4(aatype4)

    with open(save_path, "w", encoding="utf-8") as handle:
        atom_index = 1
        for model_idx, model_pos in enumerate(models, start=1):
            handle.write(f"MODEL     {model_idx}\n")
            for res_idx, resname in enumerate(residue_names, start=1):
                atom_names = nc.restype_name_to_compact_atom_names[resname]
                for atom_name_idx, atom_name in enumerate(atom_names):
                    if not atom_name:
                        continue
                    pos = model_pos[res_idx - 1, atom_name_idx]
                    if float(np.linalg.norm(pos)) <= 1e-7:
                        continue
                    element = atom_name[0]
                    name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                    handle.write(
                        f"ATOM  {atom_index:5d} {name:<4} {resname:>3} {PDB_CHAIN_ID}{res_idx:4d}    "
                        f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                        f"{1.00:6.2f}{0.00:6.2f}          {element:>2}\n"
                    )
                    atom_index += 1
            handle.write(f"TER   {atom_index:5d}      {residue_names[-1]:>3} {PDB_CHAIN_ID}{len(residue_names):4d}\n")
            atom_index += 1
            handle.write("ENDMDL\n")
        handle.write("END\n")
    return save_path
