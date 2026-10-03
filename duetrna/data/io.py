"""RNA sequence/PDB I/O and dataset split manifests.

Torch and pandas are loaded only by the functions that need them.
"""
from __future__ import annotations

import functools as fn
import os
from pathlib import Path
from typing import Iterable, List, Tuple, TYPE_CHECKING

import numpy as np
from duetrna.chemistry.residues import AATYPE4_TO_RESTYPE
from duetrna.chemistry import nucleotide_constants as nc

if TYPE_CHECKING:
    import pandas as pd
    import torch

RNA_CHAR_TO_AATYPE9 = {
    "A": 4,
    "C": 5,
    "G": 6,
    "U": 7,
    "T": 7,
}


def parse_fasta_records(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    current_name = None
    current_seq: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    records.append((current_name, "".join(current_seq)))
                current_name = line[1:].strip() or f"seq_{len(records)}"
                current_seq = []
            else:
                current_seq.append(line.replace(" ", "").upper())
    if current_name is not None:
        records.append((current_name, "".join(current_seq)))
    return records


def encode_rna_sequence(sequence: str) -> torch.Tensor:
    import torch

    seq = sequence.strip().upper().replace(" ", "")
    if not seq:
        raise ValueError("Empty RNA sequence is not allowed.")
    encoded = []
    for idx, ch in enumerate(seq):
        if ch not in RNA_CHAR_TO_AATYPE9:
            raise ValueError(
                f"Unsupported nucleotide `{ch}` at position {idx + 1} in sequence `{sequence}`. "
                "Only A/C/G/U/T are supported."
            )
        encoded.append(RNA_CHAR_TO_AATYPE9[ch])
    return torch.tensor(encoded, dtype=torch.long)


def load_sequence_records(fasta_path: str | None, inline_sequence: str | None, inline_name: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    if fasta_path:
        fasta = Path(fasta_path)
        if not fasta.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        records.extend(parse_fasta_records(str(fasta)))
    if inline_sequence:
        records.append((inline_name, str(inline_sequence)))
    if not records:
        raise ValueError("Expected `fasta_path` or `inline_sequence`.")
    return records


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
    import pandas as pd

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
