from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from duetrna_shared_core.chemistry import AATYPE4_TO_RESTYPE, nc


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


__all__ = ["write_atom23_to_pdb"]
