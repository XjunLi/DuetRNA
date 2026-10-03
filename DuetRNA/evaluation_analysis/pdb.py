"""Small dependency-light RNA PDB reader for evaluation post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


RNA_RESNAMES = {
    "A": "A",
    "U": "U",
    "G": "G",
    "C": "C",
    "DA": "A",
    "DT": "U",
    "DG": "G",
    "DC": "C",
    "ADE": "A",
    "URA": "U",
    "GUA": "G",
    "CYT": "C",
}


@dataclass(frozen=True)
class Residue:
    index: int
    chain_id: str
    resseq: int
    icode: str
    resname: str
    atoms: dict[str, np.ndarray]

    @property
    def uid(self) -> str:
        return f"{self.chain_id}:{self.resseq}{self.icode.strip()}"


def _parse_atom_line(line: str) -> tuple[str, str, str, int, str, np.ndarray] | None:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None
    atom_name = line[12:16].strip()
    altloc = line[16].strip()
    if altloc not in {"", "A", "1"}:
        return None
    resname = RNA_RESNAMES.get(line[17:20].strip())
    if resname is None:
        return None
    chain_id = line[21].strip() or "A"
    try:
        resseq = int(line[22:26])
        coordinate = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
    except ValueError:
        return None
    return atom_name, resname, chain_id, resseq, line[26].strip(), coordinate


def read_rna_pdb(path: str | Path, *, first_model_only: bool = True) -> list[Residue]:
    records: dict[tuple[str, int, str], dict[str, object]] = {}
    in_model = False
    seen_model = False
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                if first_model_only and seen_model:
                    break
                seen_model = True
                in_model = True
                continue
            if line.startswith("ENDMDL"):
                if first_model_only:
                    break
                in_model = False
                continue
            if first_model_only and seen_model and not in_model:
                continue
            parsed = _parse_atom_line(line)
            if parsed is None:
                continue
            atom_name, resname, chain_id, resseq, icode, coordinate = parsed
            key = (chain_id, resseq, icode)
            record = records.setdefault(
                key,
                {
                    "chain_id": chain_id,
                    "resseq": resseq,
                    "icode": icode,
                    "resname": resname,
                    "atoms": {},
                },
            )
            atoms = record["atoms"]
            assert isinstance(atoms, dict)
            atoms[atom_name] = coordinate

    residues: list[Residue] = []
    for index, key in enumerate(sorted(records, key=lambda item: (item[0], item[1], item[2]))):
        record = records[key]
        residues.append(
            Residue(
                index=index,
                chain_id=str(record["chain_id"]),
                resseq=int(record["resseq"]),
                icode=str(record["icode"]),
                resname=str(record["resname"]),
                atoms=dict(record["atoms"]),
            )
        )
    return residues
