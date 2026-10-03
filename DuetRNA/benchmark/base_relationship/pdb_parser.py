from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
        suffix = self.icode.strip()
        return f"{self.chain_id}:{self.resseq}{suffix}"


def _parse_atom_line(line: str) -> tuple[str, str, str, int, str, np.ndarray] | None:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None
    atom_name = line[12:16].strip()
    altloc = line[16].strip()
    if altloc not in {"", "A", "1"}:
        return None
    resname_raw = line[17:20].strip()
    resname = RNA_RESNAMES.get(resname_raw)
    if resname is None:
        return None
    chain_id = line[21].strip() or "A"
    try:
        resseq = int(line[22:26])
        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
    except ValueError:
        return None
    icode = line[26].strip()
    return atom_name, resname, chain_id, resseq, icode, xyz


def read_rna_pdb(path: str | Path, *, first_model_only: bool = True) -> list[Residue]:
    """Read RNA residues from a PDB file.

    The parser is intentionally lightweight and only keeps RNA atom records. It
    supports the single-chain files produced by the current DuetRNA benchmark path,
    but also retains chain/residue identifiers for DSSR/FR3D matching.
    """
    path = Path(path)
    residues: dict[tuple[str, int, str], dict[str, object]] = {}
    in_model = False
    seen_model = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                if first_model_only and seen_model:
                    break
                in_model = True
                seen_model = True
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
            atom_name, resname, chain_id, resseq, icode, xyz = parsed
            key = (chain_id, resseq, icode)
            rec = residues.setdefault(
                key,
                {
                    "chain_id": chain_id,
                    "resseq": resseq,
                    "icode": icode,
                    "resname": resname,
                    "atoms": {},
                },
            )
            rec["atoms"][atom_name] = xyz

    out = []
    for idx, key in enumerate(sorted(residues, key=lambda x: (x[0], x[1], x[2]))):
        rec = residues[key]
        out.append(
            Residue(
                index=idx,
                chain_id=str(rec["chain_id"]),
                resseq=int(rec["resseq"]),
                icode=str(rec["icode"]),
                resname=str(rec["resname"]),
                atoms=dict(rec["atoms"]),
            )
        )
    return out


def iter_pdb_files(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.pdb")))
        elif path.suffix.lower() == ".pdb":
            out.append(path)
    return out
