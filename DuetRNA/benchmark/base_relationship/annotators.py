from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pdb_parser import Residue


@dataclass(frozen=True)
class EdgeAnnotation:
    i: int
    j: int
    relation: str
    source: str = "annotation"

    @property
    def family(self) -> str:
        rel = self.relation.lower()
        if "stack" in rel or rel.startswith("s"):
            return "stack"
        if rel in {"pair", "base_pair"} or len(rel) == 3 and rel[1:].upper() in {"WW", "WH", "HW", "WS", "SW", "HH", "HS", "SH", "SS"}:
            return "pair"
        if "pair" in rel or rel.startswith(("c", "t")):
            return "pair"
        return rel

    @property
    def key(self) -> tuple[int, int, str]:
        a, b = sorted((self.i, self.j))
        return a, b, self.family


def _residue_lookup(residues: list[Residue]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for residue in residues:
        lookup[str(residue.index + 1)] = residue.index
        lookup[residue.uid] = residue.index
        lookup[f"{residue.chain_id}.{residue.resname}{residue.resseq}"] = residue.index
        lookup[f"{residue.chain_id}.{residue.resseq}"] = residue.index
        lookup[f"{residue.chain_id}:{residue.resname}{residue.resseq}"] = residue.index
        lookup[f"{residue.chain_id}:{residue.resseq}"] = residue.index
    return lookup


def _parse_nt_id(value: Any, lookup: dict[str, int]) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in lookup:
        return lookup[text]
    # FR3D/BGSU unit ids commonly look like "1S72|1|0|A|1193" or
    # "5NJT|1|A|G|11". Match by chain/residue number, then fall back to
    # residue number only for generated single-chain PDBs.
    if "|" in text:
        parts = text.split("|")
        if len(parts) >= 5:
            chain = parts[-3]
            resseq = parts[-1]
            for key in (f"{chain}:{resseq}", f"{chain}.{resseq}", resseq):
                if key in lookup:
                    return lookup[key]
    # DSSR often uses "A.G12" or "A.G12^M"; keep chain and numeric part.
    match = re.search(r"([A-Za-z0-9])\.?[A-Za-z]*(-?\d+)", text)
    if match:
        chain, resseq = match.group(1), match.group(2)
        for key in (f"{chain}:{resseq}", f"{chain}.{resseq}"):
            if key in lookup:
                return lookup[key]
    # Last resort: a bare integer residue index, 1-based by convention.
    match = re.search(r"-?\d+", text)
    if match and match.group(0) in lookup:
        return lookup[match.group(0)]
    return None


def _edge_from_mapping(row: dict[str, Any], lookup: dict[str, int], source: str) -> EdgeAnnotation | None:
    nt1 = (
        row.get("nt1")
        or row.get("nt1_id")
        or row.get("nt1_name")
        or row.get("i")
        or row.get("residue_i")
        or row.get("res1")
        or row.get("residue1")
        or row.get("unit_id1")
        or row.get("unit_id_1")
        or row.get("unit1")
    )
    nt2 = (
        row.get("nt2")
        or row.get("nt2_id")
        or row.get("nt2_name")
        or row.get("j")
        or row.get("residue_j")
        or row.get("res2")
        or row.get("residue2")
        or row.get("unit_id2")
        or row.get("unit_id_2")
        or row.get("unit2")
    )
    if (nt1 is None or nt2 is None) and row.get("nt_pair"):
        parts = re.split(r"[,;:\s]+", str(row["nt_pair"]).strip())
        if len(parts) >= 2:
            nt1, nt2 = parts[0], parts[1]
    i = _parse_nt_id(nt1, lookup)
    j = _parse_nt_id(nt2, lookup)
    if i is None or j is None or i == j:
        return None
    relation = (
        row.get("relation")
        or row.get("type")
        or row.get("LW")
        or row.get("lw")
        or row.get("lw_type")
        or row.get("DSSR")
        or row.get("basepair")
        or row.get("interaction")
        or row.get("name")
        or row.get("family")
        or "pair"
    )
    return EdgeAnnotation(i=i, j=j, relation=str(relation), source=source)


def _load_json_edges(path: Path, residues: list[Residue]) -> list[EdgeAnnotation]:
    lookup = _residue_lookup(residues)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(data, list):
        rows.extend(("json", row) for row in data if isinstance(row, dict))
    elif isinstance(data, dict):
        for key in ("pairs", "basePairs", "base_pairs"):
            for row in data.get(key, []) or []:
                if isinstance(row, dict):
                    row = dict(row)
                    row.setdefault("relation", row.get("LW", "pair"))
                    rows.append(("dssr_pair", row))
        for key in ("stacks", "stacking", "baseStacks"):
            for row in data.get(key, []) or []:
                if isinstance(row, dict):
                    row = dict(row)
                    row.setdefault("relation", "stack")
                    rows.append(("dssr_stack", row))
        for key in ("edges", "relations"):
            for row in data.get(key, []) or []:
                if isinstance(row, dict):
                    rows.append(("json", row))
    out = []
    for source, row in rows:
        edge = _edge_from_mapping(row, lookup, source)
        if edge is not None:
            out.append(edge)
    return out


def _load_jsonl_edges(path: Path, residues: list[Residue]) -> list[EdgeAnnotation]:
    lookup = _residue_lookup(residues)
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                edge = _edge_from_mapping(row, lookup, "jsonl")
                if edge is not None:
                    out.append(edge)
    return out


def _load_csv_edges(path: Path, residues: list[Residue]) -> list[EdgeAnnotation]:
    lookup = _residue_lookup(residues)
    out = []
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    known_headers = {
        "nt1",
        "nt2",
        "nt1_id",
        "nt2_id",
        "unit_id1",
        "unit_id2",
        "relation",
        "basepair",
        "interaction",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        if not first_line:
            return out
        handle.seek(0)
        first_fields = next(csv.reader([first_line], delimiter=delimiter), [])
        has_header = bool({field.strip().lower() for field in first_fields} & known_headers)
        if has_header:
            for row in csv.DictReader(handle, delimiter=delimiter):
                edge = _edge_from_mapping(row, lookup, "csv")
                if edge is not None:
                    out.append(edge)
        else:
            for fields in csv.reader(handle, delimiter=delimiter):
                if len(fields) < 3:
                    continue
                row = {"unit_id1": fields[0], "relation": fields[1], "unit_id2": fields[2]}
                edge = _edge_from_mapping(row, lookup, "fr3d")
                if edge is not None:
                    out.append(edge)
    return out


def load_annotations(path: str | Path | None, residues: list[Residue]) -> list[EdgeAnnotation]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        return _load_json_edges(path, residues)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return _load_jsonl_edges(path, residues)
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        return _load_csv_edges(path, residues)
    return []


def infer_annotation_path(pdb_path: str | Path, annotation_dir: str | Path | None) -> Path | None:
    if annotation_dir is None:
        return None
    annotation_dir = Path(annotation_dir)
    stem = Path(pdb_path).stem
    for suffix in (".json", ".jsonl", ".csv", ".tsv", ".txt"):
        for name in (
            f"{stem}{suffix}",
            f"{stem}_annotations{suffix}",
            f"{stem}_fr3d{suffix}",
            f"{stem}_basepairs{suffix}",
            f"{stem}_basepair{suffix}",
        ):
            candidate = annotation_dir / name
            if candidate.exists():
                return candidate
    return None
