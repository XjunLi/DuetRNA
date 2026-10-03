from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from .annotators import infer_annotation_path, load_annotations
from .features import RelationRecord, geometry_fallback_edges, records_from_jsonable, records_to_jsonable
from .features import relation_records_from_edges
from .frames import build_residue_frames
from .metrics import (
    base_pair_counts,
    bpgv_metrics,
    relation_compactness,
    relation_self_consistency,
    relation_separability,
    sequence_base_compatibility,
    stack_geometry_metrics,
    stack_reference,
    fit_relation_gaussians,
)
from .pdb_parser import iter_pdb_files, read_rna_pdb


def collect_records(
    pdb_paths: list[str | Path],
    *,
    annotation_dir: str | Path | None = None,
    geometry_fallback: bool = True,
    require_annotations: bool = False,
    skip_zero_edge_annotations: bool = False,
) -> list[RelationRecord]:
    """Collect relation records from PDB files and optional DSSR/FR3D-like annotations.

    If no annotation file is found for a PDB and ``geometry_fallback`` is true,
    approximate pair/stack candidates are inferred from base-plane geometry.
    Paper-facing runs should pass ``require_annotations=True`` and
    ``geometry_fallback=False`` so missing FR3D/DSSR annotations fail loudly.
    """
    records: list[RelationRecord] = []
    for pdb_path in iter_pdb_files(pdb_paths):
        residues = read_rna_pdb(pdb_path)
        if not residues:
            continue
        frames = build_residue_frames(residues)
        annotation_path = infer_annotation_path(pdb_path, annotation_dir)
        if require_annotations and annotation_path is None:
            raise FileNotFoundError(f"Missing annotation for {pdb_path} under {annotation_dir}")
        edges = load_annotations(annotation_path, residues)
        if require_annotations and annotation_path is not None and annotation_path.stat().st_size > 0 and not edges:
            if skip_zero_edge_annotations:
                continue
            raise RuntimeError(f"Annotation file parsed to zero edges for {pdb_path}: {annotation_path}")
        if not edges and geometry_fallback:
            edges = geometry_fallback_edges(frames)
        records.extend(relation_records_from_edges(Path(pdb_path).stem, frames, edges))
    return records


def save_records_jsonl(records: list[RelationRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records_to_jsonable(records):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def load_records_jsonl(path: str | Path) -> list[RelationRecord]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return records_from_jsonable(rows)


def build_reference(records: list[RelationRecord]) -> dict[str, Any]:
    return {
        "version": 1,
        "n_records": len(records),
        "base_gaussians": fit_relation_gaussians(records, "base", label_key="relation"),
        "sugar_gaussians": fit_relation_gaussians(records, "sugar", label_key="relation"),
        "base_family_gaussians": fit_relation_gaussians(records, "base", label_key="family"),
        "sugar_family_gaussians": fit_relation_gaussians(records, "sugar", label_key="family"),
        "stack": stack_reference(records),
        "base_pair_counts": base_pair_counts(records),
        "representation": representation_metrics(records),
    }


def save_reference(reference: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(reference, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_reference(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def representation_metrics(records: list[RelationRecord]) -> dict[str, float]:
    out: dict[str, float] = {}
    for frame in ("base", "sugar"):
        out.update(relation_compactness(records, frame))
        out.update(relation_separability(records, frame))
    return out


def evaluate_records(
    records: list[RelationRecord],
    reference: dict[str, Any],
    *,
    folded_records: list[RelationRecord] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {"n_records": float(len(records))}
    out.update(representation_metrics(records))
    base_reference = dict(reference.get("base_family_gaussians", {}))
    base_reference.update(reference.get("base_gaussians", {}))
    out.update(bpgv_metrics(records, base_reference, frame="base"))
    out.update(stack_geometry_metrics(records, reference.get("stack", {})))
    out.update(sequence_base_compatibility(records, reference.get("base_pair_counts")))
    if folded_records is not None:
        out.update(relation_self_consistency(records, folded_records))
    return out
