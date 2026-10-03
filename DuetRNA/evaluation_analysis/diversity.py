"""Validity, diversity, novelty, and mechanism analysis for evaluation runs.

The denominator semantics are intentionally explicit:

* ``G``: every final generated PDB, including samples whose folding evaluation failed.
* ``V``: the subset of ``G`` with max scTM at or above the configured threshold.
* ``D_raw = K(G) / |G|``.
* ``D_valid = K(V) / |V|``.
* ``Y_UV = K(V) / |G|``.

qTMclust is run on explicit file lists. Missing self-consistency records stay in
the raw denominator and cannot enter the valid subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation_analysis.common import (
    canonical_sample_id,
    discover_pdbs,
    dotted_get,
    file_fingerprint,
    finite_or_none,
    infer_length_from_name,
    load_cases,
    load_records,
    load_toml,
    mean,
    read_csv_rows,
    read_json,
    resolve_executable,
    resolve_path,
    sample_std,
    sha256_file,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: Path
    length: int
    sctm: float | None
    self_consistency_success: bool
    valid: bool
    nearest_train_tm: float | None


def _record_id(record: dict[str, Any]) -> str:
    for key in ("pdb_name", "sample_id", "name", "path", "pdb_path"):
        if record.get(key) not in (None, ""):
            return canonical_sample_id(record[key])
    raise KeyError("Self-consistency record has no sample identifier")


def _record_length(record: dict[str, Any]) -> int | None:
    for key in ("seqlen", "n_res", "length", "num_res", "sequence_length"):
        value = finite_or_none(record.get(key))
        if value is not None:
            return int(value)
    return None


def _record_sctm(record: dict[str, Any]) -> float | None:
    # RNA-FrameFlow's historical artifact called this min_scTM even though the
    # evaluator stores the maximum over designed sequences. Prefer the corrected
    # key but retain the legacy reader for reproducibility.
    value = record.get("max_scTM")
    if value in (None, ""):
        value = record.get("min_scTM")
    return finite_or_none(value)


def load_self_consistency(path: Path) -> dict[str, dict[str, Any]]:
    records = load_records(path)
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = _record_id(record)
        if sample_id in indexed:
            raise ValueError(f"Duplicate self-consistency record for {sample_id}")
        indexed[sample_id] = record
    return indexed


def load_nearest_train_tm(
    path: Path, *, expected_training_count: int | None = None
) -> dict[str, float]:
    """Stream US-align output and retain max TM2 per generated PDB.

    US-align ``-dir1/-dir2`` emits one contiguous block per generated input.  We
    enforce that ordering so an exhaustive multi-million-row report can be
    audited with memory proportional to one training-reference block rather
    than to the full Cartesian product.
    """
    nearest: dict[str, float] = {}
    completed_samples: set[str] = set()
    current_sample: str | None = None
    current_partners: set[str] = set()
    current_max = float("-inf")

    def finish_sample() -> None:
        nonlocal current_sample, current_partners, current_max
        if current_sample is None:
            return
        if (
            expected_training_count is not None
            and len(current_partners) != expected_training_count
        ):
            raise ValueError(
                "US-align report is not an exhaustive all-training comparison: "
                f"expected {expected_training_count} partners for {current_sample}; "
                f"observed={len(current_partners)}"
            )
        nearest[current_sample] = current_max
        completed_samples.add(current_sample)

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=2):
            raw_id = row.get("#PDBchain1") or row.get("PDBchain1")
            raw_partner = row.get("PDBchain2") or row.get("#PDBchain2")
            score = finite_or_none(row.get("TM2"))
            if raw_id in (None, "") or raw_partner in (None, "") or score is None:
                raise ValueError(f"Malformed US-align row {row_number} in {path}")
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"US-align TM2 is outside [0, 1] at row {row_number}: {score}"
                )
            sample_id = canonical_sample_id(raw_id)
            partner_id = canonical_sample_id(raw_partner)
            if current_sample != sample_id:
                finish_sample()
                if sample_id in completed_samples:
                    raise ValueError(
                        "US-align generated-sample rows must be contiguous for "
                        f"streaming audit; repeated block for {sample_id} at row {row_number}"
                    )
                current_sample = sample_id
                current_partners = set()
                current_max = float("-inf")
            if partner_id in current_partners:
                raise ValueError(
                    "Duplicate US-align pair at row "
                    f"{row_number}: {(sample_id, partner_id)}"
                )
            current_partners.add(partner_id)
            current_max = max(current_max, score)
    finish_sample()
    if not nearest:
        raise ValueError(f"No '#PDBchain1'/'TM2' records found in {path}")
    return nearest


def _input_set_signature(paths: list[Path]) -> str:
    """Hash resolved paths plus cheap file identity for provenance checks."""
    digest = hashlib.sha256()
    for path in sorted(path.resolve() for path in paths):
        stat = path.stat()
        digest.update(
            f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest()


def load_training_manifest(path: Path) -> list[Path]:
    rows = read_csv_rows(path)
    paths: list[Path] = []
    seen: set[Path] = set()
    for row_number, row in enumerate(rows, start=2):
        raw = row.get("path") or row.get("pdb_path") or row.get("raw_path")
        if not raw:
            raise ValueError(f"Training manifest row {row_number} has no path column value")
        candidate = Path(os.path.expandvars(os.path.expanduser(raw)))
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate in seen:
            raise ValueError(f"Duplicate training structure in manifest: {candidate}")
        seen.add(candidate)
        paths.append(candidate)
    if not paths:
        raise ValueError(f"Training manifest is empty: {path}")
    return paths


def run_usalign_all_train(
    generated_paths: list[Path],
    training_paths: list[Path],
    *,
    executable: str,
    output_path: Path,
    workers: int = 1,
    generated_chunk_size: int = 32,
) -> dict[str, float]:
    """Run US-align against every structure in an explicit training manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="duetrna-usalign-") as temporary:
        staging = Path(temporary)
        generated_dir = staging / "generated"
        training_dir = staging / "training"
        generated_dir.mkdir()
        training_dir.mkdir()
        generated_names: list[str] = []
        for source in generated_paths:
            name = source.name
            _safe_link(source, generated_dir / name)
            generated_names.append(name)
        training_names: list[str] = []
        for index, source in enumerate(training_paths):
            suffix = source.suffix if source.suffix else ".pdb"
            name = f"train_{index:07d}{suffix}"
            _safe_link(source, training_dir / name)
            training_names.append(name)
        training_list = staging / "training.list"
        training_list.write_text("\n".join(training_names) + "\n", encoding="utf-8")
        workers = max(1, int(workers))
        generated_chunk_size = max(1, int(generated_chunk_size))
        chunks = [
            generated_names[start : start + generated_chunk_size]
            for start in range(0, len(generated_names), generated_chunk_size)
        ]

        def run_chunk(chunk_index: int, chunk_names: list[str]):
            generated_list = staging / f"generated-{chunk_index:05d}.list"
            shard_output = staging / f"usalign-{chunk_index:05d}.tsv"
            generated_list.write_text("\n".join(chunk_names) + "\n", encoding="utf-8")
            command = [
                executable,
                "-dir1",
                str(generated_dir) + os.sep,
                str(generated_list),
                "-dir2",
                str(training_dir) + os.sep,
                str(training_list),
                "-outfmt",
                "2",
            ]
            with shard_output.open("w", encoding="utf-8") as output_handle:
                completed = subprocess.run(
                    command,
                    check=True,
                    stdout=output_handle,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            return chunk_index, shard_output, command, completed.stderr

        completed_shards = []
        with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as executor:
            futures = {
                executor.submit(run_chunk, chunk_index, chunk_names): chunk_index
                for chunk_index, chunk_names in enumerate(chunks)
            }
            for future in as_completed(futures):
                completed_shards.append(future.result())
        completed_shards.sort(key=lambda item: item[0])

        header: str | None = None
        with output_path.open("w", encoding="utf-8") as combined:
            for _index, shard_output, _command, _stderr in completed_shards:
                lines = shard_output.read_text(encoding="utf-8").splitlines()
                if not lines:
                    raise ValueError(f"US-align shard produced no output: {shard_output}")
                if header is None:
                    header = lines[0]
                    combined.write(header + "\n")
                elif lines[0] != header:
                    raise ValueError("US-align shard headers differ")
                for line in lines[1:]:
                    if line.strip():
                        combined.write(line + "\n")
        write_json(
            output_path.parent / f"{output_path.stem}.command.json",
            {
                "commands": [item[2] for item in completed_shards],
                "stderrs": [item[3] for item in completed_shards],
                "generated_count": len(generated_paths),
                "training_reference_count": len(training_paths),
                "generated_input_signature": _input_set_signature(generated_paths),
                "training_reference_signature": _input_set_signature(training_paths),
                "expected_pair_count": len(generated_paths) * len(training_paths),
                "reference_scope": "all_training_structures",
                "report_sha256": sha256_file(output_path),
                "workers": workers,
                "generated_chunk_size": generated_chunk_size,
                "shard_count": len(chunks),
            },
        )
    nearest = load_nearest_train_tm(
        output_path, expected_training_count=len(training_paths)
    )
    expected = {canonical_sample_id(path) for path in generated_paths}
    observed = set(nearest)
    if observed != expected:
        raise ValueError(
            "US-align generated-sample coverage mismatch: "
            f"missing={sorted(expected - observed)[:5]}, "
            f"unexpected={sorted(observed - expected)[:5]}"
        )
    return nearest


def parse_qtmclust_clusters(path: str | Path) -> list[list[str]]:
    """Parse qTMclust's tab-separated, one-cluster-per-line output."""
    clusters: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            members = [column.strip() for column in stripped.split("\t") if column.strip()]
            if not members:
                raise ValueError(f"Empty qTMclust cluster at line {line_number}")
            clusters.append(members)
    return clusters


def _safe_link(source: Path, destination: Path) -> None:
    try:
        destination.symlink_to(source)
    except OSError:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def run_qtmclust(
    paths: list[Path],
    *,
    executable: str,
    tm_threshold: float,
    output_path: Path,
    ter_mode: int = 0,
    split_mode: int = 0,
) -> list[list[str]]:
    """Run qTMclust on a staged, explicit set of PDBs and audit membership."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not paths:
        output_path.write_text("", encoding="utf-8")
        write_json(
            output_path.parent / f"{output_path.stem}.command.json",
            {
                "command": None,
                "input_count": 0,
                "inputs": [],
                "status": "empty_input",
            },
        )
        return []
    with tempfile.TemporaryDirectory(prefix="duetrna-qtmclust-") as tmp:
        staging = Path(tmp)
        staged_names: list[str] = []
        for index, source in enumerate(paths):
            staged_name = f"sample_{index:06d}.pdb"
            _safe_link(source, staging / staged_name)
            staged_names.append(staged_name)
        list_path = staging / "samples.list"
        list_path.write_text("\n".join(staged_names) + "\n", encoding="utf-8")
        command = [
            executable,
            "-dir",
            str(staging) + os.sep,
            str(list_path),
            "-TMcut",
            str(tm_threshold),
            "-ter",
            str(ter_mode),
            "-split",
            str(split_mode),
            "-o",
            str(output_path),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        (output_path.parent / f"{output_path.stem}.command.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "input_count": len(paths),
                    "inputs": [str(path) for path in paths],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        clusters = parse_qtmclust_clusters(output_path)
        members = [member for cluster in clusters for member in cluster]
        if len(members) != len(paths) or len(set(members)) != len(paths):
            raise ValueError(
                "qTMclust membership audit failed: "
                f"expected {len(paths)} unique members, got {len(members)} entries/"
                f"{len(set(members))} unique"
            )
        observed = {Path(member.split(":", 1)[0]).name for member in members}
        expected = set(staged_names)
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                "qTMclust membership names do not match the staged input set: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        return clusters


def cluster_statistics(clusters: list[list[str]], denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {"clusters": 0, "diversity": None, "largest_cluster_occupancy": None}
    cluster_count = len(clusters)
    largest = max((len(cluster) for cluster in clusters), default=0)
    return {
        "clusters": cluster_count,
        "diversity": cluster_count / denominator,
        "largest_cluster_occupancy": largest / denominator,
    }


def _pdb_length(path: Path) -> int:
    from benchmark.pdb import read_rna_pdb

    length = len(read_rna_pdb(path))
    if length <= 0:
        raise ValueError(f"Could not infer RNA length from {path}")
    return length


def build_samples(
    pdb_paths: list[Path],
    *,
    records: dict[str, dict[str, Any]],
    nearest_train_tm: dict[str, float],
    tm_threshold: float,
) -> list[Sample]:
    samples: list[Sample] = []
    for path in pdb_paths:
        sample_id = canonical_sample_id(path)
        record = records.get(sample_id)
        length = _record_length(record) if record is not None else None
        if length is None:
            length = infer_length_from_name(path)
        if length is None:
            length = _pdb_length(path)
        sctm = _record_sctm(record) if record is not None else None
        if sctm is not None and not 0.0 <= sctm <= 1.0:
            raise ValueError(f"Self-consistency TM score is outside [0, 1] for {sample_id}: {sctm}")
        success = record is not None and sctm is not None
        samples.append(
            Sample(
                sample_id=sample_id,
                path=path,
                length=length,
                sctm=sctm,
                self_consistency_success=success,
                valid=bool(success and sctm >= tm_threshold),
                nearest_train_tm=nearest_train_tm.get(sample_id),
            )
        )
    return samples


def _load_scalar_report(
    report_value: str | None,
    key: str | None,
    *,
    base_dir: Path,
) -> float | None:
    if not report_value:
        return None
    report_path = resolve_path(report_value, base_dir=base_dir, must_exist=True)
    assert report_path is not None
    report = read_json(report_path)
    if key:
        value = dotted_get(report, key)
    else:
        candidates = ("corrected_max_error", "max_equivariance_error", "se3_equivariance_error")
        value = None
        for candidate in candidates:
            try:
                value = dotted_get(report, candidate)
                break
            except KeyError:
                continue
        if value is None:
            raise KeyError(f"No default SE(3) error key found in {report_path}")
    parsed = finite_or_none(value)
    if parsed is None:
        raise ValueError(f"SE(3) error is not finite in {report_path}: {value!r}")
    return parsed


def _length_rows(
    samples: list[Sample],
    *,
    case_metadata: dict[str, Any],
    min_length: int,
    max_length: int,
    qtmclust_executable: str,
    tm_threshold: float,
    qtm_output_dir: Path,
    ter_mode: int,
    split_mode: int,
) -> list[dict[str, Any]]:
    by_length: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_length[sample.length].append(sample)
    rows: list[dict[str, Any]] = []
    for length in range(min_length, max_length + 1):
        current = by_length.get(length, [])
        valid = [sample for sample in current if sample.valid]
        all_clusters = run_qtmclust(
            [sample.path for sample in current],
            executable=qtmclust_executable,
            tm_threshold=tm_threshold,
            output_path=qtm_output_dir / f"length_{length:03d}_all.tsv",
            ter_mode=ter_mode,
            split_mode=split_mode,
        )
        valid_clusters = run_qtmclust(
            [sample.path for sample in valid],
            executable=qtmclust_executable,
            tm_threshold=tm_threshold,
            output_path=qtm_output_dir / f"length_{length:03d}_valid.tsv",
            ter_mode=ter_mode,
            split_mode=split_mode,
        )
        all_stats = cluster_statistics(all_clusters, len(current))
        valid_stats = cluster_statistics(valid_clusters, len(valid))
        all_nearest = [sample.nearest_train_tm for sample in current]
        valid_nearest = [sample.nearest_train_tm for sample in valid]
        rows.append(
            {
                **case_metadata,
                "length": length,
                "N_generated": len(current),
                "S_successful_self_consistency": sum(
                    sample.self_consistency_success for sample in current
                ),
                "M_valid": len(valid),
                "validity": None if not current else len(valid) / len(current),
                "K_generated": all_stats["clusters"],
                "D_raw": all_stats["diversity"],
                "largest_raw_cluster_occupancy": all_stats["largest_cluster_occupancy"],
                "K_valid": valid_stats["clusters"],
                "D_valid": valid_stats["diversity"],
                "Y_UV": None if not current else valid_stats["clusters"] / len(current),
                "largest_valid_cluster_occupancy": valid_stats[
                    "largest_cluster_occupancy"
                ],
                "nearest_train_tm_all_mean": mean(all_nearest),
                "nearest_train_tm_valid_mean": mean(valid_nearest),
                "novelty_pdbTM_all_mean": mean(all_nearest),
                "novelty_pdbTM_valid_mean": mean(valid_nearest),
                "nearest_train_tm_all_n": sum(value is not None for value in all_nearest),
                "nearest_train_tm_valid_n": sum(value is not None for value in valid_nearest),
            }
        )
    return rows


def evaluate_case(
    case: dict[str, Any],
    *,
    config_dir: Path,
    output_dir: Path,
    qtmclust_executable: str,
    tm_threshold: float,
    min_length: int,
    max_length: int,
    ter_mode: int,
    split_mode: int,
    include_sha256: bool,
    usalign_executable: str | None,
    default_training_manifest: str | None,
    default_usalign_reference_scope: str | None,
    usalign_workers: int,
    usalign_generated_chunk_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    case_id = str(case["id"])
    case_dir = output_dir / "cases" / case_id
    qtm_dir = case_dir / "qtmclust"
    case_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = resolve_path(case["generated_dir"], base_dir=config_dir, must_exist=True)
    records_path = resolve_path(
        case["self_consistency"], base_dir=config_dir, must_exist=True
    )
    assert generated_dir is not None and records_path is not None
    pdb_paths = discover_pdbs(
        generated_dir,
        pattern=str(case.get("pdb_glob", "**/*.pdb")),
        excluded_markers=tuple(case.get("exclude_markers", ("traj", "trajectory"))),
    )
    records = load_self_consistency(records_path)
    generated_sample_ids = {canonical_sample_id(path) for path in pdb_paths}
    unexpected_records = sorted(set(records) - generated_sample_ids)
    if unexpected_records:
        raise ValueError(
            f"Case {case_id} self-consistency artifact contains records for "
            f"{len(unexpected_records)} non-generated samples: {unexpected_records[:5]}"
        )
    usalign_report_value = case.get("usalign_report")
    training_manifest_value = case.get("training_manifest", default_training_manifest)
    if not training_manifest_value:
        raise KeyError(f"Case {case_id} requires the exact post-filter training_manifest")
    training_manifest_path = resolve_path(
        training_manifest_value, base_dir=config_dir, must_exist=True
    )
    assert training_manifest_path is not None
    training_paths = load_training_manifest(training_manifest_path)
    training_reference_count = len(training_paths)
    usalign_audit_path: Path
    if usalign_report_value:
        reference_scope = str(
            case.get("usalign_reference_scope", default_usalign_reference_scope or "")
        ).strip().lower()
        if reference_scope != "all_training_structures":
            raise ValueError(
                f"Case {case_id} supplies a precomputed US-align report but does not "
                "declare usalign_reference_scope='all_training_structures'. A report "
                "against cluster representatives cannot be labeled nearest-train TM."
            )
        usalign_path = resolve_path(
            usalign_report_value, base_dir=config_dir, must_exist=True
        )
        assert usalign_path is not None
        audit_value = case.get("usalign_audit")
        usalign_audit_path = (
            resolve_path(audit_value, base_dir=config_dir, must_exist=True)
            if audit_value
            else usalign_path.parent / f"{usalign_path.stem}.command.json"
        )
        assert usalign_audit_path is not None
        if not usalign_audit_path.is_file():
            raise FileNotFoundError(
                f"Precomputed US-align report requires its command audit: {usalign_audit_path}"
            )
        usalign_audit = read_json(usalign_audit_path)
        if usalign_audit.get("reference_scope") != "all_training_structures":
            raise ValueError("US-align command audit does not certify all training structures")
        if int(usalign_audit.get("training_reference_count", -1)) != training_reference_count:
            raise ValueError(
                "US-align command audit/training manifest count mismatch: "
                f"{usalign_audit.get('training_reference_count')} != {training_reference_count}"
            )
        if int(usalign_audit.get("generated_count", -1)) != len(pdb_paths):
            raise ValueError("US-align command audit/generated PDB count mismatch")
        if usalign_audit.get("generated_input_signature") != _input_set_signature(pdb_paths):
            raise ValueError("US-align command audit/generated input signature mismatch")
        if usalign_audit.get("training_reference_signature") != _input_set_signature(
            training_paths
        ):
            raise ValueError("US-align command audit/training reference signature mismatch")
        if int(usalign_audit.get("expected_pair_count", -1)) != len(pdb_paths) * len(
            training_paths
        ):
            raise ValueError("US-align command audit/pair count mismatch")
        audited_report_sha256 = usalign_audit.get("report_sha256")
        if not audited_report_sha256:
            raise ValueError("US-align command audit has no report SHA-256")
        if audited_report_sha256 != sha256_file(usalign_path):
            raise ValueError("US-align command audit/report SHA-256 mismatch")
        nearest = load_nearest_train_tm(
            usalign_path, expected_training_count=training_reference_count
        )
    else:
        if usalign_executable is None:
            raise RuntimeError("US-align executable was not resolved")
        usalign_path = case_dir / "MASTER_USALIGN_ALL_TRAIN.txt"
        nearest = run_usalign_all_train(
            pdb_paths,
            training_paths,
            executable=usalign_executable,
            output_path=usalign_path,
            workers=usalign_workers,
            generated_chunk_size=usalign_generated_chunk_size,
        )
        usalign_audit_path = case_dir / "MASTER_USALIGN_ALL_TRAIN.command.json"
    expected_sample_ids = {canonical_sample_id(path) for path in pdb_paths}
    if set(nearest) != expected_sample_ids:
        raise ValueError(
            f"Case {case_id} US-align coverage differs from generated samples: "
            f"missing={sorted(expected_sample_ids - set(nearest))[:5]}, "
            f"unexpected={sorted(set(nearest) - expected_sample_ids)[:5]}"
        )
    samples = build_samples(
        pdb_paths,
        records=records,
        nearest_train_tm=nearest,
        tm_threshold=tm_threshold,
    )
    outside_range = [sample for sample in samples if not min_length <= sample.length <= max_length]
    if outside_range:
        preview = ", ".join(f"{item.sample_id}:{item.length}" for item in outside_range[:5])
        raise ValueError(
            f"Case {case_id} contains {len(outside_range)} samples outside "
            f"[{min_length}, {max_length}]: {preview}"
        )
    valid = [sample for sample in samples if sample.valid]
    all_clusters = run_qtmclust(
        [sample.path for sample in samples],
        executable=qtmclust_executable,
        tm_threshold=tm_threshold,
        output_path=qtm_dir / "all_generated.tsv",
        ter_mode=ter_mode,
        split_mode=split_mode,
    )
    valid_clusters = run_qtmclust(
        [sample.path for sample in valid],
        executable=qtmclust_executable,
        tm_threshold=tm_threshold,
        output_path=qtm_dir / "sctm_valid.tsv",
        ter_mode=ter_mode,
        split_mode=split_mode,
    )
    all_stats = cluster_statistics(all_clusters, len(samples))
    valid_stats = cluster_statistics(valid_clusters, len(valid))
    metadata = {
        "case_id": case_id,
        "model": str(case.get("model", case_id)),
        "checkpoint": case.get("checkpoint", "unspecified"),
        "sampler": str(case.get("sampler", "unspecified")),
        "seed": case.get("seed", "unspecified"),
        "training_seed": case.get("training_seed", case.get("seed", "unspecified")),
        "inference_seed": case.get("inference_seed", case.get("seed", "unspecified")),
    }
    se3_error = _load_scalar_report(
        case.get("se3_report"),
        case.get("se3_key"),
        base_dir=config_dir,
    )
    summary = {
        **metadata,
        "tm_threshold": tm_threshold,
        "N_generated": len(samples),
        "S_successful_self_consistency": sum(
            sample.self_consistency_success for sample in samples
        ),
        "M_valid": len(valid),
        "validity": len(valid) / len(samples),
        "K_generated": all_stats["clusters"],
        "D_raw": all_stats["diversity"],
        "largest_raw_cluster_occupancy": all_stats["largest_cluster_occupancy"],
        "K_valid": valid_stats["clusters"],
        "D_valid": valid_stats["diversity"],
        "Y_UV": valid_stats["clusters"] / len(samples),
        "largest_valid_cluster_occupancy": valid_stats[
            "largest_cluster_occupancy"
        ],
        "self_consistency_success_rate": sum(
            sample.self_consistency_success for sample in samples
        )
        / len(samples),
        "nearest_train_tm_all_mean": mean(
            sample.nearest_train_tm for sample in samples
        ),
        "nearest_train_tm_all_n": sum(
            sample.nearest_train_tm is not None for sample in samples
        ),
        "nearest_train_tm_valid_mean": mean(
            sample.nearest_train_tm for sample in valid
        ),
        "novelty_pdbTM_all_mean": mean(
            sample.nearest_train_tm for sample in samples
        ),
        "novelty_pdbTM_valid_mean": mean(
            sample.nearest_train_tm for sample in valid
        ),
        "nearest_train_tm_valid_n": sum(
            sample.nearest_train_tm is not None for sample in valid
        ),
        "nearest_train_reference_scope": "all_training_structures",
        "training_reference_count": training_reference_count,
        "se3_equivariance_error": se3_error,
        "local_geometry_violation_percent": None,
    }
    sample_rows = [
        {
            **metadata,
            "sample_id": sample.sample_id,
            "path": str(sample.path),
            "length": sample.length,
            "self_consistency_success": sample.self_consistency_success,
            "max_scTM": sample.sctm,
            "valid": sample.valid,
            "nearest_train_tm": sample.nearest_train_tm,
            "novelty_pdbTM": sample.nearest_train_tm,
        }
        for sample in samples
    ]
    length_rows = _length_rows(
        samples,
        case_metadata=metadata,
        min_length=min_length,
        max_length=max_length,
        qtmclust_executable=qtmclust_executable,
        tm_threshold=tm_threshold,
        qtm_output_dir=qtm_dir / "by_length",
        ter_mode=ter_mode,
        split_mode=split_mode,
    )
    write_json(case_dir / "diversity_summary.json", summary)
    write_csv(case_dir / "samples.csv", sample_rows)
    write_csv(case_dir / "per_length.csv", length_rows)
    write_json(
        case_dir / "input_manifest.json",
        {
            "generated_dir": str(generated_dir),
            "self_consistency": file_fingerprint(
                records_path, include_sha256=include_sha256
            ),
            "usalign_report": file_fingerprint(
                usalign_path, include_sha256=include_sha256
            ),
            "training_manifest": file_fingerprint(
                training_manifest_path, include_sha256=True
            ),
            "usalign_audit": file_fingerprint(usalign_audit_path, include_sha256=True),
            "pdbs": [
                file_fingerprint(path, include_sha256=include_sha256) for path in pdb_paths
            ],
        },
    )
    return summary, length_rows, sample_rows


def _setting_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["model"]), str(row["checkpoint"]), str(row["sampler"])


def _setting_label(row: dict[str, Any]) -> str:
    return f"{row['model']} | {row['checkpoint']} | {row['sampler']}"


def aggregate_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_setting_key(row)].append(row)
    metrics = (
        "validity",
        "D_raw",
        "D_valid",
        "Y_UV",
        "largest_raw_cluster_occupancy",
        "largest_valid_cluster_occupancy",
        "nearest_train_tm_all_mean",
        "nearest_train_tm_valid_mean",
        "novelty_pdbTM_all_mean",
        "novelty_pdbTM_valid_mean",
        "se3_equivariance_error",
        "local_geometry_violation_percent",
    )
    aggregated: list[dict[str, Any]] = []
    for (model, checkpoint, sampler), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "model": model,
            "checkpoint": checkpoint,
            "sampler": sampler,
            "setting": _setting_label(group[0]),
            "n_seeds": len(group),
            "seeds": ",".join(str(row["seed"]) for row in group),
        }
        for metric in metrics:
            values = [finite_or_none(row.get(metric)) for row in group]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_sd"] = sample_std(values)
            result[f"{metric}_n"] = sum(value is not None for value in values)
        aggregated.append(result)
    return aggregated


def aggregate_lengths(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(*_setting_key(row), int(row["length"]))].append(row)
    metrics = (
        "validity",
        "D_valid",
        "nearest_train_tm_valid_mean",
        "novelty_pdbTM_valid_mean",
    )
    output: list[dict[str, Any]] = []
    for (model, checkpoint, sampler, length), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "model": model,
            "checkpoint": checkpoint,
            "sampler": sampler,
            "setting": _setting_label(group[0]),
            "length": length,
            "n_seeds": len(group),
            "N_generated_total": sum(int(row["N_generated"]) for row in group),
            "M_valid_total": sum(int(row["M_valid"]) for row in group),
        }
        for metric in metrics:
            values = [finite_or_none(row.get(metric)) for row in group]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_sd"] = sample_std(values)
            result[f"{metric}_n"] = sum(value is not None for value in values)
        output.append(result)
    return output


def merge_geometry_summary(
    summaries: list[dict[str, Any]],
    geometry_summary_path: Path | None,
) -> None:
    if geometry_summary_path is None:
        return
    geometry_rows = read_csv_rows(geometry_summary_path)
    indexed = {str(row["case_id"]): row for row in geometry_rows}
    for summary in summaries:
        match = indexed.get(str(summary["case_id"]))
        if match is not None:
            summary["local_geometry_violation_percent"] = finite_or_none(
                match.get("local_geometry_violation_percent")
            )


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Plotting requires matplotlib and numpy") from exc
    return plt, np


def _pareto_indices(points: list[tuple[float, float]]) -> list[int]:
    keep = []
    for index, (x_value, y_value) in enumerate(points):
        dominated = any(
            other_index != index
            and other_x >= x_value
            and other_y >= y_value
            and (other_x > x_value or other_y > y_value)
            for other_index, (other_x, other_y) in enumerate(points)
        )
        if not dominated:
            keep.append(index)
    return keep


def _unit_interval_plot_limits(
    centers: list[float], errors: list[float | None]
) -> tuple[float, float]:
    """Zoom a unit-interval metric while retaining uncertainty and context."""
    if not centers:
        return -0.02, 1.02
    lower = min(
        center - (error if error is not None else 0.0)
        for center, error in zip(centers, errors)
    )
    upper = max(
        center + (error if error is not None else 0.0)
        for center, error in zip(centers, errors)
    )
    padding = max(0.02, 0.15 * max(upper - lower, 1e-6))
    return max(-0.02, lower - padding), min(1.02, upper + padding)


def plot_validity_coverage_pareto(
    aggregate_rows: list[dict[str, Any]], output_dir: Path
) -> None:
    plt, _ = _require_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    panels = (("Y_UV", "Unique-valid yield $Y_{UV}$"), ("D_valid", "Valid diversity $D_{valid}$"))
    colors = plt.get_cmap("tab10")
    for axis, (metric, ylabel) in zip(axes, panels):
        valid_rows = [
            row
            for row in aggregate_rows
            if finite_or_none(row.get("validity_mean")) is not None
            and finite_or_none(row.get(f"{metric}_mean")) is not None
        ]
        points: list[tuple[float, float]] = []
        x_errors: list[float | None] = []
        y_errors: list[float | None] = []
        for index, row in enumerate(valid_rows):
            x_value = float(row["validity_mean"])
            y_value = float(row[f"{metric}_mean"])
            points.append((x_value, y_value))
            x_error = finite_or_none(row.get("validity_sd"))
            y_error = finite_or_none(row.get(f"{metric}_sd"))
            x_errors.append(x_error)
            y_errors.append(y_error)
            axis.errorbar(
                x_value,
                y_value,
                xerr=x_error,
                yerr=y_error,
                fmt="o",
                color=colors(index % 10),
                capsize=3,
                label=row["setting"],
            )
        front = _pareto_indices(points)
        if len(front) >= 2:
            ordered = sorted((points[index] for index in front), key=lambda item: item[0])
            axis.plot(
                [point[0] for point in ordered],
                [point[1] for point in ordered],
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.65,
                label="non-dominated front",
            )
        axis.set_xlabel("Validity $M/N$ (max scTM threshold)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(
            *_unit_interval_plot_limits(
                [point[0] for point in points], x_errors
            )
        )
        axis.set_ylim(
            *_unit_interval_plot_limits(
                [point[1] for point in points], y_errors
            )
        )
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    figure.suptitle("Validity–coverage trade-off across checkpoints and samplers")
    figure.savefig(output_dir / "figure1_validity_coverage_pareto.png", dpi=300)
    figure.savefig(output_dir / "figure1_validity_coverage_pareto.pdf")
    plt.close(figure)


def plot_per_length(aggregate_rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt, np = _require_matplotlib()
    figure, axes = plt.subplots(3, 1, figsize=(12.5, 10.5), sharex=True, constrained_layout=True)
    panels = (
        ("validity", "Validity $M/N$", (0.0, 1.0)),
        ("D_valid", "Valid diversity $D_{valid}$", (0.0, 1.0)),
        ("nearest_train_tm_valid_mean", "Nearest-train TM (valid only; lower is more novel)", (0.0, 1.0)),
    )
    by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_setting[str(row["setting"])].append(row)
    colors = plt.get_cmap("tab10")
    for setting_index, (setting, rows) in enumerate(sorted(by_setting.items())):
        rows = sorted(
            (
                row
                for row in rows
                if int(row.get("N_generated_total", 1)) > 0
            ),
            key=lambda row: int(row["length"]),
        )
        lengths = np.array([int(row["length"]) for row in rows])
        for axis, (metric, _label, _ylim) in zip(axes, panels):
            values = np.array(
                [
                    np.nan
                    if finite_or_none(row.get(f"{metric}_mean")) is None
                    else float(row[f"{metric}_mean"])
                    for row in rows
                ]
            )
            deviations = np.array(
                [
                    np.nan
                    if finite_or_none(row.get(f"{metric}_sd")) is None
                    else float(row[f"{metric}_sd"])
                    for row in rows
                ]
            )
            color = colors(setting_index % 10)
            axis.plot(lengths, values, marker=".", linewidth=1.2, color=color, label=setting)
            if np.isfinite(deviations).any():
                axis.fill_between(
                    lengths,
                    values - np.nan_to_num(deviations),
                    values + np.nan_to_num(deviations),
                    color=color,
                    alpha=0.12,
                )
    for axis, (_metric, label, ylim) in zip(axes, panels):
        axis.set_ylabel(label)
        axis.set_ylim(*ylim)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Generated RNA length")
    axes[-1].set_xlim(
        min(int(row["length"]) for row in aggregate_rows),
        max(int(row["length"]) for row in aggregate_rows),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    figure.suptitle("Per-length validity, valid diversity, and novelty")
    figure.savefig(output_dir / "figure2_per_length_behavior.png", dpi=300)
    figure.savefig(output_dir / "figure2_per_length_behavior.pdf")
    plt.close(figure)


def plot_mechanism(
    aggregate_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    plt, np = _require_matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), constrained_layout=True)
    panels = (
        ("se3_equivariance_error", "SE(3) equivariance error", True),
        ("largest_raw_cluster_occupancy", "Largest generated-cluster occupancy", False),
        ("local_geometry_violation_percent", "Phenix bond/angle outliers (%)", False),
    )
    x_positions = np.arange(len(aggregate_rows))
    labels = [str(row["setting"]) for row in aggregate_rows]
    case_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        case_groups[_setting_key(row)].append(row)
    for axis, (metric, ylabel, use_log) in zip(axes, panels):
        means = [finite_or_none(row.get(f"{metric}_mean")) for row in aggregate_rows]
        present = [value is not None for value in means]
        if not any(present):
            axis.text(0.5, 0.5, "not supplied", ha="center", va="center", transform=axis.transAxes)
            axis.set_ylabel(ylabel)
            continue
        present_positions = [position for position, exists in zip(x_positions, present) if exists]
        heights = [float(value) for value in means if value is not None]
        axis.bar(present_positions, heights, color="#4C78A8", alpha=0.78)
        for index, aggregate in enumerate(aggregate_rows):
            group = case_groups.get(
                (str(aggregate["model"]), str(aggregate["checkpoint"]), str(aggregate["sampler"])),
                [],
            )
            values = [finite_or_none(row.get(metric)) for row in group]
            values = [value for value in values if value is not None]
            if values:
                offsets = np.linspace(-0.09, 0.09, len(values))
                axis.scatter(index + offsets, values, s=22, color="black", zorder=3)
        if use_log and all(value is None or value > 0 for value in means):
            axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x_positions, labels, rotation=55, ha="right", fontsize=8)
    figure.suptitle("Mechanism diagnostics: symmetry, mode occupancy, and local geometry")
    figure.savefig(output_dir / "figure3_mechanism_diagnostics.png", dpi=300)
    figure.savefig(output_dir / "figure3_mechanism_diagnostics.pdf")
    plt.close(figure)


def run_from_config(config_path: str | Path, *, output_override: str | Path | None = None) -> Path:
    config_file, root = load_toml(config_path)
    config = root.get("diversity")
    if not isinstance(config, dict):
        raise KeyError("Config must contain a [diversity] table")
    config_dir = config_file.parent
    output_value = output_override or config.get("output_dir")
    output_dir = resolve_path(output_value, base_dir=config_dir)
    if output_dir is None:
        raise KeyError("diversity.output_dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    qtmclust = resolve_executable(config["qtmclust_path"], base_dir=config_dir)
    cases = load_cases(config, base_dir=config_dir, table_name="diversity")
    needs_usalign = any(not case.get("usalign_report") for case in cases)
    usalign_executable = None
    if needs_usalign:
        usalign_executable = resolve_executable(config["usalign_path"], base_dir=config_dir)
    tm_threshold = float(config.get("tm_threshold", 0.45))
    min_length = int(config.get("min_length", 40))
    max_length = int(config.get("max_length", 150))
    ter_mode = int(config.get("qtmclust_ter", 0))
    split_mode = int(config.get("qtmclust_split", 0))
    include_sha256 = bool(config.get("hash_inputs", False))
    usalign_workers = max(1, int(config.get("usalign_workers", 1)))
    usalign_generated_chunk_size = max(
        1, int(config.get("usalign_generated_chunk_size", 32))
    )
    case_ids = [str(case["id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("diversity case IDs must be unique")
    summaries: list[dict[str, Any]] = []
    all_length_rows: list[dict[str, Any]] = []
    all_sample_rows: list[dict[str, Any]] = []
    for case in cases:
        summary, length_rows, sample_rows = evaluate_case(
            case,
            config_dir=config_dir,
            output_dir=output_dir,
            qtmclust_executable=qtmclust,
            tm_threshold=tm_threshold,
            min_length=min_length,
            max_length=max_length,
            ter_mode=ter_mode,
            split_mode=split_mode,
            include_sha256=include_sha256,
            usalign_executable=usalign_executable,
            default_training_manifest=config.get("training_manifest"),
            default_usalign_reference_scope=config.get("usalign_reference_scope"),
            usalign_workers=usalign_workers,
            usalign_generated_chunk_size=usalign_generated_chunk_size,
        )
        summaries.append(summary)
        all_length_rows.extend(length_rows)
        all_sample_rows.extend(sample_rows)
    geometry_summary = resolve_path(
        config.get("geometry_summary"), base_dir=config_dir, must_exist=False
    )
    if geometry_summary is not None and not geometry_summary.exists():
        raise FileNotFoundError(geometry_summary)
    merge_geometry_summary(summaries, geometry_summary)
    for summary in summaries:
        write_json(
            output_dir / "cases" / str(summary["case_id"]) / "diversity_summary.json",
            summary,
        )
    aggregate_summary = aggregate_cases(summaries)
    aggregate_per_length = aggregate_lengths(all_length_rows)
    write_csv(output_dir / "case_summary.csv", summaries)
    write_json(output_dir / "case_summary.json", summaries)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_summary)
    write_csv(output_dir / "per_length.csv", all_length_rows)
    write_csv(output_dir / "aggregate_per_length.csv", aggregate_per_length)
    write_csv(output_dir / "samples.csv", all_sample_rows)
    write_json(
        output_dir / "run_manifest.json",
        {
            "config": file_fingerprint(config_file, include_sha256=True),
            "qtmclust_executable": qtmclust,
            "usalign_executable": usalign_executable,
            "usalign_workers": usalign_workers,
            "usalign_generated_chunk_size": usalign_generated_chunk_size,
            "tm_threshold": tm_threshold,
            "length_range": [min_length, max_length],
            "qtmclust_ter": ter_mode,
            "qtmclust_split": split_mode,
            "definitions": {
                "G": "all final generated PDBs; failed evaluations remain included",
                "V": f"generated PDBs with max scTM >= {tm_threshold}",
                "D_raw": "K(G) / N",
                "D_valid": "K(V) / M",
                "Y_UV": "K(V) / N",
                "novelty": "nearest-train TM is max US-align TM2 over every structure in the exact training manifest; lower is more novel",
                "novelty_pdbTM": "paper-table alias of nearest-train TM; it is not 1 - TM",
            },
        },
    )
    if bool(config.get("make_plots", True)):
        plot_validity_coverage_pareto(aggregate_summary, output_dir)
        plot_per_length(aggregate_per_length, output_dir)
        plot_mechanism(aggregate_summary, summaries, output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute evaluation validity/diversity/novelty metrics and figures"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = run_from_config(args.config, output_override=args.output_dir)
    print(f"Wrote diversity evaluation artifacts to {output}")


if __name__ == "__main__":
    main()
