"""RNA structure-quality evaluation with actual Phenix validation commands.

The external results in this module come only from ``phenix.rna_validate`` and
``phenix.clashscore``.  If either executable is unavailable, preflight fails;
the independent RNA conformation and heavy-atom analyses are never substituted
for a Phenix/MolProbity result.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import tempfile
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluation_analysis.common import (
    canonical_sample_id,
    discover_pdbs,
    file_fingerprint,
    finite_or_none,
    load_cases,
    load_toml,
    mean,
    parse_json_from_stdout,
    read_json,
    resolve_executable,
    resolve_path,
    sample_std,
    write_csv,
    write_json,
)
from benchmark.pdb import Residue, read_rna_pdb
PUCKER_ATOMS = (
    ("C4'", "O4'", "C1'", "C2'"),
    ("O4'", "C1'", "C2'", "C3'"),
    ("C1'", "C2'", "C3'", "C4'"),
    ("C2'", "C3'", "C4'", "O4'"),
    ("C3'", "C4'", "O4'", "C1'"),
)
CHI_ATOMS = {
    "A": ("O4'", "C1'", "N9", "C4"),
    "U": ("O4'", "C1'", "N1", "C2"),
    "G": ("O4'", "C1'", "N9", "C4"),
    "C": ("O4'", "C1'", "N1", "C2"),
}
VDW_RADIUS_ANGSTROM = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80}


@dataclass(frozen=True)
class PhenixCommands:
    rna_validate: str
    clashscore: str
    version: str | None
    version_text: str


@dataclass(frozen=True)
class PuckerMeasurement:
    phase_degrees: float
    amplitude_degrees: float
    fit_rms_degrees: float


def dihedral_degrees(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float | None:
    """Return a signed dihedral in [-180, 180], or None if degenerate."""
    b0 = -(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))
    b1 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    b2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    norm = float(np.linalg.norm(b1))
    if norm < 1e-10:
        return None
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if float(np.linalg.norm(v)) < 1e-10 or float(np.linalg.norm(w)) < 1e-10:
        return None
    x_value = float(np.dot(v, w))
    y_value = float(np.dot(np.cross(b1, v), w))
    return math.degrees(math.atan2(y_value, x_value))


def fit_sugar_pseudorotation(torsions: Sequence[float]) -> PuckerMeasurement:
    """Calculate the 1972 Altona-Sundaralingam parameters from nu0..nu4."""
    if len(torsions) != 5:
        raise ValueError(f"Expected five ribose torsions, received {len(torsions)}")
    values = np.asarray(torsions, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Sugar torsions must all be finite")
    nu0, nu1, nu2, nu3, nu4 = map(float, values)
    numerator = nu4 + nu1 - nu3 - nu0
    denominator = 2.0 * nu2 * (
        math.sin(math.radians(36.0)) + math.sin(math.radians(72.0))
    )
    phase_radians = math.atan2(numerator, denominator)
    phase = math.degrees(phase_radians) % 360.0
    cosine = math.cos(phase_radians)

    # tau_m = nu2 / cos(P). Close to cos(P)=0, use the equivalent
    # five-angle least-squares amplitude to avoid numerical amplification.
    theta = 4.0 * math.pi * (np.arange(5, dtype=float) - 2.0) / 5.0
    if abs(cosine) >= 1e-8:
        amplitude = nu2 / cosine
    else:
        basis = np.cos(phase_radians + theta)
        amplitude = float(np.dot(values, basis) / np.dot(basis, basis))
    if amplitude < 0:
        amplitude = -amplitude
        phase = (phase + 180.0) % 360.0
        phase_radians += math.pi
    fitted = amplitude * np.cos(phase_radians + theta)
    residual = float(np.sqrt(np.mean((values - fitted) ** 2)))
    return PuckerMeasurement(phase, amplitude, residual)


def sugar_pseudorotation(residue: Residue) -> PuckerMeasurement | None:
    """Measure the five ribose torsions and fit their pseudorotation phase."""
    torsions: list[float] = []
    for quartet in PUCKER_ATOMS:
        if not all(atom in residue.atoms for atom in quartet):
            return None
        angle = dihedral_degrees(*(residue.atoms[atom] for atom in quartet))
        if angle is None:
            return None
        torsions.append(angle)
    return fit_sugar_pseudorotation(torsions)


def glycosidic_chi(residue: Residue) -> float | None:
    quartet = CHI_ATOMS[residue.resname]
    if not all(atom in residue.atoms for atom in quartet):
        return None
    return dihedral_degrees(*(residue.atoms[atom] for atom in quartet))


def _element(atom_name: str) -> str:
    return next((character.upper() for character in atom_name if character.isalpha()), "C")


def rnaff_heavy_atom_clashes(
    residues: Sequence[Residue], *, tolerance_angstrom: float = 0.6
) -> dict[str, int | float]:
    """Apply RNA-FrameFlow's inter-residue heavy-atom VdW clash definition."""
    atoms: list[tuple[int, str, str, np.ndarray, float]] = []
    for residue_position, residue in enumerate(residues):
        for atom_name, coordinate in residue.atoms.items():
            element = _element(atom_name)
            radius = VDW_RADIUS_ANGSTROM.get(element)
            if radius is None:
                continue
            atoms.append((residue_position, residue.chain_id, atom_name, coordinate, radius))

    if tolerance_angstrom < 0:
        raise ValueError("RNA-FrameFlow VdW tolerance must be non-negative")

    def is_phosphodiester_link(left, right) -> bool:
        left_residue, left_chain, left_name, *_ = left
        right_residue, right_chain, right_name, *_ = right
        if left_chain != right_chain:
            return False
        return (
            right_residue == left_residue + 1
            and left_name == "O3'"
            and right_name == "P"
        ) or (
            left_residue == right_residue + 1
            and right_name == "O3'"
            and left_name == "P"
        )

    # The reported pair denominator is all non-covalent inter-residue pairs.
    # Compute it exactly without materialising the O(n^2) distance matrix.
    atom_count_by_residue: dict[int, int] = defaultdict(int)
    atom_names_by_residue: dict[int, set[str]] = defaultdict(set)
    for residue_position, _chain, atom_name, _coordinate, _radius in atoms:
        atom_count_by_residue[residue_position] += 1
        atom_names_by_residue[residue_position].add(atom_name)
    atom_count = len(atoms)
    within_residue_pairs = sum(
        count * (count - 1) // 2 for count in atom_count_by_residue.values()
    )
    excluded_link_pairs = sum(
        residues[position].chain_id == residues[position + 1].chain_id
        and "O3'" in atom_names_by_residue[position]
        and "P" in atom_names_by_residue[position + 1]
        for position in range(max(0, len(residues) - 1))
    )
    pairs = atom_count * (atom_count - 1) // 2 - within_residue_pairs - excluded_link_pairs

    # No atom pair can clash beyond the largest possible VdW threshold.  A
    # uniform spatial grid therefore gives the exact same count as the former
    # all-pairs loop while avoiding millions of Python iterations per sample.
    clashes = 0
    max_cutoff = max(
        (left_radius + right_radius - tolerance_angstrom)
        for left_radius in VDW_RADIUS_ANGSTROM.values()
        for right_radius in VDW_RADIUS_ANGSTROM.values()
    )
    if atoms and max_cutoff > 0:
        cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        atom_cells: list[tuple[int, int, int]] = []
        for atom_index, atom in enumerate(atoms):
            coordinate = np.asarray(atom[3], dtype=float)
            if not np.isfinite(coordinate).all():
                raise ValueError("RNA structure contains a non-finite atom coordinate")
            cell = tuple(np.floor(coordinate / max_cutoff).astype(np.int64).tolist())
            atom_cells.append(cell)
            cells[cell].append(atom_index)
        neighbor_offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
        for left_index, left in enumerate(atoms):
            left_residue, _left_chain, _left_name, left_xyz, left_radius = left
            left_cell = atom_cells[left_index]
            for offset in neighbor_offsets:
                neighbor_cell = tuple(left_cell[axis] + offset[axis] for axis in range(3))
                for right_index in cells.get(neighbor_cell, ()):
                    if right_index <= left_index:
                        continue
                    right = atoms[right_index]
                    right_residue, _right_chain, _right_name, right_xyz, right_radius = right
                    if left_residue == right_residue or is_phosphodiester_link(left, right):
                        continue
                    threshold = left_radius + right_radius - tolerance_angstrom
                    if threshold < 0:
                        continue
                    delta = left_xyz - right_xyz
                    if float(np.dot(delta, delta)) <= threshold * threshold:
                        clashes += 1
    return {
        "rnaff_heavy_atom_clashes": clashes,
        "rnaff_heavy_atoms": atom_count,
        "rnaff_heavy_atom_pairs": pairs,
        "rnaff_clashes_per_100_atoms": 0.0 if atom_count == 0 else 100.0 * clashes / atom_count,
    }


def _mapping_with_keys(value: Any, required: set[str]) -> Mapping[str, Any] | None:
    queue: deque[Any] = deque([value])
    while queue:
        current = queue.popleft()
        if isinstance(current, Mapping):
            if required.issubset(current):
                return current
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return None


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    direct = payload.get(name)
    if isinstance(direct, Mapping):
        return direct
    found = _mapping_with_keys(payload, {name})
    if found is not None and isinstance(found[name], Mapping):
        return found[name]
    raise KeyError(f"Phenix RNA JSON has no {name!r} section")


def _flat_results(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = section.get("flat_results", [])
    if rows is None:
        return []
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("Phenix flat_results must be a list of objects")
    return [dict(row) for row in rows]


def _is_outlier(row: Mapping[str, Any]) -> bool:
    value = row.get("outlier", row.get("is_outlier", False))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "outlier"}
    return bool(value)


def _phenix_summary_counts(
    section: Mapping[str, Any], *, total_key: str
) -> tuple[int, int]:
    summaries = section.get("summary_results")
    if not isinstance(summaries, Mapping):
        raise TypeError("Phenix RNA section has no summary_results object")
    total = 0
    outliers = 0
    for model_id, model_summary in summaries.items():
        if not isinstance(model_summary, Mapping):
            raise TypeError(f"Phenix summary for model {model_id!r} is not an object")
        if total_key not in model_summary or "num_outliers" not in model_summary:
            raise KeyError(
                f"Phenix summary for model {model_id!r} lacks "
                f"{total_key!r} or 'num_outliers'"
            )
        model_total = finite_or_none(model_summary[total_key])
        model_outliers = finite_or_none(model_summary["num_outliers"])
        if (
            model_total is None
            or model_outliers is None
            or not float(model_total).is_integer()
            or not float(model_outliers).is_integer()
            or model_total < 0
            or model_outliers < 0
            or model_outliers > model_total
        ):
            raise ValueError(f"Invalid Phenix counts for model {model_id!r}")
        total += int(model_total)
        outliers += int(model_outliers)
    return total, outliers


def summarize_phenix_section(
    section: Mapping[str, Any], *, total_key: str
) -> dict[str, Any]:
    rows = _flat_results(section)
    deltas = [finite_or_none(row.get("delta")) for row in rows]
    scores = [finite_or_none(row.get("score", row.get("sigma"))) for row in rows]
    total, outliers = _phenix_summary_counts(section, total_key=total_key)
    row_outliers = sum(_is_outlier(row) for row in rows)
    # outliers_only=False is part of the recorded command contract, so every
    # validated item (including non-outliers) must be represented in flat_results.
    if len(rows) != total or row_outliers != outliers:
        raise ValueError(
            "Phenix flat_results disagree with summary_results: "
            f"rows={len(rows)} vs total={total}, "
            f"row_outliers={row_outliers} vs outliers={outliers}"
        )
    return {
        "total": total,
        "outliers": outliers,
        "outlier_percent": None if total == 0 else 100.0 * outliers / total,
        "absolute_delta_mean": mean(abs(value) for value in deltas if value is not None),
        "absolute_score_mean": mean(abs(value) for value in scores if value is not None),
        "absolute_score_max": max((abs(value) for value in scores if value is not None), default=None),
    }


def parse_rna_validate_json(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise TypeError("phenix.rna_validate JSON root must be an object")
    summary: dict[str, Any] = {}
    details: list[dict[str, Any]] = []
    for output_name, section_name, total_key in (
        ("bond", "rna_bonds", "num_total"),
        ("angle", "rna_angles", "num_total"),
        ("pucker", "rna_puckers", "num_residues"),
        ("suite", "rna_suites", "num_suites"),
    ):
        section = _section(payload, section_name)
        statistics = summarize_phenix_section(section, total_key=total_key)
        for key, value in statistics.items():
            summary[f"phenix_{output_name}_{key}"] = value
        for row in _flat_results(section):
            details.append(
                {
                    "validation": output_name,
                    "outlier": _is_outlier(row),
                    "delta": finite_or_none(row.get("delta")),
                    "score": finite_or_none(row.get("score", row.get("sigma"))),
                    "details_json": json.dumps(row, sort_keys=True, allow_nan=False),
                }
            )
    denominator = summary["phenix_bond_total"] + summary["phenix_angle_total"]
    numerator = summary["phenix_bond_outliers"] + summary["phenix_angle_outliers"]
    summary["local_geometry_violation_percent"] = (
        None if denominator == 0 else 100.0 * numerator / denominator
    )
    return summary, details


def parse_clashscore_json(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("phenix.clashscore JSON root must be an object")
    container = _mapping_with_keys(payload, {"summary_results", "flat_results"})
    if container is None:
        raise KeyError("phenix.clashscore JSON lacks summary_results/flat_results")
    summaries = container["summary_results"]
    if not isinstance(summaries, Mapping) or len(summaries) != 1:
        raise ValueError("phenix.clashscore must contain exactly one model summary")
    model_summary = next(iter(summaries.values()))
    if not isinstance(model_summary, Mapping):
        raise TypeError("phenix.clashscore model summary must be an object")
    clashscore = finite_or_none(model_summary.get("clashscore"))
    num_clashes = finite_or_none(model_summary.get("num_clashes"))
    if (
        clashscore is None
        or clashscore < 0
        or num_clashes is None
        or num_clashes < 0
        or not float(num_clashes).is_integer()
    ):
        raise ValueError("phenix.clashscore summary has invalid clashscore/num_clashes")
    rows = _flat_results(container)
    if int(num_clashes) != len(rows):
        raise ValueError(
            "phenix.clashscore flat_results disagree with summary_results: "
            f"rows={len(rows)} vs num_clashes={int(num_clashes)}"
        )
    return {
        "molprobity_clashscore": clashscore,
        "molprobity_severe_clash_records": int(num_clashes),
    }


def resolve_phenix_commands(config: Mapping[str, Any], *, config_dir: Path) -> PhenixCommands:
    """Fail closed unless the actual Phenix RNA and clash programs resolve."""
    try:
        rna_validate = resolve_executable(
            config.get("rna_validate_command", "phenix.rna_validate"), base_dir=config_dir
        )
        clashscore = resolve_executable(
            config.get("clashscore_command", "phenix.clashscore"), base_dir=config_dir
        )
    except (FileNotFoundError, PermissionError) as exc:
        raise RuntimeError(
            "Actual Phenix/MolProbity validation is required, but a required executable "
            "is unavailable. Install/licence Phenix or configure absolute executable paths. "
            "No custom metric will be substituted."
        ) from exc

    version_value = config.get("version_command", "phenix.version")
    version_command: str | None = None
    version_text = "unavailable"
    if version_value:
        try:
            version_command = resolve_executable(version_value, base_dir=config_dir)
            completed = subprocess.run(
                [version_command], check=True, capture_output=True, text=True, timeout=60
            )
            version_text = (completed.stdout or completed.stderr).strip()
        except (FileNotFoundError, PermissionError, subprocess.SubprocessError) as exc:
            if bool(config.get("require_version_command", True)):
                raise RuntimeError("Could not record the required Phenix version") from exc
    return PhenixCommands(rna_validate, clashscore, version_command, version_text)


def _run_json_command(
    command: Sequence[str], *, timeout_seconds: float, artifact_prefix: Path
) -> Any:
    artifact_prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="duetrna-phenix-") as temporary:
        try:
            completed = subprocess.run(
                [str(part) for part in command],
                cwd=temporary,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            write_json(
                artifact_prefix.with_suffix(".command.json"),
                {
                    "command": [str(part) for part in command],
                    "returncode": None,
                    "status": "timeout",
                    "timeout_seconds": timeout_seconds,
                    "stdout": exc.stdout.decode(errors="replace")
                    if isinstance(exc.stdout, bytes)
                    else exc.stdout,
                    "stderr": exc.stderr.decode(errors="replace")
                    if isinstance(exc.stderr, bytes)
                    else exc.stderr,
                },
            )
            raise RuntimeError(
                f"External command timed out after {timeout_seconds}s: {command[0]}; "
                f"see {artifact_prefix.with_suffix('.command.json')}"
            ) from exc
        metadata = {
            "command": [str(part) for part in command],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        write_json(artifact_prefix.with_suffix(".command.json"), metadata)
        if completed.returncode != 0:
            raise RuntimeError(
                f"External command failed with exit {completed.returncode}: {command[0]}; "
                f"see {artifact_prefix.with_suffix('.command.json')}"
            )
        try:
            payload = parse_json_from_stdout(completed.stdout)
        except (json.JSONDecodeError, ValueError):
            candidates = sorted(Path(temporary).rglob("*.json"))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Could not identify one JSON result from {command[0]}; "
                    f"found {len(candidates)} JSON files"
                )
            payload = read_json(candidates[0])
    write_json(artifact_prefix.with_suffix(".result.json"), payload)
    return payload


def run_phenix_for_sample(
    pdb_path: Path,
    *,
    commands: PhenixCommands,
    cache_dir: Path,
    timeout_seconds: float,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_id = canonical_sample_id(pdb_path)
    rna_prefix = cache_dir / sample_id / "rna_validate"
    clash_prefix = cache_dir / sample_id / "clashscore"
    rna_result = rna_prefix.with_suffix(".result.json")
    clash_result = clash_prefix.with_suffix(".result.json")
    cache_manifest = cache_dir / sample_id / "cache-input.json"
    rna_command = [
        commands.rna_validate,
        f"model={pdb_path}",
        "outliers_only=False",
        "json=True",
    ]
    clash_command = [
        commands.clashscore,
        f"model={pdb_path}",
        "json=True",
        "verbose=False",
        "keep_hydrogens=False",
    ]
    cache_signature = {
        "pdb": file_fingerprint(pdb_path, include_sha256=True),
        "phenix_version": commands.version_text,
        "rna_validate_command": rna_command,
        "clashscore_command": clash_command,
    }
    cache_matches = False
    if resume and cache_manifest.exists():
        cache_matches = read_json(cache_manifest) == cache_signature
    if cache_matches and rna_result.exists() and clash_result.exists():
        rna_payload = read_json(rna_result)
        clash_payload = read_json(clash_result)
    else:
        rna_payload = _run_json_command(
            rna_command,
            timeout_seconds=timeout_seconds,
            artifact_prefix=rna_prefix,
        )
        clash_payload = _run_json_command(
            clash_command,
            timeout_seconds=timeout_seconds,
            artifact_prefix=clash_prefix,
        )
        # Write the cache signature only after both commands have completed and
        # their JSON payloads have been persisted successfully.
        write_json(cache_manifest, cache_signature)
    summary, details = parse_rna_validate_json(rna_payload)
    summary.update(parse_clashscore_json(clash_payload))
    return summary, details


def _independent_measurements(
    residues: Sequence[Residue], *, tolerance_angstrom: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for residue in residues:
        pucker = sugar_pseudorotation(residue)
        chi = glycosidic_chi(residue)
        rows.append(
            {
                "residue_index": residue.index,
                "chain_id": residue.chain_id,
                "resseq": residue.resseq,
                "icode": residue.icode,
                "resname": residue.resname,
                "pucker_phase_degrees": None if pucker is None else pucker.phase_degrees,
                "pucker_amplitude_degrees": None if pucker is None else pucker.amplitude_degrees,
                "pucker_fit_rms_degrees": None if pucker is None else pucker.fit_rms_degrees,
                "chi_degrees": chi,
            }
        )
    pucker_rows = [row for row in rows if row["pucker_phase_degrees"] is not None]
    chi_rows = [row for row in rows if row["chi_degrees"] is not None]
    # IUPAC defines the northern half of the pseudorotation cycle as
    # P = 0 +/- 90 degrees and the southern half as P = 180 +/- 90 degrees.
    north = [
        row
        for row in pucker_rows
        if row["pucker_phase_degrees"] >= 270.0 or row["pucker_phase_degrees"] < 90.0
    ]
    south = [
        row for row in pucker_rows if 90.0 <= row["pucker_phase_degrees"] < 270.0
    ]
    summary = {
        "rna_residues": len(residues),
        "pucker_phase_count": len(pucker_rows),
        "pucker_north_fraction": None if not pucker_rows else len(north) / len(pucker_rows),
        "pucker_south_fraction": None if not pucker_rows else len(south) / len(pucker_rows),
        "pucker_amplitude_mean_degrees": mean(
            row["pucker_amplitude_degrees"] for row in pucker_rows
        ),
        "pucker_fit_rms_mean_degrees": mean(
            row["pucker_fit_rms_degrees"] for row in pucker_rows
        ),
        "chi_count": len(chi_rows),
    }
    summary.update(
        rnaff_heavy_atom_clashes(residues, tolerance_angstrom=tolerance_angstrom)
    )
    return summary, rows


def evaluate_sample(
    pdb_path: Path,
    *,
    case_metadata: Mapping[str, Any],
    commands: PhenixCommands,
    cache_dir: Path,
    timeout_seconds: float,
    resume: bool,
    tolerance_angstrom: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    residues = read_rna_pdb(pdb_path)
    if not residues:
        raise ValueError(f"No RNA residues parsed from {pdb_path}")
    phenix_summary, details = run_phenix_for_sample(
        pdb_path,
        commands=commands,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
        resume=resume,
    )
    independent_summary, residue_rows = _independent_measurements(
        residues, tolerance_angstrom=tolerance_angstrom
    )
    sample_id = canonical_sample_id(pdb_path)
    metadata = {**case_metadata, "sample_id": sample_id, "path": str(pdb_path)}
    sample_row = {**metadata, **phenix_summary, **independent_summary}
    for row in residue_rows:
        row.update(metadata)
    for row in details:
        row.update(metadata)
    return sample_row, residue_rows, details


def summarize_case(case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not case_rows:
        raise ValueError("Cannot summarize an empty case")
    first = case_rows[0]
    summary = {
        key: first[key]
        for key in ("case_id", "model", "checkpoint", "sampler", "seed")
    }
    summary["n_samples"] = len(case_rows)
    for section in ("bond", "angle", "pucker", "suite"):
        total = sum(int(row[f"phenix_{section}_total"]) for row in case_rows)
        outliers = sum(int(row[f"phenix_{section}_outliers"]) for row in case_rows)
        summary[f"phenix_{section}_total"] = total
        summary[f"phenix_{section}_outliers"] = outliers
        summary[f"phenix_{section}_outlier_percent"] = (
            None if total == 0 else 100.0 * outliers / total
        )
    local_total = summary["phenix_bond_total"] + summary["phenix_angle_total"]
    local_outliers = summary["phenix_bond_outliers"] + summary["phenix_angle_outliers"]
    summary["local_geometry_violation_percent"] = (
        None if local_total == 0 else 100.0 * local_outliers / local_total
    )
    for metric in (
        "molprobity_clashscore",
        "rnaff_clashes_per_100_atoms",
        "pucker_north_fraction",
        "pucker_south_fraction",
        "pucker_amplitude_mean_degrees",
        "pucker_fit_rms_mean_degrees",
    ):
        values = [finite_or_none(row.get(metric)) for row in case_rows]
        summary[f"{metric}_mean"] = mean(values)
        summary[f"{metric}_sd"] = sample_std(values)
        summary[f"{metric}_n"] = sum(value is not None for value in values)
    return summary


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - evaluator environment
        raise RuntimeError("Plotting structure quality requires matplotlib") from exc
    return plt


def _case_label(row: Mapping[str, Any]) -> str:
    return f"{row['model']} | {row['checkpoint']} | {row['sampler']} | seed {row['seed']}"


def plot_quality_distributions(
    sample_rows: Sequence[Mapping[str, Any]],
    residue_rows: Sequence[Mapping[str, Any]],
    detail_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    plt = _require_matplotlib()
    case_ids = sorted({str(row["case_id"]) for row in sample_rows})
    labels = {
        str(row["case_id"]): _case_label(row) for row in sample_rows
    }

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for case_id in case_ids:
        for axis, validation, unit in (
            (axes[0], "bond", "Bond |delta| (Angstrom)"),
            (axes[1], "angle", "Angle |delta| (degrees)"),
        ):
            values = [
                abs(float(row["delta"]))
                for row in detail_rows
                if str(row["case_id"]) == case_id
                and row["validation"] == validation
                and finite_or_none(row.get("delta")) is not None
            ]
            if values:
                axis.hist(values, bins=60, density=True, histtype="step", linewidth=1.4, label=labels[case_id])
            axis.set_xlabel(unit)
            axis.set_ylabel("Density")
            axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    figure.suptitle("Phenix RNA covalent-geometry deviations")
    figure.savefig(output_dir / "bond_angle_deviations.png", dpi=300)
    figure.savefig(output_dir / "bond_angle_deviations.pdf")
    plt.close(figure)

    for metric, filename, xlabel, limits in (
        ("pucker_phase_degrees", "sugar_pucker_distributions", "Pseudorotation phase (degrees)", (0, 360)),
        ("chi_degrees", "chi_angle_distributions", "Glycosidic chi angle (degrees)", (-180, 180)),
    ):
        figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        for case_id in case_ids:
            values = [
                float(row[metric])
                for row in residue_rows
                if str(row["case_id"]) == case_id and finite_or_none(row.get(metric)) is not None
            ]
            if values:
                axis.hist(values, bins=72, density=True, histtype="step", linewidth=1.4, label=labels[case_id])
        axis.set_xlim(*limits)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
        figure.savefig(output_dir / f"{filename}.png", dpi=300)
        figure.savefig(output_dir / f"{filename}.pdf")
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for axis, metric, ylabel in (
        (axes[0], "molprobity_clashscore", "MolProbity clashscore"),
        (axes[1], "rnaff_clashes_per_100_atoms", "RNA-FrameFlow heavy-atom clashes / 100 atoms"),
    ):
        distributions = [
            [float(row[metric]) for row in sample_rows if str(row["case_id"]) == case_id]
            for case_id in case_ids
        ]
        axis.boxplot(distributions, tick_labels=[labels[case_id] for case_id in case_ids], showfliers=False)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=55, labelsize=7)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Actual MolProbity and independent RNA-FrameFlow clash definitions")
    figure.savefig(output_dir / "clash_statistics.png", dpi=300)
    figure.savefig(output_dir / "clash_statistics.pdf")
    plt.close(figure)


def run_from_config(config_path: str | Path, *, output_override: str | Path | None = None) -> Path:
    config_file, root = load_toml(config_path)
    config = root.get("structure_quality")
    if not isinstance(config, Mapping):
        raise KeyError("Config must contain a [structure_quality] table")
    config_dir = config_file.parent
    output_dir = resolve_path(output_override or config.get("output_dir"), base_dir=config_dir)
    if output_dir is None:
        raise KeyError("structure_quality.output_dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = resolve_phenix_commands(config, config_dir=config_dir)
    cases = load_cases(config, base_dir=config_dir, table_name="structure_quality")
    case_ids = [str(case["id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("structure_quality case IDs must be unique")

    workers = max(1, int(config.get("workers", 4)))
    timeout_seconds = float(config.get("timeout_seconds", 300))
    resume = bool(config.get("resume", True))
    tolerance = float(config.get("rnaff_vdw_tolerance_angstrom", 0.6))
    include_sha256 = bool(config.get("hash_inputs", False))
    all_samples: list[dict[str, Any]] = []
    all_residues: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case["id"])
        generated_dir = resolve_path(case["generated_dir"], base_dir=config_dir, must_exist=True)
        assert generated_dir is not None
        pdb_paths = discover_pdbs(
            generated_dir,
            pattern=str(case.get("pdb_glob", "**/*.pdb")),
            excluded_markers=tuple(case.get("exclude_markers", ("traj", "trajectory"))),
        )
        metadata = {
            "case_id": case_id,
            "model": str(case.get("model", case_id)),
            "checkpoint": case.get("checkpoint", "unspecified"),
            "sampler": str(case.get("sampler", "unspecified")),
            "seed": case.get("seed", "unspecified"),
        }
        cache_dir = output_dir / "cases" / case_id / "phenix"
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for pdb_path in pdb_paths:
                future = executor.submit(
                    evaluate_sample,
                    pdb_path,
                    case_metadata=metadata,
                    commands=commands,
                    cache_dir=cache_dir,
                    timeout_seconds=timeout_seconds,
                    resume=resume,
                    tolerance_angstrom=tolerance,
                )
                futures[future] = pdb_path
            for future in as_completed(futures):
                pdb_path = futures[future]
                try:
                    sample_row, residue_rows, detail_rows = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Structure-quality evaluation failed for {pdb_path}") from exc
                all_samples.append(sample_row)
                all_residues.extend(residue_rows)
                all_details.extend(detail_rows)
        manifests.append(
            {
                **metadata,
                "generated_dir": str(generated_dir),
                "pdbs": [
                    file_fingerprint(path, include_sha256=include_sha256) for path in pdb_paths
                ],
            }
        )

    all_samples.sort(key=lambda row: (str(row["case_id"]), str(row["sample_id"])))
    all_residues.sort(
        key=lambda row: (str(row["case_id"]), str(row["sample_id"]), int(row["residue_index"]))
    )
    all_details.sort(
        key=lambda row: (
            str(row["case_id"]),
            str(row["sample_id"]),
            str(row["validation"]),
            str(row["details_json"]),
        )
    )
    summaries = [
        summarize_case([row for row in all_samples if str(row["case_id"]) == case_id])
        for case_id in case_ids
    ]
    write_csv(output_dir / "sample_quality.csv", all_samples)
    write_csv(output_dir / "residue_conformations.csv", all_residues)
    write_csv(output_dir / "phenix_validation_details.csv", all_details)
    write_csv(output_dir / "case_quality_summary.csv", summaries)
    write_json(output_dir / "case_quality_summary.json", summaries)
    write_json(
        output_dir / "run_manifest.json",
        {
            "config": file_fingerprint(config_file, include_sha256=True),
            "phenix_version": commands.version_text,
            "rna_validate_executable": commands.rna_validate,
            "clashscore_executable": commands.clashscore,
            "rnaff_vdw_tolerance_angstrom": tolerance,
            "cases": manifests,
            "metric_boundaries": {
                "molprobity_clashscore": "phenix.clashscore using Reduce/Probe; severe overlaps per 1000 atoms",
                "phenix_rna_validation": "phenix.rna_validate bond, angle, pucker, and suite records",
                "rnaff_heavy_atom_clashes": "independent RNA-FrameFlow Appendix C.4 VdW definition; not a MolProbity value",
                "pucker_phase": "independent five-torsion Altona-Sundaralingam 1972 phase",
                "chi": "standard O4'-C1'-N9-C4 for purines and O4'-C1'-N1-C2 for pyrimidines",
            },
        },
    )
    if bool(config.get("make_plots", True)):
        plot_quality_distributions(all_samples, all_residues, all_details, output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run actual Phenix RNA validation and evaluation structure-quality analyses"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    output = run_from_config(arguments.config, output_override=arguments.output_dir)
    print(f"Wrote structure-quality artifacts to {output}")


if __name__ == "__main__":
    main()
