"""Shared configuration and artifact helpers for extended evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


TRAJECTORY_MARKERS = ("traj", "trajectory")


def load_toml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a TOML file and return its resolved path and contents."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        return config_path, tomllib.load(handle)


def resolve_path(
    value: str | Path | None,
    *,
    base_dir: Path,
    must_exist: bool = False,
) -> Path | None:
    """Expand environment variables and resolve a config-relative path."""
    if value is None or str(value).strip() == "":
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    path = expanded if expanded.is_absolute() else base_dir / expanded
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_executable(value: str | Path, *, base_dir: Path) -> str:
    """Resolve an executable name or a config-relative executable path."""
    raw = os.path.expandvars(os.path.expanduser(str(value))).strip()
    if not raw:
        raise ValueError("Executable cannot be empty")
    if "/" in raw or raw.startswith("."):
        candidate = resolve_path(raw, base_dir=base_dir, must_exist=True)
        assert candidate is not None
        if not os.access(candidate, os.X_OK):
            raise PermissionError(f"Executable is not marked executable: {candidate}")
        return str(candidate)
    resolved = shutil.which(raw)
    if resolved is None:
        raise FileNotFoundError(
            f"Required executable '{raw}' is not on PATH. Configure its absolute path."
        )
    return resolved


def canonical_sample_id(value: str | Path) -> str:
    """Normalize a generated-structure reference to its PDB stem."""
    raw = str(value).strip().replace("\\", "/")
    raw = raw.split(":", 1)[0]
    return Path(raw).stem


def is_final_pdb(path: Path, excluded_markers: Sequence[str] = TRAJECTORY_MARKERS) -> bool:
    """Return whether a path is a final PDB rather than a saved trajectory."""
    if path.suffix.lower() != ".pdb":
        return False
    stem_lower = path.stem.lower()
    return not any(marker.lower() in stem_lower for marker in excluded_markers)


def discover_pdbs(
    root: str | Path,
    *,
    pattern: str = "**/*.pdb",
    excluded_markers: Sequence[str] = TRAJECTORY_MARKERS,
) -> list[Path]:
    """Discover final generated PDBs and reject ambiguous duplicate stems."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    paths = sorted(
        path.resolve()
        for path in root_path.glob(pattern)
        if path.is_file() and is_final_pdb(path, excluded_markers)
    )
    seen: dict[str, Path] = {}
    for path in paths:
        sample_id = canonical_sample_id(path)
        previous = seen.get(sample_id)
        if previous is not None:
            raise ValueError(
                "Generated PDB stems must be unique after flattening: "
                f"'{sample_id}' maps to both {previous} and {path}"
            )
        seen[sample_id] = path
    if not paths:
        raise ValueError(f"No final PDB files matched {pattern!r} under {root_path}")
    return paths


def infer_length_from_name(value: str | Path) -> int | None:
    """Infer a sequence length from common benchmark filenames/directories."""
    path = Path(value)
    candidates = [path.stem, *reversed(path.parts[:-1])]
    patterns = (
        re.compile(r"^(\d+)(?:_|$)"),
        re.compile(r"(?:^|_)length[_-]?(\d+)(?:_|$)", re.IGNORECASE),
        re.compile(r"(?:^|_)len[_-]?(\d+)(?:_|$)", re.IGNORECASE),
    )
    for candidate in candidates:
        for pattern in patterns:
            match = pattern.search(candidate)
            if match:
                return int(match.group(1))
    return None


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path, *, include_sha256: bool = False) -> dict[str, Any]:
    file_path = Path(path).resolve()
    stat = file_path.stat()
    result: dict[str, Any] = {
        "path": str(file_path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = sha256_file(file_path)
    return result


def file_metadata(path: str | Path) -> dict[str, Any]:
    """Return the public metadata-only representation for existing callers."""
    return file_fingerprint(path)


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path, *, delimiter: str | None = None) -> list[dict[str, str]]:
    input_path = Path(path)
    if delimiter is None:
        delimiter = "\t" if input_path.suffix.lower() in {".tsv", ".tab"} else ","
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def load_cases(config: Mapping[str, Any], *, base_dir: Path, table_name: str) -> list[dict[str, Any]]:
    """Load case dictionaries from inline TOML arrays or one portable CSV/TSV."""
    inline = config.get("cases")
    cases_file_value = config.get("cases_file")
    if inline is not None and cases_file_value is not None:
        raise ValueError(f"{table_name} must use either cases or cases_file, not both")
    if cases_file_value is not None:
        cases_path = resolve_path(cases_file_value, base_dir=base_dir, must_exist=True)
        assert cases_path is not None
        rows = read_csv_rows(cases_path)
        return [
            {key: value for key, value in row.items() if value not in (None, "")}
            for row in rows
        ]
    if not isinstance(inline, list) or not inline:
        raise ValueError(
            f"{table_name} requires at least one inline cases entry or a non-empty cases_file"
        )
    if not all(isinstance(case, Mapping) for case in inline):
        raise TypeError(f"{table_name} cases must be objects")
    return [dict(case) for case in inline]


def flatten_record_container(value: Any) -> list[dict[str, Any]]:
    """Flatten EvalSuite JSON/PT containers into a list of dictionaries."""
    if isinstance(value, list):
        if not all(isinstance(item, Mapping) for item in value):
            raise TypeError("Metric record lists must contain dictionaries")
        return [dict(item) for item in value]
    if isinstance(value, Mapping):
        if "records" in value:
            return flatten_record_container(value["records"])
        flattened: list[dict[str, Any]] = []
        for item in value.values():
            if isinstance(item, list):
                flattened.extend(flatten_record_container(item))
        if flattened:
            return flattened
    raise TypeError("Could not find a list of metric records in the supplied artifact")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load self-consistency records from JSON, JSONL, CSV/TSV, or trusted PT."""
    input_path = Path(path).expanduser().resolve()
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        return flatten_record_container(read_json(input_path))
    if suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError(f"JSONL row {line_number} is not an object")
                rows.append(dict(value))
        return rows
    if suffix in {".csv", ".tsv", ".tab"}:
        return [dict(row) for row in read_csv_rows(input_path)]
    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on evaluator environment
            raise RuntimeError("Loading a .pt artifact requires PyTorch") from exc
        try:
            value = torch.load(input_path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch before the weights_only argument
            value = torch.load(input_path, map_location="cpu")
        return flatten_record_container(value)
    raise ValueError(f"Unsupported metric artifact format: {input_path}")


def parse_json_from_stdout(text: str) -> Any:
    """Parse JSON even when an external tool writes a short banner first."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        starts = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end < start:
            raise
        return json.loads(stripped[start : end + 1])


def dotted_get(value: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = value
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def run_checked(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    timeout_seconds: int | float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        cwd=None if cwd is None else str(cwd),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def finite_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def mean(values: Iterable[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return None if not valid else sum(valid) / len(valid)


def sample_std(values: Iterable[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if len(valid) < 2:
        return None
    center = sum(valid) / len(valid)
    return (sum((value - center) ** 2 for value in valid) / (len(valid) - 1)) ** 0.5
