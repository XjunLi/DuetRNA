from __future__ import annotations

import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from benchmark.pdb import iter_pdb_files


def find_fr3d_script(fr3d_root: str | Path | None = None, fr3d_script: str | Path | None = None) -> Path:
    if fr3d_script is not None:
        path = Path(fr3d_script)
    else:
        root = Path(fr3d_root or os.environ.get("FR3D_ROOT", ""))
        path = root / "fr3d" / "classifiers" / "NA_pairwise_interactions.py"
    if not path.exists():
        raise FileNotFoundError(
            "FR3D NA_pairwise_interactions.py not found. Set FR3D_ROOT or pass --fr3d-script. "
            f"Tried: {path}"
        )
    return path


def merge_fr3d_outputs(stem: str, fr3d_output_dir: Path, merged_path: Path) -> int:
    """Merge FR3D basepair/stacking outputs into the annotation TSV we parse.

    FR3D writes separate files such as ``<stem>_basepair.txt`` and
    ``<stem>_stacking.txt``.  The base-relationship parser expects one
    three-column file: unit_id1, relation, unit_id2.
    """
    rows: list[tuple[str, str, str]] = []
    found_any_output = False
    for category in ("basepair", "stacking"):
        path = fr3d_output_dir / f"{stem}_{category}.txt"
        if not path.exists():
            continue
        found_any_output = True
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 3 and fields[0] and fields[1] and fields[2]:
                    rows.append((fields[0], fields[1], fields[2]))
    if not found_any_output:
        raise RuntimeError(f"FR3D produced no basepair/stacking output files for {stem} in {fr3d_output_dir}")
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as handle:
        for unit1, relation, unit2 in rows:
            handle.write(f"{unit1}\t{relation}\t{unit2}\n")
    return len(rows)


def annotate_pdbs_with_fr3d(
    pdb_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    fr3d_root: str | Path | None = None,
    fr3d_script: str | Path | None = None,
    python_cmd: str | Path = "python",
    category: str = "basepair,stacking",
    force: bool = False,
    workers: int = 1,
    skip_failures: bool = False,
    failure_log: str | Path | None = None,
) -> dict[str, int]:
    script = find_fr3d_script(fr3d_root=fr3d_root, fr3d_script=fr3d_script)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_files = list(iter_pdb_files(list(pdb_paths)))
    if not pdb_files:
        raise RuntimeError(f"No PDB files found for FR3D annotation: {list(pdb_paths)}")

    counts: dict[str, int] = {}
    failures: list[tuple[str, str]] = []
    env = os.environ.copy()
    fr3d_pkg_root = str(script.parents[2])
    env["PYTHONPATH"] = fr3d_pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def annotate_one(pdb_path: Path, tmp_dir: Path) -> tuple[str, int]:
        pdb_path = Path(pdb_path)
        merged = output_dir / f"{pdb_path.stem}_fr3d.tsv"
        if merged.exists() and not force:
            count = sum(1 for _ in merged.open("r", encoding="utf-8"))
            if count == 0:
                raise RuntimeError(f"FR3D annotation parsed to zero basepair/stacking rows for {pdb_path}: {merged}")
            return pdb_path.stem, count

        run_dir = tmp_dir / pdb_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(python_cmd),
            str(script),
            "-c",
            category,
            # FR3D's localpath.py can ship with a machine-specific default
            # inputPath such as a Windows directory.  We pass absolute PDB
            # paths, so force an empty input path to avoid that side effect.
            "-i",
            "",
            "-o",
            str(run_dir),
            str(pdb_path),
        ]
        proc = subprocess.run(cmd, cwd=str(script.parents[2]), env=env, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"FR3D failed for {pdb_path} with exit code {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        count = merge_fr3d_outputs(pdb_path.stem, run_dir, merged)
        if count == 0:
            raise RuntimeError(f"FR3D annotation parsed to zero basepair/stacking rows for {pdb_path}: {merged}")
        return pdb_path.stem, count

    def handle_failure(pdb_path: Path, exc: BaseException) -> None:
        if not skip_failures:
            raise exc
        failures.append((str(pdb_path), str(exc).replace("\n", "\\n")))

    with tempfile.TemporaryDirectory(prefix="fr3d_out_", dir=str(output_dir)) as tmp:
        tmp_dir = Path(tmp)
        if workers <= 1:
            for pdb_path in pdb_files:
                try:
                    stem, count = annotate_one(Path(pdb_path), tmp_dir)
                    counts[stem] = count
                except Exception as exc:
                    handle_failure(Path(pdb_path), exc)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(annotate_one, Path(pdb_path), tmp_dir): Path(pdb_path) for pdb_path in pdb_files}
                for future in as_completed(futures):
                    try:
                        stem, count = future.result()
                        counts[stem] = count
                    except Exception as exc:
                        handle_failure(futures[future], exc)
    if failure_log is not None:
        log_path = Path(failure_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("pdb_path\terror\n")
            for pdb_path, error in failures:
                handle.write(f"{pdb_path}\t{error}\n")
    if not counts:
        detail = "" if not failures else f"; failures written to {failure_log}"
        raise RuntimeError(f"FR3D produced no usable annotations for {list(pdb_paths)}{detail}")
    return counts
