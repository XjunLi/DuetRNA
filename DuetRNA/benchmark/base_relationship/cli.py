from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .reference import (
    build_reference,
    collect_records,
    evaluate_records,
    load_records_jsonl,
    load_reference,
    representation_metrics,
    save_records_jsonl,
    save_reference,
)


def _json_float_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_float_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_float_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _write_json(data: dict[str, Any], output: str | Path | None) -> None:
    text = json.dumps(_json_float_safe(data), indent=2, sort_keys=True)
    if output is None:
        print(text)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(path)


def _records_from_args(args: argparse.Namespace):
    if getattr(args, "records", None):
        return load_records_jsonl(args.records)
    return collect_records(
        args.pdb,
        annotation_dir=getattr(args, "annotation_dir", None),
        geometry_fallback=not getattr(args, "no_geometry_fallback", False),
    )


def cmd_collect(args: argparse.Namespace) -> None:
    records = _records_from_args(args)
    save_records_jsonl(records, args.output)
    print(json.dumps({"output": str(args.output), "n_records": len(records)}, sort_keys=True))


def cmd_build_reference(args: argparse.Namespace) -> None:
    records = _records_from_args(args)
    reference = build_reference(records)
    save_reference(reference, args.output)
    if args.records_output:
        save_records_jsonl(records, args.records_output)
    _write_json(
        {
            "output": str(args.output),
            "records_output": None if args.records_output is None else str(args.records_output),
            "n_records": len(records),
            "representation": reference["representation"],
        },
        args.summary_output,
    )


def cmd_representation(args: argparse.Namespace) -> None:
    records = _records_from_args(args)
    out = {"n_records": len(records), **representation_metrics(records)}
    _write_json(out, args.output)


def cmd_evaluate(args: argparse.Namespace) -> None:
    reference = load_reference(args.reference)
    records = _records_from_args(args)
    folded_records = None
    if args.folded_records:
        folded_records = load_records_jsonl(args.folded_records)
    elif args.folded_pdb:
        folded_records = collect_records(
            args.folded_pdb,
            annotation_dir=args.folded_annotation_dir,
            geometry_fallback=not args.no_geometry_fallback,
        )
    out = evaluate_records(records, reference, folded_records=folded_records)
    _write_json(out, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.base_relationship.cli",
        description="Base relationship evaluation suite for RNA structures.",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    collect = subparsers.add_parser("collect", help="Convert PDBs plus annotations to relation-record JSONL.")
    collect.add_argument("--pdb", nargs="+", required=True, help="PDB file(s) or directories.")
    collect.add_argument("--annotation-dir", default=None, help="Directory with same-stem DSSR/FR3D JSON/CSV annotations.")
    collect.add_argument("--no-geometry-fallback", action="store_true", help="Do not infer approximate relations when annotations are missing.")
    collect.add_argument("--output", required=True, help="Output JSONL path.")
    collect.set_defaults(func=cmd_collect)

    build_ref = subparsers.add_parser("build-reference", help="Build reference distributions from native RNA records.")
    build_ref.add_argument("--records", default=None, help="Existing relation-record JSONL.")
    build_ref.add_argument("--pdb", nargs="+", help="PDB file(s) or directories if --records is not used.")
    build_ref.add_argument("--annotation-dir", default=None)
    build_ref.add_argument("--no-geometry-fallback", action="store_true")
    build_ref.add_argument("--output", required=True, help="Output reference pickle path.")
    build_ref.add_argument("--records-output", default=None, help="Optional copy of collected records as JSONL.")
    build_ref.add_argument("--summary-output", default=None, help="Optional JSON summary path.")
    build_ref.set_defaults(func=cmd_build_reference)

    representation = subparsers.add_parser("representation", help="Compute RCS/RSS representation metrics.")
    representation.add_argument("--records", default=None)
    representation.add_argument("--pdb", nargs="+")
    representation.add_argument("--annotation-dir", default=None)
    representation.add_argument("--no-geometry-fallback", action="store_true")
    representation.add_argument("--output", default=None)
    representation.set_defaults(func=cmd_representation)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate generated structures against a reference.")
    evaluate.add_argument("--reference", required=True, help="Reference pickle from build-reference.")
    evaluate.add_argument("--records", default=None, help="Generated relation-record JSONL.")
    evaluate.add_argument("--pdb", nargs="+", help="Generated PDB file(s) or directories if --records is not used.")
    evaluate.add_argument("--annotation-dir", default=None)
    evaluate.add_argument("--folded-records", default=None, help="Forward-folded relation-record JSONL for RSC.")
    evaluate.add_argument("--folded-pdb", nargs="+", help="Forward-folded PDB file(s) or directories for RSC.")
    evaluate.add_argument("--folded-annotation-dir", default=None)
    evaluate.add_argument("--no-geometry-fallback", action="store_true")
    evaluate.add_argument("--output", default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "records", None) is None and getattr(args, "pdb", None) is None:
        parser.error("provide either --records or --pdb")
    args.func(args)


if __name__ == "__main__":
    main()
