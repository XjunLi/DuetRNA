"""Process RNA3DB mmCIF chains into train-ready pickles and metadata.

This script mirrors the `duetrna.data.process_rna_pdb_files`
entrypoint, but targets the official RNA3DB chain-level mmCIF releases.

Expected workflow:
1. download and extract `rna3db-jsons.tar.gz`
2. download and extract `rna3db-mmcifs.tar.xz`
3. run this script on `filter.json` to build the full filtered corpus
"""

from __future__ import annotations

import argparse
import functools as fn
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Support both `python -m duetrna.data.process_rna3db_mmcif_files` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser
from tqdm import tqdm

from duetrna.data import parsing as parsers
from duetrna.data import features as utils


class RNA3DBProcessingError(RuntimeError):
    """Raised when a RNA3DB chain cannot be converted into training features."""


parser = argparse.ArgumentParser(description="RNA3DB mmCIF processing script.")
parser.add_argument(
    "--mmcif_dir",
    type=str,
    default="data/rna3db/mmcifs/2026-01-05/rna3db-mmcifs",
    help="Directory containing extracted RNA3DB mmCIF files.",
)
parser.add_argument(
    "--filter_json",
    type=str,
    default="data/rna3db/releases/2026-01-05/jsons/rna3db-jsons/filter.json",
    help="RNA3DB filter.json file that defines the full filtered chain pool.",
)
parser.add_argument(
    "--cluster_json",
    type=str,
    default="data/rna3db/releases/2026-01-05/jsons/rna3db-jsons/cluster.json",
    help="Optional RNA3DB cluster.json for recording component membership.",
)
parser.add_argument(
    "--split_json",
    type=str,
    default="data/rna3db/releases/2026-01-05/jsons/rna3db-jsons/split.json",
    help="Optional RNA3DB split.json for recording official representative split labels.",
)
parser.add_argument(
    "--write_dir",
    type=str,
    default="data/rna3db_proc/2026-01-05",
    help="Directory to write processed pickle files.",
)
parser.add_argument(
    "--metadata_path",
    type=str,
    default="metadata/rna3db_metadata_2026-01-05_filter_available.csv",
    help="Output CSV path for train-ready metadata built from filter.json ∩ available mmCIF files.",
)
parser.add_argument(
    "--missing_manifest_path",
    type=str,
    default="metadata/rna3db_missing_from_official_mmcif_2026-01-05.csv",
    help="CSV path for filter.json chains that are absent from the downloaded official mmCIF asset.",
)
parser.add_argument(
    "--extra_manifest_path",
    type=str,
    default="metadata/rna3db_present_not_in_filter_2026-01-05.csv",
    help="CSV path for official mmCIF chains that are present on disk but absent from filter.json.",
)
parser.add_argument(
    "--release_tag",
    type=str,
    default="2026-01-05-full-release",
    help="RNA3DB release label to record in metadata.",
)
parser.add_argument(
    "--num_processes",
    type=int,
    default=16,
    help="Number of worker processes.",
)
parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help="Optional limit for smoke tests. 0 means process all chains.",
)
parser.add_argument(
    "--skip_existing",
    action="store_true",
    help="Skip writing pickles that already exist. Metadata is still rebuilt.",
)
parser.add_argument("--debug", action="store_true", help="Run serially for debugging.")
parser.add_argument("--verbose", action="store_true", help="Print per-chain details.")


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_component_map(cluster_json: Path) -> Dict[str, str]:
    if not cluster_json.exists():
        return {}
    cluster = _load_json(cluster_json)
    out = {}
    for component_name, members in cluster.items():
        for chain_key in members.keys():
            out[chain_key] = component_name
    return out


def _load_split_map(split_json: Path) -> Dict[str, str]:
    if not split_json.exists():
        return {}
    split = _load_json(split_json)
    out = {}
    for split_name, components in split.items():
        for members in components.values():
            for chain_key in members.keys():
                out[chain_key] = split_name
    return out


def _index_mmcifs(mmcif_dir: Path) -> Dict[str, Path]:
    paths = sorted(mmcif_dir.rglob("*.cif"))
    if not paths:
        raise FileNotFoundError(f"No .cif files found under {mmcif_dir}")
    index: Dict[str, Path] = {}
    for path in paths:
        pdb_id, chain_token = path.stem.split("_", 1)
        canonical_chain_key = f"{pdb_id}_{chain_token.replace('-', '')}"
        if canonical_chain_key in index:
            raise ValueError(f"Duplicate canonical RNA3DB chain key detected: {canonical_chain_key}")
        index[canonical_chain_key] = path
    return index


def _compute_radius_of_gyration(atom_positions: np.ndarray, atom_mask: np.ndarray) -> float:
    coords = atom_positions[atom_mask.astype(bool)]
    if coords.size == 0:
        return float("nan")
    center = np.mean(coords, axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=-1))))


def _process_chain(
    task: tuple[str, Dict[str, Any], str],
    *,
    mmcif_index: Dict[str, Path],
    write_dir: Path,
    skip_existing: bool,
    verbose: bool,
    release_tag: str,
    component_map: Dict[str, str],
    split_map: Dict[str, str],
) -> Dict[str, Any]:
    chain_key, filter_row, source_label = task
    pdb_id, chain_id = chain_key.split("_", 1)
    if chain_key not in mmcif_index:
        raise RNA3DBProcessingError(f"Missing mmCIF for {chain_key}: no file named {chain_key}.cif")

    file_path = mmcif_index[chain_key]
    out_subdir = write_dir / pdb_id[1:3].lower()
    out_subdir.mkdir(parents=True, exist_ok=True)
    processed_path = out_subdir / f"{chain_key}.pkl"

    if not (skip_existing and processed_path.exists()):
        structure = MMCIFParser(QUIET=True).get_structure(pdb_id, str(file_path))
        struct_chains = {chain.id: chain for chain in structure.get_chains()}
        if chain_id not in struct_chains:
            available = ", ".join(sorted(struct_chains.keys())[:20])
            raise RNA3DBProcessingError(
                f"Chain {chain_key} not found in {file_path}. Available chains start with: {available}"
            )

        chain = struct_chains[chain_id]
        chain_index = utils.chain_str_to_int(chain_id)
        chain_mol = parsers.process_chain_pdb(chain, chain_index, chain_id, verbose=verbose)
        if chain_mol is None:
            raise RNA3DBProcessingError(f"Chain {chain_key} could not be parsed as NA/protein")
        if chain_mol[-1]["molecule_type"] != "na":
            raise RNA3DBProcessingError(f"Chain {chain_key} parsed as non-RNA molecule")
        na_natype = chain_mol[-2]

        chain_constants = chain_mol[-1]["molecule_constants"]
        chain_backbone_atom_name = chain_mol[-1]["molecule_backbone_atom_name"]
        chain_dict = parsers.macromolecule_outputs_to_dict(chain_mol)
        chain_dict = utils.parse_chain_feats_pdb(
            chain_feats=chain_dict,
            molecule_constants=chain_constants,
            molecule_backbone_atom_name=chain_backbone_atom_name,
        )
        complex_feats = utils.concat_np_features([chain_dict], add_batch_dim=False)
        if complex_feats["bb_mask"].sum() < 1.0:
            raise RNA3DBProcessingError(f"Chain {chain_key} produced an empty backbone mask")

        complex_aatype = complex_feats["aatype"]
        modeled_idx = np.where((complex_aatype != 20) & (complex_aatype != 26))[0]
        na_modeled_idx = np.where(na_natype != 26)[0]
        if len(modeled_idx) == 0:
            raise RNA3DBProcessingError(f"Chain {chain_key} has no modeled residues after conversion")
        if len(na_modeled_idx) == 0:
            raise RNA3DBProcessingError(f"Chain {chain_key} has no modeled NA residues after conversion")
        complex_feats["modeled_idx"] = modeled_idx
        complex_feats["na_modeled_idx"] = na_modeled_idx
        complex_feats["inter_chain_interacting_idx"] = np.array([], dtype=np.int64)
        utils.write_pkl(str(processed_path), complex_feats)
    else:
        complex_feats = utils.read_pkl(str(processed_path))

    complex_aatype = complex_feats["aatype"]
    modeled_idx = np.where((complex_aatype != 20) & (complex_aatype != 26))[0]
    if len(modeled_idx) == 0:
        raise RNA3DBProcessingError(f"Chain {chain_key} has no modeled residues after conversion")
    na_modeled_idx = complex_feats.get("na_modeled_idx")
    if na_modeled_idx is None:
        na_modeled_idx = np.where(complex_aatype != 26)[0]
    if len(na_modeled_idx) == 0:
        raise RNA3DBProcessingError(f"Chain {chain_key} has no modeled NA residues after conversion")

    metadata = {
        "pdb_name": chain_key,
        "processed_path": _repo_relative(processed_path),
        "raw_path": _repo_relative(file_path),
        "num_chains": 1,
        "quaternary_category": "homomer",
        "num_protein_chains": 0,
        "num_na_chains": 1,
        "seq_len": len(complex_aatype),
        "na_seq_len": len(complex_aatype),
        "modeled_seq_len": int(np.max(modeled_idx) - np.min(modeled_idx) + 1),
        "modeled_protein_seq_len": 0,
        "modeled_na_seq_len": int(np.max(na_modeled_idx) - np.min(na_modeled_idx) + 1),
        "coil_percent": np.nan,
        "helix_percent": np.nan,
        "strand_percent": np.nan,
        "radius_gyration": _compute_radius_of_gyration(
            complex_feats["atom_positions"], complex_feats["atom_mask"]
        ),
        "rfam": "",
        "eq_class": "",
        "type": "RNA3DB",
        "cluster_seqid0.8": "",
        "cluster_structsim0.45": "",
        "rna3db_release_tag": release_tag,
        "rna3db_release_date": str(filter_row.get("release_date", "")),
        "rna3db_structure_method": str(filter_row.get("structure_method", "")),
        "rna3db_resolution": filter_row.get("resolution", ""),
        "rna3db_length": filter_row.get("length", ""),
        "rna3db_sequence": str(filter_row.get("sequence", "")),
        "rna3db_component": component_map.get(chain_key, ""),
        "rna3db_official_split": split_map.get(chain_key, ""),
        "source_dataset": source_label,
    }
    return metadata


def _process_task(
    task: tuple[str, Dict[str, Any], str],
    *,
    mmcif_index: Dict[str, Path],
    write_dir: Path,
    skip_existing: bool,
    verbose: bool,
    release_tag: str,
    component_map: Dict[str, str],
    split_map: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    try:
        start_time = time.time()
        row = _process_chain(
            task,
            mmcif_index=mmcif_index,
            write_dir=write_dir,
            skip_existing=skip_existing,
            verbose=verbose,
            release_tag=release_tag,
            component_map=component_map,
            split_map=split_map,
        )
        if verbose:
            print(f"Finished {task[0]} in {time.time() - start_time:2.2f}s")
        return row
    except Exception as exc:
        if verbose:
            print(f"Failed {task[0]}: {exc}")
        return {
            "pdb_name": task[0],
            "processed_path": "",
            "raw_path": "",
            "num_chains": "",
            "quaternary_category": "",
            "num_protein_chains": "",
            "num_na_chains": "",
            "seq_len": "",
            "na_seq_len": "",
            "modeled_seq_len": "",
            "modeled_protein_seq_len": "",
            "modeled_na_seq_len": "",
            "coil_percent": "",
            "helix_percent": "",
            "strand_percent": "",
            "radius_gyration": "",
            "rfam": "",
            "eq_class": "",
            "type": "",
            "cluster_seqid0.8": "",
            "cluster_structsim0.45": "",
            "rna3db_release_tag": release_tag,
            "rna3db_release_date": "",
            "rna3db_structure_method": "",
            "rna3db_resolution": "",
            "rna3db_length": "",
            "rna3db_sequence": "",
            "rna3db_component": "",
            "rna3db_official_split": "",
            "source_dataset": "RNA3DB",
            "processing_error": str(exc),
        }


def _build_tasks(filter_rows: Dict[str, Dict[str, Any]], limit: int) -> list[tuple[str, Dict[str, Any], str]]:
    tasks = [(chain_key, row, "RNA3DB") for chain_key, row in sorted(filter_rows.items())]
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def main(args: argparse.Namespace) -> None:
    mmcif_dir = Path(args.mmcif_dir)
    filter_json = Path(args.filter_json)
    cluster_json = Path(args.cluster_json)
    split_json = Path(args.split_json)
    write_dir = Path(args.write_dir)
    metadata_path = Path(args.metadata_path)
    missing_manifest_path = Path(args.missing_manifest_path)
    extra_manifest_path = Path(args.extra_manifest_path)

    write_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    missing_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    extra_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    filter_rows = _load_json(filter_json)
    component_map = _load_component_map(cluster_json)
    split_map = _load_split_map(split_json)
    mmcif_index = _index_mmcifs(mmcif_dir)
    available_keys = set(mmcif_index)
    filter_keys = set(filter_rows)
    available_filter_keys = sorted(filter_keys & available_keys)
    missing_filter_keys = sorted(filter_keys - available_keys)
    extra_available_keys = sorted(available_keys - filter_keys)

    pd.DataFrame({"pdb_name": missing_filter_keys}).to_csv(
        missing_manifest_path, index=False
    )
    pd.DataFrame({"pdb_name": extra_available_keys}).to_csv(
        extra_manifest_path, index=False
    )

    available_filter_rows = {key: filter_rows[key] for key in available_filter_keys}
    tasks = _build_tasks(available_filter_rows, args.limit)

    print(f"Indexed {len(mmcif_index)} mmCIF files from {mmcif_dir}")
    print(f"filter.json chains: {len(filter_rows)}")
    print(f"filter.json ∩ available mmCIF files: {len(available_filter_rows)}")
    print(f"filter.json chains missing from official mmCIF asset: {len(missing_filter_keys)}")
    print(f"official mmCIF chains not in filter.json: {len(extra_available_keys)}")
    print(f"Missing manifest written to {missing_manifest_path}")
    print(f"Extra-chain manifest written to {extra_manifest_path}")
    print(f"Will process {len(tasks)} RNA3DB chains from the available filter overlap")

    process_one = fn.partial(
        _process_task,
        mmcif_index=mmcif_index,
        write_dir=write_dir,
        skip_existing=args.skip_existing,
        verbose=args.verbose,
        release_tag=args.release_tag,
        component_map=component_map,
        split_map=split_map,
    )

    if args.debug or args.num_processes == 1:
        rows = [process_one(task) for task in tqdm(tasks)]
    else:
        with mp.Pool(processes=args.num_processes) as pool:
            rows = list(tqdm(pool.imap(process_one, tasks), total=len(tasks)))

    df = pd.DataFrame(rows)
    df.to_csv(metadata_path, index=False)
    failures = int(df["processing_error"].notna().sum()) if "processing_error" in df.columns else 0
    successes = len(df) - failures
    print(f"Wrote metadata to {metadata_path}")
    print(f"Finished processing {successes}/{len(tasks)} RNA3DB chains")
    if failures:
        print(f"Failures recorded in metadata CSV: {failures}")


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main(parser.parse_args())
