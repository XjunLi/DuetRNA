"""Export the exact post-filter training cohort used by a model configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evaluation_analysis.common import file_fingerprint, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export exact RNA training-reference paths")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    from omegaconf import OmegaConf
    import pandas as pd

    from duetrna.runtime.data.dataset import (
        PDBNABaseDataset,
    )

    config_path = arguments.model_config.expanduser().resolve()
    data_root = arguments.data_root.expanduser().resolve()
    output_path = arguments.output.expanduser().resolve()
    config = OmegaConf.load(config_path)
    data_config = config.data_cfg
    csv_path = Path(str(data_config.csv_path))
    if not csv_path.is_absolute():
        csv_path = data_root / csv_path
    data_config.csv_path = str(csv_path.resolve())
    metadata = pd.read_csv(csv_path)
    filtering = data_config.filtering
    length_filtered = metadata[
        (metadata["modeled_na_seq_len"] >= int(filtering.min_len))
        & (metadata["modeled_na_seq_len"] <= int(filtering.max_len))
    ]
    homomer_filtered = length_filtered[
        length_filtered["quaternary_category"] == "homomer"
    ]
    protein_free_filtered = homomer_filtered[
        homomer_filtered["num_protein_chains"] == 0
    ]

    previous_cwd = Path.cwd()
    try:
        os.chdir(data_root)
        dataset = PDBNABaseDataset(data_config, is_training=True)
    finally:
        os.chdir(previous_cwd)

    rows = []
    seen_ids: set[str] = set()
    for record in dataset.csv.to_dict(orient="records"):
        sample_id = str(record["pdb_name"])
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate training sample ID after filtering: {sample_id}")
        seen_ids.add(sample_id)
        raw_path = Path(str(record["raw_path"]))
        if not raw_path.is_absolute():
            raw_path = data_root / raw_path
        raw_path = raw_path.resolve()
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        rows.append({"sample_id": sample_id, "path": str(raw_path)})
    rows.sort(key=lambda row: row["sample_id"])
    final_ids = {row["sample_id"] for row in rows}
    metadata_filtered_ids = set(protein_free_filtered["pdb_name"].astype(str))
    write_csv(output_path, rows)
    write_json(
        output_path.with_suffix(".manifest.json"),
        {
            "model_config": file_fingerprint(config_path, include_sha256=True),
            "metadata_csv": file_fingerprint(csv_path, include_sha256=True),
            "data_root": str(data_root),
            "training_structure_count": len(rows),
            "cohort_counts": {
                "metadata_total": int(len(metadata)),
                "length_filtered": int(len(length_filtered)),
                "homomer_filtered": int(len(homomer_filtered)),
                "protein_free_filtered": int(len(protein_free_filtered)),
                "configured_dataset_final": len(rows),
            },
            "removed_by_configured_dataset_after_metadata_filters": sorted(
                metadata_filtered_ids - final_ids
            ),
            "definition": "exact PDBNABaseDataset training cohort after configured filters",
        },
    )
    print(f"Wrote {len(rows)} training structures to {output_path}")


if __name__ == "__main__":
    main()
