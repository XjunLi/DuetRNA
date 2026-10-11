"""Run the portable evaluation post-processing pipeline in dependency order."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_analysis.diversity import run_from_config as run_diversity
from evaluation_analysis.structure_quality import run_from_config as run_structure_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structure quality, then diversity figures")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--skip-structure-quality",
        action="store_true",
        help="Use an already complete case_quality_summary.csv referenced by the config",
    )
    arguments = parser.parse_args()
    if not arguments.skip_structure_quality:
        run_structure_quality(arguments.config)
    run_diversity(arguments.config)


if __name__ == "__main__":
    main()
