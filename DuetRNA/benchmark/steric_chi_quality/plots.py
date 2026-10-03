"""Plotting functions for steric clash and chi angle evaluation.

Generates Figure 16-style bar plots with sequence length bucketing.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path


def bucket_by_length(stats_list: list[dict]) -> dict[int, dict]:
    """Bucket statistics by sequence length (10-unit intervals)."""
    buckets = defaultdict(list)
    for stat in stats_list:
        if "length" in stat and stat["length"] > 0:
            length_bin = (stat["length"] // 10) * 10
            buckets[length_bin].append(stat)
    return dict(buckets)


def summarize_bucket(bucket_stats: list[dict], key: str) -> tuple[float, float]:
    """Compute mean and std for a metric in a bucket."""
    values = [s.get(key, float("nan")) for s in bucket_stats]
    values = [v for v in values if not np.isnan(v)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def plot_clash_by_length(
    cases_stats: dict[str, list[dict]],  # {case_label: list of per-sample stats}
    output_path: Path | str,
    title: str = "Steric clashes per 100 atoms by sequence length",
) -> None:
    """
    Plot steric clashes per 100 atoms vs sequence length.

    Figure 16 style: bar plot with length bins on x-axis.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all unique bins
    all_bins = set()
    for stats in cases_stats.values():
        for s in stats:
            if "length" in s and s["length"] > 0:
                all_bins.add((s["length"] // 10) * 10)
    bins = sorted(all_bins)

    if len(bins) == 0:
        print("No data to plot")
        return

    # Prepare data for each case
    case_data = {}
    for case_label, stats in cases_stats.items():
        bucketed = bucket_by_length(stats)
        means = []
        stds = []
        for b in bins:
            if b in bucketed:
                m, s = summarize_bucket(bucketed[b], "clashes_per_100_atoms")
                means.append(m)
                stds.append(s)
            else:
                means.append(float("nan"))
                stds.append(float("nan"))
        case_data[case_label] = (means, stds)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    bar_width = 8
    n_cases = len(cases_stats)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    x = np.array(bins)
    offset = -(n_cases - 1) * bar_width / 2

    for i, (case_label, (means, stds)) in enumerate(case_data.items()):
        color = colors[i % len(colors)]
        x_pos = x + offset + i * bar_width
        # Filter NaN
        valid_mask = ~np.isnan(means)
        if np.any(valid_mask):
            ax.bar(
                x_pos[valid_mask],
                np.array(means)[valid_mask],
                width=bar_width,
                yerr=np.array(stds)[valid_mask],
                color=color,
                label=case_label,
                capsize=3,
                alpha=0.8,
            )

    # Add reference line for RNAsolo (paper value: 10.03±1.52)
    ax.axhline(10.03, color="gray", linestyle="--", linewidth=1.5, label="RNAsolo native (10.03±1.52)")

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Steric clashes per 100 atoms")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b}-{b+9}" for b in bins], rotation=25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_chi_by_length(
    cases_stats: dict[str, list[dict]],
    output_path: Path | str,
    metric_key: str = "chi_mae",  # or "chi_within_5deg"
    title: str = "Chi angle deviation by sequence length",
    ylabel: str = "Chi angle MAE (degrees)",
) -> None:
    """
    Plot chi angle metrics vs sequence length.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all unique bins
    all_bins = set()
    for stats in cases_stats.values():
        for s in stats:
            if "length" in s and s["length"] > 0:
                all_bins.add((s["length"] // 10) * 10)
    bins = sorted(all_bins)

    if len(bins) == 0:
        print("No data to plot")
        return

    # Prepare data for each case
    case_data = {}
    for case_label, stats in cases_stats.items():
        bucketed = bucket_by_length(stats)
        means = []
        stds = []
        for b in bins:
            if b in bucketed:
                m, s = summarize_bucket(bucketed[b], metric_key)
                means.append(m)
                stds.append(s)
            else:
                means.append(float("nan"))
                stds.append(float("nan"))
        case_data[case_label] = (means, stds)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    bar_width = 8
    n_cases = len(cases_stats)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    x = np.array(bins)
    offset = -(n_cases - 1) * bar_width / 2

    for i, (case_label, (means, stds)) in enumerate(case_data.items()):
        color = colors[i % len(colors)]
        x_pos = x + offset + i * bar_width
        valid_mask = ~np.isnan(means)
        if np.any(valid_mask):
            ax.bar(
                x_pos[valid_mask],
                np.array(means)[valid_mask],
                width=bar_width,
                yerr=np.array(stds)[valid_mask],
                color=color,
                label=case_label,
                capsize=3,
                alpha=0.8,
            )

    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b}-{b+9}" for b in bins], rotation=25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_summary_tsv(
    cases_stats: dict[str, list[dict]],
    output_path: Path | str,
) -> None:
    """Save per-case per-bucket summary as TSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Header
        headers = ["case", "length_bin", "n_samples", "clash_mean", "clash_std", "chi_mae_mean", "chi_mae_std", "chi_5deg_pct"]
        f.write("\t".join(headers) + "\n")

        for case_label, stats in cases_stats.items():
            bucketed = bucket_by_length(stats)
            for bin_len in sorted(bucketed.keys()):
                bucket_stats = bucketed[bin_len]
                n = len(bucket_stats)
                clash_m, clash_s = summarize_bucket(bucket_stats, "clashes_per_100_atoms")
                chi_m, chi_s = summarize_bucket(bucket_stats, "chi_mae")
                chi_5_m, _ = summarize_bucket(bucket_stats, "chi_within_5deg")

                row = [
                    case_label,
                    str(bin_len),
                    str(n),
                    f"{clash_m:.2f}" if not np.isnan(clash_m) else "nan",
                    f"{clash_s:.2f}" if not np.isnan(clash_s) else "nan",
                    f"{chi_m:.2f}" if not np.isnan(chi_m) else "nan",
                    f"{chi_s:.2f}" if not np.isnan(chi_s) else "nan",
                    f"{chi_5_m:.2%}" if not np.isnan(chi_5_m) else "nan",
                ]
                f.write("\t".join(row) + "\n")

    print(f"Saved: {output_path}")
