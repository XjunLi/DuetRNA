"""Steric clash and chi angle quality evaluation module.

This module implements:
- Steric clash counting using VdW-radius-based thresholds (RNA-FrameFlow paper Appendix C.4)
- Chi angle (glycosidic torsion) extraction and comparison
- Figure 16 style bar plots with sequence length bucketing
"""

from .steric import count_steric_clashes, VDW_RADIUS, VDW_TOLERANCE
from .chi_angles import extract_chi_angle_deg, compute_chi_metrics
from .plots import plot_clash_by_length, plot_chi_by_length, save_summary_tsv

__all__ = [
    "count_steric_clashes",
    "VDW_RADIUS",
    "VDW_TOLERANCE",
    "extract_chi_angle_deg",
    "compute_chi_metrics",
    "plot_clash_by_length",
    "plot_chi_by_length",
    "save_summary_tsv",
]
