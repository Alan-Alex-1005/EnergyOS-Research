"""
anomaly/robust.py — Day 2 Part J/K: Robust Residual Statistics + Severity

Median / Median Absolute Deviation (MAD) / robust z-score, fit
EXCLUSIVELY on TRAINING residuals (never on the residuals being
classified) — this is what makes anomaly thresholds leakage-free
(Part D, Part J).

    m   = median(training residuals)
    MAD = median(|training residuals - m|)
    z_t = 0.6745 * (e_t - m) / MAD

0.6745 rescales MAD to be a consistent estimator of the standard
deviation under a normal distribution, so z behaves like a standard
z-score for roughly-normal residuals — a standard, well-documented
robust-statistics convention (not invented here).

Severity thresholds (Part K) are centralized in SEVERITY_BAND_RATIOS
below, explicitly documented as PROTOTYPE DECISION THRESHOLDS chosen
for this demonstrator — not universal scientific constants — and
configurable via a single base threshold (spec's `anomaly_threshold`
config key).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_BASE_THRESHOLD = 3.0

# Severity band cutoffs as multiples of the base threshold. These ratios
# are a PROTOTYPE DESIGN CHOICE for this demonstrator, not a scientific
# standard — documented here so they can be inspected, tested, and
# adjusted. A single `base_threshold` (LOW cutoff) scales the whole
# ladder proportionally.
SEVERITY_BAND_RATIOS = {
    "LOW": 1.0,
    "MEDIUM": 1.33,
    "HIGH": 1.83,
    "CRITICAL": 2.33,
}
SEVERITY_ORDER = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Only reached in the fully-degenerate edge case where BOTH the training
# MAD and the training standard deviation are exactly 0 (i.e. every
# training residual is bit-for-bit identical). A z-score is mathematically
# undefined there (0/0); this sentinel is a clearly-flagged stand-in, not
# a computed statistic, and is documented wherever it can appear.
DEGENERATE_SENTINEL_Z = 10.0


@dataclass
class RobustResidualStats:
    median: float
    mad: float
    scale_method: str  # "MAD" or a documented fallback
    note: Optional[str] = None


def fit_robust_stats(train_residuals: np.ndarray) -> RobustResidualStats:
    """Fit (median, MAD) from TRAINING residuals only. Handles MAD == 0 safely."""
    train_residuals = np.asarray(train_residuals, dtype=float)
    train_residuals = train_residuals[~np.isnan(train_residuals)]

    if len(train_residuals) == 0:
        return RobustResidualStats(median=0.0, mad=0.0, scale_method="undefined", note="No training residuals available.")

    median = float(np.median(train_residuals))
    abs_dev = np.abs(train_residuals - median)
    mad = float(np.median(abs_dev))

    if mad > 0:
        return RobustResidualStats(median=median, mad=mad, scale_method="MAD")

    # MAD == 0: at least half the training residuals are identical to the
    # median. Documented fallback: use the training residual standard
    # deviation, rescaled to be MAD-equivalent (MAD ~ 0.6745*std under
    # normality, so std_as_mad = std / 0.6745 keeps the same 0.6745 formula
    # meaningful downstream).
    std = float(np.std(train_residuals))
    if std > 0:
        return RobustResidualStats(
            median=median,
            mad=std / 0.6745,
            scale_method="MAD=0 fallback to std-based scale",
            note="Training residual MAD was 0 (many identical residuals); fell back to a standard-deviation-derived scale.",
        )

    return RobustResidualStats(
        median=median,
        mad=0.0,
        scale_method="degenerate (MAD=0 and std=0)",
        note=(
            "Training residuals have zero spread (all identical). Robust z-score is mathematically "
            f"undefined (0/0); any deviation from the median is scored with a fixed sentinel z="
            f"{DEGENERATE_SENTINEL_Z} rather than dividing by zero."
        ),
    )


def robust_z_scores(residuals: np.ndarray, stats: RobustResidualStats) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    if stats.mad > 0:
        return 0.6745 * (residuals - stats.median) / stats.mad
    # Degenerate case: 0 for exact matches, sentinel for any deviation.
    return np.where(residuals == stats.median, 0.0, np.sign(residuals - stats.median) * DEGENERATE_SENTINEL_Z)


def severity_cutoffs(base_threshold: float = DEFAULT_BASE_THRESHOLD) -> dict:
    if base_threshold <= 0:
        raise ValueError(f"anomaly base_threshold must be positive; got {base_threshold}")
    return {label: base_threshold * ratio for label, ratio in SEVERITY_BAND_RATIOS.items()}


def classify_severity(abs_z: float, base_threshold: float = DEFAULT_BASE_THRESHOLD) -> str:
    cutoffs = severity_cutoffs(base_threshold)
    if abs_z >= cutoffs["CRITICAL"]:
        return "CRITICAL"
    if abs_z >= cutoffs["HIGH"]:
        return "HIGH"
    if abs_z >= cutoffs["MEDIUM"]:
        return "MEDIUM"
    if abs_z >= cutoffs["LOW"]:
        return "LOW"
    return "NORMAL"
