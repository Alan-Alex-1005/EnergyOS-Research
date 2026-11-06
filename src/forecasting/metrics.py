"""
forecasting/metrics.py — Day 2 Part G: Forecast Metrics

Deterministic metric functions only — no fitting, no randomness. Every
edge case (zero actuals for MAPE, constant series for R2) is handled
explicitly and documented rather than silently producing inf/NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MetricResult:
    value: Optional[float]
    available: bool
    note: Optional[str] = None


def mae(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    actual, predicted = _align(actual, predicted)
    if len(actual) == 0:
        return MetricResult(value=None, available=False, note="No paired observations.")
    value = float(np.mean(np.abs(actual - predicted)))
    return MetricResult(value=value, available=True)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    actual, predicted = _align(actual, predicted)
    if len(actual) == 0:
        return MetricResult(value=None, available=False, note="No paired observations.")
    value = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return MetricResult(value=value, available=True)


def mape(actual: np.ndarray, predicted: np.ndarray, zero_epsilon: float = 0.0) -> MetricResult:
    """
    MAPE is undefined wherever actual == 0 (division by zero). We do NOT
    silently divide by a fudge factor — we compute MAPE only over the
    subset of observations where actual != 0, and report exactly how
    many were excluded. If EVERY actual value is zero, MAPE is reported
    as unavailable rather than returning a misleading number.
    """
    actual, predicted = _align(actual, predicted)
    if len(actual) == 0:
        return MetricResult(value=None, available=False, note="No paired observations.")

    nonzero_mask = actual != 0
    excluded = int(len(actual) - nonzero_mask.sum())
    if not nonzero_mask.any():
        return MetricResult(
            value=None,
            available=False,
            note=f"MAPE unavailable: all {len(actual)} actual value(s) are zero (division by zero).",
        )

    pct_errors = np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])
    value = float(100.0 * np.mean(pct_errors))
    note = None
    if excluded:
        note = f"{excluded} observation(s) with actual == 0 excluded from MAPE (division by zero)."
    return MetricResult(value=value, available=True, note=note)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    """
    Standard coefficient of determination: 1 - SS_res / SS_tot.
    A constant actual series makes SS_tot == 0 (undefined R2 in the
    classical sense — the mean explains everything trivially since
    there's no variance to explain). We report this explicitly instead
    of returning +-inf or NaN silently.
    """
    actual, predicted = _align(actual, predicted)
    if len(actual) < 2:
        return MetricResult(value=None, available=False, note="R2 requires at least 2 paired observations.")

    ss_res = float(np.sum((actual - predicted) ** 2))
    mean_actual = float(np.mean(actual))
    ss_tot = float(np.sum((actual - mean_actual) ** 2))

    if ss_tot == 0:
        if ss_res == 0:
            return MetricResult(
                value=1.0, available=True, note="Actual series is constant; predictions match it exactly."
            )
        return MetricResult(
            value=None,
            available=False,
            note="R2 undefined: actual series is constant (zero variance) and predictions differ from it.",
        )

    value = float(1.0 - (ss_res / ss_tot))
    return MetricResult(value=value, available=True)


def _align(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"actual and predicted must be the same shape; got {actual.shape} vs {predicted.shape}")
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    return actual[mask], predicted[mask]
