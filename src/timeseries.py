"""
timeseries.py — Day 2 Part C: Time-Series Preparation

Turns a raw AnalysisSeries into a model-ready, chronologically-prepared
series: ordered, deduplicated, with gaps/irregularity/missing-value
reporting, and split into train/test WITHOUT shuffling (Part D: no
future information may leak into training).

This is the single choke point every forecasting/anomaly component goes
through — none of them touch AnalysisSeries.timestamps/values directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .analysis_series import AnalysisSeries, MIN_SAMPLES_FOR_FORECAST

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_TRAIN_VAL_TEST_RATIOS = (0.70, 0.15, 0.15)


@dataclass
class PreparedSeries:
    series_id: str
    role: str
    unit: Optional[str]
    timestamps: pd.Series  # chronological, deduplicated, non-null values only
    values: pd.Series
    sampling_interval_seconds: Optional[float]
    sampling_interval_label: Optional[str]
    is_regular: bool  # True if sampling interval is consistent enough to trust
    gap_count: int  # number of gaps larger than ~1.5x the median interval
    largest_gap_seconds: Optional[float]
    missing_value_count: int  # NaN observations in the ORIGINAL series (before dropping)
    duplicate_timestamp_count: int  # true within-series duplicates, already resolved (kept first)
    original_sample_count: int
    prepared_sample_count: int
    sufficient: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrainTestSplit:
    series_id: str
    train_ratio: float
    train_timestamps: pd.Series
    train_values: pd.Series
    test_timestamps: pd.Series
    test_values: pd.Series
    train_start: Optional[str]
    train_end: Optional[str]
    test_start: Optional[str]
    test_end: Optional[str]
    train_samples: int
    test_samples: int
    warnings: list[str] = field(default_factory=list)


class TimeSeriesPrepError(Exception):
    """Raised when a series cannot be defensibly prepared/split."""


def prepare_series(series: AnalysisSeries, min_samples: int = MIN_SAMPLES_FOR_FORECAST) -> PreparedSeries:
    """
    Chronological ordering + duplicate handling + sampling-interval
    detection + gap detection + missing-value reporting (Part C).
    Never fabricates a value: rows with NaN are reported, then dropped
    only for the purpose of modeling (kept out of `values`), never
    imputed.
    """
    warnings: list[str] = []
    original_sample_count = int(len(series.values))
    missing_value_count = int(series.values.isna().sum())

    ts, vals = series.non_null()  # already chronological (AnalysisSeries guarantees this) + NaNs dropped

    # True duplicate timestamps within this series: keep first occurrence
    # only, exactly like Day-1 cleaning.py's REMOVED_DUPLICATE philosophy,
    # and report the count rather than silently absorbing it.
    dup_mask = ts.duplicated(keep="first")
    duplicate_timestamp_count = int(dup_mask.sum())
    if duplicate_timestamp_count:
        ts = ts[~dup_mask].reset_index(drop=True)
        vals = vals[~dup_mask].reset_index(drop=True)
        warnings.append(
            f"{duplicate_timestamp_count} true duplicate timestamp(s) within this series; kept first occurrence."
        )

    # Chronological ordering is already guaranteed by AnalysisSeries
    # construction, but re-assert it defensively — this function is the
    # dedicated preparation choke point and must never trust an upstream
    # invariant silently.
    if not ts.is_monotonic_increasing:
        order = ts.argsort(kind="stable")
        ts = ts.iloc[order].reset_index(drop=True)
        vals = vals.iloc[order].reset_index(drop=True)
        warnings.append("Timestamps were not chronologically ordered; re-sorted before analysis.")

    sampling_interval_seconds, sampling_interval_label, is_regular = _infer_sampling_regularity(ts)

    gap_count, largest_gap_seconds = _detect_gaps(ts, sampling_interval_seconds)
    if gap_count:
        warnings.append(
            f"{gap_count} gap(s) larger than 1.5x the typical sampling interval detected "
            f"(largest: {largest_gap_seconds:.0f}s)."
        )
    if missing_value_count:
        warnings.append(f"{missing_value_count} missing (NaN) observation(s) in the raw series were excluded, not filled.")
    if not is_regular:
        warnings.append(
            "Sampling interval is irregular or could not be determined; seasonal-naive forecasting "
            "and time-based integration (excess energy) will be unavailable or degraded."
        )

    prepared_sample_count = int(len(vals))
    sufficient = prepared_sample_count >= min_samples
    if not sufficient:
        warnings.append(f"Only {prepared_sample_count} usable observation(s); minimum required is {min_samples}.")

    return PreparedSeries(
        series_id=series.series_id,
        role=series.role,
        unit=series.unit,
        timestamps=ts,
        values=vals,
        sampling_interval_seconds=sampling_interval_seconds,
        sampling_interval_label=sampling_interval_label,
        is_regular=is_regular,
        gap_count=gap_count,
        largest_gap_seconds=largest_gap_seconds,
        missing_value_count=missing_value_count,
        duplicate_timestamp_count=duplicate_timestamp_count,
        original_sample_count=original_sample_count,
        prepared_sample_count=prepared_sample_count,
        sufficient=sufficient,
        warnings=warnings,
    )


def _infer_sampling_regularity(ts: pd.Series, regularity_tolerance: float = 0.2) -> tuple:
    """
    Returns (median_interval_seconds, human_label, is_regular).
    `is_regular` requires the interquartile spread of intervals to be
    small relative to the median (within `regularity_tolerance`) — a
    documented, simple test, not a guess.
    """
    if len(ts) < 3:
        return None, None, False

    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None, None, False

    seconds = diffs.dt.total_seconds()
    median = float(seconds.median())
    if median <= 0:
        return None, None, False

    q1, q3 = seconds.quantile(0.25), seconds.quantile(0.75)
    iqr = q3 - q1
    is_regular = bool((iqr / median) <= regularity_tolerance) if median else False

    if median < 60:
        label = f"{median:.0f} seconds"
    elif median < 3600:
        label = f"{median / 60:.0f} minutes"
    elif median < 86400:
        label = f"{median / 3600:.0f} hours"
    else:
        label = f"{median / 86400:.1f} days"

    return median, label, is_regular


def _detect_gaps(ts: pd.Series, median_interval_seconds: Optional[float]) -> tuple:
    if median_interval_seconds is None or len(ts) < 2:
        return 0, None
    diffs = ts.diff().dropna().dt.total_seconds()
    threshold = median_interval_seconds * 1.5
    gaps = diffs[diffs > threshold]
    if gaps.empty:
        return 0, None
    return int(len(gaps)), float(gaps.max())


@dataclass
class TrainValidationTestSplit:
    """
    Chronological 3-way split. VALIDATION is used exclusively for model
    SELECTION (Part F); TEST remains completely untouched until final,
    already-selected-model evaluation. Order is always
    train -> validation -> test by position (never shuffled).
    """

    series_id: str
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    train_timestamps: pd.Series
    train_values: pd.Series
    validation_timestamps: pd.Series
    validation_values: pd.Series
    test_timestamps: pd.Series
    test_values: pd.Series
    train_start: Optional[str]
    train_end: Optional[str]
    validation_start: Optional[str]
    validation_end: Optional[str]
    test_start: Optional[str]
    test_end: Optional[str]
    train_samples: int
    validation_samples: int
    test_samples: int
    warnings: list[str] = field(default_factory=list)


def chronological_train_validation_test_split(
    prepared: PreparedSeries,
    train_ratio: float = DEFAULT_TRAIN_VAL_TEST_RATIOS[0],
    validation_ratio: float = DEFAULT_TRAIN_VAL_TEST_RATIOS[1],
    test_ratio: Optional[float] = None,
) -> TrainValidationTestSplit:
    """
    Chronological (never random) 3-way split: TRAIN -> VALIDATION -> TEST,
    strictly in that time order. Default 70/15/15. `test_ratio` defaults
    to whatever remains after train+validation; pass it explicitly to
    validate all three sum to 1.0.
    """
    if test_ratio is None:
        test_ratio = round(1.0 - train_ratio - validation_ratio, 6)

    for name, r in (("train_ratio", train_ratio), ("validation_ratio", validation_ratio), ("test_ratio", test_ratio)):
        if not (0.0 < r < 1.0):
            raise TimeSeriesPrepError(f"{name} must be strictly between 0 and 1; got {r}.")
    total = round(train_ratio + validation_ratio + test_ratio, 6)
    if abs(total - 1.0) > 1e-6:
        raise TimeSeriesPrepError(
            f"train_ratio + validation_ratio + test_ratio must sum to 1.0; got {total} "
            f"({train_ratio} + {validation_ratio} + {test_ratio})."
        )

    n = len(prepared.values)
    if n < 3:
        raise TimeSeriesPrepError(
            f"Series '{prepared.series_id}' has too few observations ({n}) for a train/validation/test split."
        )

    train_end_idx = int(round(n * train_ratio))
    val_end_idx = int(round(n * (train_ratio + validation_ratio)))

    # Guarantee every segment gets at least 1 sample, preserving order.
    train_end_idx = max(1, min(n - 2, train_end_idx))
    val_end_idx = max(train_end_idx + 1, min(n - 1, val_end_idx))

    def _slice(a, b):
        return (
            prepared.timestamps.iloc[a:b].reset_index(drop=True),
            prepared.values.iloc[a:b].reset_index(drop=True),
        )

    train_ts, train_vals = _slice(0, train_end_idx)
    val_ts, val_vals = _slice(train_end_idx, val_end_idx)
    test_ts, test_vals = _slice(val_end_idx, n)

    warnings: list[str] = []
    if len(val_vals) < 5:
        warnings.append(f"Validation period has only {len(val_vals)} observation(s); model selection may be unstable.")
    if len(test_vals) < 5:
        warnings.append(f"Test period has only {len(test_vals)} observation(s); final metrics may be unstable.")

    return TrainValidationTestSplit(
        series_id=prepared.series_id,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        train_timestamps=train_ts,
        train_values=train_vals,
        validation_timestamps=val_ts,
        validation_values=val_vals,
        test_timestamps=test_ts,
        test_values=test_vals,
        train_start=str(train_ts.iloc[0]) if len(train_ts) else None,
        train_end=str(train_ts.iloc[-1]) if len(train_ts) else None,
        validation_start=str(val_ts.iloc[0]) if len(val_ts) else None,
        validation_end=str(val_ts.iloc[-1]) if len(val_ts) else None,
        test_start=str(test_ts.iloc[0]) if len(test_ts) else None,
        test_end=str(test_ts.iloc[-1]) if len(test_ts) else None,
        train_samples=int(len(train_vals)),
        validation_samples=int(len(val_vals)),
        test_samples=int(len(test_vals)),
        warnings=warnings,
    )


def chronological_train_test_split(
    prepared: PreparedSeries, train_ratio: float = DEFAULT_TRAIN_RATIO
) -> TrainTestSplit:
    """
    Part C + Part D: chronological (NEVER random) split. The first
    `train_ratio` fraction of observations (by position, since the
    series is already chronologically ordered) becomes training data;
    the remainder becomes the held-out test period. This is the ONLY
    place a train/test boundary is established — every downstream
    model-fitting and threshold-fitting step must use `train_values`
    only, never peek at `test_values` (see docs/RESEARCH_SPEC.md, Data
    Leakage Rule).
    """
    if not (0.0 < train_ratio < 1.0):
        raise TimeSeriesPrepError(f"train_ratio must be strictly between 0 and 1; got {train_ratio}.")

    n = len(prepared.values)
    if n < 2:
        raise TimeSeriesPrepError(f"Series '{prepared.series_id}' has too few observations ({n}) to split.")

    split_idx = int(round(n * train_ratio))
    split_idx = max(1, min(n - 1, split_idx))  # guarantee both sides non-empty

    train_ts = prepared.timestamps.iloc[:split_idx].reset_index(drop=True)
    train_vals = prepared.values.iloc[:split_idx].reset_index(drop=True)
    test_ts = prepared.timestamps.iloc[split_idx:].reset_index(drop=True)
    test_vals = prepared.values.iloc[split_idx:].reset_index(drop=True)

    warnings: list[str] = []
    if len(test_vals) < 5:
        warnings.append(f"Test period has only {len(test_vals)} observation(s); metrics may be unstable.")

    return TrainTestSplit(
        series_id=prepared.series_id,
        train_ratio=train_ratio,
        train_timestamps=train_ts,
        train_values=train_vals,
        test_timestamps=test_ts,
        test_values=test_vals,
        train_start=str(train_ts.iloc[0]) if len(train_ts) else None,
        train_end=str(train_ts.iloc[-1]) if len(train_ts) else None,
        test_start=str(test_ts.iloc[0]) if len(test_ts) else None,
        test_end=str(test_ts.iloc[-1]) if len(test_ts) else None,
        train_samples=int(len(train_vals)),
        test_samples=int(len(test_vals)),
        warnings=warnings,
    )
