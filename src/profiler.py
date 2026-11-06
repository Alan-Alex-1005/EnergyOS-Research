"""
profiler.py — Day 1: Automatic Data Profiler

Produces a structured Profile object describing the shape, types, and
statistical characteristics of a dataframe, without assuming which
columns mean what (that's role_detection.py's job).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    null_ratio: float
    unique_count: int
    cardinality_ratio: float  # unique_count / row_count
    is_numeric: bool
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    numeric_mean: Optional[float] = None
    suspected_categorical: bool = False
    suspected_identifier: bool = False
    suspected_binary_status: bool = False
    suspected_timestamp: bool = False
    sample_values: list = field(default_factory=list)


@dataclass
class Profile:
    row_count: int
    column_count: int
    column_names: list[str]
    columns: dict[str, ColumnProfile]
    duplicate_row_count: int
    format_hint: str  # "WIDE" or "LONG" or "UNKNOWN"
    long_format_columns: dict  # detected timestamp/id/metric/value columns if LONG


def _numeric_stats(series: pd.Series) -> tuple:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return None, None, None
    return float(valid.min()), float(valid.max()), float(valid.mean())


def _profile_column(name: str, series: pd.Series, row_count: int) -> ColumnProfile:
    null_count = int(series.isna().sum())
    null_ratio = null_count / row_count if row_count else 0.0
    non_null = series.dropna()
    unique_count = int(non_null.nunique())
    cardinality_ratio = unique_count / row_count if row_count else 0.0

    is_numeric = pd.api.types.is_numeric_dtype(series)
    numeric_min = numeric_max = numeric_mean = None
    if is_numeric:
        numeric_min, numeric_max, numeric_mean = _numeric_stats(series)

    # Suspected identifier: near-unique, non-numeric-ish or high-cardinality strings/ints
    suspected_identifier = (
        row_count > 1
        and cardinality_ratio > 0.9
        and null_ratio < 0.5
        and not _looks_like_free_float(series)
    )

    # Suspected categorical: low cardinality relative to row count
    suspected_categorical = (
        row_count > 1
        and 1 < unique_count <= max(20, int(row_count * 0.05))
        and not suspected_identifier
    )

    # Suspected binary status column: exactly 2 (or 1) distinct values,
    # values look like on/off, 0/1, true/false, yes/no
    suspected_binary_status = _looks_like_status(non_null)

    sample_values = non_null.unique()[:5].tolist()

    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        null_count=null_count,
        null_ratio=round(null_ratio, 4),
        unique_count=unique_count,
        cardinality_ratio=round(cardinality_ratio, 4),
        is_numeric=is_numeric,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
        numeric_mean=numeric_mean,
        suspected_categorical=suspected_categorical,
        suspected_identifier=suspected_identifier,
        suspected_binary_status=suspected_binary_status,
        sample_values=sample_values,
    )


def _looks_like_free_float(series: pd.Series) -> bool:
    """True if column is a continuous-looking float (not an ID)."""
    if not pd.api.types.is_float_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    # Continuous measurements rarely have every value unique by coincidence
    # unless it's genuinely a measurement; treat float columns as non-ID.
    return True


_STATUS_TOKENS = {
    "on",
    "off",
    "0",
    "1",
    "true",
    "false",
    "yes",
    "no",
    "open",
    "closed",
    "active",
    "inactive",
    "running",
    "stopped",
}


def _looks_like_status(series: pd.Series) -> bool:
    if series.empty:
        return False
    unique_vals = series.unique()
    if len(unique_vals) > 4:
        return False
    tokens = {str(v).strip().lower() for v in unique_vals}
    if tokens.issubset(_STATUS_TOKENS) and len(tokens) > 0:
        return True
    # numeric 0/1 only
    try:
        numeric_tokens = {float(t) for t in tokens}
        if numeric_tokens.issubset({0.0, 1.0}):
            return True
    except (ValueError, TypeError):
        pass
    return False


def _detect_duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated(keep="first").sum())


def _detect_format(df: pd.DataFrame, columns: dict[str, ColumnProfile]) -> tuple:
    """
    Heuristic LONG vs WIDE detection.

    LONG format signature: a small-cardinality "metric"/"parameter"-like
    categorical column paired with a single generic "value" numeric column,
    typically alongside an id column (meter/device) and a timestamp.

    WIDE format signature: multiple distinct numeric measurement columns
    as separate columns.
    """
    candidate_metric_cols = []
    candidate_value_cols = []
    candidate_unit_cols = []
    for name, col in columns.items():
        lname = name.lower()
        if col.suspected_categorical and lname in (
            "metric",
            "parameter",
            "measurement",
            "type",
            "field",
            "variable",
        ):
            candidate_metric_cols.append(name)
        if col.is_numeric and lname in ("value", "reading", "val", "measurement_value"):
            candidate_value_cols.append(name)
        # Optional unit column (section 6/9): only ever used if the source
        # data actually carries one — never fabricated if absent.
        if lname in ("unit", "units", "uom", "unit_of_measure"):
            candidate_unit_cols.append(name)

    if candidate_metric_cols and candidate_value_cols:
        long_format_columns = {
            "metric_column": candidate_metric_cols[0],
            "value_column": candidate_value_cols[0],
        }
        if candidate_unit_cols:
            long_format_columns["unit_column"] = candidate_unit_cols[0]
        return "LONG", long_format_columns

    # Otherwise: count "real" numeric measurement-looking columns
    numeric_measure_cols = [
        name
        for name, col in columns.items()
        if col.is_numeric and not col.suspected_identifier and not col.suspected_binary_status
    ]
    if len(numeric_measure_cols) >= 1:
        return "WIDE", {}

    return "UNKNOWN", {}


def profile_dataframe(df: pd.DataFrame) -> Profile:
    row_count = int(df.shape[0])
    column_count = int(df.shape[1])
    column_names = list(df.columns.astype(str))

    columns = {
        name: _profile_column(name, df[name], row_count) for name in df.columns
    }

    duplicate_row_count = _detect_duplicate_rows(df) if row_count > 0 else 0
    format_hint, long_format_columns = _detect_format(df, columns)

    return Profile(
        row_count=row_count,
        column_count=column_count,
        column_names=column_names,
        columns=columns,
        duplicate_row_count=duplicate_row_count,
        format_hint=format_hint,
        long_format_columns=long_format_columns,
    )
