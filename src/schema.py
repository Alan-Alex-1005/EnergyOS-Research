"""
schema.py — Day 1: Timestamp Detection & Schema Understanding

Detects the timestamp column (if any) using name hints, dtype, and
parseability — WITHOUT ever fallback-parsing plain numeric columns as
dates. Also derives sampling interval, start/end range, monotonicity,
and duplicate-timestamp counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

TIMESTAMP_NAME_HINTS = [
    "timestamp",
    "datetime",
    "date_time",
    "date time",
    "date",
    "time",
    "recordedat",
    "recorded_at",
    "readingtime",
    "reading_time",
    "ts",
    "dt",
]


@dataclass
class TimestampInfo:
    detected: bool
    column: Optional[str] = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    parsed_series: Optional[pd.Series] = None
    start: Optional[str] = None
    end: Optional[str] = None
    sampling_interval: Optional[str] = None
    sampling_interval_seconds: Optional[float] = None
    is_monotonic: Optional[bool] = None
    duplicate_timestamp_count: int = 0
    is_timezone_aware: Optional[bool] = None
    assumed_timezone: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


def _name_hint_score(col_name: str) -> float:
    lname = col_name.strip().lower().replace("-", "_")
    lname_nospace = lname.replace(" ", "_")
    for hint in TIMESTAMP_NAME_HINTS:
        hint_norm = hint.replace(" ", "_")
        if lname_nospace == hint_norm:
            return 1.0
        if hint_norm in lname_nospace:
            return 0.7
    return 0.0


def _is_pure_numeric(series: pd.Series) -> bool:
    """True if the column is an int/float dtype (not datetime, not bool)."""
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )


def _has_explicit_timezone(series: pd.Series) -> bool:
    """
    Return True if any non-null string value ends with an explicit timezone
    marker such as Z, +01:00, +0200, -05:00, etc.
    """
    non_null = series.dropna()
    if non_null.empty:
        return False

    text = non_null.astype(str).str.strip()
    return bool(
        text.str.contains(
            r"(?:Z|[+-]\d{2}:?\d{2})$",
            regex=True,
            na=False,
        ).any()
    )


def _try_parse_datetime(series: pd.Series) -> tuple[Optional[pd.Series], float]:
    """
    Attempt to parse a column as datetimes.

    Rules:
    - Genuine datetime64 columns are preserved exactly as supplied.
    - Pure numeric columns are never interpreted as timestamps.
    - Timezone-naive string timestamps remain timezone-naive.
    - Explicit offset-aware strings are normalized to UTC so DST-changing
      offsets such as +01:00 and +02:00 can coexist safely.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, 1.0

    if _is_pure_numeric(series):
        return None, 0.0

    non_null = series.dropna()
    if non_null.empty:
        return None, 0.0

    explicit_timezone = _has_explicit_timezone(series)

    try:
        if explicit_timezone:
            parsed = pd.to_datetime(
                series,
                errors="coerce",
                format="mixed",
                utc=True,
            )
        else:
            parsed = pd.to_datetime(
                series,
                errors="coerce",
                format="mixed",
            )
    except (ValueError, TypeError):
        try:
            if explicit_timezone:
                parsed = pd.to_datetime(
                    series,
                    errors="coerce",
                    utc=True,
                )
            else:
                parsed = pd.to_datetime(
                    series,
                    errors="coerce",
                )
        except Exception:  # noqa: BLE001
            return None, 0.0

    success_ratio = (
        parsed.notna().sum() / len(non_null)
        if len(non_null)
        else 0.0
    )
    return parsed, float(success_ratio)


def detect_timestamp(df: pd.DataFrame) -> TimestampInfo:
    """
    Detect the most likely timestamp column in `df`.

    Strategy:
      1. name hints
      2. actual datetime64 dtype
      3. parseability of string/object columns

    Numeric (int/float) columns are NEVER treated as timestamps unless
    they are already a genuine datetime dtype.
    """
    candidates = []

    for col in df.columns:
        series = df[col]
        name_score = _name_hint_score(str(col))
        already_datetime = pd.api.types.is_datetime64_any_dtype(series)

        if _is_pure_numeric(series) and not already_datetime:
            continue

        parsed, parse_ratio = _try_parse_datetime(series)
        if parsed is None or parse_ratio == 0.0:
            continue

        evidence = []
        if already_datetime:
            evidence.append("column dtype is already datetime64")
        if name_score > 0:
            evidence.append(
                f"column name matches timestamp naming pattern (score={name_score})"
            )
        if parse_ratio > 0:
            evidence.append(
                f"{parse_ratio * 100:.1f}% of values parse as valid datetimes"
            )

        confidence = round(0.4 * name_score + 0.6 * parse_ratio, 3)
        if name_score == 0.0 and parse_ratio >= 0.99:
            confidence = max(confidence, 0.55)

        candidates.append((confidence, col, parsed, evidence, parse_ratio))

    if not candidates:
        return TimestampInfo(
            detected=False,
            warnings=["No timestamp column detected."],
        )

    candidates.sort(key=lambda c: c[0], reverse=True)
    confidence, col, parsed, evidence, parse_ratio = candidates[0]

    warnings = []
    if confidence < 0.5:
        warnings.append(
            f"Low-confidence timestamp detection ({confidence:.2f}) for column '{col}'; "
            "consider an explicit config override."
        )

    if parse_ratio < 1.0:
        warnings.append(
            f"{(1 - parse_ratio) * 100:.1f}% of values in '{col}' failed to parse as timestamps "
            "and will be treated as INVALID_TIMESTAMP during cleaning."
        )

    valid = parsed.dropna()
    is_tz_aware = None
    assumed_timezone = None

    if not valid.empty:
        try:
            is_tz_aware = valid.dt.tz is not None
        except (AttributeError, TypeError):
            is_tz_aware = False

        if not is_tz_aware:
            warnings.append(
                "Timestamps are timezone-naive; preserved as naive (no timezone was invented)."
            )

    start = str(valid.min()) if not valid.empty else None
    end = str(valid.max()) if not valid.empty else None
    is_monotonic = bool(valid.is_monotonic_increasing) if len(valid) > 1 else None
    duplicate_ts_count = (
        int(valid.duplicated(keep="first").sum())
        if not valid.empty
        else 0
    )

    sampling_interval, sampling_seconds = _infer_sampling_interval(valid)

    return TimestampInfo(
        detected=True,
        column=str(col),
        confidence=confidence,
        evidence=evidence,
        parsed_series=parsed,
        start=start,
        end=end,
        sampling_interval=sampling_interval,
        sampling_interval_seconds=sampling_seconds,
        is_monotonic=is_monotonic,
        duplicate_timestamp_count=duplicate_ts_count,
        is_timezone_aware=is_tz_aware,
        assumed_timezone=assumed_timezone,
        warnings=warnings,
    )


def _infer_sampling_interval(valid_timestamps: pd.Series) -> tuple:
    if len(valid_timestamps) < 3:
        return None, None

    sorted_ts = valid_timestamps.sort_values()
    diffs = sorted_ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]

    if diffs.empty:
        return None, None

    median_diff = diffs.median()
    seconds = median_diff.total_seconds()

    if seconds < 60:
        label = f"{seconds:.0f} seconds"
    elif seconds < 3600:
        label = f"{seconds / 60:.0f} minutes"
    elif seconds < 86400:
        label = f"{seconds / 3600:.0f} hours"
    else:
        label = f"{seconds / 86400:.1f} days"

    return label, seconds
