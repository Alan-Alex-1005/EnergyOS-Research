"""
cleaning.py — Day 1: Cleaning / Validation

Conservative cleaning: rows are only dropped for clearly-defined reasons
(exact duplicate rows, unparseable timestamps). Missing values are NEVER
silently filled. Out-of-range numeric values are flagged, not altered.
Every action is classified and counted in a CleaningReport so nothing
happens silently (spec section 11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .mapping import MappingResult
from .schema import TimestampInfo

ACTION_TYPES = [
    "REMOVED_DUPLICATE",
    "INVALID_TIMESTAMP",
    "MISSING_VALUE",
    "OUT_OF_RANGE",
    "UNRESOLVED",
]

# Conservative domain range rules: (min, max) inclusive. None = unbounded.
_RANGE_RULES = {
    "POWER": (0, None),
    "ENERGY": (0, None),
    "DEMAND": (0, None),
    "VOLTAGE": (0, None),
    "CURRENT": (0, None),
    "FREQUENCY": (0, None),
    "HUMIDITY": (0, 100),
}


@dataclass
class CleaningReport:
    rows_before: int
    rows_after: int
    action_counts: dict[str, int]
    per_column_missing: dict[str, int]
    per_column_out_of_range: dict[str, int]
    notes: list[str] = field(default_factory=list)


def clean_dataset(
    df: pd.DataFrame,
    timestamp_info: TimestampInfo,
    mapping_result: MappingResult,
) -> tuple[pd.DataFrame, CleaningReport]:
    rows_before = int(df.shape[0])
    action_counts = {a: 0 for a in ACTION_TYPES}
    notes: list[str] = []
    working = df.copy()

    # 1. Attach parsed timestamp column if one was detected, and drop rows
    #    whose timestamp failed to parse (INVALID_TIMESTAMP).
    if timestamp_info.detected and timestamp_info.parsed_series is not None:
        working["_parsed_timestamp"] = timestamp_info.parsed_series.values
        invalid_mask = working["_parsed_timestamp"].isna()
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            action_counts["INVALID_TIMESTAMP"] = invalid_count
            notes.append(
                f"{invalid_count} row(s) had an unparseable value in timestamp column "
                f"'{timestamp_info.column}' and were excluded from the canonical dataset."
            )
        working = working.loc[~invalid_mask].copy()

    # 2. Remove exact duplicate rows (based on original columns only).
    original_cols = [c for c in df.columns]
    dup_mask = working.duplicated(subset=original_cols, keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count:
        action_counts["REMOVED_DUPLICATE"] = dup_count
        notes.append(f"{dup_count} exact duplicate row(s) removed (kept first occurrence).")
    working = working.loc[~dup_mask].copy()

    # 3. Duplicate timestamps: a row that shares a timestamp with another
    #    row is only a problem if they also share the same logical series.
    #    Repeated timestamps across DIFFERENT meters/devices/equipment are
    #    normal (spec Day-2 Part A2) — e.g. two meters both reporting at
    #    09:00 is not a data-quality issue. We only have identifier-column
    #    granularity at this stage (EQUIPMENT_ID/DEVICE_ID/METER_ID); LONG
    #    format's additional per-METRIC granularity isn't resolved until
    #    long_format.py runs later in the pipeline, so a LONG dataset can
    #    still show some over-counting here — the authoritative per-series
    #    figure is `long_format.LogicalSeriesMapping.duplicate_timestamp_count`
    #    and quality.py's LONG-aware temporal-consistency factor, both of
    #    which have true metric-level granularity. This is documented, not
    #    hidden (see docs/ARCHITECTURE.md).
    identifier_cols = [
        col
        for col, m in mapping_result.mapped.items()
        if m.role in ("EQUIPMENT_ID", "DEVICE_ID", "METER_ID") and col in working.columns
    ]
    if timestamp_info.detected and "_parsed_timestamp" in working.columns:
        if identifier_cols:
            dup_ts_mask = working.duplicated(subset=identifier_cols + ["_parsed_timestamp"], keep=False)
            dup_ts_count = int(dup_ts_mask.sum())
            scope_note = f"within the same series (grouped by {', '.join(identifier_cols)})"
        else:
            dup_ts_count = int(working["_parsed_timestamp"].duplicated(keep=False).sum())
            scope_note = "dataset-wide (no equipment/device/meter identifier column was mapped)"
        if dup_ts_count:
            action_counts["UNRESOLVED"] += dup_ts_count
            notes.append(
                f"{dup_ts_count} row(s) share a timestamp with another row {scope_note}. "
                f"Left unresolved — may indicate a genuine conflict."
            )

    # 4. Missing values in mapped (non-timestamp) columns — counted, never filled.
    per_column_missing: dict[str, int] = {}
    for col, m in mapping_result.mapped.items():
        if m.role == "TIMESTAMP":
            continue
        if col not in working.columns:
            continue
        missing = int(working[col].isna().sum())
        if missing:
            per_column_missing[col] = missing
            action_counts["MISSING_VALUE"] += missing

    # 5. Out-of-range numeric values for roles with clear domain rules.
    per_column_out_of_range: dict[str, int] = {}
    for col, m in mapping_result.mapped.items():
        rule = _RANGE_RULES.get(m.role)
        if rule is None or col not in working.columns:
            continue
        lo, hi = rule
        numeric = pd.to_numeric(working[col], errors="coerce")
        mask = pd.Series(False, index=numeric.index)
        if lo is not None:
            mask = mask | (numeric < lo)
        if hi is not None:
            mask = mask | (numeric > hi)
        mask = mask & numeric.notna()
        count = int(mask.sum())
        if count:
            per_column_out_of_range[col] = count
            action_counts["OUT_OF_RANGE"] += count
            notes.append(
                f"{count} value(s) in '{col}' (role {m.role}) fall outside the expected "
                f"range {lo}-{hi if hi is not None else '∞'}; values preserved as-is and flagged."
            )

    # 6. Sort by timestamp if available (stable sort; preserves row order
    #    for ties, which matters for multi-series long-format data).
    if timestamp_info.detected and "_parsed_timestamp" in working.columns:
        working = working.sort_values("_parsed_timestamp", kind="stable").reset_index(drop=True)
    else:
        working = working.reset_index(drop=True)

    rows_after = int(working.shape[0])

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=rows_after,
        action_counts=action_counts,
        per_column_missing=per_column_missing,
        per_column_out_of_range=per_column_out_of_range,
        notes=notes,
    )
    return working, report
