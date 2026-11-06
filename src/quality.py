"""
quality.py — Day 1: Data Quality Score

A simple, fully-documented 0-100 score built from six weighted factors.
Nothing here is an opaque AI-generated number — every factor and weight
is listed below and returned in the breakdown (spec section 12).

    Overall = 25% * Timestamp Validity
            + 20% * Mapping Coverage
            + 20% * Completeness
            + 15% * Duplicate Quality
            + 10% * Unit Confidence
            + 10% * Temporal Consistency

Each factor is itself a plain percentage (0-100) computed from counts
already produced by profiler.py / schema.py / mapping.py / cleaning.py —
no hidden heuristics.

Day 2 Part A1/A2 fix: for LONG-format datasets, a physical column like
"metric" or "value" is intrinsically structural — it can never itself
carry a "role", because the role lives in each individual metric VALUE
(see long_format.py). Scoring those columns as "unresolved" the same way
an ordinary unmapped WIDE column would be scored made a fully-resolved
LONG dataset look artificially mediocre. When an applicable
`LongFormatMappingResult` is supplied, four of the six factors
(mapping_coverage, completeness, unit_confidence, temporal_consistency)
are computed from the resolved LOGICAL SERIES instead of physical
columns. This does NOT inflate scores for datasets that genuinely have
unresolved metrics — an unresolved metric still counts against coverage,
exactly as an unmapped column would. WIDE-format behavior (no
long_format_mapping, or one with applicable=False) is completely
unchanged from the original per-column formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .cleaning import CleaningReport
from .mapping import MappingResult
from .profiler import Profile
from .schema import TimestampInfo
from . import units as units_mod

WEIGHTS = {
    "timestamp_validity": 0.25,
    "mapping_coverage": 0.20,
    "completeness": 0.20,
    "duplicate_quality": 0.15,
    "unit_confidence": 0.10,
    "temporal_consistency": 0.10,
}

MEASURE_ROLES_REQUIRING_UNITS = {
    "POWER",
    "ENERGY",
    "DEMAND",
    "VOLTAGE",
    "CURRENT",
    "FREQUENCY",
    "TEMPERATURE",
    "HUMIDITY",
}


@dataclass
class QualityResult:
    overall: float
    factors: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))
    status: str = "UNKNOWN"


def _status_for(score: float) -> str:
    if score >= 85:
        return "GOOD"
    if score >= 60:
        return "FAIR"
    return "POOR"


def compute_quality(
    profile: Profile,
    timestamp_info: TimestampInfo,
    mapping_result: MappingResult,
    cleaning_report: CleaningReport,
    long_format_mapping: Optional[object] = None,
) -> QualityResult:
    rows_before = max(cleaning_report.rows_before, 1)
    rows_after = max(cleaning_report.rows_after, 1)

    use_long = bool(long_format_mapping is not None and getattr(long_format_mapping, "applicable", False))

    # 1. Timestamp validity: share of original rows whose timestamp parsed cleanly.
    #    (Row-level concept; identical for WIDE and LONG — unchanged.)
    if timestamp_info.detected:
        invalid = cleaning_report.action_counts.get("INVALID_TIMESTAMP", 0)
        timestamp_validity = 100.0 * (1 - invalid / rows_before)
    else:
        timestamp_validity = 0.0

    if use_long:
        mapping_coverage = _long_mapping_coverage(profile, mapping_result, long_format_mapping)
        completeness = _long_completeness(long_format_mapping)
        unit_confidence = _long_unit_confidence(long_format_mapping)
    else:
        # 2. Mapping coverage: share of all columns that received a role mapping.
        total_columns = max(profile.column_count, 1)
        mapping_coverage = 100.0 * (mapping_result.mapped_count / total_columns)

        # 3. Completeness: share of mapped (non-timestamp) cells that are non-null.
        non_ts_mapped_cols = [c for c, m in mapping_result.mapped.items() if m.role != "TIMESTAMP"]
        if non_ts_mapped_cols:
            total_cells = len(non_ts_mapped_cols) * rows_after
            total_missing = sum(cleaning_report.per_column_missing.values())
            completeness = 100.0 * (1 - (total_missing / total_cells)) if total_cells else 100.0
        else:
            completeness = 0.0

        # 5. Unit confidence: share of unit-bearing measure columns with a known unit.
        unit_bearing = [
            m for m in mapping_result.mapped.values() if m.role in MEASURE_ROLES_REQUIRING_UNITS
        ]
        if unit_bearing:
            known = sum(1 for m in unit_bearing if units_mod.is_known_unit(m.unit, m.role))
            unit_confidence = 100.0 * (known / len(unit_bearing))
        else:
            unit_confidence = 100.0  # no unit-bearing columns => nothing to penalize

    # 4. Duplicate quality: share of original rows that were NOT exact duplicates.
    #    (Row-level concept; identical for WIDE and LONG — unchanged.)
    removed_dupes = cleaning_report.action_counts.get("REMOVED_DUPLICATE", 0)
    duplicate_quality = 100.0 * (1 - removed_dupes / rows_before)

    # 6. Temporal consistency.
    if use_long:
        temporal_consistency = _long_temporal_consistency(long_format_mapping)
    elif timestamp_info.detected and rows_after > 0:
        dup_ts_ratio = timestamp_info.duplicate_timestamp_count / rows_after
        temporal_consistency = 100.0 * (1 - min(1.0, dup_ts_ratio))
        if timestamp_info.is_monotonic is False:
            temporal_consistency = max(0.0, temporal_consistency - 20.0)
    else:
        temporal_consistency = 0.0

    factors = {
        "timestamp_validity": round(max(0.0, min(100.0, timestamp_validity)), 1),
        "mapping_coverage": round(max(0.0, min(100.0, mapping_coverage)), 1),
        "completeness": round(max(0.0, min(100.0, completeness)), 1),
        "duplicate_quality": round(max(0.0, min(100.0, duplicate_quality)), 1),
        "unit_confidence": round(max(0.0, min(100.0, unit_confidence)), 1),
        "temporal_consistency": round(max(0.0, min(100.0, temporal_consistency)), 1),
    }

    overall = sum(factors[k] * WEIGHTS[k] for k in WEIGHTS)
    overall = round(overall, 1)

    return QualityResult(overall=overall, factors=factors, weights=dict(WEIGHTS), status=_status_for(overall))


def _long_mapping_coverage(profile: Profile, mapping_result: MappingResult, long_format_mapping) -> float:
    """
    Blend physical-column coverage (timestamp, identifiers, context — the
    things that ARE resolvable column-by-column) with metric-value
    coverage (the things LONG format resolves per-value, not per-column).
    The "metric"/"value"/"unit" columns themselves are excluded from the
    physical-column denominator since they have no column-level role by
    design — their resolution is measured via metric coverage instead.
    """
    structural_cols = {
        c for c in (long_format_mapping.metric_column, long_format_mapping.value_column, long_format_mapping.unit_column)
        if c
    }
    physical_total = max(profile.column_count - len(structural_cols), 1)
    physical_resolved = min(mapping_result.mapped_count, physical_total)

    role_by_metric: dict[str, str] = {}
    for s in long_format_mapping.series:
        role_by_metric.setdefault(s.metric_value, s.role)
    total_metrics = max(len(role_by_metric), 1)
    resolved_metrics = sum(1 for r in role_by_metric.values() if r != "UNKNOWN")

    total_items = physical_total + total_metrics
    resolved_items = physical_resolved + resolved_metrics
    return 100.0 * (resolved_items / total_items) if total_items else 100.0


def _long_completeness(long_format_mapping) -> float:
    """Null ratio among RESOLVED metric-value canonical rows only —
    unresolved metrics are a coverage concern, not a completeness one."""
    canonical_df = long_format_mapping.canonical_df
    if canonical_df is None or canonical_df.empty:
        return 0.0
    total_cells = len(canonical_df)
    missing = int(canonical_df["raw_value"].isna().sum())
    return 100.0 * (1 - (missing / total_cells)) if total_cells else 100.0


def _long_unit_confidence(long_format_mapping) -> float:
    unit_bearing = [
        s for s in long_format_mapping.series if s.role in MEASURE_ROLES_REQUIRING_UNITS and s.role != "UNKNOWN"
    ]
    if not unit_bearing:
        return 100.0
    known = sum(1 for s in unit_bearing if units_mod.is_known_unit(s.unit, s.role))
    return 100.0 * (known / len(unit_bearing))


def _long_temporal_consistency(long_format_mapping) -> float:
    """
    Per-series true duplicate-timestamp ratio (spec Part A2): a repeat
    timestamp across two DIFFERENT logical series (different metric
    and/or meter) is normal and never enters this calculation — only
    `LogicalSeriesMapping.duplicate_timestamp_count`, which is computed
    strictly WITHIN each series' own rows, does. Weighted by row count
    so a large series' duplicates aren't swamped by many small clean
    ones (or vice versa). Assumes each series' rows are already in
    timestamp order (guaranteed by cleaning.py's dataset-wide sort, of
    which every series is a sub-sequence) — monotonicity is not
    re-checked per series.
    """
    resolved = [s for s in long_format_mapping.series if s.role != "UNKNOWN"]
    total_rows = sum(s.row_count for s in resolved)
    if total_rows <= 0:
        return 0.0
    total_dupes = sum(s.duplicate_timestamp_count for s in resolved)
    return 100.0 * (1 - min(1.0, total_dupes / total_rows))
