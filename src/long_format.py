"""
long_format.py — Day 1: Long-Format Semantic Mapping

WIDE-format role detection (role_detection.py) works at the physical
COLUMN level — one role per column. That breaks down for LONG-format
data:

    timestamp | meter_id | metric | value | unit

...where a single "metric" column's *values* (e.g. "power_kw",
"energy_kwh") each denote a different physical quantity sharing one
generic "value" column. A column-level role detector correctly (and
conservatively) leaves "metric" and "value" UNKNOWN — neither column
name means anything on its own. This module adds the missing piece:
semantic mapping at the METRIC-VALUE level, grouped into logical series
by (metric, identifier columns present).

Nothing here fabricates a role or unit that isn't grounded in the
metric text (or an explicit config override / unit column) — a metric
value that doesn't match any known pattern stays UNKNOWN and is left
unresolved, exactly like an unmapped physical column (spec section 18).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .canonical import CONTEXT_ROLES
from .mapping import MappingConfigError, MappingResult
from .profiler import Profile
from .role_detection import LOW_CONFIDENCE_THRESHOLD, ROLES, match_role_by_label
from . import units as units_mod

IDENTIFIER_ROLES = ("EQUIPMENT_ID", "DEVICE_ID", "METER_ID")

_CANONICAL_COLUMNS = (
    ["timestamp", "source_column", "role", "raw_value", "raw_unit", "normalized_value", "normalized_unit"]
    + [r.lower() for r in CONTEXT_ROLES]
    + ["quality", "mapping_confidence", "mapping_source"]
)


@dataclass
class LogicalSeriesMapping:
    """One resolved (or unresolved) logical series: a single metric value,
    optionally scoped to a specific meter/device/equipment identifier."""

    metric_value: str
    group_keys: dict[str, str] = field(default_factory=dict)  # e.g. {"meter_id": "M1"}
    role: str = "UNKNOWN"
    confidence: float = 0.0
    unit: Optional[str] = None
    source: str = "AUTO"  # "AUTO" or "CONFIG"
    evidence: list[str] = field(default_factory=list)
    row_count: int = 0
    duplicate_timestamp_count: int = 0


@dataclass
class LongFormatMappingResult:
    applicable: bool
    metric_column: Optional[str] = None
    value_column: Optional[str] = None
    unit_column: Optional[str] = None
    identifier_columns: list[str] = field(default_factory=list)
    series: list[LogicalSeriesMapping] = field(default_factory=list)
    unresolved_metrics: list[str] = field(default_factory=list)
    canonical_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_CANONICAL_COLUMNS))
    warnings: list[str] = field(default_factory=list)

    def resolved_roles(self) -> set[str]:
        """Roles contributed by resolved (non-UNKNOWN, non-low-confidence) logical series."""
        return {s.role for s in self.series if s.role != "UNKNOWN" and s.confidence >= LOW_CONFIDENCE_THRESHOLD}


def _validate_metric_overrides(metric_overrides: dict) -> None:
    for metric_val, spec in metric_overrides.items():
        if not isinstance(spec, dict) or "role" not in spec:
            raise MappingConfigError(
                f"Config 'metric_mappings' entry '{metric_val}' must be an object containing at least 'role'."
            )
        role = spec["role"]
        if role not in ROLES:
            raise MappingConfigError(
                f"Config 'metric_mappings' entry '{metric_val}' has invalid role '{role}'. "
                f"Valid roles: {', '.join(ROLES)}"
            )
        unit = spec.get("unit")
        if unit is not None and not isinstance(unit, str):
            raise MappingConfigError(f"Config 'metric_mappings' entry '{metric_val}' has non-string unit.")


def build_long_format_mapping(
    cleaned_df: pd.DataFrame,
    profile: Profile,
    mapping_result: MappingResult,
    config: Optional[dict] = None,
) -> LongFormatMappingResult:
    """
    Detect metric/value/unit columns from the profiler's LONG-format
    hint, group rows into logical series by (metric value, identifier
    columns already resolved at the physical-column level — meter/
    device/equipment id), and infer a role + unit per logical series.

    Returns a LongFormatMappingResult with applicable=False (and an
    empty canonical_df) when the dataset isn't LONG format, so callers
    can unconditionally merge its output without a format check.
    """
    empty_result = LongFormatMappingResult(applicable=False)

    if profile.format_hint != "LONG":
        return empty_result

    lf_cols = profile.long_format_columns or {}
    metric_col = lf_cols.get("metric_column")
    value_col = lf_cols.get("value_column")
    unit_col = lf_cols.get("unit_column")

    if not metric_col or not value_col:
        return empty_result
    if metric_col not in cleaned_df.columns or value_col not in cleaned_df.columns:
        return empty_result

    config = config or {}
    metric_overrides = config.get("metric_mappings", {}) or {}
    if not isinstance(metric_overrides, dict):
        raise MappingConfigError("Config 'metric_mappings' must be a JSON object of metric -> {role, unit}.")
    _validate_metric_overrides(metric_overrides)

    # Identifier columns already resolved at the physical-column level
    # (e.g. "meter_id" -> METER_ID) become logical-series grouping keys,
    # per spec: "grouped by metric + meter/device/equipment identifiers
    # where available".
    identifier_columns = [
        col for col, m in mapping_result.mapped.items() if m.role in IDENTIFIER_ROLES and col in cleaned_df.columns
    ]

    ts_col = "_parsed_timestamp" if "_parsed_timestamp" in cleaned_df.columns else mapping_result.timestamp_column

    group_cols = [metric_col] + identifier_columns
    series_list: list[LogicalSeriesMapping] = []
    unresolved_metrics: list[str] = []
    canonical_frames: list[pd.DataFrame] = []

    for group_key, group_df in cleaned_df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metric_value = str(group_key[0])
        group_keys = {identifier_columns[i]: str(group_key[i + 1]) for i in range(len(identifier_columns))}

        role, confidence, unit, source, evidence = _resolve_metric(
            metric_value, group_df, unit_col, metric_overrides
        )

        # True duplicate-timestamp count WITHIN this logical series (spec
        # Part A2): this is the metric-aware granularity cleaning.py can't
        # see (it runs before metric roles are resolved) — this is the
        # authoritative figure quality.py uses for LONG-format temporal
        # consistency.
        dup_ts_count = 0
        if ts_col in group_df.columns:
            dup_ts_count = int(group_df[ts_col].duplicated(keep="first").sum())

        series_entry = LogicalSeriesMapping(
            metric_value=metric_value,
            group_keys=group_keys,
            role=role,
            confidence=confidence,
            unit=unit,
            source=source,
            evidence=evidence,
            row_count=int(len(group_df)),
            duplicate_timestamp_count=dup_ts_count,
        )
        series_list.append(series_entry)

        if role == "UNKNOWN" or confidence < LOW_CONFIDENCE_THRESHOLD:
            unresolved_metrics.append(metric_value)
            continue  # never fabricate a canonical series for an unresolved metric

        canonical_frames.append(
            _build_canonical_frame(
                group_df=group_df,
                ts_col=ts_col,
                metric_value=metric_value,
                role=role,
                unit=unit,
                value_col=value_col,
                confidence=confidence,
                source=source,
                identifier_columns=identifier_columns,
                mapping_result=mapping_result,
            )
        )

    canonical_df = (
        pd.concat(canonical_frames, ignore_index=True) if canonical_frames else pd.DataFrame(columns=_CANONICAL_COLUMNS)
    )

    warnings: list[str] = []
    distinct_unresolved = sorted(set(unresolved_metrics))
    if distinct_unresolved:
        warnings.append(
            f"{len(distinct_unresolved)} metric value(s) in long-format column '{metric_col}' could not be "
            f"mapped to a known role and were left UNKNOWN: {distinct_unresolved}"
        )

    return LongFormatMappingResult(
        applicable=True,
        metric_column=metric_col,
        value_column=value_col,
        unit_column=unit_col,
        identifier_columns=identifier_columns,
        series=series_list,
        unresolved_metrics=distinct_unresolved,
        canonical_df=canonical_df,
        warnings=warnings,
    )


def _resolve_metric(
    metric_value: str,
    group_df: pd.DataFrame,
    unit_col: Optional[str],
    metric_overrides: dict,
) -> tuple[str, float, Optional[str], str, list[str]]:
    """Resolve (role, confidence, unit, source, evidence) for one metric value."""
    if metric_value in metric_overrides:
        spec = metric_overrides[metric_value]
        role = spec["role"]
        unit = spec.get("unit")
        confidence = 1.0
        source = "CONFIG"
        evidence = ["explicit 'metric_mappings' override in config"]
    else:
        suggestion = match_role_by_label(metric_value)
        role = suggestion.role
        confidence = suggestion.confidence
        source = "AUTO"
        evidence = list(suggestion.evidence)
        unit = suggestion.inferred_unit

    # An explicit unit column (if present) is trusted over a name-inferred
    # unit, but ONLY when it's consistent across the whole group — never
    # guessed when it's missing or ambiguous.
    if unit_col and unit_col in group_df.columns:
        observed_units = [u for u in group_df[unit_col].dropna().unique().tolist()]
        if len(observed_units) == 1:
            unit = str(observed_units[0])
            evidence = evidence + [f"unit column reports a consistent unit '{unit}' for this series"]
        elif len(observed_units) > 1:
            evidence = evidence + [
                f"unit column has inconsistent values {observed_units} for this series; unit left as name-inferred"
            ]

    return role, confidence, unit, source, evidence


def _build_canonical_frame(
    group_df: pd.DataFrame,
    ts_col: str,
    metric_value: str,
    role: str,
    unit: Optional[str],
    value_col: str,
    confidence: float,
    source: str,
    identifier_columns: list[str],
    mapping_result: MappingResult,
) -> pd.DataFrame:
    sub = pd.DataFrame()
    sub["timestamp"] = group_df[ts_col] if ts_col in group_df.columns else pd.NaT
    sub["source_column"] = metric_value
    sub["role"] = role
    sub["raw_value"] = group_df[value_col]
    sub["raw_unit"] = unit

    if unit and role in units_mod.CANONICAL_UNITS and units_mod.is_known_unit(unit, role):
        numeric_raw = pd.to_numeric(group_df[value_col], errors="coerce")
        conv = units_mod.normalize_value(1.0, unit, role)  # probe value to get the deterministic factor
        factor = conv.normalized_value if conv.converted else None
        if factor is not None:
            sub["normalized_value"] = numeric_raw * factor
            sub["normalized_unit"] = units_mod.CANONICAL_UNITS[role]
        else:
            sub["normalized_value"] = pd.NA
            sub["normalized_unit"] = pd.NA
    else:
        sub["normalized_value"] = pd.NA
        sub["normalized_unit"] = pd.NA

    # Context fields: identifier columns that are logical-series grouping
    # keys (meter/device/equipment) come straight from this group; any
    # other context role (site/building/floor/zone/room) mapped at the
    # physical-column level is carried through unchanged. Never fabricated.
    identifier_role_by_col = {col: mapping_result.mapped[col].role for col in identifier_columns}
    for role_name in CONTEXT_ROLES:
        field_name = role_name.lower()
        matching_id_col = next((c for c, r in identifier_role_by_col.items() if r == role_name), None)
        if matching_id_col is not None:
            sub[field_name] = group_df[matching_id_col].iloc[0] if len(group_df) else None
            continue
        phys_col = next((c for c, m in mapping_result.mapped.items() if m.role == role_name), None)
        sub[field_name] = group_df[phys_col] if phys_col and phys_col in group_df.columns else None

    sub["quality"] = "MEASURED"
    sub["mapping_confidence"] = confidence
    sub["mapping_source"] = source

    return sub
