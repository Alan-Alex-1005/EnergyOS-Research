"""
analysis_series.py — Day 2 Part B: Analytical Series Selection

The forecasting/anomaly engines must never touch a raw dataframe column
directly — they consume a small, uniform `AnalysisSeries` object built
from Day 1's `canonical_df` (which is already unified across WIDE and
LONG input shapes; see docs/ARCHITECTURE.md). This module is the only
place that knows how to turn canonical rows into series.

Also implements Day 2 Part A3: forecasting/anomaly sufficiency must be
judged PER SERIES, not from the whole dataset's row count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

SUPPORTED_ROLES = ("POWER", "ENERGY", "DEMAND")
ROLE_PREFERENCE = {"POWER": 0, "DEMAND": 1, "ENERGY": 2}
MIN_SAMPLES_FOR_FORECAST = 20

CONTEXT_FIELDS = (
    "equipment_id",
    "device_id",
    "meter_id",
    "site",
    "building",
    "floor",
    "zone",
    "room",
)


@dataclass
class AnalysisSeries:
    series_id: str
    role: str
    unit: Optional[str]
    timestamps: pd.Series
    values: pd.Series
    used_normalized: bool
    equipment_id: Optional[str] = None
    device_id: Optional[str] = None
    meter_id: Optional[str] = None
    site: Optional[str] = None
    building: Optional[str] = None
    floor: Optional[str] = None
    zone: Optional[str] = None
    room: Optional[str] = None
    sample_count: int = 0
    missing_ratio: float = 0.0
    duplicate_timestamp_count: int = 0
    quality: str = "MEASURED"
    provenance: dict = field(default_factory=dict)

    def label(self) -> str:
        """Short human-readable identity for reports: 'role / context'."""
        ctx = self.meter_id or self.device_id or self.equipment_id
        return f"{self.role}" + (f" ({ctx})" if ctx else "")

    def non_null(self) -> tuple[pd.Series, pd.Series]:
        """Timestamps/values with missing observations dropped."""
        mask = self.values.notna()
        return (
            self.timestamps[mask].reset_index(drop=True),
            self.values[mask].reset_index(drop=True),
        )


@dataclass
class SeriesSufficiency:
    series_id: str
    role: str
    sample_count: int
    status: str
    reason: str


def _has_explicit_timezone(series: pd.Series) -> bool:
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


def _parse_analysis_timestamps(series: pd.Series) -> pd.Series:
    """
    Parse canonical timestamps without inventing timezone information.

    - Existing datetime64 values are preserved.
    - Naive strings remain naive.
    - Explicit offset-aware strings are normalized to UTC.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.copy()

    explicit_timezone = _has_explicit_timezone(series)

    try:
        if explicit_timezone:
            return pd.to_datetime(
                series,
                errors="coerce",
                format="mixed",
                utc=True,
            )
        return pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )
    except (ValueError, TypeError):
        if explicit_timezone:
            return pd.to_datetime(
                series,
                errors="coerce",
                utc=True,
            )
        return pd.to_datetime(
            series,
            errors="coerce",
        )


def build_analysis_series(
    canonical_df: pd.DataFrame,
    roles: tuple = SUPPORTED_ROLES,
) -> list[AnalysisSeries]:
    """
    Group canonical rows into one AnalysisSeries per distinct
    (role, source_column, meter_id/device_id/equipment_id) combination.
    """
    if canonical_df is None or canonical_df.empty:
        return []

    relevant = canonical_df[canonical_df["role"].isin(roles)].copy()
    if relevant.empty:
        return []

    id_cols = [
        c
        for c in ("meter_id", "device_id", "equipment_id")
        if c in relevant.columns
    ]
    group_cols = ["role", "source_column"] + id_cols

    series_list: list[AnalysisSeries] = []

    for group_key, group_df in relevant.groupby(
        group_cols,
        dropna=False,
        sort=False,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        role = group_key[0]
        source_column = group_key[1]
        id_values = {
            id_cols[i]: group_key[2 + i]
            for i in range(len(id_cols))
        }

        ordered = (
            group_df.sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )

        use_normalized = ordered["normalized_value"].notna().any()

        if use_normalized:
            values = pd.to_numeric(
                ordered["normalized_value"],
                errors="coerce",
            )
            unit = _first_non_null(ordered["normalized_unit"])
        else:
            values = pd.to_numeric(
                ordered["raw_value"],
                errors="coerce",
            )
            unit = _first_non_null(ordered["raw_unit"])

        timestamps = _parse_analysis_timestamps(ordered["timestamp"])

        sample_count = int(len(ordered))
        missing_ratio = (
            round(float(values.isna().sum()) / sample_count, 4)
            if sample_count
            else 1.0
        )
        dup_ts_count = int(
            timestamps.dropna().duplicated(keep="first").sum()
        )

        context = {
            f: _first_non_null(ordered[f]) if f in ordered.columns else None
            for f in CONTEXT_FIELDS
        }

        for f, v in id_values.items():
            if v is not None and not _is_nan(v):
                context[f] = v

        series_id = ":".join(
            [role, str(source_column)]
            + [
                f"{k}={v}"
                for k, v in id_values.items()
                if v is not None and not _is_nan(v)
            ]
        )

        mapping_confidence = _first_non_null(
            ordered["mapping_confidence"]
        )
        mapping_source = _first_non_null(
            ordered["mapping_source"]
        )

        series_list.append(
            AnalysisSeries(
                series_id=series_id,
                role=role,
                unit=unit,
                timestamps=timestamps,
                values=values,
                used_normalized=bool(use_normalized),
                equipment_id=context.get("equipment_id"),
                device_id=context.get("device_id"),
                meter_id=context.get("meter_id"),
                site=context.get("site"),
                building=context.get("building"),
                floor=context.get("floor"),
                zone=context.get("zone"),
                room=context.get("room"),
                sample_count=sample_count,
                missing_ratio=missing_ratio,
                duplicate_timestamp_count=dup_ts_count,
                quality="MEASURED",
                provenance={
                    "source_column": source_column,
                    "mapping_confidence": mapping_confidence,
                    "mapping_source": mapping_source,
                },
            )
        )

    return series_list


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else None


def _is_nan(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def assess_series_sufficiency(
    series: AnalysisSeries,
    min_samples: int = MIN_SAMPLES_FOR_FORECAST,
) -> SeriesSufficiency:
    """
    Decide whether a given series has enough history for analysis.
    """
    non_null_count = int(series.values.notna().sum())

    if non_null_count >= min_samples:
        return SeriesSufficiency(
            series_id=series.series_id,
            role=series.role,
            sample_count=non_null_count,
            status="AVAILABLE",
            reason=(
                f"{non_null_count} non-null observations >= "
                f"minimum required ({min_samples})."
            ),
        )

    return SeriesSufficiency(
        series_id=series.series_id,
        role=series.role,
        sample_count=non_null_count,
        status="INSUFFICIENT_DATA",
        reason=(
            f"only {non_null_count} non-null observation(s); "
            f"minimum required is {min_samples}."
        ),
    )


def select_primary_series(
    series_list: list[AnalysisSeries],
    requested_series_id: Optional[str] = None,
) -> tuple[Optional[AnalysisSeries], str]:
    """
    Deterministic series selection.
    """
    if not series_list:
        return (
            None,
            "no POWER/ENERGY/DEMAND analytical series were found in this dataset.",
        )

    if requested_series_id:
        for s in series_list:
            if s.series_id == requested_series_id:
                return (
                    s,
                    f"explicitly requested via --series {requested_series_id!r}.",
                )

        available = ", ".join(s.series_id for s in series_list)
        return (
            None,
            f"requested series {requested_series_id!r} not found. "
            f"Available series: {available}",
        )

    sufficient = [
        s
        for s in series_list
        if assess_series_sufficiency(s).status == "AVAILABLE"
    ]
    pool = sufficient if sufficient else series_list

    def sort_key(s: AnalysisSeries):
        return (
            ROLE_PREFERENCE.get(s.role, 99),
            -int(s.values.notna().sum()),
        )

    chosen = sorted(pool, key=sort_key)[0]

    if sufficient:
        reason = (
            "selected deterministically: role preference "
            "(POWER > DEMAND > ENERGY), then most non-null samples "
            "among series with sufficient data."
        )
    else:
        reason = (
            "no series met the minimum sample threshold; selected the "
            "largest available series anyway (forecast/anomaly steps "
            "will report INSUFFICIENT_DATA)."
        )

    return chosen, reason
