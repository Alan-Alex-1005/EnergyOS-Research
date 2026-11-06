"""
canonical.py — Day 1: Canonical Energy Model

Builds the canonical (long-format) energy representation from the
cleaned, mapped dataset. Never fabricates missing equipment/location
context — those fields are left null (None / NaN) if the source data
does not contain them (spec section 9).

The canonical dataset is represented as a pandas DataFrame (vectorized,
scales to large inputs) with one row per (timestamp, measured column).
`to_canonical_records` converts a (optionally capped) slice of it into
CanonicalRecord dataclass instances for inspection/tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .mapping import MappingResult
from . import units as units_mod

CONTEXT_ROLES = [
    "SITE",
    "BUILDING",
    "FLOOR",
    "ZONE",
    "ROOM",
    "EQUIPMENT_ID",
    "DEVICE_ID",
    "METER_ID",
]

# Roles that become measured canonical series (everything mapped except
# TIMESTAMP and the structural CONTEXT_ROLES above).
_EXCLUDED_FROM_MEASURES = set(CONTEXT_ROLES) | {"TIMESTAMP"}


@dataclass
class CanonicalRecord:
    timestamp: Optional[str]
    source_column: str
    role: str
    raw_value: object
    raw_unit: Optional[str]
    normalized_value: Optional[float]
    normalized_unit: Optional[str]
    equipment_id: Optional[str]
    device_id: Optional[str]
    meter_id: Optional[str]
    site: Optional[str]
    building: Optional[str]
    floor: Optional[str]
    zone: Optional[str]
    room: Optional[str]
    quality: str
    provenance: dict


def build_canonical(
    df_clean: pd.DataFrame,
    mapping_result: MappingResult,
    source_filename: str,
) -> pd.DataFrame:
    """
    Build the canonical long-format dataset as a DataFrame with columns:
    timestamp, source_column, role, raw_value, raw_unit, normalized_value,
    normalized_unit, equipment_id, device_id, meter_id, site, building,
    floor, zone, room, quality, mapping_confidence, mapping_source.
    """
    ts_col = "_parsed_timestamp" if "_parsed_timestamp" in df_clean.columns else mapping_result.timestamp_column

    context_col_by_role = {
        m.role: col for col, m in mapping_result.mapped.items() if m.role in CONTEXT_ROLES
    }

    measure_cols = [
        (col, m) for col, m in mapping_result.mapped.items() if m.role not in _EXCLUDED_FROM_MEASURES
    ]

    frames = []
    for col, m in measure_cols:
        if col not in df_clean.columns:
            continue
        sub = pd.DataFrame()
        sub["timestamp"] = df_clean[ts_col] if ts_col in df_clean.columns else pd.NaT
        sub["source_column"] = col
        sub["role"] = m.role
        sub["raw_value"] = df_clean[col]
        sub["raw_unit"] = m.unit

        if m.unit and m.role in units_mod.CANONICAL_UNITS and units_mod.is_known_unit(m.unit, m.role):
            numeric_raw = pd.to_numeric(df_clean[col], errors="coerce")
            conv = units_mod.normalize_value(1.0, m.unit, m.role)  # get factor via a probe value
            factor = conv.normalized_value if conv.converted else None
            if factor is not None:
                sub["normalized_value"] = numeric_raw * factor
                sub["normalized_unit"] = units_mod.CANONICAL_UNITS[m.role]
            else:
                sub["normalized_value"] = pd.NA
                sub["normalized_unit"] = pd.NA
        else:
            sub["normalized_value"] = pd.NA
            sub["normalized_unit"] = pd.NA

        for role, ctx_col in context_col_by_role.items():
            field_name = _role_to_context_field(role)
            sub[field_name] = df_clean[ctx_col] if ctx_col in df_clean.columns else None

        for role in CONTEXT_ROLES:
            field_name = _role_to_context_field(role)
            if field_name not in sub.columns:
                sub[field_name] = None

        sub["quality"] = "MEASURED"
        sub["mapping_confidence"] = m.confidence
        sub["mapping_source"] = m.source

        frames.append(sub)

    if not frames:
        columns = (
            ["timestamp", "source_column", "role", "raw_value", "raw_unit",
             "normalized_value", "normalized_unit"]
            + [_role_to_context_field(r) for r in CONTEXT_ROLES]
            + ["quality", "mapping_confidence", "mapping_source"]
        )
        return pd.DataFrame(columns=columns)

    canonical_df = pd.concat(frames, ignore_index=True)
    canonical_df.attrs["source_filename"] = source_filename
    return canonical_df


def _role_to_context_field(role: str) -> str:
    return role.lower()


def to_canonical_records(canonical_df: pd.DataFrame, limit: Optional[int] = None) -> list[CanonicalRecord]:
    subset = canonical_df.head(limit) if limit is not None else canonical_df
    records = []
    for _, row in subset.iterrows():
        records.append(
            CanonicalRecord(
                timestamp=str(row["timestamp"]) if pd.notna(row.get("timestamp")) else None,
                source_column=row["source_column"],
                role=row["role"],
                raw_value=row["raw_value"],
                raw_unit=row["raw_unit"] if pd.notna(row.get("raw_unit")) else None,
                normalized_value=(
                    float(row["normalized_value"]) if pd.notna(row.get("normalized_value")) else None
                ),
                normalized_unit=row["normalized_unit"] if pd.notna(row.get("normalized_unit")) else None,
                equipment_id=_safe_str(row.get("equipment_id")),
                device_id=_safe_str(row.get("device_id")),
                meter_id=_safe_str(row.get("meter_id")),
                site=_safe_str(row.get("site")),
                building=_safe_str(row.get("building")),
                floor=_safe_str(row.get("floor")),
                zone=_safe_str(row.get("zone")),
                room=_safe_str(row.get("room")),
                quality=row["quality"],
                provenance={
                    "mapping_confidence": row["mapping_confidence"],
                    "mapping_source": row["mapping_source"],
                },
            )
        )
    return records


def _safe_str(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)
