"""
mapping.py — Day 1: Mapping System

Combines automatic role detection with optional explicit JSON config
overrides into a final column -> (role, unit) mapping. Config overrides
always win over automatic inference. Low-confidence UNKNOWN columns are
left unresolved rather than silently auto-mapped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .role_detection import ROLES, RoleSuggestion, LOW_CONFIDENCE_THRESHOLD


class MappingConfigError(Exception):
    """Raised for invalid/malformed mapping config."""


@dataclass
class ColumnMapping:
    column: str
    role: str
    unit: Optional[str]
    confidence: float
    source: str  # "AUTO" or "CONFIG"
    evidence: list[str] = field(default_factory=list)


@dataclass
class MappingResult:
    timestamp_column: Optional[str]
    mapped: dict[str, ColumnMapping]
    unresolved: list[str]
    mapped_count: int
    unresolved_count: int
    warnings: list[str] = field(default_factory=list)


def load_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise MappingConfigError(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise MappingConfigError(f"Config file '{path}' is not valid JSON: {e}")

    if not isinstance(data, dict):
        raise MappingConfigError("Config root must be a JSON object.")

    mappings = data.get("mappings", {})
    if not isinstance(mappings, dict):
        raise MappingConfigError("Config 'mappings' must be a JSON object of column -> {role, unit}.")

    for col, spec in mappings.items():
        if not isinstance(spec, dict) or "role" not in spec:
            raise MappingConfigError(
                f"Config mapping for column '{col}' must be an object containing at least 'role'."
            )
        role = spec["role"]
        if role not in ROLES:
            raise MappingConfigError(
                f"Config mapping for column '{col}' has invalid role '{role}'. "
                f"Valid roles: {', '.join(ROLES)}"
            )
        unit = spec.get("unit")
        if unit is not None and not isinstance(unit, str):
            raise MappingConfigError(f"Config mapping for column '{col}' has non-string unit.")

    ts_col = data.get("timestamp_column")
    if ts_col is not None and not isinstance(ts_col, str):
        raise MappingConfigError("Config 'timestamp_column' must be a string.")

    return data


def build_mapping(
    all_columns: list[str],
    timestamp_column: Optional[str],
    role_suggestions: list[RoleSuggestion],
    config: Optional[dict] = None,
) -> MappingResult:
    config = config or {}
    config_ts = config.get("timestamp_column")
    config_mappings = config.get("mappings", {})

    effective_ts_column = config_ts if config_ts else timestamp_column
    warnings: list[str] = []

    if config_ts and config_ts not in all_columns:
        raise MappingConfigError(
            f"Config 'timestamp_column' = '{config_ts}' does not exist in the dataset."
        )
    for col in config_mappings:
        if col not in all_columns:
            raise MappingConfigError(
                f"Config mapping references column '{col}' which does not exist in the dataset."
            )

    suggestions_by_col = {s.column: s for s in role_suggestions}

    mapped: dict[str, ColumnMapping] = {}
    unresolved: list[str] = []

    for col in all_columns:
        if col == effective_ts_column:
            mapped[col] = ColumnMapping(
                column=col,
                role="TIMESTAMP",
                unit=None,
                confidence=1.0 if config_ts else _timestamp_confidence(timestamp_column, col),
                source="CONFIG" if config_ts else "AUTO",
                evidence=["designated as the timestamp column"],
            )
            continue

        if col in config_mappings:
            spec = config_mappings[col]
            mapped[col] = ColumnMapping(
                column=col,
                role=spec["role"],
                unit=spec.get("unit"),
                confidence=1.0,
                source="CONFIG",
                evidence=["explicit user override in config"],
            )
            continue

        suggestion = suggestions_by_col.get(col)
        if suggestion is None:
            unresolved.append(col)
            continue

        if suggestion.role == "UNKNOWN" or suggestion.confidence < LOW_CONFIDENCE_THRESHOLD:
            unresolved.append(col)
            continue

        mapped[col] = ColumnMapping(
            column=col,
            role=suggestion.role,
            unit=suggestion.inferred_unit,
            confidence=suggestion.confidence,
            source="AUTO",
            evidence=suggestion.evidence,
        )

    return MappingResult(
        timestamp_column=effective_ts_column,
        mapped=mapped,
        unresolved=unresolved,
        mapped_count=len(mapped),
        unresolved_count=len(unresolved),
        warnings=warnings,
    )


def _timestamp_confidence(detected_ts_column: Optional[str], col: str) -> float:
    return 1.0 if detected_ts_column == col else 0.0
