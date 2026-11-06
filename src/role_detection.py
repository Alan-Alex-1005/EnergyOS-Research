"""
role_detection.py — Day 1: Column Role Detection

Assigns each non-timestamp column a suggested role from a fixed taxonomy,
using name-hint patterns combined with profiler evidence (numeric-ness,
cardinality, binary-status shape, embedded units in the column name).

Every suggestion carries a confidence score and evidence list. Low
confidence columns are left as UNKNOWN rather than silently auto-mapped
(spec section 7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .profiler import ColumnProfile

ROLES = [
    "TIMESTAMP",
    "POWER",
    "ENERGY",
    "DEMAND",
    "VOLTAGE",
    "CURRENT",
    "POWER_FACTOR",
    "FREQUENCY",
    "EQUIPMENT_ID",
    "DEVICE_ID",
    "METER_ID",
    "SITE",
    "BUILDING",
    "FLOOR",
    "ZONE",
    "ROOM",
    "STATUS",
    "OCCUPANCY",
    "TEMPERATURE",
    "HUMIDITY",
    "WEATHER",
    "TARIFF",
    "CARBON_FACTOR",
    "UNKNOWN",
]

LOW_CONFIDENCE_THRESHOLD = 0.5

# Ordered (pattern, role, unit_hint, base_confidence) rules.
# Patterns are matched against a normalized column name (lowercase,
# separators collapsed to single underscore).
_NAME_RULES: list[tuple[str, str, Optional[str], float]] = [
    # Energy / power family (order matters — energy before power to avoid
    # "energy_kw" style names being caught by a generic power rule first)
    (r"energy.?kwh|kwh", "ENERGY", "kWh", 0.9),
    (r"energy.?mwh|mwh", "ENERGY", "MWh", 0.9),
    (r"energy.?wh(?!o)", "ENERGY", "Wh", 0.85),
    (r"\benergy\b|consumption", "ENERGY", None, 0.75),
    (r"active.?power|power.?kw|\bkw\b", "POWER", "kW", 0.85),
    (r"power.?mw|\bmw\b", "POWER", "MW", 0.85),
    (r"\bpower\b|\bload\b", "POWER", None, 0.7),
    (r"demand", "DEMAND", None, 0.75),
    (r"voltage|\bvolt\b|\bv_\b|_v$", "VOLTAGE", "V", 0.7),
    (r"current|\bamp(s)?\b|_a$", "CURRENT", "A", 0.6),
    (r"power.?factor|\bpf\b", "POWER_FACTOR", None, 0.8),
    (r"frequency|\bhz\b", "FREQUENCY", "Hz", 0.75),
    (r"tariff|rate.?per.?kwh|price.?kwh", "TARIFF", None, 0.75),
    (r"carbon|co2|emission.?factor", "CARBON_FACTOR", None, 0.75),
    # Identifiers
    (r"equipment.?id|equipment$", "EQUIPMENT_ID", None, 0.7),
    (r"device.?id|device$", "DEVICE_ID", None, 0.7),
    (r"meter.?id|meter$", "METER_ID", None, 0.7),
    # Location hierarchy
    (r"\bsite\b", "SITE", None, 0.7),
    (r"\bbuilding\b", "BUILDING", None, 0.7),
    (r"\bfloor\b", "FLOOR", None, 0.7),
    (r"\bzone\b", "ZONE", None, 0.7),
    (r"\broom\b", "ROOM", None, 0.7),
    # State / context
    (r"occupancy|occupied", "OCCUPANCY", None, 0.7),
    (r"status|state|on_off|on/off", "STATUS", None, 0.6),
    (r"temp(erature)?", "TEMPERATURE", "C", 0.7),
    (r"humidity", "HUMIDITY", "%", 0.75),
    (r"weather", "WEATHER", None, 0.6),
]


@dataclass
class RoleSuggestion:
    column: str
    role: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    inferred_unit: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


def _normalize(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"[\s\-]+", "_", n)
    n = re.sub(r"[^a-z0-9_/]", "", n)
    return n


def _detect_unit_from_name(name: str) -> Optional[str]:
    n = _normalize(name)
    unit_patterns = [
        (r"kwh", "kWh"),
        (r"mwh", "MWh"),
        (r"\bwh\b", "Wh"),
        (r"kw\b", "kW"),
        (r"mw\b", "MW"),
        (r"\bw\b", "W"),
        (r"celsius|_c$|\bc\b", "C"),
        (r"fahrenheit|_f$|\bf\b", "F"),
        (r"percent|pct|%", "%"),
        (r"\bv\b|volt", "V"),
        (r"\ba\b|amp", "A"),
        (r"hz", "Hz"),
    ]
    for pattern, unit in unit_patterns:
        if re.search(pattern, n):
            return unit
    return None


def detect_role(
    column_name: str, col_profile: ColumnProfile
) -> RoleSuggestion:
    normalized = _normalize(column_name)
    evidence: list[str] = []
    warnings: list[str] = []

    best_role = None
    best_confidence = 0.0
    best_unit = None

    for pattern, role, unit_hint, base_conf in _NAME_RULES:
        if re.search(pattern, normalized):
            confidence = base_conf
            local_evidence = [f"column name matches pattern for {role} (/{pattern}/)"]

            # Cross-check against profiler evidence to raise/lower confidence.
            if role in (
                "POWER",
                "ENERGY",
                "DEMAND",
                "VOLTAGE",
                "CURRENT",
                "POWER_FACTOR",
                "FREQUENCY",
                "TEMPERATURE",
                "HUMIDITY",
                "TARIFF",
                "CARBON_FACTOR",
            ):
                if col_profile.is_numeric:
                    confidence = min(1.0, confidence + 0.1)
                    local_evidence.append("column dtype is numeric, consistent with a measurement")
                else:
                    confidence = max(0.0, confidence - 0.3)
                    local_evidence.append("column is not numeric, weakening confidence for a measurement role")

            if role in ("EQUIPMENT_ID", "DEVICE_ID", "METER_ID"):
                if col_profile.suspected_identifier or col_profile.suspected_categorical:
                    confidence = min(1.0, confidence + 0.15)
                    local_evidence.append("column has identifier/categorical cardinality shape")

            if role in ("SITE", "BUILDING", "FLOOR", "ZONE", "ROOM"):
                if col_profile.suspected_categorical or col_profile.suspected_identifier:
                    confidence = min(1.0, confidence + 0.1)
                    local_evidence.append("column has categorical/location-like cardinality")

            if role in ("STATUS", "OCCUPANCY"):
                if col_profile.suspected_binary_status:
                    confidence = min(1.0, confidence + 0.2)
                    local_evidence.append("column values look binary/status-like (on/off, 0/1, yes/no)")
                elif not col_profile.suspected_categorical:
                    confidence = max(0.0, confidence - 0.2)

            unit = unit_hint or _detect_unit_from_name(column_name)

            if confidence > best_confidence:
                best_confidence = round(confidence, 3)
                best_role = role
                best_unit = unit
                evidence = local_evidence

    # Binary-status columns with no name hint at all still deserve a
    # low-confidence STATUS suggestion based purely on shape.
    if best_role is None and col_profile.suspected_binary_status:
        best_role = "STATUS"
        best_confidence = 0.4
        best_unit = None
        evidence = ["column values are binary/status-shaped (0/1, on/off) with no matching name hint"]

    if best_role is None:
        best_role = "UNKNOWN"
        best_confidence = round(_unknown_confidence(col_profile), 3)
        evidence = ["no name pattern or shape evidence matched any known role"]

    if best_confidence < LOW_CONFIDENCE_THRESHOLD and best_role != "UNKNOWN":
        warnings.append(
            f"Confidence {best_confidence:.2f} is below the auto-map threshold "
            f"({LOW_CONFIDENCE_THRESHOLD}); this column will NOT be auto-mapped."
        )

    return RoleSuggestion(
        column=column_name,
        role=best_role,
        confidence=best_confidence,
        evidence=evidence,
        inferred_unit=best_unit,
        warnings=warnings,
    )


def match_role_by_label(label: str, assume_numeric: bool = True) -> RoleSuggestion:
    """
    Like detect_role, but for a free-standing text LABEL rather than a
    physical column name — used to classify LONG-format metric values
    such as "power_kw", "active_power", "energy_kwh", "consumption".

    Reuses the exact same _NAME_RULES table as detect_role (so a metric
    value is judged by the same evidence as a column name would be), but
    since there is no per-column profile for a metric *value* (the shared
    "value" column holds many different metrics), the numeric cross-check
    is a simple assumption rather than a measured dtype check: LONG-format
    value columns are numeric by construction, so `assume_numeric=True`
    is the correct default. Never invents a role or unit that isn't
    grounded in the label text (spec section 18).
    """
    normalized = _normalize(label)
    best_role = None
    best_confidence = 0.0
    best_unit = None
    evidence: list[str] = []

    for pattern, role, unit_hint, base_conf in _NAME_RULES:
        if re.search(pattern, normalized):
            confidence = base_conf
            local_evidence = [f"metric label matches pattern for {role} (/{pattern}/)"]

            if role in (
                "POWER",
                "ENERGY",
                "DEMAND",
                "VOLTAGE",
                "CURRENT",
                "POWER_FACTOR",
                "FREQUENCY",
                "TEMPERATURE",
                "HUMIDITY",
                "TARIFF",
                "CARBON_FACTOR",
            ):
                if assume_numeric:
                    confidence = min(1.0, confidence + 0.1)
                    local_evidence.append(
                        "the long-format value column holding this metric is numeric, "
                        "consistent with a measurement"
                    )

            unit = unit_hint or _detect_unit_from_name(label)

            if confidence > best_confidence:
                best_confidence = round(confidence, 3)
                best_role = role
                best_unit = unit
                evidence = local_evidence

    if best_role is None:
        best_role = "UNKNOWN"
        best_confidence = 0.25
        evidence = ["no name pattern matched this metric label"]

    warnings: list[str] = []
    if best_confidence < LOW_CONFIDENCE_THRESHOLD and best_role != "UNKNOWN":
        warnings.append(
            f"Confidence {best_confidence:.2f} is below the auto-map threshold "
            f"({LOW_CONFIDENCE_THRESHOLD}); this metric will NOT be auto-mapped."
        )

    return RoleSuggestion(
        column=label,
        role=best_role,
        confidence=best_confidence,
        evidence=evidence,
        inferred_unit=best_unit,
        warnings=warnings,
    )


def _unknown_confidence(col_profile: ColumnProfile) -> float:
    """
    'Confidence' for an UNKNOWN classification: how confidently we believe
    this column does NOT match any known role (not a mapping confidence).
    """
    if col_profile.null_ratio > 0.9:
        return 0.8
    return 0.25


def detect_roles(column_profiles: dict[str, ColumnProfile]) -> list[RoleSuggestion]:
    return [detect_role(name, prof) for name, prof in column_profiles.items()]
