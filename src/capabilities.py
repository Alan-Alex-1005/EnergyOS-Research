"""
capabilities.py — Day 1: Dataset Capability Report

Determines which future (Day 2+) analyses are possible, purely from the
roles that were actually mapped. Every verdict carries a reason grounded
in the mapping result — nothing here is guessed (spec section 13).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .mapping import MappingResult
from .schema import TimestampInfo

MIN_ROWS_FOR_TIME_SERIES = 20

MEASURE_ROLES = {"POWER", "ENERGY", "DEMAND"}
IDENTIFIER_ROLES = {"EQUIPMENT_ID", "DEVICE_ID", "METER_ID"}


@dataclass
class Capability:
    name: str
    status: str  # "AVAILABLE" | "LIMITED" | "UNAVAILABLE"
    reason: str


@dataclass
class CapabilityReport:
    capabilities: list[Capability] = field(default_factory=list)

    def as_dict(self) -> dict[str, Capability]:
        return {c.name: c for c in self.capabilities}


def assess_capabilities(
    mapping_result: MappingResult,
    timestamp_info: TimestampInfo,
    row_count: int,
    extra_roles: Optional[set[str]] = None,
) -> CapabilityReport:
    """
    `extra_roles` lets callers contribute roles that were resolved outside
    the physical column mapping — e.g. LONG-format logical series, where
    a role is assigned per metric VALUE rather than per column (see
    long_format.py). Capabilities must reflect what the dataset can
    actually support, not just what physical columns happened to map,
    so these are folded into the same roles_present set used below.
    """
    roles_present = {m.role for m in mapping_result.mapped.values()} | set(extra_roles or set())
    has_measure = bool(roles_present & MEASURE_ROLES)
    has_identifier = bool(roles_present & IDENTIFIER_ROLES)
    has_timestamp = timestamp_info.detected
    enough_rows = row_count >= MIN_ROWS_FOR_TIME_SERIES

    capabilities = []

    # Forecasting
    if has_timestamp and has_measure and enough_rows:
        capabilities.append(Capability("Forecasting", "AVAILABLE", "timestamp + measured energy/power data present with enough rows."))
    elif not has_timestamp:
        capabilities.append(Capability("Forecasting", "UNAVAILABLE", "no timestamp column detected."))
    elif not has_measure:
        capabilities.append(Capability("Forecasting", "UNAVAILABLE", "no power/energy/demand column mapped."))
    else:
        capabilities.append(Capability("Forecasting", "UNAVAILABLE", f"fewer than {MIN_ROWS_FOR_TIME_SERIES} rows after cleaning."))

    # Anomaly Detection
    if has_timestamp and has_measure and enough_rows:
        capabilities.append(Capability("Anomaly Detection", "AVAILABLE", "timestamp + measured energy/power data present with enough rows."))
    elif not has_timestamp:
        capabilities.append(Capability("Anomaly Detection", "UNAVAILABLE", "no timestamp column detected."))
    elif not has_measure:
        capabilities.append(Capability("Anomaly Detection", "UNAVAILABLE", "no power/energy/demand column mapped."))
    else:
        capabilities.append(Capability("Anomaly Detection", "UNAVAILABLE", f"fewer than {MIN_ROWS_FOR_TIME_SERIES} rows after cleaning."))

    # Equipment Analysis
    if has_identifier and has_measure:
        capabilities.append(Capability("Equipment Analysis", "AVAILABLE", "an equipment/device/meter identifier is mapped alongside measured data."))
    elif has_measure:
        capabilities.append(Capability("Equipment Analysis", "LIMITED", "measured data present but no equipment/device/meter identifier column — dataset is treated as a single asset."))
    else:
        capabilities.append(Capability("Equipment Analysis", "UNAVAILABLE", "no measured energy/power data mapped."))

    # Schedule Analysis
    if has_timestamp and ("STATUS" in roles_present or "OCCUPANCY" in roles_present):
        capabilities.append(Capability("Schedule Analysis", "AVAILABLE", "status/occupancy column present alongside a timestamp."))
    else:
        capabilities.append(Capability("Schedule Analysis", "UNAVAILABLE", "no status or occupancy column mapped."))

    # Cost Analysis
    if "TARIFF" in roles_present and has_measure:
        capabilities.append(Capability("Cost Analysis", "AVAILABLE", "tariff data mapped alongside measured energy/power."))
    else:
        capabilities.append(Capability("Cost Analysis", "UNAVAILABLE", "no tariff data or configured tariff."))

    # Carbon Analysis
    if "CARBON_FACTOR" in roles_present and has_measure:
        capabilities.append(Capability("Carbon Analysis", "AVAILABLE", "carbon factor mapped alongside measured energy/power."))
    else:
        capabilities.append(Capability("Carbon Analysis", "UNAVAILABLE", "no carbon factor configured."))

    # Weather Context
    if roles_present & {"TEMPERATURE", "HUMIDITY", "WEATHER"}:
        capabilities.append(Capability("Weather Context", "AVAILABLE", "temperature/humidity/weather column(s) mapped."))
    else:
        capabilities.append(Capability("Weather Context", "UNAVAILABLE", "no temperature, humidity, or weather column mapped."))

    # Occupancy Context
    if "OCCUPANCY" in roles_present:
        capabilities.append(Capability("Occupancy Context", "AVAILABLE", "occupancy column mapped."))
    else:
        capabilities.append(Capability("Occupancy Context", "UNAVAILABLE", "no occupancy column mapped."))

    return CapabilityReport(capabilities=capabilities)
