"""
units.py — Day 1: Unit Normalization

Deterministic, conservative unit conversion. Never guesses an unknown
unit — if a unit can't be recognized, the raw value/unit are preserved
and normalization is skipped with a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Canonical target unit per physical quantity.
CANONICAL_UNITS = {
    "POWER": "kW",
    "ENERGY": "kWh",
    "DEMAND": "kW",
    "TEMPERATURE": "C",
    "VOLTAGE": "V",
    "CURRENT": "A",
    "FREQUENCY": "Hz",
    "HUMIDITY": "%",
    "POWER_FACTOR": None,  # dimensionless, no conversion needed
}

# Deterministic multiplicative conversion factors to the canonical unit.
_POWER_TO_KW = {"W": 0.001, "kW": 1.0, "MW": 1000.0}
_ENERGY_TO_KWH = {"Wh": 0.001, "kWh": 1.0, "MWh": 1000.0}
_VOLTAGE_TO_V = {"V": 1.0, "mV": 0.001, "kV": 1000.0}
_CURRENT_TO_A = {"A": 1.0, "mA": 0.001}
_FREQUENCY_TO_HZ = {"Hz": 1.0, "kHz": 1000.0}


@dataclass
class ConversionResult:
    raw_value: float
    raw_unit: Optional[str]
    normalized_value: Optional[float]
    normalized_unit: Optional[str]
    converted: bool
    note: Optional[str] = None


def normalize_value(value: float, unit: Optional[str], role: str) -> ConversionResult:
    """
    Convert `value` (given in `unit`) to the canonical unit for `role`.
    If the unit is missing/unrecognized, the raw value is preserved and
    `converted` is False — normalization is never guessed.
    """
    if unit is None:
        return ConversionResult(
            raw_value=value,
            raw_unit=None,
            normalized_value=None,
            normalized_unit=None,
            converted=False,
            note="No unit available; cannot normalize.",
        )

    unit_clean = unit.strip()

    if role == "POWER" or role == "DEMAND":
        table = _POWER_TO_KW
    elif role == "ENERGY":
        table = _ENERGY_TO_KWH
    elif role == "VOLTAGE":
        table = _VOLTAGE_TO_V
    elif role == "CURRENT":
        table = _CURRENT_TO_A
    elif role == "FREQUENCY":
        table = _FREQUENCY_TO_HZ
    elif role == "TEMPERATURE":
        return _normalize_temperature(value, unit_clean)
    else:
        return ConversionResult(
            raw_value=value,
            raw_unit=unit_clean,
            normalized_value=None,
            normalized_unit=None,
            converted=False,
            note=f"No normalization rule defined for role '{role}'.",
        )

    factor = table.get(unit_clean)
    if factor is None:
        return ConversionResult(
            raw_value=value,
            raw_unit=unit_clean,
            normalized_value=None,
            normalized_unit=None,
            converted=False,
            note=f"Unrecognized unit '{unit_clean}' for role '{role}'; left unconverted.",
        )

    canonical = CANONICAL_UNITS[role]
    return ConversionResult(
        raw_value=value,
        raw_unit=unit_clean,
        normalized_value=value * factor,
        normalized_unit=canonical,
        converted=True,
    )


def _normalize_temperature(value: float, unit: str) -> ConversionResult:
    """
    Temperature is only converted when the unit is explicit. Fahrenheit
    is only ever converted if EXPLICITLY provided/configured (spec
    section 10) — never inferred.
    """
    u = unit.strip().upper()
    if u in ("C", "CELSIUS", "°C"):
        return ConversionResult(
            raw_value=value, raw_unit=unit, normalized_value=value,
            normalized_unit="C", converted=True,
        )
    if u in ("F", "FAHRENHEIT", "°F"):
        celsius = (value - 32.0) * 5.0 / 9.0
        return ConversionResult(
            raw_value=value, raw_unit=unit, normalized_value=celsius,
            normalized_unit="C", converted=True,
            note="Converted from explicitly-configured Fahrenheit.",
        )
    return ConversionResult(
        raw_value=value, raw_unit=unit, normalized_value=None,
        normalized_unit=None, converted=False,
        note=f"Unrecognized temperature unit '{unit}'; left unconverted.",
    )


def is_known_unit(unit: Optional[str], role: str) -> bool:
    if unit is None:
        return False
    if role in ("POWER", "DEMAND"):
        return unit in _POWER_TO_KW
    if role == "ENERGY":
        return unit in _ENERGY_TO_KWH
    if role == "VOLTAGE":
        return unit in _VOLTAGE_TO_V
    if role == "CURRENT":
        return unit in _CURRENT_TO_A
    if role == "FREQUENCY":
        return unit in _FREQUENCY_TO_HZ
    if role == "TEMPERATURE":
        return unit.strip().upper() in ("C", "CELSIUS", "°C", "F", "FAHRENHEIT", "°F")
    return False
