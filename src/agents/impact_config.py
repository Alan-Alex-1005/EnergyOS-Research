"""
agents/impact_config.py — Day 3: Optional Cost/Carbon Config

Extends the JSON config with an optional "impact" block. Both fields are
optional and independent — missing either one simply makes the
corresponding agent report UNAVAILABLE. Nothing here invents a tariff or
carbon factor; it only validates what the user explicitly provided.

    {
      "impact": {
        "tariff_per_kwh": 0.15,
        "tariff_currency": "USD",
        "carbon_factor_kg_per_kwh": 0.4
      }
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class ImpactConfigError(Exception):
    pass


@dataclass
class ImpactConfig:
    tariff_per_kwh: Optional[float] = None
    tariff_currency: str = "USD"
    carbon_factor_kg_per_kwh: Optional[float] = None


def parse_impact_config(config: Optional[dict]) -> ImpactConfig:
    block = (config or {}).get("impact", {}) or {}
    if not isinstance(block, dict):
        raise ImpactConfigError("config 'impact' must be a JSON object.")

    tariff = block.get("tariff_per_kwh")
    currency = block.get("tariff_currency", "USD")
    carbon_factor = block.get("carbon_factor_kg_per_kwh")

    if tariff is not None and (not isinstance(tariff, (int, float)) or tariff <= 0):
        raise ImpactConfigError(f"config impact.tariff_per_kwh must be a positive number; got {tariff!r}.")
    if carbon_factor is not None and (not isinstance(carbon_factor, (int, float)) or carbon_factor <= 0):
        raise ImpactConfigError(f"config impact.carbon_factor_kg_per_kwh must be a positive number; got {carbon_factor!r}.")
    if not isinstance(currency, str):
        raise ImpactConfigError("config impact.tariff_currency must be a string.")

    return ImpactConfig(
        tariff_per_kwh=float(tariff) if tariff is not None else None,
        tariff_currency=currency,
        carbon_factor_kg_per_kwh=float(carbon_factor) if carbon_factor is not None else None,
    )
