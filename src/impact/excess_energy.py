"""
impact/excess_energy.py — Day 2 Part N: Excess Energy Mathematics

    E_excess = sum_t max(P_actual,t - P_expected,t, 0) * dt

Only computed for POWER series (never cumulative ENERGY — integrating
an already-cumulative quantity again would be dimensionally wrong), and
only when the unit is the canonical kW and the sampling interval is
known and regular enough to trust a fixed dt for every sample. Any
precondition failure returns an explicit "unavailable + reason" result
— never a silently wrong number.

If power is in kW and dt is in hours, the result is kWh (spec's own
example).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

REQUIRED_UNIT = "kW"


@dataclass
class ExcessEnergyResult:
    available: bool
    reason: str
    excess_energy_kwh: Optional[float] = None
    basis: Optional[str] = None
    interval_hours: Optional[float] = None
    contributing_samples: int = 0
    provenance: str = "ESTIMATED"


def compute_excess_energy(
    role: str,
    unit: Optional[str],
    sampling_interval_seconds: Optional[float],
    is_regular: bool,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> ExcessEnergyResult:
    if role != "POWER":
        return ExcessEnergyResult(
            available=False,
            reason=f"excess energy is only computed for POWER series (this series is {role}); "
            f"integrating an already-cumulative ENERGY series again would be dimensionally invalid.",
        )

    if unit != REQUIRED_UNIT:
        return ExcessEnergyResult(
            available=False,
            reason=f"unit is {unit!r}, not the canonical {REQUIRED_UNIT!r} required for a defensible kWh integration.",
        )

    if not sampling_interval_seconds or not is_regular:
        return ExcessEnergyResult(
            available=False,
            reason="sampling interval is unknown or irregular; time integration would not be defensible.",
        )

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        return ExcessEnergyResult(available=False, reason="actual and expected series have mismatched lengths.")

    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if not mask.any():
        return ExcessEnergyResult(available=False, reason="no paired actual/expected observations available.")

    excess_power = np.maximum(actual[mask] - predicted[mask], 0.0)  # negative residuals (below expected) ignored
    interval_hours = sampling_interval_seconds / 3600.0
    excess_kwh = float(np.sum(excess_power) * interval_hours)

    return ExcessEnergyResult(
        available=True,
        reason="",
        excess_energy_kwh=round(excess_kwh, 3),
        basis="Actual power above forecast baseline, integrated over the test period.",
        interval_hours=interval_hours,
        contributing_samples=int(mask.sum()),
    )
