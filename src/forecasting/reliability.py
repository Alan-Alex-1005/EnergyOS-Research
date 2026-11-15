
"""
EnergyOS Forecast Reliability Safety Gate.

Provides preliminary diagnostic checks for forecast quality.
A passing assessment is not proof of real-world forecasting skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForecastResult


@dataclass(frozen=True)
class ReliabilityAssessment:
    status: str
    suitable_for_anomaly_evidence: bool
    reasons: tuple[str, ...]


def assess_forecast_reliability(
    result: ForecastResult,
) -> ReliabilityAssessment:

    actual = np.asarray(result.actual, dtype=float)
    predicted = np.asarray(result.predicted, dtype=float)

    if (
        actual.size == 0
        or actual.shape != predicted.shape
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(predicted))
    ):
        return ReliabilityAssessment(
            status="INVALID",
            suitable_for_anomaly_evidence=False,
            reasons=(
                "Missing, mismatched or non-finite forecast values.",
            ),
        )

    reasons: list[str] = []

    # Constant readings cannot demonstrate skill at forecasting
    # changes in energy demand.
    if np.allclose(
        actual,
        actual[0],
        rtol=0,
        atol=1e-10,
    ):
        reasons.append(
            "Constant test period: forecasting skill under "
            "varying loads cannot be established."
        )

    # Negative R2 indicates weak performance against
    # the test-period mean benchmark.
    if result.r2 is not None:
        r2_value = float(result.r2)

        if not np.isfinite(r2_value):
            reasons.append(
                "Non-finite test R2 requires review."
            )
        elif r2_value < 0:
            reasons.append(
                "Negative test R2: forecast performs worse "
                "than the test-period mean benchmark."
            )

    # Relative RMSE is a diagnostic, not a universal standard.
    mean_absolute_load = float(
        np.mean(np.abs(actual))
    )

    if (
        mean_absolute_load > 1e-10
        and result.rmse is not None
    ):
        rmse_value = float(result.rmse)

        if not np.isfinite(rmse_value):
            reasons.append(
                "Non-finite RMSE requires review."
            )
        elif rmse_value < 0:
            reasons.append(
                "Negative RMSE is invalid."
            )
        else:
            relative_rmse = (
                rmse_value / mean_absolute_load
            )

            if relative_rmse > 0.30:
                reasons.append(
                    f"RMSE is {relative_rmse:.1%} of mean "
                    "absolute load; requires domain-specific review."
                )

    if reasons:
        return ReliabilityAssessment(
            status="REVIEW_REQUIRED",
            suitable_for_anomaly_evidence=False,
            reasons=tuple(reasons),
        )

    return ReliabilityAssessment(
        status="PROVISIONALLY_ACCEPTABLE",
        suitable_for_anomaly_evidence=True,
        reasons=(
            "Basic reliability checks passed; this does "
            "not establish forecasting skill or confirm anomalies.",
        ),
    )
