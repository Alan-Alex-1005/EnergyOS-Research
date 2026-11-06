"""
anomaly/engine.py — Day 2 Part J/K/L: Anomaly Detection Assembly

Uses the SAME selected forecast model's in-sample TRAINING residuals to
fit robust statistics (median/MAD), then scores the model's genuine
out-of-sample TEST residuals against those training-only statistics.
This is the concrete mechanism behind Part D's leakage rule for anomaly
detection: the expected-residual distribution never sees a single test
observation while being fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..forecasting.base import ForecastModel, ForecastResult
from ..timeseries import TrainTestSplit
from .events import AnomalyEvent, AnomalyObservation, group_anomaly_events
from .robust import DEFAULT_BASE_THRESHOLD, RobustResidualStats, classify_severity, fit_robust_stats, robust_z_scores


@dataclass
class AnomalyEngineResult:
    series_id: str
    applicable: bool
    reason: str
    base_threshold: float
    train_stats: Optional[RobustResidualStats] = None
    observations: list[AnomalyObservation] = field(default_factory=list)
    events: list[AnomalyEvent] = field(default_factory=list)
    observations_analyzed: int = 0
    anomalies_detected: int = 0
    anomaly_rate: float = 0.0
    peak_score: Optional[float] = None
    highest_severity: str = "NORMAL"
    warnings: list[str] = field(default_factory=list)


def run_anomaly_engine(
    series_id: str,
    selected_model: Optional[ForecastModel],
    split: TrainTestSplit,
    forecast_result: Optional[ForecastResult],
    base_threshold: float = DEFAULT_BASE_THRESHOLD,
) -> AnomalyEngineResult:
    if selected_model is None or forecast_result is None:
        return AnomalyEngineResult(
            series_id=series_id,
            applicable=False,
            reason="no forecast model was selected; anomaly detection requires a fitted model's residuals.",
            base_threshold=base_threshold,
        )

    train_residuals = selected_model.in_sample_residuals(split.train_values)
    if len(train_residuals) < 5:
        return AnomalyEngineResult(
            series_id=series_id,
            applicable=False,
            reason=(
                f"only {len(train_residuals)} in-sample training residual(s) available for "
                f"'{selected_model.name}'; need at least 5 to fit a defensible expected-residual distribution."
            ),
            base_threshold=base_threshold,
        )

    train_stats = fit_robust_stats(train_residuals)

    test_residuals = np.asarray(forecast_result.residuals, dtype=float)
    z_scores = robust_z_scores(test_residuals, train_stats)

    observations: list[AnomalyObservation] = []
    for i, (ts, residual, z) in enumerate(zip(forecast_result.test_timestamps, test_residuals, z_scores)):
        severity = classify_severity(abs(z), base_threshold) if not np.isnan(z) else "NORMAL"
        observations.append(
            AnomalyObservation(
                index=i,
                timestamp=ts,
                residual=float(residual) if not np.isnan(residual) else float("nan"),
                z_score=float(z) if not np.isnan(z) else 0.0,
                severity=severity,
                is_anomalous=severity != "NORMAL",
            )
        )

    events = group_anomaly_events(observations, series_id)

    anomalies_detected = sum(1 for o in observations if o.is_anomalous)
    observations_analyzed = len(observations)
    anomaly_rate = round(100.0 * anomalies_detected / observations_analyzed, 2) if observations_analyzed else 0.0
    peak_score = max((abs(o.z_score) for o in observations), default=None)
    severity_rank = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    highest_severity = max((o.severity for o in observations), key=lambda s: severity_rank.get(s, 0), default="NORMAL")

    warnings = []
    if train_stats.note:
        warnings.append(train_stats.note)

    return AnomalyEngineResult(
        series_id=series_id,
        applicable=True,
        reason="",
        base_threshold=base_threshold,
        train_stats=train_stats,
        observations=observations,
        events=events,
        observations_analyzed=observations_analyzed,
        anomalies_detected=anomalies_detected,
        anomaly_rate=anomaly_rate,
        peak_score=peak_score,
        highest_severity=highest_severity,
        warnings=warnings,
    )
