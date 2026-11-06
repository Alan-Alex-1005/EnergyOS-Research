"""
anomaly/events.py — Day 2 Part L/M: Anomaly Events + Temporal Persistence

Groups temporally ADJACENT anomalous observations (consecutive positions
in the ordered test sequence) into one AnomalyEvent, rather than treating
every anomalous timestamp as a separate incident. Distinguishes an
isolated spike from a sustained deviation using plain, inspectable
counts (consecutive observations, duration) — never an opaque score
(Part M).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Transparent persistence classification thresholds (by observation
# count within one event) — documented, not an opaque AI score.
SUSTAINED_MIN_OBSERVATIONS = 4  # >= this many consecutive anomalous points = "sustained"


@dataclass
class AnomalyObservation:
    index: int  # position within the scored (test) sequence
    timestamp: str
    residual: float
    z_score: float
    severity: str  # NORMAL / LOW / MEDIUM / HIGH / CRITICAL
    is_anomalous: bool


@dataclass
class AnomalyEvent:
    event_id: str
    series_id: str
    start_time: str
    end_time: str
    duration_seconds: Optional[float]
    peak_score: float  # max |z| within the event
    mean_score: float
    peak_deviation: float  # max |residual| within the event
    direction: str  # "ABOVE_EXPECTED" / "BELOW_EXPECTED" / "MIXED"
    severity: str  # highest severity attained within the event
    observation_count: int
    persistence_class: str  # "ISOLATED_SPIKE" or "SUSTAINED" (transparent, count-based)
    evidence: list[str] = field(default_factory=list)


_SEVERITY_RANK = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def group_anomaly_events(observations: list[AnomalyObservation], series_id: str) -> list[AnomalyEvent]:
    """Group consecutive (by `index`) anomalous observations into events."""
    events: list[AnomalyEvent] = []
    run: list[AnomalyObservation] = []

    def flush(run: list[AnomalyObservation]) -> None:
        if not run:
            return
        events.append(_build_event(run, series_id, event_id=f"{series_id}:evt{len(events) + 1}"))

    prev_index: Optional[int] = None
    for obs in observations:
        if not obs.is_anomalous:
            flush(run)
            run = []
            prev_index = None
            continue
        if prev_index is not None and obs.index != prev_index + 1:
            flush(run)
            run = []
        run.append(obs)
        prev_index = obs.index
    flush(run)

    return events


def _build_event(run: list[AnomalyObservation], series_id: str, event_id: str) -> AnomalyEvent:
    abs_zs = [abs(o.z_score) for o in run]
    residuals = [o.residual for o in run]
    peak_score = float(max(abs_zs))
    mean_score = float(sum(abs_zs) / len(abs_zs))
    peak_idx = abs_zs.index(peak_score)
    peak_deviation = float(abs(residuals[peak_idx]))

    positive = any(r > 0 for r in residuals)
    negative = any(r < 0 for r in residuals)
    if positive and negative:
        direction = "MIXED"
    elif positive:
        direction = "ABOVE_EXPECTED"
    else:
        direction = "BELOW_EXPECTED"

    severity = max((o.severity for o in run), key=lambda s: _SEVERITY_RANK.get(s, 0))

    duration_seconds = None
    try:
        import pandas as pd

        start = pd.Timestamp(run[0].timestamp)
        end = pd.Timestamp(run[-1].timestamp)
        duration_seconds = float((end - start).total_seconds())
    except Exception:  # noqa: BLE001 — duration is best-effort, never fatal
        duration_seconds = None

    observation_count = len(run)
    persistence_class = "SUSTAINED" if observation_count >= SUSTAINED_MIN_OBSERVATIONS else "ISOLATED_SPIKE"

    evidence = [
        f"{observation_count} consecutive anomalous observation(s) from {run[0].timestamp} to {run[-1].timestamp}",
        f"peak |z|={peak_score:.2f} ({run[peak_idx].severity}), peak deviation={peak_deviation:.4g}",
        f"direction: {direction.lower().replace('_', ' ')}",
    ]

    return AnomalyEvent(
        event_id=event_id,
        series_id=series_id,
        start_time=run[0].timestamp,
        end_time=run[-1].timestamp,
        duration_seconds=duration_seconds,
        peak_score=peak_score,
        mean_score=mean_score,
        peak_deviation=peak_deviation,
        direction=direction,
        severity=severity,
        observation_count=observation_count,
        persistence_class=persistence_class,
        evidence=evidence,
    )
