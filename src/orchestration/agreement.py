"""
orchestration/agreement.py — Day 4 Part 4: Agent Agreement Score

TRANSPARENT, DOCUMENTED, NOT a black-box percentage.

Participants: ONLY forecast_agent and anomaly_agent. These are the two
agents that make an independent DIRECTIONAL read of "is consumption
behaving unexpectedly, and which way" from the same test-period data.

Explicitly EXCLUDED from this score, and why:
  - equipment_agent: purely contextual (identifiers/location), has no
    directional stance on whether a deviation occurred, so it cannot
    "agree" or "disagree" with anything.
  - cost_agent / carbon_agent: mechanically DERIVED from the exact same
    excess-energy number that forecast_agent's own residuals produce
    (excess_energy = sum of positive residuals). Counting them here
    would double-count the forecast signal, not add independent evidence.

Numerator: count of AVAILABLE participants whose directional read is
COMPATIBLE with the other available participant(s) (see the compatibility
table in `_agree` below).
Denominator: count of AVAILABLE participants (status == "DONE"). An
UNAVAILABLE/INSUFFICIENT_DATA participant is dropped from BOTH numerator
and denominator — it is never counted as a disagreement.

If 0 participants are available: score is None ("NO_PARTICIPANTS").
If exactly 1 participant is available: score is None ("SINGLE_AGENT") —
agreement is mathematically meaningless with only one voice.
If 2 participants are available: score is 0, 50, or 100 depending on the
compatibility table below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..agents.base import AgentResult

FORECAST_DEVIATION_EPSILON = 1e-9  # residuals within this of 0 count as NEUTRAL, not directional noise


@dataclass
class AgreementResult:
    score: Optional[float]  # 0-100, or None if not computable
    status: str  # "FULL_AGREEMENT" / "PARTIAL_AGREEMENT" / "DISAGREEMENT" / "SINGLE_AGENT" / "NO_PARTICIPANTS"
    participants: list = field(default_factory=list)
    forecast_direction: Optional[str] = None
    anomaly_direction: Optional[str] = None
    notes: list = field(default_factory=list)


def _forecast_direction(forecast_agent: Optional[AgentResult]) -> Optional[str]:
    if forecast_agent is None or forecast_agent.status != "DONE":
        return None
    mean_residual = forecast_agent.evidence.get("mean_residual")
    if mean_residual is None:
        return None
    if mean_residual > FORECAST_DEVIATION_EPSILON:
        return "ABOVE_EXPECTED"
    if mean_residual < -FORECAST_DEVIATION_EPSILON:
        return "BELOW_EXPECTED"
    return "NEUTRAL"


def _anomaly_direction(anomaly_agent: Optional[AgentResult]) -> Optional[str]:
    if anomaly_agent is None or anomaly_agent.status != "DONE":
        return None
    events = anomaly_agent.evidence.get("events", [])
    if not events:
        return "NONE"
    directions = {e["direction"] for e in events}
    if directions == {"ABOVE_EXPECTED"}:
        return "ABOVE_EXPECTED"
    if directions == {"BELOW_EXPECTED"}:
        return "BELOW_EXPECTED"
    return "MIXED"


def compute_agreement(agents: dict) -> AgreementResult:
    forecast_agent = agents.get("forecast_agent")
    anomaly_agent = agents.get("anomaly_agent")

    available = []
    if forecast_agent is not None and forecast_agent.status == "DONE":
        available.append("forecast_agent")
    if anomaly_agent is not None and anomaly_agent.status == "DONE":
        available.append("anomaly_agent")

    f_dir = _forecast_direction(forecast_agent)
    a_dir = _anomaly_direction(anomaly_agent)

    if len(available) == 0:
        return AgreementResult(
            score=None,
            status="NO_PARTICIPANTS",
            participants=[],
            notes=["Neither forecast_agent nor anomaly_agent is available; agreement cannot be computed."],
        )

    if len(available) == 1:
        return AgreementResult(
            score=None,
            status="SINGLE_AGENT",
            participants=available,
            forecast_direction=f_dir,
            anomaly_direction=a_dir,
            notes=[f"Only {available[0]} is available; agreement requires at least 2 independent participants."],
        )

    # Both available — apply the documented compatibility table.
    if a_dir == "NONE" and f_dir == "NEUTRAL":
        agree, status, note = 2, "FULL_AGREEMENT", "Both agents indicate no unusual deviation."
    elif a_dir == "NONE" and f_dir in ("ABOVE_EXPECTED", "BELOW_EXPECTED"):
        # A mild average residual with no anomaly flag isn't a contradiction —
        # anomaly detection is a stricter (statistical) threshold than "any
        # nonzero average residual". Treated as compatible, not conflicting.
        agree, status, note = 2, "FULL_AGREEMENT", "No anomaly flagged; forecast shows only a mild, sub-threshold residual — not contradictory."
    elif a_dir == "MIXED":
        agree, status, note = 1, "PARTIAL_AGREEMENT", "Anomaly events point in mixed directions; partial credit given."
    elif f_dir == "NEUTRAL" and a_dir in ("ABOVE_EXPECTED", "BELOW_EXPECTED"):
        agree, status, note = 1, "PARTIAL_AGREEMENT", "Anomaly detected a direction while forecast residual averaged to neutral."
    elif f_dir == a_dir:
        agree, status, note = 2, "FULL_AGREEMENT", f"Both agents agree the deviation direction is {f_dir}."
    else:
        agree, status, note = 0, "DISAGREEMENT", f"Forecast indicates {f_dir} while anomaly events indicate {a_dir}."

    return AgreementResult(
        score=round(100.0 * agree / 2, 1),
        status=status,
        participants=available,
        forecast_direction=f_dir,
        anomaly_direction=a_dir,
        notes=[note],
    )
