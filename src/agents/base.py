"""
agents/base.py — Day 3: Common Agent Contract

Every agent (Forecast, Anomaly, Equipment, Cost, Carbon) returns exactly
this structure, so Day 4's orchestrator can treat them uniformly without
knowing each agent's internals. No arbitrary agent-level confidence
percentages here yet (that's explicitly deferred).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Capability-aware execution states (spec: must be visibly distinguished)
STATUS_DONE = "DONE"
STATUS_LIMITED = "LIMITED"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STATUS_ERROR = "ERROR"

VALID_STATUSES = {STATUS_DONE, STATUS_LIMITED, STATUS_UNAVAILABLE, STATUS_INSUFFICIENT_DATA, STATUS_ERROR}


@dataclass
class AgentResult:
    agent_id: str
    status: str  # one of VALID_STATUSES
    finding: str  # one-line human-readable summary — never fabricated
    evidence: dict = field(default_factory=dict)  # the concrete numbers/facts backing `finding`
    metrics: dict = field(default_factory=dict)  # numeric metrics, JSON-safe
    provenance: dict = field(default_factory=dict)  # e.g. {"expected": "ESTIMATED", "count": "CALCULATED"}
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"AgentResult.status must be one of {VALID_STATUSES}; got {self.status!r}")
