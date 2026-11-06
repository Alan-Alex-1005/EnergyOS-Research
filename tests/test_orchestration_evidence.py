from src.agents.base import AgentResult
from src.orchestration.agreement import compute_agreement
from src.orchestration.correlation import correlate_incidents
from src.orchestration.evidence import SharedEvidence, extract_evidence
from src.analysis_pipeline import run_day2_analysis
from src.agents.runner import run_all_agents
from tests.conftest import fixture_path


def _agents_and_series(fixture_name, impact_cfg=None, tmp_path="/tmp/day4_test_out"):
    run = run_day2_analysis(fixture_path(fixture_name), output_root=str(tmp_path), generate_figures=False)
    agents = run_all_agents(run, impact_cfg)
    return agents, run.primary


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

def test_extract_evidence_rich_dataset_with_impact(tmp_path):
    agents, series_result = _agents_and_series(
        "wide_energy.csv", {"impact": {"tariff_per_kwh": 0.15, "carbon_factor_kg_per_kwh": 0.4}}, tmp_path
    )
    evidence = extract_evidence(agents, series_result)
    types = {e.evidence_type for e in evidence}
    assert "FORECAST_DEVIATION" in types
    assert "ANOMALY" in types
    assert "EQUIPMENT_CONTEXT" in types
    assert "ENERGY_IMPACT" in types
    assert "COST_IMPACT" in types
    assert "CARBON_IMPACT" in types
    for e in evidence:
        assert e.series_id == series_result.series.series_id


def test_extract_evidence_no_series_returns_empty():
    assert extract_evidence({}, None) == []


def test_extract_evidence_skips_unavailable_cost_carbon(tmp_path):
    agents, series_result = _agents_and_series("wide_energy.csv", None, tmp_path)
    evidence = extract_evidence(agents, series_result)
    types = {e.evidence_type for e in evidence}
    assert "COST_IMPACT" not in types  # no tariff configured -> agent UNAVAILABLE -> no evidence fabricated
    assert "CARBON_IMPACT" not in types


# ---------------------------------------------------------------------------
# Incident correlation
# ---------------------------------------------------------------------------

def _ev(evidence_type, series_id="S1", t0="2026-01-01", t1="2026-01-02", value=1.0):
    return SharedEvidence(
        source_agent="x", evidence_type=evidence_type, series_id=series_id, time_start=t0, time_end=t1,
        metric="m", value=value, unit=None, provenance="CALCULATED", availability="DONE", description="d"
    )


def test_no_incident_when_no_anomalies():
    evidence = [_ev("FORECAST_DEVIATION"), _ev("ANOMALY", value=0.0)]
    incidents = correlate_incidents(evidence)
    assert incidents == []


def test_incident_created_when_anomaly_present():
    evidence = [_ev("FORECAST_DEVIATION"), _ev("ANOMALY", value=3.0), _ev("ENERGY_IMPACT")]
    incidents = correlate_incidents(evidence)
    assert len(incidents) == 1
    assert incidents[0].series_id == "S1"
    assert "ANOMALY" in incidents[0].evidence_types_present
    assert "FORECAST_DEVIATION" in incidents[0].evidence_types_present
    assert "ENERGY_IMPACT" in incidents[0].evidence_types_present


def test_unrelated_series_not_falsely_correlated():
    evidence = [_ev("ANOMALY", series_id="S1", value=2.0), _ev("FORECAST_DEVIATION", series_id="S2")]
    incidents = correlate_incidents(evidence)
    assert len(incidents) == 1
    assert incidents[0].series_id == "S1"
    assert all(e.series_id == "S1" for e in incidents[0].all_evidence)


def test_non_overlapping_time_window_not_attached_as_supporting():
    trigger = _ev("ANOMALY", t0="2026-02-01", t1="2026-02-02", value=1.0)
    unrelated = _ev("COST_IMPACT", t0="2026-05-01", t1="2026-05-02")
    incidents = correlate_incidents([trigger, unrelated])
    assert len(incidents) == 1
    assert unrelated not in incidents[0].supporting_evidence


def test_equipment_context_always_attaches_no_time_window():
    trigger = _ev("ANOMALY", value=1.0)
    equip = _ev("EQUIPMENT_CONTEXT", t0=None, t1=None)
    incidents = correlate_incidents([trigger, equip])
    assert equip in incidents[0].supporting_evidence


# ---------------------------------------------------------------------------
# Agent agreement
# ---------------------------------------------------------------------------

def _fake_agent(status, evidence=None, metrics=None):
    return AgentResult(agent_id="x", status=status, finding="f", evidence=evidence or {}, metrics=metrics or {})


def test_agreement_no_participants():
    result = compute_agreement({"forecast_agent": _fake_agent("UNAVAILABLE"), "anomaly_agent": _fake_agent("UNAVAILABLE")})
    assert result.score is None
    assert result.status == "NO_PARTICIPANTS"


def test_agreement_single_agent():
    result = compute_agreement({"forecast_agent": _fake_agent("DONE", {"mean_residual": 1.0}), "anomaly_agent": _fake_agent("UNAVAILABLE")})
    assert result.score is None
    assert result.status == "SINGLE_AGENT"
    assert result.participants == ["forecast_agent"]


def test_agreement_full_agreement_both_normal():
    result = compute_agreement(
        {
            "forecast_agent": _fake_agent("DONE", {"mean_residual": 0.0}),
            "anomaly_agent": _fake_agent("DONE", {"events": [], "anomalies_detected": 0}),
        }
    )
    assert result.score == 100.0
    assert result.status == "FULL_AGREEMENT"


def test_agreement_full_agreement_both_above():
    result = compute_agreement(
        {
            "forecast_agent": _fake_agent("DONE", {"mean_residual": 5.0}),
            "anomaly_agent": _fake_agent("DONE", {"events": [{"direction": "ABOVE_EXPECTED"}]}),
        }
    )
    assert result.score == 100.0
    assert result.status == "FULL_AGREEMENT"


def test_agreement_disagreement_opposite_directions():
    result = compute_agreement(
        {
            "forecast_agent": _fake_agent("DONE", {"mean_residual": 5.0}),
            "anomaly_agent": _fake_agent("DONE", {"events": [{"direction": "BELOW_EXPECTED"}]}),
        }
    )
    assert result.score == 0.0
    assert result.status == "DISAGREEMENT"


def test_agreement_partial_mixed_anomaly_directions():
    result = compute_agreement(
        {
            "forecast_agent": _fake_agent("DONE", {"mean_residual": 5.0}),
            "anomaly_agent": _fake_agent("DONE", {"events": [{"direction": "ABOVE_EXPECTED"}, {"direction": "BELOW_EXPECTED"}]}),
        }
    )
    assert result.score == 50.0
    assert result.status == "PARTIAL_AGREEMENT"


def test_agreement_unavailable_agent_never_counted_as_disagreement():
    result = compute_agreement({"forecast_agent": _fake_agent("DONE", {"mean_residual": 5.0}), "anomaly_agent": _fake_agent("INSUFFICIENT_DATA")})
    assert result.status == "SINGLE_AGENT"
    assert "anomaly_agent" not in result.participants


def test_agreement_real_dataset_end_to_end(tmp_path):
    agents, _ = _agents_and_series("wide_energy.csv", None, tmp_path)
    result = compute_agreement(agents)
    assert result.status in ("FULL_AGREEMENT", "PARTIAL_AGREEMENT", "DISAGREEMENT")
    assert result.score is not None
