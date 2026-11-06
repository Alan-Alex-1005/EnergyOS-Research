import pandas as pd

from src.agents.base import AgentResult
from src.analysis_pipeline import run_day2_analysis
from src.orchestration.completeness import compute_completeness
from src.orchestration.confidence import compute_confidence
from src.orchestration.energy_engineer import ROOT_CAUSE_UNCONFIRMED, generate_narrative
from src.orchestration.orchestrator import run_orchestration
from src.orchestration.recommendations import generate_recommendations
from tests.conftest import fixture_path


def _fake_agent(status, evidence=None, metrics=None, finding="f", limitations=None):
    return AgentResult(agent_id="x", status=status, finding=finding, evidence=evidence or {}, metrics=metrics or {}, limitations=limitations or [])


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

def test_completeness_all_done_scores_100():
    agents = {
        "forecast_agent": _fake_agent("DONE"),
        "anomaly_agent": _fake_agent("DONE"),
        "equipment_agent": _fake_agent("DONE"),
        "cost_agent": _fake_agent("DONE"),
        "carbon_agent": _fake_agent("DONE"),
    }

    class FakeSeriesResult:
        class excess_energy:
            available = True

    result = compute_completeness(agents, FakeSeriesResult())
    assert result.score == 100.0


def test_completeness_missing_optional_only_small_penalty():
    agents = {
        "forecast_agent": _fake_agent("DONE"),
        "anomaly_agent": _fake_agent("DONE"),
        "equipment_agent": _fake_agent("LIMITED"),
        "cost_agent": _fake_agent("UNAVAILABLE"),
        "carbon_agent": _fake_agent("UNAVAILABLE"),
    }

    class FakeSeriesResult:
        class excess_energy:
            available = True

    result = compute_completeness(agents, FakeSeriesResult())
    assert result.score >= 75.0
    assert result.category_status["cost"] == "UNAVAILABLE"


def test_completeness_no_series_result():
    agents = {"forecast_agent": _fake_agent("DONE")}
    result = compute_completeness(agents, None)
    assert result.category_status["energy_impact"] == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def test_confidence_all_components_available():
    result = compute_confidence(data_quality=90.0, agreement_score=100.0, completeness_score=80.0)
    expected = 0.45 * 90 + 0.30 * 100 + 0.25 * 80
    assert abs(result.score - expected) < 0.1
    assert "agent_agreement" in result.components


def test_confidence_agreement_none_redistributes_weight():
    result = compute_confidence(data_quality=90.0, agreement_score=None, completeness_score=80.0)
    expected = (0.45 * 90 + 0.25 * 80) / 0.70
    assert abs(result.score - expected) < 0.1
    assert "agent_agreement" not in result.components
    assert result.notes


def test_confidence_label_not_probability():
    result = compute_confidence(80, 80, 80)
    assert "not" in result.label.lower() and "probability" in result.label.lower()


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def test_recommendations_no_incidents():
    recs = generate_recommendations([], {})
    assert len(recs) == 1
    assert recs[0].possible_explanation is None
    assert "no investigation" in recs[0].recommended_action.lower()


def test_recommendations_never_fabricate_explanation_without_equipment():
    from src.orchestration.correlation import Incident
    from src.orchestration.evidence import SharedEvidence

    trigger = SharedEvidence(
        source_agent="anomaly_agent", evidence_type="ANOMALY", series_id="S1", time_start="t0", time_end="t1",
        metric="anomalies_detected", value=3.0, unit="count", provenance="INFERRED", availability="DONE",
        description="3 anomalies", detail={"events": [{"persistence_class": "ISOLATED_SPIKE"}]},
    )
    incident = Incident(incident_id="i1", series_id="S1", time_start="t0", time_end="t1", trigger_evidence=[trigger], supporting_evidence=[])
    recs = generate_recommendations([incident], {})
    assert recs[0].possible_explanation is None


def test_recommendations_hypothesis_labeled_when_equipment_present():
    from src.orchestration.correlation import Incident
    from src.orchestration.evidence import SharedEvidence

    trigger = SharedEvidence(
        source_agent="anomaly_agent", evidence_type="ANOMALY", series_id="S1", time_start="t0", time_end="t1",
        metric="anomalies_detected", value=3.0, unit="count", provenance="INFERRED", availability="DONE",
        description="3 anomalies", detail={"events": [{"persistence_class": "SUSTAINED"}]},
    )
    equip = SharedEvidence(
        source_agent="equipment_agent", evidence_type="EQUIPMENT_CONTEXT", series_id="S1", time_start=None, time_end=None,
        metric="identifier_context", value=None, unit=None, provenance="MEASURED", availability="DONE",
        description="equipment present", detail={"identifier_context": {"meter_id": "M1"}},
    )
    incident = Incident(incident_id="i1", series_id="S1", time_start="t0", time_end="t1", trigger_evidence=[trigger], supporting_evidence=[equip])
    recs = generate_recommendations([incident], {})
    assert recs[0].possible_explanation is not None
    assert "HYPOTHESIS" in recs[0].possible_explanation
    assert "unconfirmed" in recs[0].possible_explanation.lower()


# ---------------------------------------------------------------------------
# Orchestrator + AI Energy Engineer (end to end)
# ---------------------------------------------------------------------------

def test_orchestrator_normal_operation(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    collab = run_orchestration(run)
    assert collab.data_quality == run.day1.quality_result.overall
    assert collab.confidence is not None
    assert 0 <= collab.confidence.score <= 100


def test_orchestrator_no_series(tmp_path):
    run = run_day2_analysis(fixture_path("no_energy_data.csv"), output_root=str(tmp_path), generate_figures=False)
    collab = run_orchestration(run)
    assert collab.incidents == []
    assert collab.evidence == []


def test_energy_engineer_root_cause_always_unconfirmed(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    collab = run_orchestration(run)
    narrative = generate_narrative(run, collab)
    assert narrative.root_cause == ROOT_CAUSE_UNCONFIRMED


def test_energy_engineer_degraded_output_no_series(tmp_path):
    run = run_day2_analysis(fixture_path("no_energy_data.csv"), output_root=str(tmp_path), generate_figures=False)
    collab = run_orchestration(run)
    narrative = generate_narrative(run, collab)
    assert "no analytical series" in narrative.what.lower()
    assert narrative.root_cause == ROOT_CAUSE_UNCONFIRMED


def test_energy_engineer_missing_context_lists_unavailable_agents(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    collab = run_orchestration(run)
    narrative = generate_narrative(run, collab)
    assert any("cost_agent" in m or "carbon_agent" in m for m in narrative.missing_context)


def test_energy_engineer_confidence_string_matches_collab(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    collab = run_orchestration(run)
    narrative = generate_narrative(run, collab)
    assert f"{collab.confidence.score:.1f}" in narrative.confidence
