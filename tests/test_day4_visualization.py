from src.agents.base import AgentResult
from src.orchestration.agreement import AgreementResult
from src.orchestration.completeness import CompletenessResult
from src.orchestration.confidence import ConfidenceResult
from src.orchestration.correlation import Incident
from src.orchestration.evidence import SharedEvidence
from src.visualization.research_plots import (
    plot_agent_agreement,
    plot_agent_status,
    plot_decision_confidence,
    plot_evidence_completeness,
    plot_incident_summary,
)


def test_agent_status_generated(tmp_path):
    agents = {"forecast_agent": AgentResult(agent_id="x", status="DONE", finding="f")}
    result = plot_agent_status(str(tmp_path), agents)
    assert result.generated is True


def test_agent_status_skipped_when_empty(tmp_path):
    result = plot_agent_status(str(tmp_path), {})
    assert result.generated is False


def test_agent_agreement_generated_when_score_present(tmp_path):
    ag = AgreementResult(score=75.0, status="PARTIAL_AGREEMENT", participants=["forecast_agent", "anomaly_agent"])
    result = plot_agent_agreement(str(tmp_path), ag)
    assert result.generated is True


def test_agent_agreement_skipped_when_score_none(tmp_path):
    ag = AgreementResult(score=None, status="SINGLE_AGENT", participants=["forecast_agent"])
    result = plot_agent_agreement(str(tmp_path), ag)
    assert result.generated is False
    assert "SINGLE_AGENT" in result.reason


def test_evidence_completeness_generated(tmp_path):
    comp = CompletenessResult(score=85.0, category_credit={"forecast": 1.0, "anomaly": 1.0, "cost": 0.0})
    result = plot_evidence_completeness(str(tmp_path), comp)
    assert result.generated is True


def test_incident_summary_skipped_when_no_incidents(tmp_path):
    result = plot_incident_summary(str(tmp_path), [])
    assert result.generated is False
    assert "normal operation" in result.reason.lower()


def test_incident_summary_generated_with_incident(tmp_path):
    incident = Incident(incident_id="i1", series_id="S1", time_start="t0", time_end="t1")
    incident.trigger_evidence = [
        SharedEvidence(source_agent="a", evidence_type="ANOMALY", series_id="S1", time_start="t0", time_end="t1", metric="m", value=1.0, unit=None, provenance="INFERRED", availability="DONE", description="d")
    ]
    result = plot_incident_summary(str(tmp_path), [incident])
    assert result.generated is True


def test_decision_confidence_generated(tmp_path):
    conf = ConfidenceResult(score=88.0, components={"data_quality": 90, "evidence_completeness": 80}, weights_used={"data_quality": 0.64, "evidence_completeness": 0.36})
    result = plot_decision_confidence(str(tmp_path), conf)
    assert result.generated is True


def test_decision_confidence_skipped_when_no_components(tmp_path):
    conf = ConfidenceResult(score=0.0, components={})
    result = plot_decision_confidence(str(tmp_path), conf)
    assert result.generated is False
