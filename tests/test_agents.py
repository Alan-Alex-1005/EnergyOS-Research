import pytest

from src.agents.base import STATUS_DONE, STATUS_INSUFFICIENT_DATA, STATUS_LIMITED, STATUS_UNAVAILABLE, AgentResult
from src.agents.impact_config import ImpactConfigError, parse_impact_config
from src.agents.runner import run_all_agents
from src.analysis_pipeline import run_day2_analysis
from tests.conftest import fixture_path


def test_agent_result_rejects_invalid_status():
    with pytest.raises(ValueError):
        AgentResult(agent_id="x", status="NOT_A_REAL_STATUS", finding="x")


def test_parse_impact_config_defaults():
    cfg = parse_impact_config(None)
    assert cfg.tariff_per_kwh is None
    assert cfg.carbon_factor_kg_per_kwh is None
    assert cfg.tariff_currency == "USD"


def test_parse_impact_config_valid():
    cfg = parse_impact_config({"impact": {"tariff_per_kwh": 0.2, "tariff_currency": "EUR", "carbon_factor_kg_per_kwh": 0.45}})
    assert cfg.tariff_per_kwh == 0.2
    assert cfg.tariff_currency == "EUR"
    assert cfg.carbon_factor_kg_per_kwh == 0.45


def test_parse_impact_config_rejects_negative_tariff():
    with pytest.raises(ImpactConfigError):
        parse_impact_config({"impact": {"tariff_per_kwh": -1}})


def test_parse_impact_config_rejects_bad_carbon_factor():
    with pytest.raises(ImpactConfigError):
        parse_impact_config({"impact": {"carbon_factor_kg_per_kwh": 0}})


# ---------------------------------------------------------------------------
# Rich-context dataset (no tariff/carbon configured) — forecast/anomaly DONE,
# equipment LIMITED (single asset), cost/carbon UNAVAILABLE.
# ---------------------------------------------------------------------------

def test_agents_rich_dataset_no_impact_config(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    agents = run_all_agents(run)

    assert agents["forecast_agent"].status == STATUS_DONE
    assert "selected_model" in agents["forecast_agent"].evidence
    assert agents["forecast_agent"].metrics["rmse"] is not None

    assert agents["anomaly_agent"].status == STATUS_DONE
    assert "observations_analyzed" in agents["anomaly_agent"].evidence

    assert agents["equipment_agent"].status == STATUS_LIMITED  # wide_energy.csv has no meter/device column

    assert agents["cost_agent"].status == STATUS_UNAVAILABLE
    assert "tariff" in agents["cost_agent"].limitations[0].lower()

    assert agents["carbon_agent"].status == STATUS_UNAVAILABLE


def test_agents_rich_dataset_with_impact_config(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    impact_cfg = {"impact": {"tariff_per_kwh": 0.15, "carbon_factor_kg_per_kwh": 0.4}}
    agents = run_all_agents(run, impact_cfg)

    excess_kwh = run.primary.excess_energy.excess_energy_kwh
    if run.primary.excess_energy.available:
        assert agents["cost_agent"].status == STATUS_DONE
        assert agents["cost_agent"].metrics["estimated_cost"] == round(excess_kwh * 0.15, 2)
        assert agents["cost_agent"].provenance["tariff"] == "CONFIGURED"

        assert agents["carbon_agent"].status == STATUS_DONE
        assert agents["carbon_agent"].metrics["estimated_carbon_kg"] == round(excess_kwh * 0.4, 2)


def test_agents_equipment_done_when_meter_present(tmp_path):
    run = run_day2_analysis(fixture_path("long_format_meter.csv"), output_root=str(tmp_path), generate_figures=False)
    agents = run_all_agents(run)
    assert agents["equipment_agent"].status == STATUS_DONE
    assert "meter_id" in agents["equipment_agent"].evidence["identifier_context"]


# ---------------------------------------------------------------------------
# Sparse dataset (insufficient samples) — every agent must degrade honestly,
# never crash, never fabricate a finding.
# ---------------------------------------------------------------------------

def test_agents_sparse_dataset_insufficient_data(tmp_path):
    import pandas as pd

    tiny_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=5, freq="h"), "power_kw": [1, 2, 3, 4, 5]}
    ).to_csv(tiny_path, index=False)
    run = run_day2_analysis(str(tiny_path), output_root=str(tmp_path / "out"), generate_figures=False)
    agents = run_all_agents(run)

    assert agents["forecast_agent"].status == STATUS_INSUFFICIENT_DATA
    assert agents["anomaly_agent"].status == STATUS_INSUFFICIENT_DATA
    assert agents["cost_agent"].status == STATUS_UNAVAILABLE
    assert agents["carbon_agent"].status == STATUS_UNAVAILABLE


def test_agents_no_series_all_unavailable(tmp_path):
    run = run_day2_analysis(fixture_path("no_energy_data.csv"), output_root=str(tmp_path), generate_figures=False)
    agents = run_all_agents(run)
    for agent in agents.values():
        assert agent.status == STATUS_UNAVAILABLE


def test_no_agent_fabricates_equipment_when_absent(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    agents = run_all_agents(run)
    eq = agents["equipment_agent"]
    assert eq.evidence["identifier_context"] == {}  # never fabricated
    assert "no equipment/device/meter identifier" in eq.finding.lower()


def test_no_agent_fabricates_schedule_deviation(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    agents = run_all_agents(run)
    eq = agents["equipment_agent"]
    assert any("schedule" in lim.lower() and "never" in lim.lower() for lim in eq.limitations)


def test_forecast_agent_provenance_never_labels_prediction_as_measured(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    agents = run_all_agents(run)
    fa = agents["forecast_agent"]
    assert fa.provenance.get("predicted") != "MEASURED"
    assert fa.provenance.get("predicted") == "ESTIMATED"


def test_agents_wide_and_long_both_supported(tmp_path):
    run_wide = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path / "w"), generate_figures=False)
    run_long = run_day2_analysis(fixture_path("long_format_meter.csv"), output_root=str(tmp_path / "l"), generate_figures=False)
    agents_wide = run_all_agents(run_wide)
    agents_long = run_all_agents(run_long)
    assert agents_wide["forecast_agent"].status == STATUS_DONE
    assert agents_long["forecast_agent"].status == STATUS_DONE
