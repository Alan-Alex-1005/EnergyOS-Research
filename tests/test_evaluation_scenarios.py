import os

from src.evaluation.comparison import build_comparison_rows, write_comparison_csv
from src.evaluation.robustness import build_robustness_matrix
from src.evaluation.scenario_runner import run_scenario
from src.evaluation.scenarios import build_scenarios


def test_build_scenarios_returns_five(tmp_path):
    specs = build_scenarios(str(tmp_path / "fixtures"), repo_root=os.path.dirname(os.path.dirname(__file__)))
    assert len(specs) == 5
    ids = {s.scenario_id for s in specs}
    assert len(ids) == 5


def test_scenario_2_spike_dataset_written_and_synthetic_labeled(tmp_path):
    specs = build_scenarios(str(tmp_path / "fixtures"), repo_root=os.path.dirname(os.path.dirname(__file__)))
    spike = next(s for s in specs if s.scenario_id == "scenario_2_unexplained_spike")
    assert os.path.exists(spike.path)
    assert "SYNTHETIC" in spike.data_kind or "SYNTHETIC" in os.path.basename(spike.path)


def test_run_scenario_normal_operation(tmp_path):
    specs = build_scenarios(str(tmp_path / "fixtures"), repo_root=os.path.dirname(os.path.dirname(__file__)))
    spec = next(s for s in specs if s.scenario_id == "scenario_1_normal_operation")
    result = run_scenario(
        spec.scenario_id, spec.description, spec.data_kind, spec.input_format, spec.path,
        analysis_config=spec.analysis_config, output_root=str(tmp_path / "out"),
    )
    assert result.run.primary is not None
    assert result.level3_coverage.percentage >= result.level1_coverage.percentage
    assert result.level3_coverage.denominator == result.level1_coverage.denominator == result.level2_coverage.denominator


def test_run_scenario_insufficient_data(tmp_path):
    specs = build_scenarios(str(tmp_path / "fixtures"), repo_root=os.path.dirname(os.path.dirname(__file__)))
    spec = next(s for s in specs if s.scenario_id == "scenario_4_insufficient_data")
    result = run_scenario(
        spec.scenario_id, spec.description, spec.data_kind, spec.input_format, spec.path,
        output_root=str(tmp_path / "out"),
    )
    assert result.run.primary.skipped_reason is not None
    assert result.level1_coverage.numerator == 0
    # Level 3's core analytical dimensions must still be False — no
    # fabricated forecast/anomaly/impact evidence just because the
    # framework exists. Only meta-level "the framework degrades honestly"
    # capabilities (data-quality awareness, an honest agreement/completeness
    # assessment, a coherent degraded narrative) may remain True.
    assert result.level3_support["forecast_context"] is False
    assert result.level3_support["abnormal_behavior_detection"] is False
    assert result.level3_support["cost_impact"] is False
    assert result.level3_support["carbon_impact"] is False
    assert result.level3_support["excess_energy_impact"] is False
    assert result.level3_support["cross_agent_evidence_correlation"] is False
    assert result.level3_support["evidence_backed_recommendation"] is False


def test_run_scenario_long_format(tmp_path):
    specs = build_scenarios(str(tmp_path / "fixtures"), repo_root=os.path.dirname(os.path.dirname(__file__)))
    spec = next(s for s in specs if s.scenario_id == "scenario_5_long_format")
    result = run_scenario(
        spec.scenario_id, spec.description, spec.data_kind, spec.input_format, spec.path,
        series_id_override=spec.series_id_override, output_root=str(tmp_path / "out"),
    )
    assert result.run.selected_series_id == spec.series_id_override
    assert result.level1_support["forecast_context"] is True


def test_robustness_matrix_uses_actual_statuses(tmp_path):
    specs = build_scenarios(str(tmp_path / "fixtures"), repo_root=os.path.dirname(os.path.dirname(__file__)))
    results = []
    for spec in specs[:2]:
        results.append(
            run_scenario(
                spec.scenario_id, spec.description, spec.data_kind, spec.input_format, spec.path,
                analysis_config=spec.analysis_config, series_id_override=spec.series_id_override,
                output_root=str(tmp_path / "out"),
            )
        )
    matrix = build_robustness_matrix(results)
    assert len(matrix) == 2
    for row in matrix:
        assert row["forecast"] in ("DONE", "UNAVAILABLE", "INSUFFICIENT_DATA", "LIMITED", "ERROR")


def test_comparison_rows_and_csv(tmp_path):
    specs = build_scenarios(str(tmp_path / "fixtures"), repo_root=os.path.dirname(os.path.dirname(__file__)))
    results = [
        run_scenario(
            s.scenario_id, s.description, s.data_kind, s.input_format, s.path,
            analysis_config=s.analysis_config, series_id_override=s.series_id_override,
            output_root=str(tmp_path / "out"),
        )
        for s in specs[:2]
    ]
    rows = build_comparison_rows(results)
    assert any(r["dimension"] == "MEAN_EXPLANATION_COVERAGE_PCT" for r in rows)
    csv_path = write_comparison_csv(results, str(tmp_path / "research_comparison.csv"))
    assert os.path.exists(csv_path)
    assert os.path.getsize(csv_path) > 0


def test_comparison_schedule_evidence_always_zero_applicable(tmp_path):
    specs = build_scenarios(str(tmp_path / "fixtures"), repo_root=os.path.dirname(os.path.dirname(__file__)))
    results = [
        run_scenario(
            s.scenario_id, s.description, s.data_kind, s.input_format, s.path,
            analysis_config=s.analysis_config, series_id_override=s.series_id_override,
            output_root=str(tmp_path / "out"),
        )
        for s in specs[:2]
    ]
    rows = build_comparison_rows(results)
    schedule_row = next(r for r in rows if r["dimension"] == "schedule_evidence")
    assert schedule_row["applicable_scenarios"] == 0
