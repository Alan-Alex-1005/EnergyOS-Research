import json
import os

import pytest

from src.analysis_pipeline import (
    AnalysisConfigError,
    build_results_dict,
    parse_analysis_config,
    run_day2_analysis,
    write_results_json,
)
from tests.conftest import fixture_path


def test_parse_analysis_config_defaults():
    cfg = parse_analysis_config(None)
    assert cfg.train_ratio == 0.70
    assert cfg.validation_ratio == 0.15
    assert cfg.anomaly_threshold == 3.0


def test_parse_analysis_config_overrides():
    cfg = parse_analysis_config({"analysis": {"train_ratio": 0.6, "validation_ratio": 0.2, "test_ratio": 0.2, "anomaly_threshold": 4.0, "series": "POWER:x"}})
    assert cfg.train_ratio == 0.6
    assert cfg.test_ratio == 0.2
    assert cfg.anomaly_threshold == 4.0
    assert cfg.series_id == "POWER:x"


def test_parse_analysis_config_rejects_bad_ratio():
    with pytest.raises(AnalysisConfigError):
        parse_analysis_config({"analysis": {"train_ratio": 1.5}})


def test_parse_analysis_config_rejects_ratios_not_summing_to_one():
    with pytest.raises(AnalysisConfigError):
        parse_analysis_config({"analysis": {"train_ratio": 0.5, "validation_ratio": 0.3, "test_ratio": 0.3}})


def test_parse_analysis_config_rejects_bad_threshold():
    with pytest.raises(AnalysisConfigError):
        parse_analysis_config({"analysis": {"anomaly_threshold": -1}})


def test_rich_dataset_full_analysis(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path))
    assert run.selected_series_id is not None
    assert run.primary is not None
    assert run.primary.forecast is not None
    assert run.primary.forecast.selected_model is not None
    assert run.primary.anomaly is not None
    assert run.output_dir and os.path.isdir(run.output_dir)

    generated = [f for f in run.primary.figures.values() if f.generated]
    assert len(generated) >= 3
    for f in generated:
        assert os.path.getsize(f.path) > 0


def test_results_json_written_and_valid(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path))
    path = write_results_json(run)
    assert path is not None
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)  # must not raise — fully JSON-safe
    assert data["dataset"]["file"] == "wide_energy.csv"
    assert data["selected_series"] is not None
    assert "forecast" in data["analysis"]


def test_insufficient_data_case_reported_not_crashed(tmp_path):
    import pandas as pd

    tiny_path = tmp_path / "tiny.csv"
    pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=5, freq="h"), "power_kw": [1, 2, 3, 4, 5]}).to_csv(tiny_path, index=False)
    run = run_day2_analysis(str(tiny_path), output_root=str(tmp_path / "out"))
    assert run.primary is not None
    assert run.primary.sufficiency.status == "INSUFFICIENT_DATA"
    assert run.primary.skipped_reason is not None
    assert run.primary.forecast is None  # never attempted on insufficient data


def test_no_series_at_all(tmp_path):
    run = run_day2_analysis(fixture_path("no_energy_data.csv"), output_root=str(tmp_path))
    assert run.selected_series_id is None
    assert run.primary is None
    assert len(run.warnings) == 1


def test_multiple_series_long_format_lists_all(tmp_path):
    run = run_day2_analysis(fixture_path("long_format_meter.csv"), output_root=str(tmp_path))
    assert len(run.all_series) == 4  # 2 meters x 2 metrics
    ids = {s.series_id for s in run.all_series}
    assert len(ids) == 4


def test_explicit_series_selection(tmp_path):
    run_all = run_day2_analysis(fixture_path("long_format_meter.csv"), output_root=str(tmp_path), generate_figures=False)
    target_id = run_all.all_series[0].series_id
    run = run_day2_analysis(
        fixture_path("long_format_meter.csv"), output_root=str(tmp_path), series_id_override=target_id, generate_figures=False
    )
    assert run.selected_series_id == target_id
    assert "explicitly requested" in run.selection_reason


def test_seasonal_period_config_override_applied(tmp_path):
    run = run_day2_analysis(
        fixture_path("wide_energy.csv"),
        analysis_config={"analysis": {"seasonal_period": 4}},
        output_root=str(tmp_path),
        generate_figures=False,
    )
    seasonal_eval = next((e for e in run.primary.forecast.comparison if e.model_name == "Seasonal Naive"), None)
    assert seasonal_eval is not None
    assert seasonal_eval.metadata.get("seasonal_period") == 4 or seasonal_eval.applicable is False


def test_generate_figures_false_skips_png_creation(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    assert run.output_dir is None
    assert run.primary.figures == {}


def test_excess_energy_present_for_power_series_with_regular_sampling(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    assert run.primary.excess_energy is not None
    if run.primary.series.role == "POWER":
        assert run.primary.excess_energy.available is True
        assert run.primary.excess_energy.provenance == "ESTIMATED"


def test_day1_bad_config_still_raises(tmp_path):
    from src.pipeline import PipelineError

    with pytest.raises(PipelineError):
        run_day2_analysis(
            fixture_path("wide_energy.csv"), day1_config_path=fixture_path("does_not_exist.json"), output_root=str(tmp_path)
        )
