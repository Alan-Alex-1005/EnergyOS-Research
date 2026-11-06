import os

import pandas as pd
import pytest

from src.analysis_pipeline import run_day2_analysis
from src.evaluation.provenance import build_provenance
from src.evaluation.real_data_runner import (
    RealDataError,
    RealDatasetSpec,
    apply_long_format_transform,
    run_real_data_evaluation,
)
from tests.conftest import fixture_path


def test_build_provenance_derives_real_fields(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    prov = build_provenance(run, "Test Dataset", "Test Provider", "https://example.org/dataset")
    assert prov.dataset_name == "Test Dataset"
    assert prov.filename == "wide_energy.csv"
    assert prov.input_format == "WIDE"
    assert prov.row_count == 200
    assert prov.selected_series is not None
    assert prov.unit == "kW"
    assert prov.time_start is not None and prov.time_end is not None


def test_real_dataset_spec_rejects_missing_name():
    with pytest.raises(RealDataError, match="dataset_name"):
        RealDatasetSpec(
            scenario_id="s1", dataset_name="", source_provider="X", source_reference="https://x",
            category="normal_operation", path=fixture_path("wide_energy.csv"),
        )


def test_real_dataset_spec_rejects_missing_source_provider():
    with pytest.raises(RealDataError, match="source_provider"):
        RealDatasetSpec(
            scenario_id="s1", dataset_name="D", source_provider="", source_reference="https://x",
            category="normal_operation", path=fixture_path("wide_energy.csv"),
        )


def test_real_dataset_spec_rejects_missing_source_reference():
    with pytest.raises(RealDataError, match="source_reference"):
        RealDatasetSpec(
            scenario_id="s1", dataset_name="D", source_provider="X", source_reference="",
            category="normal_operation", path=fixture_path("wide_energy.csv"),
        )


def test_real_dataset_spec_rejects_missing_file():
    with pytest.raises(RealDataError, match="not found"):
        RealDatasetSpec(
            scenario_id="s1", dataset_name="D", source_provider="X", source_reference="https://x",
            category="normal_operation", path="/tmp/does_not_exist_12345.csv",
        )


def test_run_real_data_evaluation_rejects_empty_list(tmp_path):
    with pytest.raises(RealDataError, match="No real dataset specs"):
        run_real_data_evaluation([], str(tmp_path))


def test_long_format_transform_preserves_values_exactly(tmp_path):
    src = tmp_path / "src.csv"
    pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=5, freq="h"), "power_kw": [1.0, 2.0, 3.0, 4.0, 5.0], "temp_c": [10, 11, 12, 13, 14]}
    ).to_csv(src, index=False)
    out = tmp_path / "long.csv"
    apply_long_format_transform(str(src), str(out), "timestamp", ["power_kw", "temp_c"])
    long_df = pd.read_csv(out)
    assert len(long_df) == 10
    power_vals = sorted(long_df[long_df["metric"] == "power_kw"]["value"].tolist())
    assert power_vals == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_long_format_transform_rejects_missing_columns(tmp_path):
    src = tmp_path / "src.csv"
    pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="h"), "power_kw": [1, 2, 3]}).to_csv(src, index=False)
    with pytest.raises(RealDataError, match="not found"):
        apply_long_format_transform(str(src), str(tmp_path / "out.csv"), "timestamp", ["does_not_exist"])


def test_run_real_data_evaluation_end_to_end_with_test_fixture_as_stand_in(tmp_path):
    """
    Uses an existing local test fixture only to exercise the REAL-DATA
    RUNNER'S MECHANICS (provenance requirements, category tracking,
    honest N/A marking) — this is a software test, not a research result,
    and is never presented as such.
    """
    spec = RealDatasetSpec(
        scenario_id="mechanics_check",
        dataset_name="Local fixture (mechanics check only, NOT a research dataset)",
        source_provider="Internal test fixture",
        source_reference="tests/fixtures/wide_energy.csv",
        category="normal_operation",
        path=fixture_path("wide_energy.csv"),
    )
    result = run_real_data_evaluation([spec], str(tmp_path))

    assert os.path.exists(result["csv_path"])
    assert os.path.exists(result["json_path"])
    assert result["category_status"]["normal_operation"] == "AVAILABLE"
    assert "NOT_AVAILABLE" in result["category_status"]["natural_anomaly"]
    assert "NOT_AVAILABLE" in result["category_status"]["rich_context"]
    assert "NOT_AVAILABLE" in result["category_status"]["insufficient_data"]
    assert "NOT_AVAILABLE" in result["category_status"]["long_format"]

    for key, fig in result["figures"].items():
        if fig.generated:
            assert os.path.getsize(fig.path) > 0

    assert result["provenance_records"][0].dataset_name.startswith("Local fixture")


def test_real_data_evaluation_never_defaults_a_tariff(tmp_path):
    spec = RealDatasetSpec(
        scenario_id="no_tariff_check",
        dataset_name="Local fixture (mechanics check only)",
        source_provider="Internal test fixture",
        source_reference="tests/fixtures/wide_energy.csv",
        category="normal_operation",
        path=fixture_path("wide_energy.csv"),
        analysis_config=None,
    )
    result = run_real_data_evaluation([spec], str(tmp_path))
    sr = result["scenario_results"][0]
    assert sr.agents["cost_agent"].status == "UNAVAILABLE"
    assert sr.agents["carbon_agent"].status == "UNAVAILABLE"
