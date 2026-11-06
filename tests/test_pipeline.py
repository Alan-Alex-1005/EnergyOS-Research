import pytest

from src.pipeline import PipelineError, run_pipeline
from tests.conftest import fixture_path


def test_full_pipeline_simple_csv():
    result = run_pipeline(fixture_path("simple_timestamp_energy.csv"))
    assert result.timestamp_info.detected is True
    assert "energy_kwh" in result.mapping_result.mapped
    assert result.mapping_result.mapped["energy_kwh"].role == "ENERGY"
    assert result.quality_result.overall > 0
    assert len(result.capability_report.capabilities) == 8


def test_full_pipeline_wide_csv():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    assert result.profile.format_hint == "WIDE"
    assert result.mapping_result.mapped_count >= 3
    assert not result.canonical_df.empty


def test_full_pipeline_long_format_csv():
    result = run_pipeline(fixture_path("long_format_meter.csv"))
    assert result.profile.format_hint == "LONG"
    assert result.timestamp_info.detected is True


def test_full_pipeline_xlsx():
    result = run_pipeline(fixture_path("sample_single_sheet.xlsx"))
    assert result.load_result.file_type == "XLSX"
    assert result.load_result.selected_sheet == "Data"


def test_full_pipeline_xlsx_multi_sheet_needs_selection():
    with pytest.raises(PipelineError, match="multiple sheets"):
        run_pipeline(fixture_path("sample_multi_sheet.xlsx"))


def test_full_pipeline_xlsx_multi_sheet_with_selection():
    result = run_pipeline(fixture_path("sample_multi_sheet.xlsx"), sheet="SiteA")
    assert result.load_result.selected_sheet == "SiteA"


def test_full_pipeline_missing_values():
    result = run_pipeline(fixture_path("missing_values.csv"))
    assert result.cleaning_report.action_counts["MISSING_VALUE"] == 4


def test_full_pipeline_duplicate_rows_removed():
    result = run_pipeline(fixture_path("duplicate_rows.csv"))
    assert result.cleaning_report.action_counts["REMOVED_DUPLICATE"] == 2


def test_full_pipeline_duplicate_timestamps_across_series_not_flagged():
    """Part A2: legitimate cross-series (different meter) repeats aren't corruption."""
    result = run_pipeline(fixture_path("duplicate_timestamps.csv"))
    assert result.cleaning_report.action_counts["UNRESOLVED"] == 0


def test_full_pipeline_true_duplicate_timestamp_within_series_flagged():
    """Part A2: a genuine repeat within the SAME meter is still flagged."""
    result = run_pipeline(fixture_path("duplicate_timestamps_same_series.csv"))
    assert result.cleaning_report.action_counts["UNRESOLVED"] == 4


def test_full_pipeline_ambiguous_numeric_not_treated_as_timestamp():
    result = run_pipeline(fixture_path("ambiguous_numeric.csv"))
    assert result.timestamp_info.column == "timestamp"


def test_full_pipeline_status_on_off():
    result = run_pipeline(fixture_path("status_on_off.csv"))
    assert result.mapping_result.mapped["chiller_status"].role == "STATUS"


def test_full_pipeline_strange_energy_name_unresolved():
    result = run_pipeline(fixture_path("strange_energy_name.csv"))
    assert "ChillerLoadReading" in result.mapping_result.unresolved


def test_full_pipeline_no_timestamp_file():
    result = run_pipeline(fixture_path("no_timestamp.csv"))
    assert result.timestamp_info.detected is False
    caps = result.capability_report.as_dict()
    assert caps["Forecasting"].status == "UNAVAILABLE"


def test_full_pipeline_no_energy_data_file():
    result = run_pipeline(fixture_path("no_energy_data.csv"))
    caps = result.capability_report.as_dict()
    assert caps["Forecasting"].status == "UNAVAILABLE"
    assert caps["Equipment Analysis"].status == "UNAVAILABLE"


def test_full_pipeline_mixed_known_unknown_columns():
    result = run_pipeline(fixture_path("mixed_known_unknown.csv"))
    assert result.mapping_result.mapped["power_kw"].role == "POWER"
    assert result.mapping_result.unresolved_count >= 1


def test_full_pipeline_explicit_config_override():
    result = run_pipeline(
        fixture_path("config_override_base.csv"),
        config_path=fixture_path("config_override.json"),
    )
    assert result.mapping_result.mapped["Main_Load"].role == "POWER"
    assert result.mapping_result.mapped["Main_Load"].source == "CONFIG"
    assert result.mapping_result.mapped["AHU_State"].role == "STATUS"


def test_pipeline_missing_file_raises_clean_error():
    with pytest.raises(PipelineError, match="not found"):
        run_pipeline(fixture_path("nope_does_not_exist.csv"))


def test_pipeline_bad_config_raises_clean_error():
    with pytest.raises(PipelineError):
        run_pipeline(
            fixture_path("simple_timestamp_energy.csv"),
            config_path=fixture_path("does_not_exist.json"),
        )


def test_pipeline_result_is_deterministic():
    r1 = run_pipeline(fixture_path("wide_energy.csv"))
    r2 = run_pipeline(fixture_path("wide_energy.csv"))
    assert r1.quality_result.overall == r2.quality_result.overall
    assert r1.mapping_result.mapped_count == r2.mapping_result.mapped_count
