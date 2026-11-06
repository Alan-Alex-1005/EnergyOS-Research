from src.long_format import build_long_format_mapping
from src.mapping import MappingConfigError
from src.pipeline import run_pipeline
from src.profiler import profile_dataframe
from tests.conftest import fixture_path


def test_long_format_power_and_energy_make_forecasting_available():
    """
    Core regression test for the gap this module fixes: a long-format
    dataset whose metric column contains recognizable power/energy
    labels must make Forecasting and Anomaly Detection AVAILABLE, not
    UNAVAILABLE just because the physical 'metric'/'value' columns
    themselves carry no fixed role.
    """
    result = run_pipeline(fixture_path("long_format_meter.csv"))
    caps = result.capability_report.as_dict()
    assert caps["Forecasting"].status == "AVAILABLE"
    assert caps["Anomaly Detection"].status == "AVAILABLE"
    assert caps["Equipment Analysis"].status == "AVAILABLE"  # meter_id is mapped as an identifier


def test_long_format_mapping_result_applicable_and_grounded():
    result = run_pipeline(fixture_path("long_format_meter.csv"))
    lf = result.long_format_mapping
    assert lf.applicable is True
    assert lf.metric_column == "metric"
    assert lf.value_column == "value"
    assert lf.identifier_columns == ["meter_id"]

    roles_by_metric = {s.metric_value: s.role for s in lf.series}
    assert roles_by_metric["power_kw"] == "POWER"
    assert roles_by_metric["energy_kwh"] == "ENERGY"

    # every resolved series carries confidence + evidence + provenance
    for s in lf.series:
        assert 0.0 <= s.confidence <= 1.0
        assert isinstance(s.evidence, list) and len(s.evidence) >= 1
        assert s.source in ("AUTO", "CONFIG")


def test_long_format_grouped_by_meter_produces_one_series_per_meter():
    result = run_pipeline(fixture_path("long_format_meter.csv"))
    lf = result.long_format_mapping
    # 2 metrics x 2 meters = 4 distinct logical series
    assert len(lf.series) == 4
    meters_seen = {s.group_keys.get("meter_id") for s in lf.series}
    assert meters_seen == {"M1", "M2"}


def test_long_format_canonical_rows_have_correct_units_and_context():
    result = run_pipeline(fixture_path("long_format_meter.csv"))
    canon = result.canonical_df
    power_rows = canon[canon["source_column"] == "power_kw"]
    assert not power_rows.empty
    assert (power_rows["role"] == "POWER").all()
    assert (power_rows["normalized_unit"] == "kW").all()
    assert power_rows["normalized_value"].notna().all()
    # meter_id context is preserved per logical series, never fabricated
    assert set(power_rows["meter_id"].dropna().unique()) == {"M1", "M2"}


def test_long_format_mixed_recognized_and_unrecognized_metrics():
    result = run_pipeline(fixture_path("long_format_mixed_metrics.csv"))
    lf = result.long_format_mapping
    assert lf.applicable is True

    roles_by_metric = {s.metric_value: s.role for s in lf.series if s.metric_value == "active_power"}
    assert roles_by_metric["active_power"] == "POWER"

    energy_roles = {s.role for s in lf.series if s.metric_value == "consumption"}
    assert energy_roles == {"ENERGY"}

    # the unrecognized metric must NEVER be fabricated into a real role
    unknown_roles = {s.role for s in lf.series if s.metric_value == "widget_index"}
    assert unknown_roles == {"UNKNOWN"}
    assert "widget_index" in lf.unresolved_metrics

    # unresolved metrics never produce canonical rows
    assert "widget_index" not in set(result.canonical_df["source_column"])

    # capabilities are still available thanks to the recognized metrics
    caps = result.capability_report.as_dict()
    assert caps["Forecasting"].status == "AVAILABLE"
    assert caps["Anomaly Detection"].status == "AVAILABLE"


def test_long_format_multiple_meters_each_get_independent_series():
    result = run_pipeline(fixture_path("long_format_mixed_metrics.csv"))
    lf = result.long_format_mapping
    # 3 recognized/unrecognized metrics x 3 meters = 9 distinct logical series
    assert len(lf.series) == 9
    devices_seen = {s.group_keys.get("device_id") for s in lf.series}
    assert devices_seen == {"M1", "M2", "M3"}
    # each device has its own resolved POWER series with its own row count
    power_series = [s for s in lf.series if s.metric_value == "active_power"]
    assert len(power_series) == 3
    assert all(s.row_count == 40 for s in power_series)


def test_long_format_optional_unit_column_used_when_name_has_no_hint():
    result = run_pipeline(fixture_path("long_format_with_unit_column.csv"))
    lf = result.long_format_mapping
    assert lf.unit_column == "unit"

    by_metric = {s.metric_value: s for s in lf.series}
    assert by_metric["power"].role == "POWER"
    assert by_metric["power"].unit == "kW"
    assert by_metric["energy"].role == "ENERGY"
    assert by_metric["energy"].unit == "kWh"

    canon = result.canonical_df
    assert (canon[canon["source_column"] == "power"]["normalized_unit"] == "kW").all()
    assert (canon[canon["source_column"] == "energy"]["normalized_unit"] == "kWh").all()


def test_long_format_metric_config_override():
    config = {"metric_mappings": {"widget_index": {"role": "OCCUPANCY"}}}
    result = run_pipeline(fixture_path("long_format_mixed_metrics.csv"), config_path=None)
    # sanity check without config: widget_index is unresolved
    assert "widget_index" in result.long_format_mapping.unresolved_metrics

    import json
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    try:
        result2 = run_pipeline(fixture_path("long_format_mixed_metrics.csv"), config_path=config_path)
    finally:
        os.unlink(config_path)

    by_metric = {s.metric_value: s for s in result2.long_format_mapping.series if s.metric_value == "widget_index"}
    assert all(s.role == "OCCUPANCY" and s.source == "CONFIG" and s.confidence == 1.0 for s in by_metric.values())
    assert "widget_index" not in result2.long_format_mapping.unresolved_metrics


def test_long_format_invalid_metric_config_role_rejected():
    from src.pipeline import PipelineError
    import json
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"metric_mappings": {"widget_index": {"role": "NOT_A_ROLE"}}}, f)
        config_path = f.name
    try:
        try:
            run_pipeline(fixture_path("long_format_mixed_metrics.csv"), config_path=config_path)
            assert False, "expected PipelineError for invalid metric role"
        except PipelineError as e:
            assert "invalid role" in str(e)
    finally:
        os.unlink(config_path)


def test_wide_format_unaffected_by_long_format_module():
    """Regression guard: WIDE-format datasets must behave exactly as before."""
    result = run_pipeline(fixture_path("wide_energy.csv"))
    assert result.long_format_mapping.applicable is False
    assert result.long_format_mapping.series == []
    assert result.long_format_mapping.canonical_df.empty
    # capabilities/canonical behavior for WIDE data is untouched
    caps = result.capability_report.as_dict()
    assert caps["Weather Context"].status == "AVAILABLE"


def test_build_long_format_mapping_not_applicable_for_wide_profile():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3]})
    profile = profile_dataframe(df)
    from src.mapping import MappingResult

    empty_mapping = MappingResult(timestamp_column=None, mapped={}, unresolved=["a"], mapped_count=0, unresolved_count=1)
    result = build_long_format_mapping(df, profile, empty_mapping)
    assert result.applicable is False
    assert result.canonical_df.empty


def test_build_long_format_mapping_rejects_invalid_metric_mappings_shape():
    result = run_pipeline_config_error_helper()
    assert result


def run_pipeline_config_error_helper():
    # Confirms MappingConfigError is raised (not silently ignored) for a
    # malformed metric_mappings config passed directly to the module.
    import pandas as pd

    from src.mapping import MappingResult, ColumnMapping

    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00", "2026-01-01 01:00"],
            "meter_id": ["M1", "M1"],
            "metric": ["power_kw", "power_kw"],
            "value": [1.0, 2.0],
        }
    )
    profile = profile_dataframe(df)
    # Force the LONG-format hint directly rather than relying on the
    # profiler's cardinality heuristics (this test targets the config
    # validation path inside build_long_format_mapping, not detection).
    profile.format_hint = "LONG"
    profile.long_format_columns = {"metric_column": "metric", "value_column": "value"}
    mapping_result = MappingResult(
        timestamp_column="timestamp",
        mapped={
            "timestamp": ColumnMapping(column="timestamp", role="TIMESTAMP", unit=None, confidence=1.0, source="AUTO"),
            "meter_id": ColumnMapping(column="meter_id", role="METER_ID", unit=None, confidence=0.85, source="AUTO"),
        },
        unresolved=["metric", "value"],
        mapped_count=2,
        unresolved_count=2,
    )
    try:
        build_long_format_mapping(df, profile, mapping_result, config={"metric_mappings": "not_an_object"})
        return False
    except MappingConfigError:
        return True
