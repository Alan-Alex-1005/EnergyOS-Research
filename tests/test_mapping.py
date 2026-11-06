import pytest

from src.loader import load_file
from src.mapping import (
    MappingConfigError,
    build_mapping,
    load_config,
)
from src.profiler import profile_dataframe
from src.role_detection import detect_roles
from src.schema import detect_timestamp
from tests.conftest import fixture_path


def _pipeline_inputs(fixture_name):
    result = load_file(fixture_path(fixture_name))
    df = result.dataframe
    profile = profile_dataframe(df)
    ts_info = detect_timestamp(df)
    non_ts_cols = {c: profile.columns[c] for c in df.columns if c != ts_info.column}
    suggestions = detect_roles(non_ts_cols)
    return df, ts_info, suggestions


def test_automatic_mapping_maps_known_columns():
    df, ts_info, suggestions = _pipeline_inputs("wide_energy.csv")
    result = build_mapping(list(df.columns), ts_info.column, suggestions)
    assert result.timestamp_column == "DateTime"
    assert result.mapped["Main_Power_kW"].role == "POWER"
    assert result.mapped["Main_Power_kW"].source == "AUTO"


def test_low_confidence_columns_are_unresolved():
    df, ts_info, suggestions = _pipeline_inputs("strange_energy_name.csv")
    result = build_mapping(list(df.columns), ts_info.column, suggestions)
    assert "ChillerLoadReading" in result.unresolved


def test_config_override_wins_over_automatic():
    df, ts_info, suggestions = _pipeline_inputs("config_override_base.csv")
    config = load_config(fixture_path("config_override.json"))
    result = build_mapping(list(df.columns), ts_info.column, suggestions, config=config)
    assert result.mapped["Main_Load"].role == "POWER"
    assert result.mapped["Main_Load"].source == "CONFIG"
    assert result.mapped["Main_Load"].confidence == 1.0
    assert result.mapped["AHU_State"].role == "STATUS"
    assert result.timestamp_column == "DateTime"


def test_load_config_rejects_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(MappingConfigError, match="not valid JSON"):
        load_config(str(bad))


def test_load_config_rejects_invalid_role(tmp_path):
    bad = tmp_path / "bad_role.json"
    bad.write_text('{"mappings": {"Col": {"role": "NOT_A_REAL_ROLE"}}}')
    with pytest.raises(MappingConfigError, match="invalid role"):
        load_config(str(bad))


def test_build_mapping_rejects_config_column_not_in_dataset():
    df, ts_info, suggestions = _pipeline_inputs("config_override_base.csv")
    config = {"mappings": {"DoesNotExist": {"role": "POWER"}}}
    with pytest.raises(MappingConfigError, match="does not exist"):
        build_mapping(list(df.columns), ts_info.column, suggestions, config=config)


def test_mapped_and_unresolved_counts_consistent():
    df, ts_info, suggestions = _pipeline_inputs("mixed_known_unknown.csv")
    result = build_mapping(list(df.columns), ts_info.column, suggestions)
    assert result.mapped_count == len(result.mapped)
    assert result.unresolved_count == len(result.unresolved)
    assert result.mapped_count + result.unresolved_count == len(df.columns)
