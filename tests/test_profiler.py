import pandas as pd

from src.loader import load_file
from src.profiler import profile_dataframe
from tests.conftest import fixture_path


def test_profile_basic_shape():
    result = load_file(fixture_path("wide_energy.csv"))
    profile = profile_dataframe(result.dataframe)
    assert profile.row_count == 200
    assert profile.column_count == 5
    assert set(profile.column_names) == set(result.dataframe.columns)


def test_profile_detects_wide_format():
    result = load_file(fixture_path("wide_energy.csv"))
    profile = profile_dataframe(result.dataframe)
    assert profile.format_hint == "WIDE"


def test_profile_detects_long_format():
    result = load_file(fixture_path("long_format_meter.csv"))
    profile = profile_dataframe(result.dataframe)
    assert profile.format_hint == "LONG"
    assert profile.long_format_columns["metric_column"] == "metric"
    assert profile.long_format_columns["value_column"] == "value"


def test_profile_null_counts():
    result = load_file(fixture_path("missing_values.csv"))
    profile = profile_dataframe(result.dataframe)
    col = profile.columns["power_kw"]
    assert col.null_count == 4
    assert col.null_ratio == round(4 / 30, 4)


def test_profile_duplicate_row_count():
    result = load_file(fixture_path("duplicate_rows.csv"))
    profile = profile_dataframe(result.dataframe)
    assert profile.duplicate_row_count == 2


def test_profile_status_column_detected_as_binary():
    result = load_file(fixture_path("status_on_off.csv"))
    profile = profile_dataframe(result.dataframe)
    assert profile.columns["chiller_status"].suspected_binary_status is True


def test_profile_identifier_column_detected():
    result = load_file(fixture_path("no_timestamp.csv"))
    profile = profile_dataframe(result.dataframe)
    assert profile.columns["device_id"].suspected_identifier is True


def test_profile_handles_empty_dataframe_columns_only():
    df = pd.DataFrame({"a": [], "b": []})
    profile = profile_dataframe(df)
    assert profile.row_count == 0
    assert profile.column_count == 2
