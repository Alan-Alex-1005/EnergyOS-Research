from src.loader import load_file
from src.schema import detect_timestamp
from tests.conftest import fixture_path


def test_detects_standard_timestamp_column():
    result = load_file(fixture_path("simple_timestamp_energy.csv"))
    info = detect_timestamp(result.dataframe)
    assert info.detected is True
    assert info.column == "timestamp"
    assert info.confidence > 0.5


def test_detects_weird_timestamp_column_name():
    result = load_file(fixture_path("weird_timestamp_name.csv"))
    info = detect_timestamp(result.dataframe)
    assert info.detected is True
    assert info.column == "ReadingTime"


def test_numeric_column_never_treated_as_timestamp():
    """
    Critical rule (spec section 5): a plain numeric column must NEVER be
    fallback-parsed as a Unix-epoch timestamp, even if it looks epoch-like
    or is named suggestively.
    """
    result = load_file(fixture_path("ambiguous_numeric.csv"))
    info = detect_timestamp(result.dataframe)
    assert info.detected is True
    assert info.column == "timestamp"  # the real datetime column, not reading_id
    assert info.column != "reading_id"


def test_no_timestamp_column_present():
    result = load_file(fixture_path("no_timestamp.csv"))
    info = detect_timestamp(result.dataframe)
    assert info.detected is False


def test_sampling_interval_inferred():
    result = load_file(fixture_path("simple_timestamp_energy.csv"))
    info = detect_timestamp(result.dataframe)
    assert info.sampling_interval is not None
    assert "hour" in info.sampling_interval


def test_duplicate_timestamps_counted():
    result = load_file(fixture_path("duplicate_timestamps.csv"))
    info = detect_timestamp(result.dataframe)
    assert info.detected is True
    assert info.duplicate_timestamp_count > 0


def test_naive_timestamps_preserved_as_naive():
    result = load_file(fixture_path("simple_timestamp_energy.csv"))
    info = detect_timestamp(result.dataframe)
    assert info.is_timezone_aware is False
    assert info.assumed_timezone is None
