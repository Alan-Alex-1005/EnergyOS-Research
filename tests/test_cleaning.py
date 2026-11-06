from src.cleaning import clean_dataset
from src.loader import load_file
from src.mapping import build_mapping
from src.profiler import profile_dataframe
from src.role_detection import detect_roles
from src.schema import detect_timestamp
from tests.conftest import fixture_path


def _run_to_mapping(fixture_name):
    result = load_file(fixture_path(fixture_name))
    df = result.dataframe
    profile = profile_dataframe(df)
    ts_info = detect_timestamp(df)
    non_ts_cols = {c: profile.columns[c] for c in df.columns if c != ts_info.column}
    suggestions = detect_roles(non_ts_cols)
    mapping_result = build_mapping(list(df.columns), ts_info.column, suggestions)
    return df, ts_info, mapping_result


def test_duplicate_rows_removed():
    df, ts_info, mapping_result = _run_to_mapping("duplicate_rows.csv")
    cleaned, report = clean_dataset(df, ts_info, mapping_result)
    assert report.action_counts["REMOVED_DUPLICATE"] == 2
    assert report.rows_after == report.rows_before - 2
    assert cleaned.shape[0] == report.rows_after


def test_missing_values_counted_not_filled():
    df, ts_info, mapping_result = _run_to_mapping("missing_values.csv")
    cleaned, report = clean_dataset(df, ts_info, mapping_result)
    assert report.action_counts["MISSING_VALUE"] == 4
    assert report.per_column_missing.get("power_kw") == 4
    # never silently filled — nulls still present in cleaned data
    assert cleaned["power_kw"].isna().sum() == 4


def test_duplicate_timestamps_across_different_series_not_flagged():
    """
    Part A2 fix: two DIFFERENT meters legitimately reporting at the same
    wall-clock timestamp is normal, not corruption — must NOT be flagged
    as an unresolved duplicate-timestamp conflict.
    """
    df, ts_info, mapping_result = _run_to_mapping("duplicate_timestamps.csv")
    cleaned, report = clean_dataset(df, ts_info, mapping_result)
    assert report.action_counts["UNRESOLVED"] == 0
    # rows are NOT dropped just for sharing a timestamp across series
    assert cleaned.shape[0] == df.shape[0]


def test_true_duplicate_timestamp_within_same_series_still_flagged():
    """
    Part A2 counterpart: a genuine repeat WITHIN the same meter/series
    must still be flagged as unresolved — this is real conflict, not a
    cross-series artifact.
    """
    df, ts_info, mapping_result = _run_to_mapping("duplicate_timestamps_same_series.csv")
    cleaned, report = clean_dataset(df, ts_info, mapping_result)
    assert report.action_counts["UNRESOLVED"] == 4  # 2 timestamps x 2 rows each
    assert cleaned.shape[0] == df.shape[0]  # still not dropped, just flagged


def test_sorted_by_timestamp():
    df, ts_info, mapping_result = _run_to_mapping("simple_timestamp_energy.csv")
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    from src.schema import detect_timestamp as dt

    ts_info2 = dt(shuffled)
    from src.profiler import profile_dataframe as pf
    from src.role_detection import detect_roles as dr
    from src.mapping import build_mapping as bm

    profile = pf(shuffled)
    non_ts = {c: profile.columns[c] for c in shuffled.columns if c != ts_info2.column}
    suggestions = dr(non_ts)
    mapping_result2 = bm(list(shuffled.columns), ts_info2.column, suggestions)
    cleaned, _ = clean_dataset(shuffled, ts_info2, mapping_result2)
    assert cleaned["_parsed_timestamp"].is_monotonic_increasing


def test_no_timestamp_dataset_skips_invalid_timestamp_action():
    df, ts_info, mapping_result = _run_to_mapping("no_timestamp.csv")
    cleaned, report = clean_dataset(df, ts_info, mapping_result)
    assert report.action_counts["INVALID_TIMESTAMP"] == 0
    assert cleaned.shape[0] == df.shape[0]


def test_out_of_range_values_flagged_not_dropped_or_altered():
    import pandas as pd

    from src.mapping import build_mapping
    from src.profiler import profile_dataframe
    from src.role_detection import detect_roles
    from src.schema import detect_timestamp

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
            "power_kw": [5, -3, 10, 8, -1, 6, 7, 9, 4, 3],
        }
    )
    ts_info = detect_timestamp(df)
    profile = profile_dataframe(df)
    non_ts = {c: profile.columns[c] for c in df.columns if c != ts_info.column}
    suggestions = detect_roles(non_ts)
    mapping_result = build_mapping(list(df.columns), ts_info.column, suggestions)

    cleaned, report = clean_dataset(df, ts_info, mapping_result)
    assert report.action_counts["OUT_OF_RANGE"] == 2
    assert report.per_column_out_of_range["power_kw"] == 2
    # values are preserved as-is, never dropped or clamped
    assert cleaned.shape[0] == 10
    assert -3 in cleaned["power_kw"].values
    assert -1 in cleaned["power_kw"].values


def test_invalid_timestamp_rows_excluded_and_counted():
    import pandas as pd

    from src.mapping import build_mapping
    from src.profiler import profile_dataframe
    from src.role_detection import detect_roles
    from src.schema import detect_timestamp

    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "not-a-date", "2026-01-01 02:00:00", "also-bad"],
            "power_kw": [5, 6, 7, 8],
        }
    )
    ts_info = detect_timestamp(df)
    profile = profile_dataframe(df)
    non_ts = {c: profile.columns[c] for c in df.columns if c != ts_info.column}
    suggestions = detect_roles(non_ts)
    mapping_result = build_mapping(list(df.columns), ts_info.column, suggestions)

    cleaned, report = clean_dataset(df, ts_info, mapping_result)
    assert report.action_counts["INVALID_TIMESTAMP"] == 2
    assert cleaned.shape[0] == 2
