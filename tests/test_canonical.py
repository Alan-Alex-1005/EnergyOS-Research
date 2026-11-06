import pandas as pd

from src.canonical import to_canonical_records
from src.pipeline import run_pipeline
from tests.conftest import fixture_path


def test_canonical_dataframe_has_expected_columns():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    expected = {
        "timestamp", "source_column", "role", "raw_value", "raw_unit",
        "normalized_value", "normalized_unit", "quality",
    }
    assert expected.issubset(set(result.canonical_df.columns))


def test_canonical_normalizes_kw_and_kwh():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    power_rows = result.canonical_df[result.canonical_df["source_column"] == "Main_Power_kW"]
    assert (power_rows["normalized_unit"] == "kW").all()
    assert power_rows["normalized_value"].notna().all()


def test_no_equipment_context_left_null_not_fabricated():
    """
    wide_energy.csv has no equipment/device/meter/site/building column at
    all — canonical records must leave that context null, never invent it.
    """
    result = run_pipeline(fixture_path("wide_energy.csv"))
    assert result.canonical_df["equipment_id"].isna().all()
    assert result.canonical_df["device_id"].isna().all()
    assert result.canonical_df["site"].isna().all()


def test_long_format_meter_id_preserved_as_context():
    result = run_pipeline(fixture_path("long_format_meter.csv"))
    # meter_id column should map to METER_ID and appear as context, not fabricated
    if "meter_id" in result.canonical_df.columns:
        pass
    assert "meter_id" in result.canonical_df.columns


def test_to_canonical_records_respects_limit():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    records = to_canonical_records(result.canonical_df, limit=5)
    assert len(records) == 5
    assert records[0].quality == "MEASURED"
