"""
generate_fixtures.py — TEST FIXTURE GENERATOR (not a test itself)

Regenerates every fixture file under tests/fixtures/. Run manually with:

    python tests/fixtures/generate_fixtures.py

All files produced here are TEST DATA ONLY — synthetic, small, and
clearly not research results.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _path(name: str) -> str:
    return os.path.join(HERE, name)


def gen_simple_timestamp_energy():
    # 1. simple timestamp + energy CSV
    ts = pd.date_range("2026-01-01", periods=48, freq="h")
    df = pd.DataFrame({"timestamp": ts, "energy_kwh": np.round(np.random.uniform(5, 20, 48), 2)})
    df.to_csv(_path("simple_timestamp_energy.csv"), index=False)


def gen_wide_energy():
    # 2. wide energy CSV
    ts = pd.date_range("2026-02-01", periods=200, freq="15min")
    df = pd.DataFrame(
        {
            "DateTime": ts,
            "Main_Power_kW": np.round(np.random.uniform(50, 150, 200), 2),
            "Energy_kWh": np.round(np.random.uniform(10, 40, 200), 2),
            "OutdoorTemp_C": np.round(np.random.uniform(15, 35, 200), 1),
            "AHU_State": np.random.choice(["ON", "OFF"], 200),
        }
    )
    df.to_csv(_path("wide_energy.csv"), index=False)


def gen_long_format_meter():
    # 3. long-format meter CSV — multiple metrics per meter/timestamp so the
    # "metric" column is genuinely categorical (>1 distinct value).
    ts = pd.date_range("2026-03-01", periods=50, freq="30min")
    meters = ["M1", "M2"]
    metrics = ["power_kw", "energy_kwh"]
    rows = []
    for m in meters:
        for t in ts:
            for metric in metrics:
                value = round(np.random.uniform(10, 60), 2) if metric == "power_kw" else round(np.random.uniform(1, 10), 2)
                rows.append({"timestamp": t, "meter_id": m, "metric": metric, "value": value})
    df = pd.DataFrame(rows)
    df.to_csv(_path("long_format_meter.csv"), index=False)


def gen_long_format_mixed_metrics():
    # long-format with recognized metrics (power, energy) mixed with an
    # unrecognized one, across 3 meters — tests that mixed
    # recognized/unrecognized metrics AND multiple meters are handled
    # correctly and independently per logical series.
    ts = pd.date_range("2026-04-01", periods=40, freq="30min")
    meters = ["M1", "M2", "M3"]
    metrics = ["active_power", "consumption", "widget_index"]  # last one is unrecognized
    rows = []
    for m in meters:
        for t in ts:
            for metric in metrics:
                if metric == "active_power":
                    value = round(np.random.uniform(10, 60), 2)
                elif metric == "consumption":
                    value = round(np.random.uniform(1, 10), 2)
                else:
                    value = round(np.random.uniform(0, 1), 3)
                rows.append({"timestamp": t, "device_id": m, "metric": metric, "value": value})
    df = pd.DataFrame(rows)
    df.to_csv(_path("long_format_mixed_metrics.csv"), index=False)


def gen_long_format_with_unit_column():
    # long-format where the metric names alone carry NO unit hint, but an
    # explicit "unit" column supplies it — tests optional unit-column
    # detection and per-series unit resolution from that column.
    ts = pd.date_range("2026-05-01", periods=30, freq="h")
    rows = []
    for t in ts:
        rows.append({"timestamp": t, "meter_id": "M1", "metric": "power", "value": round(np.random.uniform(1, 5), 2), "unit": "kW"})
        rows.append({"timestamp": t, "meter_id": "M1", "metric": "energy", "value": round(np.random.uniform(1, 5), 2), "unit": "kWh"})
    df = pd.DataFrame(rows)
    df.to_csv(_path("long_format_with_unit_column.csv"), index=False)


def gen_xlsx_single_sheet():
    # 4. XLSX (single sheet)
    ts = pd.date_range("2026-01-01", periods=30, freq="h")
    df = pd.DataFrame({"RecordedAt": ts, "Active_Power_kW": np.round(np.random.uniform(1, 5, 30), 2)})
    with pd.ExcelWriter(_path("sample_single_sheet.xlsx"), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)


def gen_xlsx_multi_sheet():
    # 4b. XLSX (multi sheet, ambiguous unless --sheet given)
    ts = pd.date_range("2026-01-01", periods=20, freq="h")
    df1 = pd.DataFrame({"RecordedAt": ts, "Power_kW": np.round(np.random.uniform(1, 5, 20), 2)})
    df2 = pd.DataFrame({"RecordedAt": ts, "Power_kW": np.round(np.random.uniform(1, 5, 20), 2)})
    with pd.ExcelWriter(_path("sample_multi_sheet.xlsx"), engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="SiteA", index=False)
        df2.to_excel(writer, sheet_name="SiteB", index=False)


def gen_weird_timestamp_name():
    # 5. weird timestamp column name
    ts = pd.date_range("2026-01-01", periods=25, freq="h")
    df = pd.DataFrame({"ReadingTime": ts, "load_kw": np.round(np.random.uniform(2, 9, 25), 2)})
    df.to_csv(_path("weird_timestamp_name.csv"), index=False)


def gen_missing_values():
    # 6. missing values
    ts = pd.date_range("2026-01-01", periods=30, freq="h")
    power = np.round(np.random.uniform(2, 9, 30), 2)
    power = power.astype(object)
    for i in (3, 7, 15, 22):
        power[i] = None
    df = pd.DataFrame({"timestamp": ts, "power_kw": power})
    df.to_csv(_path("missing_values.csv"), index=False)


def gen_duplicate_rows():
    # 7. duplicate rows (exact duplicates)
    ts = pd.date_range("2026-01-01", periods=15, freq="h")
    df = pd.DataFrame({"timestamp": ts, "power_kw": np.round(np.random.uniform(2, 9, 15), 2)})
    df = pd.concat([df, df.iloc[[2, 5]]], ignore_index=True)
    df.to_csv(_path("duplicate_rows.csv"), index=False)


def gen_duplicate_timestamps():
    # 8. duplicate timestamps (different data, same timestamp — e.g. two meters)
    #    This is a LEGITIMATE cross-series repeat, not corruption: two
    #    different meters both reporting at the same wall-clock time.
    ts = pd.date_range("2026-01-01", periods=10, freq="h")
    rows = []
    for t in ts:
        rows.append({"timestamp": t, "meter_id": "A", "power_kw": round(np.random.uniform(2, 9), 2)})
        rows.append({"timestamp": t, "meter_id": "B", "power_kw": round(np.random.uniform(2, 9), 2)})
    df = pd.DataFrame(rows)
    df.to_csv(_path("duplicate_timestamps.csv"), index=False)


def gen_duplicate_timestamps_same_series():
    # 8b. TRUE duplicate timestamp WITHIN the same series (same meter) —
    # this is genuine corruption/conflict and must still be flagged.
    ts = list(pd.date_range("2026-01-01", periods=10, freq="h"))
    ts_with_dupe = ts + [ts[3], ts[7]]  # repeat two timestamps for the SAME meter
    rows = [{"timestamp": t, "meter_id": "A", "power_kw": round(np.random.uniform(2, 9), 2)} for t in ts_with_dupe]
    df = pd.DataFrame(rows)
    df.to_csv(_path("duplicate_timestamps_same_series.csv"), index=False)


def gen_ambiguous_numeric():
    # 9. ambiguous numeric column that must NOT be treated as a timestamp
    ts = pd.date_range("2026-01-01", periods=20, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "reading_id": np.arange(1700000000, 1700000000 + 20),  # looks epoch-like but is just an ID
            "power_kw": np.round(np.random.uniform(2, 9, 20), 2),
        }
    )
    df.to_csv(_path("ambiguous_numeric.csv"), index=False)


def gen_status_on_off():
    # 10. status ON/OFF column
    ts = pd.date_range("2026-01-01", periods=20, freq="h")
    df = pd.DataFrame({"timestamp": ts, "chiller_status": np.random.choice(["ON", "OFF"], 20)})
    df.to_csv(_path("status_on_off.csv"), index=False)


def gen_strange_energy_name():
    # 11. strange/non-standard energy column name (no recognizable hint)
    ts = pd.date_range("2026-01-01", periods=20, freq="h")
    df = pd.DataFrame({"timestamp": ts, "ChillerLoadReading": np.round(np.random.uniform(10, 90, 20), 2)})
    df.to_csv(_path("strange_energy_name.csv"), index=False)


def gen_no_timestamp():
    # 12. file with no timestamp column at all
    df = pd.DataFrame(
        {
            "device_id": [f"D{i}" for i in range(10)],
            "power_kw": np.round(np.random.uniform(1, 10, 10), 2),
        }
    )
    df.to_csv(_path("no_timestamp.csv"), index=False)


def gen_no_energy_data():
    # 13. file with no energy/power data at all
    ts = pd.date_range("2026-01-01", periods=15, freq="h")
    df = pd.DataFrame({"timestamp": ts, "room": np.random.choice(["101", "102", "103"], 15)})
    df.to_csv(_path("no_energy_data.csv"), index=False)


def gen_mixed_known_unknown():
    # 14. mixed useful and unknown columns
    ts = pd.date_range("2026-01-01", periods=25, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "power_kw": np.round(np.random.uniform(2, 9, 25), 2),
            "zone": np.random.choice(["Z1", "Z2"], 25),
            "misc_code": np.random.choice(["X1", "X2", "X3", "X4", "X5", "X6", "X7"], 25),
        }
    )
    df.to_csv(_path("mixed_known_unknown.csv"), index=False)


def gen_config_override_base():
    # 15. explicit config override target
    ts = pd.date_range("2026-01-01", periods=20, freq="h")
    df = pd.DataFrame(
        {
            "DateTime": ts,
            "Main_Load": np.round(np.random.uniform(2, 9, 20), 2),
            "AHU_State": np.random.choice(["ON", "OFF"], 20),
        }
    )
    df.to_csv(_path("config_override_base.csv"), index=False)


def gen_no_columns_and_edge_files():
    # extra edge-case fixtures used by loader tests
    with open(_path("empty_file.csv"), "w") as f:
        pass  # 0 bytes

    with open(_path("malformed.csv"), "w") as f:
        f.write('a,b,c\n1,2\n"unterminated quote,3,4\n')


def main():
    np.random.seed(42)
    gen_simple_timestamp_energy()
    gen_wide_energy()
    gen_long_format_meter()
    gen_long_format_mixed_metrics()
    gen_long_format_with_unit_column()
    gen_xlsx_single_sheet()
    gen_xlsx_multi_sheet()
    gen_weird_timestamp_name()
    gen_missing_values()
    gen_duplicate_rows()
    gen_duplicate_timestamps()
    gen_duplicate_timestamps_same_series()
    gen_ambiguous_numeric()
    gen_status_on_off()
    gen_strange_energy_name()
    gen_no_timestamp()
    gen_no_energy_data()
    gen_mixed_known_unknown()
    gen_config_override_base()
    gen_no_columns_and_edge_files()
    print("Fixtures generated in", HERE)


if __name__ == "__main__":
    main()
