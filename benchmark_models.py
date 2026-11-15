
"""Read-only pilot benchmark using EnergyOS' existing forecast engine.

Run from project root: python benchmark_models.py
No forecasting or TrustAI code is changed.
Train/validation/test = 70/15/15 chronological.
"""

from pathlib import Path
import csv
import pandas as pd

from src.analysis_series import AnalysisSeries
from src.timeseries import (
    prepare_series,
    chronological_train_validation_test_split,
)
from src.forecasting.engine import run_forecast_engine
from src.forecasting.reliability import assess_forecast_reliability


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark_results"


# Two independent sources, four distinct real meter-series.
# No cumulative-counter series.
DATASETS = [
    (
        "COLLECTiEF B01 main",
        "B01_meters.csv",
        "Time",
        "main",
        "ENERGY",
        "kWh",
        "COLLECTiEF",
    ),
    (
        "COLLECTiEF B01 hp2",
        "B01_meters.csv",
        "Time",
        "hp2",
        "ENERGY",
        "kWh",
        "COLLECTiEF",
    ),
    (
        "L08 B01 ES08 power",
        "data/L08/L08_B01/1h/adrenalin.1h.L08.B01.ES08.P.csv",
        "timestamp [UTC]",
        "value P [Total active power (kW)]",
        "POWER",
        "kW",
        "L08",
    ),
    (
        "L08 B01 ES09 power",
        "data/L08/L08_B01/1h/adrenalin.1h.L08.B01.ES09.P.csv",
        "timestamp [UTC]",
        "value P [Total active power (kW)]",
        "POWER",
        "kW",
        "L08",
    ),
]


def main():
    OUT.mkdir(exist_ok=True)

    selections = []
    candidates = []

    for (
        name,
        filename,
        time_col,
        value_col,
        role,
        unit,
        source,
    ) in DATASETS:

        df = pd.read_csv(ROOT / filename)

        ts = pd.to_datetime(
            df[time_col],
            errors="coerce",
            utc=(source == "L08"),
        )

        vals = pd.to_numeric(
            df[value_col],
            errors="coerce",
        )

        valid_ts = ts.notna()

        # Preserve missing load values for the preparation
        # module to assess.
        df2 = (
            pd.DataFrame(
                {
                    "ts": ts[valid_ts],
                    "value": vals[valid_ts],
                }
            )
            .sort_values("ts", kind="stable")
            .reset_index(drop=True)
        )

        if df2.empty:
            raise ValueError(
                f"{name}: no usable timestamps"
            )

        series = AnalysisSeries(
            series_id=name,
            role=role,
            unit=unit,
            timestamps=df2.ts,
            values=df2.value,
            used_normalized=False,
            sample_count=len(df2),
        )

        prepared = prepare_series(series)

        split = chronological_train_validation_test_split(
            prepared
        )

        result = run_forecast_engine(
            prepared,
            split,
        )

        reliability = (
            assess_forecast_reliability(
                result.forecast_result
            )
            if result.forecast_result is not None
            else None
        )

        selected = next(
            (
                e
                for e in result.comparison
                if e.model_name == result.selected_model
            ),
            None,
        )

        selections.append(
            {
                "dataset": name,
                "source": source,
                "unit": unit,
                "original_rows": len(df),
                "usable_rows": prepared.prepared_sample_count,
                "missing_values": prepared.missing_value_count,
                "duplicate_timestamps": (
                    prepared.duplicate_timestamp_count
                ),
                "gap_count": prepared.gap_count,
                "is_regular": prepared.is_regular,
                "train_n": split.train_samples,
                "validation_n": split.validation_samples,
                "test_n": split.test_samples,
                "selected_model": (
                    result.selected_model or "NONE"
                ),
                "validation_rmse": (
                    selected.validation_rmse
                    if selected else None
                ),
                "test_rmse": (
                    result.forecast_result.rmse
                    if result.forecast_result else None
                ),
                "test_mae": (
                    result.forecast_result.mae
                    if result.forecast_result else None
                ),
                "test_r2": (
                    result.forecast_result.r2
                    if result.forecast_result else None
                ),
                "trustai": (
                    reliability.status
                    if reliability else "UNAVAILABLE"
                ),
                "trustai_reasons": (
                    " | ".join(reliability.reasons)
                    if reliability
                    else result.selection_reason
                ),
            }
        )

        for e in result.comparison:
            candidates.append(
                {
                    "dataset": name,
                    "model": e.model_name,
                    "applicable": e.applicable,
                    "attempted": e.attempted,
                    "reason": e.reason,
                    "validation_rmse": e.validation_rmse,
                    "test_rmse": e.rmse,
                    "test_mae": e.mae,
                    "test_r2": e.r2,
                    "selected": (
                        e.model_name == result.selected_model
                    ),
                }
            )

    for fname, rows in [
        ("benchmark_summary.csv", selections),
        ("benchmark_all_models.csv", candidates),
    ]:
        with (OUT / fname).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(rows)

        print("Wrote:", OUT / fname)

    for row in selections:
        print(
            f"{row['dataset']}: "
            f"{row['selected_model']} | "
            f"val RMSE={row['validation_rmse']} | "
            f"test RMSE={row['test_rmse']} | "
            f"R2={row['test_r2']} | "
            f"TrustAI={row['trustai']}"
        )


if __name__ == "__main__":
    main()
