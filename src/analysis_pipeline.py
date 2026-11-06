"""
analysis_pipeline.py — Day 2: End-to-End Analysis Orchestrator

DATA UNDERSTANDING -> ENERGY SERIES SELECTION -> TIME-SERIES PREPARATION
-> FORECAST MODEL -> EXPECTED BEHAVIOUR -> FORECAST METRICS -> ANOMALY
ANALYSIS -> ANOMALY SEVERITY -> EXCESS-ENERGY ESTIMATION -> RESEARCH PNGs
-> results.json

This module contains ZERO analytical logic of its own — it only calls
into analysis_series/timeseries/forecasting/anomaly/impact/visualization
and assembles their outputs. The terminal report and results.json both
consume exactly this one result object (Part H/R: no independent
recomputation in different UI layers).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from . import pipeline as day1_pipeline
from .analysis_series import (
    AnalysisSeries,
    SeriesSufficiency,
    assess_series_sufficiency,
    build_analysis_series,
    select_primary_series,
)
from .anomaly.engine import AnomalyEngineResult, run_anomaly_engine
from .anomaly.robust import DEFAULT_BASE_THRESHOLD
from .forecasting.engine import ForecastEngineResult, run_forecast_engine
from .impact.excess_energy import ExcessEnergyResult, compute_excess_energy
from .timeseries import (
    DEFAULT_TRAIN_VAL_TEST_RATIOS,
    PreparedSeries,
    TimeSeriesPrepError,
    TrainValidationTestSplit,
    chronological_train_validation_test_split,
    prepare_series,
)
from .visualization import research_plots as viz


class AnalysisConfigError(Exception):
    """Raised for invalid Day 2 'analysis' config values."""


@dataclass
class AnalysisConfig:
    train_ratio: float = DEFAULT_TRAIN_VAL_TEST_RATIOS[0]
    validation_ratio: float = DEFAULT_TRAIN_VAL_TEST_RATIOS[1]
    test_ratio: Optional[float] = None
    series_id: Optional[str] = None
    seasonal_period: Optional[int] = None
    anomaly_threshold: float = DEFAULT_BASE_THRESHOLD


def parse_analysis_config(config: Optional[dict]) -> AnalysisConfig:
    """Validate and extract the optional 'analysis' block from a Day-2 JSON config (Part U)."""
    block = (config or {}).get("analysis", {}) or {}
    if not isinstance(block, dict):
        raise AnalysisConfigError("config 'analysis' must be a JSON object.")

    train_ratio = block.get("train_ratio", DEFAULT_TRAIN_VAL_TEST_RATIOS[0])
    validation_ratio = block.get("validation_ratio", DEFAULT_TRAIN_VAL_TEST_RATIOS[1])
    test_ratio = block.get("test_ratio")
    series_id = block.get("series")
    seasonal_period = block.get("seasonal_period")
    anomaly_threshold = block.get("anomaly_threshold", DEFAULT_BASE_THRESHOLD)

    for name, value in (("train_ratio", train_ratio), ("validation_ratio", validation_ratio)):
        if not isinstance(value, (int, float)) or not (0.0 < value < 1.0):
            raise AnalysisConfigError(f"config analysis.{name} must be a number strictly between 0 and 1; got {value!r}.")
    if test_ratio is not None:
        if not isinstance(test_ratio, (int, float)) or not (0.0 < test_ratio < 1.0):
            raise AnalysisConfigError(f"config analysis.test_ratio must be a number strictly between 0 and 1; got {test_ratio!r}.")
        total = round(train_ratio + validation_ratio + test_ratio, 6)
        if abs(total - 1.0) > 1e-6:
            raise AnalysisConfigError(
                f"config analysis train_ratio + validation_ratio + test_ratio must sum to 1.0; got {total}."
            )
    if seasonal_period is not None and (not isinstance(seasonal_period, int) or seasonal_period <= 0):
        raise AnalysisConfigError(f"config analysis.seasonal_period must be a positive integer; got {seasonal_period!r}.")
    if not isinstance(anomaly_threshold, (int, float)) or anomaly_threshold <= 0:
        raise AnalysisConfigError(f"config analysis.anomaly_threshold must be a positive number; got {anomaly_threshold!r}.")
    if series_id is not None and not isinstance(series_id, str):
        raise AnalysisConfigError("config analysis.series must be a string series_id.")

    return AnalysisConfig(
        train_ratio=float(train_ratio),
        validation_ratio=float(validation_ratio),
        test_ratio=float(test_ratio) if test_ratio is not None else None,
        series_id=series_id,
        seasonal_period=seasonal_period,
        anomaly_threshold=float(anomaly_threshold),
    )


@dataclass
class SeriesAnalysisResult:
    series: AnalysisSeries
    sufficiency: SeriesSufficiency
    prepared: Optional[PreparedSeries] = None
    split: Optional[TrainValidationTestSplit] = None
    forecast: Optional[ForecastEngineResult] = None
    anomaly: Optional[AnomalyEngineResult] = None
    excess_energy: Optional[ExcessEnergyResult] = None
    figures: dict = field(default_factory=dict)
    skipped_reason: Optional[str] = None


@dataclass
class AnalysisRunResult:
    run_id: str
    output_dir: Optional[str]
    day1: object  # day1_pipeline.PipelineResult
    all_series: list[AnalysisSeries]
    all_sufficiency: list[SeriesSufficiency]
    selected_series_id: Optional[str]
    selection_reason: str
    primary: Optional[SeriesAnalysisResult]
    config: AnalysisConfig
    warnings: list[str] = field(default_factory=list)


def run_day2_analysis(
    path: str,
    sheet: Optional[str] = None,
    day1_config_path: Optional[str] = None,
    analysis_config: Optional[dict] = None,
    series_id_override: Optional[str] = None,
    output_root: str = "outputs",
    generate_figures: bool = True,
    day1_result=None,
) -> AnalysisRunResult:
    if day1_result is None:
        day1_result = day1_pipeline.run_pipeline(path, sheet=sheet, config_path=day1_config_path)

    cfg = parse_analysis_config(analysis_config)
    requested_series = series_id_override or cfg.series_id

    all_series = build_analysis_series(day1_result.canonical_df)
    all_sufficiency = [assess_series_sufficiency(s) for s in all_series]

    selected, selection_reason = select_primary_series(all_series, requested_series_id=requested_series)

    run_id = _make_run_id(path)
    output_dir = None
    if generate_figures:
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

    warnings: list[str] = []
    primary_result: Optional[SeriesAnalysisResult] = None

    if selected is not None:
        primary_result = _analyze_series(selected, cfg, output_dir if generate_figures else None)
    else:
        warnings.append(selection_reason)

    return AnalysisRunResult(
        run_id=run_id,
        output_dir=output_dir,
        day1=day1_result,
        all_series=all_series,
        all_sufficiency=all_sufficiency,
        selected_series_id=selected.series_id if selected else None,
        selection_reason=selection_reason,
        primary=primary_result,
        config=cfg,
        warnings=warnings,
    )


def _make_run_id(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    safe_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{safe_base}_{timestamp}"


def _analyze_series(series: AnalysisSeries, cfg: AnalysisConfig, output_dir: Optional[str]) -> SeriesAnalysisResult:
    sufficiency = assess_series_sufficiency(series)
    result = SeriesAnalysisResult(series=series, sufficiency=sufficiency)

    if sufficiency.status != "AVAILABLE":
        result.skipped_reason = sufficiency.reason
        return result

    prepared = prepare_series(series)
    result.prepared = prepared

    try:
        split = chronological_train_validation_test_split(
            prepared, train_ratio=cfg.train_ratio, validation_ratio=cfg.validation_ratio, test_ratio=cfg.test_ratio
        )
    except TimeSeriesPrepError as e:
        result.skipped_reason = str(e)
        return result
    result.split = split

    forecast = run_forecast_engine(prepared, split, seasonal_period_override=cfg.seasonal_period)
    result.forecast = forecast

    anomaly = run_anomaly_engine(
        series.series_id,
        forecast.selected_model_instance,
        split,
        forecast.forecast_result,
        base_threshold=cfg.anomaly_threshold,
    )
    result.anomaly = anomaly

    if forecast.forecast_result is not None:
        excess = compute_excess_energy(
            role=series.role,
            unit=prepared.unit,
            sampling_interval_seconds=prepared.sampling_interval_seconds,
            is_regular=prepared.is_regular,
            actual=np.asarray(forecast.forecast_result.actual),
            predicted=np.asarray(forecast.forecast_result.predicted),
        )
    else:
        excess = ExcessEnergyResult(available=False, reason="no forecast was produced for this series.")
    result.excess_energy = excess

    if output_dir:
        result.figures = _generate_figures(output_dir, series, prepared, forecast, anomaly, excess, cfg)

    return result


def _generate_figures(output_dir, series, prepared, forecast, anomaly, excess, cfg) -> dict:
    figures = {}
    profile_summary = {
        "file": series.provenance.get("source_column"),
        "rows": series.sample_count,
        "mapped_series": series.series_id,
        "time_start": str(prepared.timestamps.iloc[0]) if len(prepared.timestamps) else "n/a",
        "time_end": str(prepared.timestamps.iloc[-1]) if len(prepared.timestamps) else "n/a",
        "sampling_interval": prepared.sampling_interval_label or "irregular/unknown",
        "missing_ratio": series.missing_ratio,
        "quality_score": None,
    }
    figures["01"] = viz.plot_dataset_profile(output_dir, profile_summary)
    figures["02"] = viz.plot_energy_timeseries(output_dir, prepared.timestamps, prepared.values, series.label(), series.unit)
    figures["03"] = viz.plot_daily_load_profile(
        output_dir, prepared.timestamps, prepared.values, series.label(), series.unit, prepared.is_regular
    )

    fr = forecast.forecast_result
    if fr is not None:
        figures["04"] = viz.plot_forecast_vs_actual(
            output_dir, fr.test_timestamps, fr.actual, fr.predicted, fr.model_name, series.label(), series.unit
        )
        figures["05"] = viz.plot_forecast_residuals(output_dir, fr.test_timestamps, fr.residuals, series.unit)
    else:
        figures["04"] = viz.FigureResult("Forecast vs Actual", "04_forecast_vs_actual.png", False, reason="no forecast model was fit for this series.")
        figures["05"] = viz.FigureResult("Forecast Residuals", "05_forecast_residuals.png", False, reason="no forecast residuals available.")

    if anomaly.applicable:
        figures["06"] = viz.plot_anomaly_score_timeline(output_dir, anomaly.observations, cfg.anomaly_threshold)
        figures["07"] = viz.plot_anomaly_distribution(output_dir, anomaly.observations)
    else:
        figures["06"] = viz.FigureResult("Anomaly Score Timeline", "06_anomaly_score_timeline.png", False, reason=anomaly.reason)
        figures["07"] = viz.FigureResult("Anomaly Distribution", "07_anomaly_distribution.png", False, reason=anomaly.reason)

    if excess.available and fr is not None:
        figures["08"] = viz.plot_excess_energy_analysis(
            output_dir, fr.test_timestamps, fr.actual, fr.predicted, excess.excess_energy_kwh, series.unit
        )
    else:
        figures["08"] = viz.FigureResult(
            "Excess Energy Analysis", "08_excess_energy_analysis.png", False, reason=excess.reason or "unavailable."
        )

    return figures


# ---------------------------------------------------------------------------
# results.json (Part R)
# ---------------------------------------------------------------------------


def _json_safe(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def build_results_dict(run: AnalysisRunResult) -> dict:
    results = {
        "run_id": run.run_id,
        "dataset": {
            "file": run.day1.load_result.source_filename,
            "rows": run.day1.profile.row_count,
            "quality_score": run.day1.quality_result.overall,
            "quality_status": run.day1.quality_result.status,
        },
        "available_series": [
            {"series_id": s.series_id, "role": s.role, "sample_count": suf.sample_count, "status": suf.status, "reason": suf.reason}
            for s, suf in zip(run.all_series, run.all_sufficiency)
        ],
        "selected_series": run.selected_series_id,
        "selection_reason": run.selection_reason,
        "warnings": list(run.warnings),
    }

    p = run.primary
    if p is None:
        results["analysis"] = None
        return _json_safe(results)

    analysis: dict = {
        "series_id": p.series.series_id,
        "role": p.series.role,
        "unit": p.series.unit,
        "sufficiency": {"status": p.sufficiency.status, "sample_count": p.sufficiency.sample_count, "reason": p.sufficiency.reason},
    }

    if p.skipped_reason:
        analysis["skipped_reason"] = p.skipped_reason
        results["analysis"] = analysis
        return _json_safe(results)

    prep = p.prepared
    analysis["preparation"] = {
        "sampling_interval_label": prep.sampling_interval_label,
        "is_regular": prep.is_regular,
        "gap_count": prep.gap_count,
        "missing_value_count": prep.missing_value_count,
        "duplicate_timestamp_count": prep.duplicate_timestamp_count,
        "warnings": prep.warnings,
    }

    if p.split:
        analysis["split"] = {
            "train_ratio": p.split.train_ratio,
            "validation_ratio": p.split.validation_ratio,
            "test_ratio": p.split.test_ratio,
            "train_samples": p.split.train_samples,
            "validation_samples": p.split.validation_samples,
            "test_samples": p.split.test_samples,
            "train_start": p.split.train_start,
            "train_end": p.split.train_end,
            "test_start": p.split.test_start,
            "test_end": p.split.test_end,
        }

    if p.forecast:
        fr = p.forecast.forecast_result
        analysis["forecast"] = {
            "selected_model": p.forecast.selected_model,
            "selection_reason": p.forecast.selection_reason,
            "selection_criterion": p.forecast.selection_criterion,
            "comparison": [
                {
                    "model": e.model_name,
                    "applicable": e.applicable,
                    "attempted": e.attempted,
                    "reason": e.reason or None,
                    "validation_rmse": e.validation_rmse,
                    "test_mae": e.mae,
                    "test_rmse": e.rmse,
                    "test_mape": e.mape,
                    "test_mape_note": e.mape_note,
                    "test_r2": e.r2,
                    "test_r2_note": e.r2_note,
                }
                for e in p.forecast.comparison
            ],
            "metrics": (
                {"mae": fr.mae, "rmse": fr.rmse, "mape": fr.mape, "mape_note": fr.mape_note, "r2": fr.r2, "r2_note": fr.r2_note}
                if fr
                else None
            ),
            "provenance": fr.provenance if fr else None,
        }

    if p.anomaly:
        a = p.anomaly
        analysis["anomaly"] = {
            "applicable": a.applicable,
            "reason": a.reason or None,
            "base_threshold": a.base_threshold,
            "train_stats": (
                {"median": a.train_stats.median, "mad": a.train_stats.mad, "scale_method": a.train_stats.scale_method, "note": a.train_stats.note}
                if a.train_stats
                else None
            ),
            "observations_analyzed": a.observations_analyzed,
            "anomalies_detected": a.anomalies_detected,
            "anomaly_rate_pct": a.anomaly_rate,
            "peak_score": a.peak_score,
            "highest_severity": a.highest_severity,
            "events": [
                {
                    "event_id": e.event_id,
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                    "duration_seconds": e.duration_seconds,
                    "peak_score": e.peak_score,
                    "mean_score": e.mean_score,
                    "peak_deviation": e.peak_deviation,
                    "direction": e.direction,
                    "severity": e.severity,
                    "observation_count": e.observation_count,
                    "persistence_class": e.persistence_class,
                    "evidence": e.evidence,
                }
                for e in a.events
            ],
            "provenance": "INFERRED",
        }

    if p.excess_energy:
        ee = p.excess_energy
        analysis["excess_energy"] = {
            "available": ee.available,
            "reason": ee.reason or None,
            "estimated_excess_kwh": ee.excess_energy_kwh,
            "basis": ee.basis,
            "provenance": ee.provenance if ee.available else None,
        }

    if p.figures:
        analysis["figures"] = {
            key: {"filename": f.filename, "generated": f.generated, "reason": f.reason} for key, f in p.figures.items()
        }

    results["analysis"] = analysis
    return _json_safe(results)


def write_results_json(run: AnalysisRunResult, extra: Optional[dict] = None) -> Optional[str]:
    if not run.output_dir:
        return None
    os.makedirs(run.output_dir, exist_ok=True)
    path = os.path.join(run.output_dir, "results.json")
    combined = build_results_dict(run)
    if extra:
        combined.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=str)
    return path
