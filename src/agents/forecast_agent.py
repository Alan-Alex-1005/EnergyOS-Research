"""
agents/forecast_agent.py — Day 3: Forecast Agent

Wraps the EXISTING Day-2 ForecastEngineResult. Computes nothing new —
every number here was already produced by forecasting/engine.py.
"""

from __future__ import annotations

from ..analysis_pipeline import SeriesAnalysisResult
from .base import STATUS_DONE, STATUS_INSUFFICIENT_DATA, STATUS_UNAVAILABLE, AgentResult

AGENT_ID = "forecast_agent"


def run_forecast_agent(series_result: SeriesAnalysisResult) -> AgentResult:
    if series_result.sufficiency.status != "AVAILABLE":
        return AgentResult(
            agent_id=AGENT_ID,
            status=STATUS_INSUFFICIENT_DATA,
            finding=f"Insufficient data to forecast '{series_result.series.series_id}'.",
            evidence={"sample_count": series_result.sufficiency.sample_count},
            limitations=[series_result.sufficiency.reason],
        )

    fc = series_result.forecast
    if fc is None or fc.forecast_result is None:
        return AgentResult(
            agent_id=AGENT_ID,
            status=STATUS_UNAVAILABLE,
            finding="No forecast model could be fit for this series.",
            limitations=[fc.selection_reason if fc else "forecasting was not attempted."],
        )

    fr = fc.forecast_result
    residuals = fr.residuals
    mean_residual = sum(residuals) / len(residuals) if residuals else None
    max_abs_residual = max((abs(r) for r in residuals), default=None)

    finding = (
        f"Selected '{fc.selected_model}' ({fc.selection_criterion}); "
        f"test RMSE={fr.rmse:.4g}" + (f", MAPE={fr.mape:.2f}%" if fr.mape is not None else "") + "."
    )

    limitations = list(fr.warnings)
    if fr.mape is None and fr.mape_note:
        limitations.append(f"MAPE: {fr.mape_note}")
    if fr.r2 is None and fr.r2_note:
        limitations.append(f"R2: {fr.r2_note}")

    return AgentResult(
        agent_id=AGENT_ID,
        status=STATUS_DONE,
        finding=finding,
        evidence={
            "selected_model": fc.selected_model,
            "selection_reason": fc.selection_reason,
            "model_comparison": [
                {"model": e.model_name, "validation_rmse": e.validation_rmse, "test_rmse": e.rmse, "attempted": e.attempted}
                for e in fc.comparison
            ],
            "train_samples": fr.train_samples,
            "test_samples": fr.test_samples,
            "mean_residual": mean_residual,
            "max_abs_residual": max_abs_residual,
        },
        metrics={"mae": fr.mae, "rmse": fr.rmse, "mape": fr.mape, "r2": fr.r2},
        provenance=dict(fr.provenance),
        limitations=limitations,
    )
