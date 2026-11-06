"""
forecasting/engine.py — Day 2 Part F/H: Model Selection + ForecastResult

Fits every APPLICABLE baseline model on the training split, evaluates
all of them on the exact same held-out test split, and selects the
model with the lowest test RMSE (explicitly documented criterion — not
"whichever sounds sophisticated"). Returns the winning model's full
ForecastResult plus a transparent comparison table for every model that
was tried (including ones skipped, with why).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..timeseries import PreparedSeries, TrainValidationTestSplit
from .base import ForecastModel, ForecastResult
from .linear_lag import LinearLagModel
from .metrics import mae as mae_fn
from .metrics import mape as mape_fn
from .metrics import r_squared
from .metrics import rmse as rmse_fn
from .naive import PersistenceModel
from .seasonal_naive import SeasonalNaiveModel, infer_seasonal_period

SELECTION_CRITERION = "lowest VALIDATION RMSE (test set is never used for model selection)"


@dataclass
class ModelEvaluation:
    model_name: str
    attempted: bool
    applicable: bool
    reason: str
    validation_rmse: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    mape_note: Optional[str] = None
    r2: Optional[float] = None
    r2_note: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ForecastEngineResult:
    series_id: str
    selected_model: Optional[str]
    selection_reason: str
    selection_criterion: str
    forecast_result: Optional[ForecastResult]
    comparison: list[ModelEvaluation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    selected_model_instance: Optional[ForecastModel] = None


def _build_candidate_models(prepared: PreparedSeries, seasonal_period_override: Optional[int]) -> list[ForecastModel]:
    period = seasonal_period_override or infer_seasonal_period(prepared.sampling_interval_seconds)
    return [
        PersistenceModel(),
        SeasonalNaiveModel(seasonal_period=period),
        LinearLagModel(),
    ]


def run_forecast_engine(
    prepared: PreparedSeries,
    split: TrainValidationTestSplit,
    seasonal_period_override: Optional[int] = None,
) -> ForecastEngineResult:
    warnings: list[str] = list(split.warnings)

    if split.train_samples < 1 or split.validation_samples < 1 or split.test_samples < 1:
        return ForecastEngineResult(
            series_id=prepared.series_id,
            selected_model=None,
            selection_reason="train, validation, or test split has zero samples.",
            selection_criterion=SELECTION_CRITERION,
            forecast_result=None,
            warnings=warnings,
        )

    candidates = _build_candidate_models(prepared, seasonal_period_override)
    evaluations: list[ModelEvaluation] = []
    # model_name -> (model, predicted_validation, predicted_test)
    fitted_models: dict[str, tuple[ForecastModel, np.ndarray, np.ndarray]] = {}

    combined_horizon = pd.concat(
        [split.validation_timestamps, split.test_timestamps], ignore_index=True
    )
    n_val = split.validation_samples

    for model in candidates:
        applicable, reason = model.is_applicable(
            split.train_timestamps,
            split.train_values,
            is_regular=prepared.is_regular,
            sampling_interval_seconds=prepared.sampling_interval_seconds,
        )
        if not applicable:
            evaluations.append(
                ModelEvaluation(model_name=model.name, attempted=False, applicable=False, reason=reason)
            )
            continue

        try:
            model.fit(split.train_timestamps, split.train_values)
            # Predict the WHOLE remaining horizon (validation + test) in one
            # call so recursive/stateful models (e.g. linear-lag's rollout,
            # seasonal-naive's cycle position) stay mathematically
            # consistent with predicting validation then test separately
            # would NOT be — then split the result. Test values are never
            # read here, only test TIMESTAMPS (for their count/position).
            predicted_combined = model.predict(combined_horizon)
        except Exception as e:  # noqa: BLE001 — a model failing to fit is reported, not fatal
            evaluations.append(
                ModelEvaluation(
                    model_name=model.name, attempted=True, applicable=True, reason=f"fit/predict failed: {e}"
                )
            )
            continue

        predicted_validation = predicted_combined[:n_val]
        predicted_test = predicted_combined[n_val:]

        validation_actual = np.asarray(split.validation_values, dtype=float)
        validation_rmse = rmse_fn(validation_actual, predicted_validation)

        test_actual = np.asarray(split.test_values, dtype=float)
        mae_r = mae_fn(test_actual, predicted_test)
        rmse_r = rmse_fn(test_actual, predicted_test)
        mape_r = mape_fn(test_actual, predicted_test)
        r2_r = r_squared(test_actual, predicted_test)

        evaluations.append(
            ModelEvaluation(
                model_name=model.name,
                attempted=True,
                applicable=True,
                reason="",
                validation_rmse=validation_rmse.value,
                mae=mae_r.value,
                rmse=rmse_r.value,
                mape=mape_r.value,
                mape_note=mape_r.note,
                r2=r2_r.value,
                r2_note=r2_r.note,
                metadata=model.metadata(),
            )
        )
        fitted_models[model.name] = (model, predicted_validation, predicted_test)

    # Selection uses VALIDATION RMSE only — test metrics above were
    # computed for reporting, never consulted for the selection decision.
    scored = [e for e in evaluations if e.attempted and e.validation_rmse is not None]
    if not scored:
        return ForecastEngineResult(
            series_id=prepared.series_id,
            selected_model=None,
            selection_reason="no candidate model could be fit/evaluated for this series.",
            selection_criterion=SELECTION_CRITERION,
            forecast_result=None,
            comparison=evaluations,
            warnings=warnings,
        )

    best = min(scored, key=lambda e: e.validation_rmse)
    best_model, _, best_predicted_test = fitted_models[best.model_name]
    actual = np.asarray(split.test_values, dtype=float)
    residuals = actual - best_predicted_test

    forecast_result = ForecastResult(
        series_id=prepared.series_id,
        model_name=best.model_name,
        train_start=split.train_start,
        train_end=split.train_end,
        test_start=split.test_start,
        test_end=split.test_end,
        train_samples=split.train_samples,
        test_samples=split.test_samples,
        actual=actual.tolist(),
        predicted=best_predicted_test.tolist(),
        residuals=residuals.tolist(),
        test_timestamps=[str(t) for t in split.test_timestamps],
        mae=best.mae,
        rmse=best.rmse,
        mape=best.mape,
        mape_note=best.mape_note,
        r2=best.r2,
        r2_note=best.r2_note,
        model_metadata=best.metadata,
        warnings=warnings,
    )

    return ForecastEngineResult(
        series_id=prepared.series_id,
        selected_model=best.model_name,
        selection_reason=(
            f"lowest VALIDATION RMSE ({best.validation_rmse:.4g}) among {len(scored)} evaluated model(s); "
            f"test set was not consulted for this decision."
        ),
        selection_criterion=SELECTION_CRITERION,
        forecast_result=forecast_result,
        comparison=evaluations,
        warnings=warnings,
        selected_model_instance=best_model,
    )
