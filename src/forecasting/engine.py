
"""
forecasting/engine.py — EnergyOS Forecast Engine

Fits forecasting candidates using training data only.

Model selection:
    Lowest validation RMSE.

Test data:
    Used only for final retrospective evaluation.

Includes:
    Persistence
    Seasonal Naive
    Linear Lag AR1
    Calendar Profile

Includes TrustAI reliability assessment.

No synthetic data is generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..timeseries import PreparedSeries, TrainValidationTestSplit

from .base import ForecastModel, ForecastResult
from .calendar_profile import CalendarProfileModel
from .linear_lag import LinearLagModel
from .metrics import mae as mae_fn
from .metrics import mape as mape_fn
from .metrics import r_squared
from .metrics import rmse as rmse_fn
from .naive import PersistenceModel
from .reliability import assess_forecast_reliability
from .seasonal_naive import (
    SeasonalNaiveModel,
    infer_seasonal_period,
)


SELECTION_CRITERION = (
    "lowest VALIDATION RMSE "
    "(test set is never used for model selection)"
)


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
    comparison: list[ModelEvaluation] = field(
        default_factory=list
    )
    warnings: list[str] = field(default_factory=list)
    selected_model_instance: Optional[ForecastModel] = None


def _build_candidate_models(
    prepared: PreparedSeries,
    seasonal_period_override: Optional[int],
) -> list[ForecastModel]:

    period = (
        seasonal_period_override
        if seasonal_period_override is not None
        else infer_seasonal_period(
            prepared.sampling_interval_seconds
        )
    )

    return [
        PersistenceModel(),
        SeasonalNaiveModel(seasonal_period=period),
        LinearLagModel(),
        CalendarProfileModel(),
    ]


def run_forecast_engine(
    prepared: PreparedSeries,
    split: TrainValidationTestSplit,
    seasonal_period_override: Optional[int] = None,
) -> ForecastEngineResult:

    warnings: list[str] = list(split.warnings)

    # ---------------------------------------------------------
    # 1. Validate chronological split
    # ---------------------------------------------------------

    if (
        split.train_samples < 1
        or split.validation_samples < 1
        or split.test_samples < 1
    ):
        return ForecastEngineResult(
            series_id=prepared.series_id,
            selected_model=None,
            selection_reason=(
                "train, validation, or test split "
                "has zero samples."
            ),
            selection_criterion=SELECTION_CRITERION,
            forecast_result=None,
            warnings=warnings,
        )

    # ---------------------------------------------------------
    # 2. Build forecasting candidates
    # ---------------------------------------------------------

    candidates = _build_candidate_models(
        prepared,
        seasonal_period_override,
    )

    evaluations: list[ModelEvaluation] = []

    fitted_models: dict[
        str,
        tuple[ForecastModel, np.ndarray, np.ndarray],
    ] = {}

    combined_horizon = pd.concat(
        [
            split.validation_timestamps,
            split.test_timestamps,
        ],
        ignore_index=True,
    )

    n_val = split.validation_samples

    validation_actual = np.asarray(
        split.validation_values,
        dtype=float,
    )

    test_actual = np.asarray(
        split.test_values,
        dtype=float,
    )

    # ---------------------------------------------------------
    # 3. Fit and evaluate each candidate
    # ---------------------------------------------------------

    for model in candidates:

        applicable, reason = model.is_applicable(
            split.train_timestamps,
            split.train_values,
            is_regular=prepared.is_regular,
            sampling_interval_seconds=(
                prepared.sampling_interval_seconds
            ),
        )

        if not applicable:
            evaluations.append(
                ModelEvaluation(
                    model_name=model.name,
                    attempted=False,
                    applicable=False,
                    reason=reason,
                )
            )
            continue

        try:
            model.fit(
                split.train_timestamps,
                split.train_values,
            )

            predicted_combined = np.asarray(
                model.predict(combined_horizon),
                dtype=float,
            )

            if (
                predicted_combined.ndim != 1
                or predicted_combined.size
                != len(combined_horizon)
            ):
                raise ValueError(
                    "Prediction length or shape mismatch."
                )

            if not np.all(
                np.isfinite(predicted_combined)
            ):
                raise ValueError(
                    "Model produced non-finite predictions."
                )

        except Exception as exc:
            evaluations.append(
                ModelEvaluation(
                    model_name=model.name,
                    attempted=True,
                    applicable=True,
                    reason=f"fit/predict failed: {exc}",
                )
            )
            continue

        predicted_validation = predicted_combined[:n_val]
        predicted_test = predicted_combined[n_val:]

        # Validation only determines model selection.
        validation_rmse = rmse_fn(
            validation_actual,
            predicted_validation,
        )

        # Test metrics are reported, never used to select.
        mae_r = mae_fn(
            test_actual,
            predicted_test,
        )

        rmse_r = rmse_fn(
            test_actual,
            predicted_test,
        )

        mape_r = mape_fn(
            test_actual,
            predicted_test,
        )

        r2_r = r_squared(
            test_actual,
            predicted_test,
        )

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

        fitted_models[model.name] = (
            model,
            predicted_validation,
            predicted_test,
        )

    # ---------------------------------------------------------
    # 4. Select using validation RMSE only
    # ---------------------------------------------------------

    scored = [
        evaluation
        for evaluation in evaluations
        if (
            evaluation.attempted
            and evaluation.validation_rmse is not None
            and np.isfinite(evaluation.validation_rmse)
            and evaluation.model_name in fitted_models
        )
    ]

    if not scored:
        return ForecastEngineResult(
            series_id=prepared.series_id,
            selected_model=None,
            selection_reason=(
                "no candidate model could be fit/evaluated "
                "for this series."
            ),
            selection_criterion=SELECTION_CRITERION,
            forecast_result=None,
            comparison=evaluations,
            warnings=warnings,
        )

    best = min(
        scored,
        key=lambda evaluation: evaluation.validation_rmse,
    )

    best_model, _, best_predicted_test = (
        fitted_models[best.model_name]
    )

    residuals = test_actual - best_predicted_test

    # ---------------------------------------------------------
    # 5. Construct forecast result
    # ---------------------------------------------------------

    forecast_result = ForecastResult(
        series_id=prepared.series_id,
        model_name=best.model_name,
        train_start=split.train_start,
        train_end=split.train_end,
        test_start=split.test_start,
        test_end=split.test_end,
        train_samples=split.train_samples,
        test_samples=split.test_samples,
        actual=test_actual.tolist(),
        predicted=best_predicted_test.tolist(),
        residuals=residuals.tolist(),
        test_timestamps=[
            str(timestamp)
            for timestamp in split.test_timestamps
        ],
        mae=best.mae,
        rmse=best.rmse,
        mape=best.mape,
        mape_note=best.mape_note,
        r2=best.r2,
        r2_note=best.r2_note,
        model_metadata=best.metadata,
        warnings=warnings,
    )

    # ---------------------------------------------------------
    # 6. TrustAI Forecast Reliability Safety Gate
    # ---------------------------------------------------------

    assessment = assess_forecast_reliability(
        forecast_result
    )

    warnings.append(
        f"Forecast reliability: {assessment.status}"
    )

    if not assessment.suitable_for_anomaly_evidence:
        for reason in assessment.reasons:
            warnings.append(
                f"Reliability warning: {reason}"
            )

    forecast_result.warnings = warnings

    # ---------------------------------------------------------
    # 7. Return complete engine result
    # ---------------------------------------------------------

    return ForecastEngineResult(
        series_id=prepared.series_id,
        selected_model=best.model_name,
        selection_reason=(
            f"lowest VALIDATION RMSE "
            f"({best.validation_rmse:.4g}) among "
            f"{len(scored)} evaluated model(s); "
            f"test set was not consulted for this decision."
        ),
        selection_criterion=SELECTION_CRITERION,
        forecast_result=forecast_result,
        comparison=evaluations,
        warnings=warnings,
        selected_model_instance=best_model,
    )
