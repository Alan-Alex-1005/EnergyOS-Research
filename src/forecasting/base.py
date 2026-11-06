"""
forecasting/base.py — Day 2 Part E/H: Forecasting Abstraction + Result Contract

ForecastModel: fit()/predict()/metadata() — every model (naive, seasonal
naive, and any future model) implements this same interface so the
engine (engine.py) can compare them uniformly.

ForecastResult: the ONE structured object the terminal report, results.json,
and PNG charts all consume. Nothing downstream recomputes its own numbers
independently (Part H requirement).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """Base class for all Day 2 forecasting models."""

    name: str = "ForecastModel"

    @abstractmethod
    def fit(self, train_timestamps: pd.Series, train_values: pd.Series) -> "ForecastModel":
        """Fit using TRAINING data only (Part D: no future information may leak in)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_timestamps: pd.Series) -> np.ndarray:
        """Produce one forecast per test timestamp, using only what fit() learned."""
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict:
        """Transparent description of what this model does and any parameters it used."""
        raise NotImplementedError

    @abstractmethod
    def in_sample_residuals(self, train_values: pd.Series) -> np.ndarray:
        """
        Residuals this model would have produced applied WITHIN the
        training set itself (e.g. persistence's lag-1 diffs, seasonal
        naive's lag-s diffs, linear-lag's OLS residuals). This is the
        ONLY source of "expected residual distribution" statistics for
        anomaly detection (Part J) — computed entirely from training
        data, so anomaly thresholds are never fit on the test period.
        """
        raise NotImplementedError

    def is_applicable(self, train_timestamps: pd.Series, train_values: pd.Series, **kwargs) -> tuple[bool, str]:
        """Override to reject a model when it isn't defensible for this series (e.g. seasonal
        naive with no defensible period). Default: always applicable."""
        return True, ""


@dataclass
class ForecastResult:
    series_id: str
    model_name: str
    train_start: Optional[str]
    train_end: Optional[str]
    test_start: Optional[str]
    test_end: Optional[str]
    train_samples: int
    test_samples: int
    actual: list
    predicted: list
    residuals: list  # e_t = actual - predicted (Part I)
    test_timestamps: list
    mae: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    mape_note: Optional[str]
    r2: Optional[float]
    r2_note: Optional[str]
    model_metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=lambda: {"predicted": "ESTIMATED", "residuals": "CALCULATED", "metrics": "CALCULATED"})
