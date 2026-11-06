"""
forecasting/naive.py — Day 2 Part E, Model 1: Persistence / Naive

y_hat_t = y_{t-1} (the last observed TRAINING value, held constant
across the whole test horizon). The simplest defensible baseline —
every other model must beat this to be worth using.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastModel


class PersistenceModel(ForecastModel):
    name = "Persistence"

    def __init__(self) -> None:
        self._last_value: float | None = None

    def fit(self, train_timestamps: pd.Series, train_values: pd.Series) -> "PersistenceModel":
        if len(train_values) == 0:
            raise ValueError("PersistenceModel requires at least one training observation.")
        self._last_value = float(train_values.iloc[-1])
        return self

    def predict(self, test_timestamps: pd.Series) -> np.ndarray:
        if self._last_value is None:
            raise RuntimeError("PersistenceModel.predict() called before fit().")
        return np.full(len(test_timestamps), self._last_value, dtype=float)

    def metadata(self) -> dict:
        return {
            "model": self.name,
            "equation": "y_hat_t = y_(t-1)",
            "last_training_value": self._last_value,
        }

    def in_sample_residuals(self, train_values: pd.Series) -> np.ndarray:
        v = np.asarray(train_values, dtype=float)
        if len(v) < 2:
            return np.array([], dtype=float)
        return v[1:] - v[:-1]  # y_t - y_hat_t where y_hat_t = y_(t-1)
