"""
forecasting/linear_lag.py — Day 2 Part E, OPTIONAL Model 3: Linear Autoregression

y_t = a * y_{t-1} + b, fit by ordinary least squares on TRAINING data
only. Deliberately the simplest possible transparent ML baseline (no
LSTM/Transformer per spec) — a single-lag linear regression.

Multi-step forecasting uses a recursive rollout: each predicted value
becomes the "previous" input for the next step. This uses only the
fitted (a, b) and the model's OWN prior forecasts — never a test-set
actual — so there is no leakage risk (Part D), matching the same
design principle as the seasonal-naive model's season-repeat rollout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastModel

MIN_TRAIN_SAMPLES = 10  # need a meaningful number of (y_{t-1}, y_t) pairs to fit a stable line


class LinearLagModel(ForecastModel):
    name = "Linear Lag (AR1)"

    def __init__(self) -> None:
        self._a: float | None = None
        self._b: float | None = None
        self._last_value: float | None = None

    def is_applicable(self, train_timestamps: pd.Series, train_values: pd.Series, **kwargs) -> tuple[bool, str]:
        if len(train_values) < MIN_TRAIN_SAMPLES:
            return False, f"training data has only {len(train_values)} sample(s); need at least {MIN_TRAIN_SAMPLES}."
        return True, ""

    def fit(self, train_timestamps: pd.Series, train_values: pd.Series) -> "LinearLagModel":
        applicable, reason = self.is_applicable(train_timestamps, train_values)
        if not applicable:
            raise ValueError(reason)
        values = np.asarray(train_values, dtype=float)
        x = values[:-1]  # y_{t-1}
        y = values[1:]  # y_t
        # Ordinary least squares for y = a*x + b (numpy's own linear algebra —
        # no external ML framework needed for a 1-feature fit).
        a, b = np.polyfit(x, y, deg=1)
        self._a = float(a)
        self._b = float(b)
        self._last_value = float(values[-1])
        return self

    def predict(self, test_timestamps: pd.Series) -> np.ndarray:
        if self._a is None:
            raise RuntimeError("LinearLagModel.predict() called before fit().")
        preds = []
        prev = self._last_value
        for _ in range(len(test_timestamps)):
            nxt = self._a * prev + self._b
            preds.append(nxt)
            prev = nxt  # recursive rollout using the model's own forecast, never test actuals
        return np.asarray(preds, dtype=float)

    def metadata(self) -> dict:
        return {
            "model": self.name,
            "equation": "y_t = a*y_(t-1) + b (OLS on training data), recursive multi-step rollout",
            "a": self._a,
            "b": self._b,
        }

    def in_sample_residuals(self, train_values: pd.Series) -> np.ndarray:
        if self._a is None:
            return np.array([], dtype=float)
        v = np.asarray(train_values, dtype=float)
        if len(v) < 2:
            return np.array([], dtype=float)
        x, y = v[:-1], v[1:]
        predicted = self._a * x + self._b
        return y - predicted  # the OLS fit's own in-sample residuals
