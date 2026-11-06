"""
forecasting/seasonal_naive.py — Day 2 Part E, Model 2: Seasonal Naive

y_hat_t = y_{t-s}, where s is a detected/configured seasonal period.

Multi-step-ahead implementation: the forecast repeats the LAST FULL
SEASON observed in the training data, cycling as needed to cover the
whole test horizon (the standard textbook "snaive" generalization for
horizons longer than one season). This uses ONLY training data — for
the first `s` test points this is mathematically identical to
y_hat_t = y_{t-s}; for later points it repeats rather than reaching
into the test set, so there is zero risk of test-data leakage (Part D).

Seasonal period is only ever assumed for sampling intervals explicitly
named in the spec (15-minute -> 96/day, 30-minute -> 48/day, hourly ->
24/day), and ONLY when the series' sampling is regular. Anything else
requires an explicit `seasonal_period` (e.g. from config) — otherwise
this model reports itself as inapplicable with a reason.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import ForecastModel

# (seconds, period) — matches spec section E's named examples exactly.
# Tolerance keeps this from mis-firing on close-but-not-exact intervals.
_KNOWN_INTERVALS = [
    (900, 96),  # 15 minutes -> 96 observations/day
    (1800, 48),  # 30 minutes -> 48 observations/day
    (3600, 24),  # hourly -> 24 observations/day
]
_TOLERANCE = 0.02  # 2% relative tolerance on the interval match


def infer_seasonal_period(sampling_interval_seconds: Optional[float]) -> Optional[int]:
    if not sampling_interval_seconds or sampling_interval_seconds <= 0:
        return None
    for known_seconds, period in _KNOWN_INTERVALS:
        if abs(sampling_interval_seconds - known_seconds) / known_seconds <= _TOLERANCE:
            return period
    return None


class SeasonalNaiveModel(ForecastModel):
    name = "Seasonal Naive"

    def __init__(self, seasonal_period: Optional[int] = None) -> None:
        self.seasonal_period = seasonal_period
        self._last_season: np.ndarray | None = None

    def is_applicable(self, train_timestamps: pd.Series, train_values: pd.Series, **kwargs) -> tuple[bool, str]:
        is_regular = kwargs.get("is_regular", True)
        sampling_interval_seconds = kwargs.get("sampling_interval_seconds")

        period = self.seasonal_period or infer_seasonal_period(sampling_interval_seconds)
        if period is None:
            return False, (
                "no defensible seasonal period: sampling interval is unknown/irregular, or doesn't match "
                "a known cadence (15-min, 30-min, hourly), and no seasonal_period was configured."
            )
        if not is_regular:
            return False, "sampling interval is irregular; seasonal-naive requires regular sampling."
        if len(train_values) < period:
            return False, (
                f"training data has only {len(train_values)} sample(s), fewer than one full "
                f"season ({period} samples) — cannot form a defensible seasonal baseline."
            )
        return True, ""

    def fit(self, train_timestamps: pd.Series, train_values: pd.Series) -> "SeasonalNaiveModel":
        if self.seasonal_period is None:
            raise ValueError(
                "SeasonalNaiveModel.fit() requires seasonal_period to be set (use the engine, which "
                "resolves it from the series' sampling interval, or pass it explicitly)."
            )
        if len(train_values) < self.seasonal_period:
            raise ValueError(
                f"training data has only {len(train_values)} sample(s), fewer than one full season "
                f"({self.seasonal_period})."
            )
        self._last_season = np.asarray(train_values.iloc[-self.seasonal_period:], dtype=float)
        return self

    def predict(self, test_timestamps: pd.Series) -> np.ndarray:
        if self._last_season is None:
            raise RuntimeError("SeasonalNaiveModel.predict() called before fit().")
        s = self.seasonal_period
        return np.array([self._last_season[k % s] for k in range(len(test_timestamps))], dtype=float)

    def metadata(self) -> dict:
        return {
            "model": self.name,
            "equation": "y_hat_t = y_(t-s), repeating the last full training season for h > s",
            "seasonal_period": self.seasonal_period,
        }

    def in_sample_residuals(self, train_values: pd.Series) -> np.ndarray:
        v = np.asarray(train_values, dtype=float)
        s = self.seasonal_period
        if s is None or len(v) <= s:
            return np.array([], dtype=float)
        return v[s:] - v[:-s]  # y_t - y_hat_t where y_hat_t = y_(t-s)
