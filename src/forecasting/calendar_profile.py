
"""
EnergyOS Research Demo v2
Calendar Profile Forecasting Model

A deterministic calendar-based forecasting baseline.

Features:
- Weekday and hour-of-day consumption profiles
- Training-only parameter estimation
- Missing observation support without imputation
- Chronological timestamp validation
- Safe fallback for sparsely observed calendar groups
- No validation/test leakage
- Compatible with EnergyOS ForecastModel interface

Limitations:
- Designed for hourly building-energy observations
- Does not account for weather, holidays, occupancy, or drift
- In-sample residuals are not independently calibrated
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastModel


MIN_TRAIN_SAMPLES = 168
MIN_OBSERVED_DAYS = 7
MIN_PROFILE_OBSERVATIONS = 3


class CalendarProfileModel(ForecastModel):

    name = "Calendar Profile (Hour x Weekday)"

    def __init__(self) -> None:
        self._weekday_hour_means: dict = {}
        self._weekday_hour_counts: dict = {}
        self._hour_means: dict = {}

        self._global_mean: float | None = None
        self._training_values: np.ndarray | None = None
        self._training_predictions: np.ndarray | None = None

        self._training_samples = 0
        self._observed_days = 0
        self._fitted = False

    @staticmethod
    def _parse_timestamps(
        timestamps,
    ) -> pd.DatetimeIndex:

        result = pd.DatetimeIndex(
            pd.to_datetime(
                timestamps,
                errors="coerce",
            )
        )

        if result.isna().any():
            raise ValueError(
                "Invalid or missing timestamps."
            )

        return result

    def is_applicable(
        self,
        train_timestamps: pd.Series,
        train_values: pd.Series,
        **kwargs,
    ) -> tuple[bool, str]:

        if len(train_timestamps) != len(train_values):
            return (
                False,
                "Training timestamps and values "
                "have different lengths.",
            )

        if len(train_values) < MIN_TRAIN_SAMPLES:
            return (
                False,
                f"At least {MIN_TRAIN_SAMPLES} "
                "training observations are required.",
            )

        interval = kwargs.get(
            "sampling_interval_seconds"
        )

        if interval is not None:
            try:
                interval = float(interval)
            except (TypeError, ValueError):
                return False, "Invalid sampling interval."

            if (
                not np.isfinite(interval)
                or abs(interval - 3600.0) > 72.0
            ):
                return (
                    False,
                    "Calendar Profile requires an "
                    "underlying hourly sampling cadence.",
                )

        try:
            timestamps = self._parse_timestamps(
                train_timestamps
            )

            values = np.asarray(
                train_values,
                dtype=float,
            )

            if values.ndim != 1:
                return (
                    False,
                    "Training values must be one-dimensional.",
                )

            if not np.all(np.isfinite(values)):
                return (
                    False,
                    "Training contains non-finite values.",
                )

            if timestamps.has_duplicates:
                return (
                    False,
                    "Duplicate training timestamps detected.",
                )

            if not timestamps.is_monotonic_increasing:
                return (
                    False,
                    "Training timestamps must be chronological.",
                )

            observed_days = len(
                timestamps.normalize().unique()
            )

            if observed_days < MIN_OBSERVED_DAYS:
                return (
                    False,
                    "Insufficient distinct observed days.",
                )

            # Missing hourly observations are allowed.
            # No gap filling or synthetic values are used.

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            return (
                False,
                f"Invalid training data: {exc}",
            )

        return True, ""

    def fit(
        self,
        train_timestamps: pd.Series,
        train_values: pd.Series,
    ) -> "CalendarProfileModel":

        applicable, reason = self.is_applicable(
            train_timestamps,
            train_values,
        )

        if not applicable:
            raise ValueError(reason)

        timestamps = self._parse_timestamps(
            train_timestamps
        )

        values = np.asarray(
            train_values,
            dtype=float,
        )

        frame = pd.DataFrame(
            {
                "weekday": timestamps.dayofweek,
                "hour": timestamps.hour,
                "value": values,
            }
        )

        grouped = frame.groupby(
            ["weekday", "hour"]
        )["value"]

        means = grouped.mean()
        counts = grouped.count()

        self._weekday_hour_means = {
            (int(day), int(hour)): float(mean)
            for (day, hour), mean in means.items()
        }

        self._weekday_hour_counts = {
            (int(day), int(hour)): int(count)
            for (day, hour), count in counts.items()
        }

        hourly_means = frame.groupby(
            "hour"
        )["value"].mean()

        self._hour_means = {
            int(hour): float(mean)
            for hour, mean in hourly_means.items()
        }

        self._global_mean = float(
            np.mean(values)
        )

        self._training_values = values.copy()
        self._training_samples = len(values)

        self._observed_days = len(
            timestamps.normalize().unique()
        )

        self._fitted = True

        self._training_predictions = self.predict(
            train_timestamps
        )

        return self

    def predict(
        self,
        test_timestamps: pd.Series,
    ) -> np.ndarray:

        if not self._fitted:
            raise RuntimeError(
                "CalendarProfileModel.predict() "
                "called before fit()."
            )

        timestamps = self._parse_timestamps(
            test_timestamps
        )

        predictions = np.empty(
            len(timestamps),
            dtype=float,
        )

        for index, timestamp in enumerate(timestamps):

            weekday = int(timestamp.dayofweek)
            hour = int(timestamp.hour)

            key = (weekday, hour)

            observation_count = (
                self._weekday_hour_counts.get(
                    key,
                    0,
                )
            )

            if (
                observation_count
                >= MIN_PROFILE_OBSERVATIONS
            ):
                value = self._weekday_hour_means[key]

            elif hour in self._hour_means:
                value = self._hour_means[hour]

            else:
                value = self._global_mean

            predictions[index] = float(value)

        return predictions

    def in_sample_residuals(
        self,
        train_values: pd.Series,
    ) -> np.ndarray:

        if not self._fitted:
            return np.array([], dtype=float)

        values = np.asarray(
            train_values,
            dtype=float,
        )

        if (
            self._training_values is None
            or self._training_predictions is None
        ):
            return np.array([], dtype=float)

        if (
            values.shape != self._training_values.shape
            or not np.array_equal(
                values,
                self._training_values,
            )
        ):
            raise ValueError(
                "Residual calculation requires "
                "the original training values."
            )

        return (
            values - self._training_predictions
        ).copy()

    def metadata(self) -> dict:

        return {
            "model": self.name,
            "method": (
                "Training-only weekday-hour "
                "mean consumption profile"
            ),
            "training_samples": self._training_samples,
            "observed_training_days": self._observed_days,
            "weekday_hour_profiles": len(
                self._weekday_hour_means
            ),
            "hour_profiles": len(
                self._hour_means
            ),
            "global_training_mean": self._global_mean,
            "minimum_profile_observations": (
                MIN_PROFILE_OBSERVATIONS
            ),
            "missing_observations": (
                "Allowed without imputation"
            ),
            "uses_future_actuals": False,
            "residual_method": (
                "In-sample fitted residuals; "
                "not independently calibrated"
            ),
            "limitations": (
                "No weather, occupancy, holiday, "
                "equipment-operation or drift features."
            ),
        }
