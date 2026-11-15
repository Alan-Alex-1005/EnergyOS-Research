
"""EnergyOS evidence exporter.

Exports already-calculated test predictions and anomaly results.
Does not refit models or modify analytical results.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def _finite(value):
    """Preserve finite numbers; leave unavailable values blank."""
    if value is None:
        return ""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""

    return number if math.isfinite(number) else ""


def export_forecast_evidence(run):
    """
    Export aligned test observations from an EnergyOS run.

    Returns the output path, or None when forecast
    evidence is unavailable.

    Raises ValueError if arrays are not aligned.
    """

    if not run.output_dir or run.primary is None:
        return None

    primary = run.primary
    engine = primary.forecast

    forecast = (
        engine.forecast_result
        if engine is not None
        else None
    )

    if forecast is None:
        return None

    timestamps = list(forecast.test_timestamps)
    actual = list(forecast.actual)
    predicted = list(forecast.predicted)
    residuals = list(forecast.residuals)

    count = len(timestamps)

    if not (
        count == len(actual)
        == len(predicted)
        == len(residuals)
    ):
        raise ValueError(
            "Forecast evidence arrays are not aligned."
        )

    observations = []

    if (
        primary.anomaly is not None
        and primary.anomaly.applicable
    ):
        observations = list(
            primary.anomaly.observations
        )

        if len(observations) != count:
            raise ValueError(
                "Anomaly observations do not match "
                "the forecast test period."
            )

        for index, observation in enumerate(observations):
            if str(observation.timestamp) != str(
                timestamps[index]
            ):
                raise ValueError(
                    "Anomaly timestamps are not aligned "
                    "with forecast timestamps."
                )

    destination = (
        Path(run.output_dir) / "forecast_evidence.csv"
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        ".csv.tmp"
    )

    try:
        with temporary.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "actual",
                "predicted",
                "residual",
                "anomaly_score",
                "is_anomaly",
                "severity",
            ])

            for index in range(count):

                observation = (
                    observations[index]
                    if observations
                    else None
                )

                writer.writerow([
                    timestamps[index],
                    _finite(actual[index]),
                    _finite(predicted[index]),
                    _finite(residuals[index]),
                    (
                        _finite(observation.z_score)
                        if observation else ""
                    ),
                    (
                        int(bool(observation.is_anomalous))
                        if observation else ""
                    ),
                    (
                        str(observation.severity)
                        if observation else ""
                    ),
                ])

        temporary.replace(destination)

    finally:
        temporary.unlink(missing_ok=True)

    return destination
