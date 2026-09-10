"""Time-aware temporal anomaly scores from causal residuals and rates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _squash(series: pd.Series, scale: float) -> pd.Series:
    return (series.abs() / scale).clip(upper=4.0) / 4.0


def temporal_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z = pd.concat(
        [
            out["temperature_zscore"].abs(),
            out["humidity_zscore"].abs(),
            out["pressure_zscore"].abs(),
        ],
        axis=1,
    ).max(axis=1)
    rate = pd.concat(
        [
            _squash(out["temperature_rate"], 8.0),
            _squash(out["humidity_rate"], 25.0),
            _squash(out["pressure_rate"], 5.0),
        ],
        axis=1,
    ).max(axis=1)
    out["temporal_score"] = pd.concat([_squash(z, 4.0), rate], axis=1).max(axis=1, skipna=True).clip(0, 1)
    out["temporal_z_max"] = z
    residual_strength = pd.concat(
        [
            out["temperature_residual"].abs() / 6.0,
            out["humidity_residual"].abs() / 20.0,
            out["pressure_residual"].abs() / 5.0,
        ],
        axis=1,
    ).mean(axis=1).clip(0, 1)
    # Estimate causal residual trend over the recent observation window. The
    # bounded transform makes the score comparable with other evidence while
    # retaining the existing 0..1 contract.
    drift_score = pd.Series(0.0, index=out.index)
    for _, group in out.groupby("station_id", sort=False):
        group = group.sort_values("timestamp")
        for index, timestamp in zip(group.index, group["timestamp"]):
            window = group.loc[
                (group["timestamp"] >= timestamp - pd.Timedelta("24h"))
                & (group["timestamp"] <= timestamp),
                ["timestamp", "temperature_residual"],
            ].dropna()
            if len(window) < 3:
                continue
            hours = (window["timestamp"] - window["timestamp"].iloc[0]).dt.total_seconds() / 3600.0
            duration = float(hours.iloc[-1])
            if duration <= 0:
                continue
            values = window["temperature_residual"].to_numpy(dtype=float)
            slope = float(np.polyfit(hours.to_numpy(), values, 1)[0])
            spread = float(np.std(values, ddof=1))
            normalized_slope = abs(slope) * duration / max(spread, 1.0)
            differences = np.diff(values)
            direction = np.sign(slope)
            persistence = (
                float(np.mean(np.sign(differences) == direction))
                if differences.size and direction != 0
                else 0.0
            )
            drift_score.loc[index] = np.clip(
                (1.0 - np.exp(-normalized_slope)) * persistence,
                0.0,
                1.0,
            )
    out["drift_score"] = drift_score
    out["drift_slope"] = drift_score
    return out
