"""Rolling sensor health and maintenance recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_sensor_health(df: pd.DataFrame, window: str = "30D") -> pd.DataFrame:
    """Attach rolling risk, health state, and a maintenance recommendation."""
    out = df.sort_values(["station_id", "timestamp"]).copy()
    signal = (
        0.35 * out["temporal_score"].fillna(0.0)
        + 0.25 * out["isolation_score"].fillna(0.0)
        + 0.20 * out["multivariate_score"].fillna(0.0)
        + 0.20 * out["rule_qc_score"].fillna(0.0)
    )
    risk = pd.Series(index=out.index, dtype=float)
    for _, group in out.groupby("station_id", sort=False):
        values = pd.Series(signal.loc[group.index].to_numpy(), index=group["timestamp"])
        rolling = values.rolling(window, min_periods=1).mean()
        risk.loc[group.index] = rolling.to_numpy()
    persistence = out.get("stuck_score", pd.Series(0.0, index=out.index)).fillna(0.0)
    drift = out.get("drift_score", pd.Series(0.0, index=out.index)).fillna(0.0)
    out["sensor_anomaly_rate"] = risk
    out["sensor_health_score"] = (1.0 - risk).clip(0.0, 1.0)
    out["sensor_health_state"] = np.select(
        [out["sensor_health_score"] < 0.35, out["sensor_health_score"] < 0.65],
        ["critical", "degraded"],
        default="healthy",
    )
    out.loc[(out["sensor_health_score"] < 0.8) & (out["sensor_health_state"] == "healthy"), "sensor_health_state"] = "watch"
    out["maintenance_recommendation"] = np.select(
        [
            out["sensor_health_state"].eq("critical"),
            (out["sensor_health_state"].eq("degraded") | (persistence > 0.7) | (drift > 0.7)),
            out["sensor_health_state"].eq("watch"),
        ],
        ["inspect station and sensor immediately", "schedule calibration or engineering review", "monitor recent behavior"],
        default="no immediate maintenance action",
    )
    return out.sort_index()