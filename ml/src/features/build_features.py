"""Time-aware feature engineering for T / RH / P. Rolling stats use past time only."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.src.config import PipelineConfig
from ml.src.paths import CLEAN_FEATURES, PROCESSED_OBSERVATIONS, ensure_dirs
from ml.src.preprocessing import load_processed_observations

LOGGER = logging.getLogger(__name__)

CORE_COLUMNS = [
    "timestamp",
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "elevation_m",
    "temperature_c",
    "relative_humidity_pct",
    "pressure_hpa",
]


def _hours_since_previous(timestamps: pd.Series) -> pd.Series:
    delta = timestamps.diff().dt.total_seconds() / 3600.0
    return delta.replace(0, np.nan)


def _causal_time_rolling(group: pd.DataFrame, column: str, window: str, min_periods: int) -> tuple[pd.Series, pd.Series]:
    """Past-only rolling mean/std using a time window, never future rows."""
    indexed = group.sort_values("timestamp").set_index("timestamp")
    past = indexed[column].shift(1)
    rolled = past.rolling(window, min_periods=min_periods)
    return rolled.mean(), rolled.std()


def add_temporal_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    df["gap_hours"] = df.groupby("station_id")["timestamp"].transform(_hours_since_previous)

    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby("station_id", sort=False):
        group = group.sort_values("timestamp").copy()
        hours = group["gap_hours"]
        for src, prefix in (
            ("temperature_c", "temperature"),
            ("relative_humidity_pct", "humidity"),
            ("pressure_hpa", "pressure"),
        ):
            group[f"{prefix}_change"] = group[src].diff()
            group[f"{prefix}_rate"] = group[f"{prefix}_change"] / hours
            mean, std = _causal_time_rolling(
                group, src, config.rolling_window, config.rolling_min_periods
            )
            group[f"{prefix}_rolling_mean"] = mean.to_numpy()
            group[f"{prefix}_rolling_std"] = std.to_numpy()
            group[f"{prefix}_zscore"] = (
                (group[src] - group[f"{prefix}_rolling_mean"])
                / group[f"{prefix}_rolling_std"].replace(0, np.nan)
            )
            group[f"{prefix}_residual"] = group[src] - group[f"{prefix}_rolling_mean"]
        pieces.append(group)

    out = pd.concat(pieces, ignore_index=True)
    rate_cols = ["temperature_rate", "humidity_rate", "pressure_rate"]
    out[rate_cols] = out[rate_cols].replace([np.inf, -np.inf], np.nan)
    return out


def add_persistence_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby("station_id", sort=False):
        group = group.sort_values("timestamp").copy()
        for src, prefix in (
            ("temperature_c", "temperature"),
            ("relative_humidity_pct", "humidity"),
            ("pressure_hpa", "pressure"),
        ):
            same_as_prev = group[src].eq(group[src].shift(1)) & group[src].notna()
            group[f"{prefix}_flatline"] = same_as_prev.astype(int)
            run_id = (group[src] != group[src].shift(1)).cumsum()
            group[f"{prefix}_flatline_run"] = same_as_prev.groupby(run_id).cumsum()
            run_start = group["timestamp"].where(
                group[src].ne(group[src].shift(1)) & group[src].notna()
            ).ffill()
            duration_hours = (
                group["timestamp"] - run_start
            ).dt.total_seconds() / 3600.0
            group[f"{prefix}_persistence_hours"] = duration_hours.where(group[src].notna())
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def add_multivariate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temperature_humidity_product"] = df["temperature_c"] * df["relative_humidity_pct"]
    df["temperature_pressure_ratio"] = df["temperature_c"] / df["pressure_hpa"].replace(0, np.nan)
    df["humidity_pressure_ratio"] = df["relative_humidity_pct"] / df["pressure_hpa"].replace(0, np.nan)
    z_t = df["temperature_zscore"]
    z_h = df["humidity_zscore"]
    z_p = df["pressure_zscore"]
    absolute_z = pd.concat([z_t.abs(), z_h.abs(), z_p.abs()], axis=1)
    df["multivariate_abs_z_max"] = absolute_z.max(axis=1, skipna=True)
    df["multivariate_abs_z_mean"] = absolute_z.mean(axis=1, skipna=True)
    df["variable_disagreement"] = (
        z_t.abs().fillna(0)
        - 0.5 * (z_h.abs().fillna(0) + z_p.abs().fillna(0))
    ).clip(lower=0)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366.0)
    return df


def add_data_quality_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    df["source_sampling_gap"] = (df["gap_hours"] > config.communication_gap_hours).astype(int)
    df["communication_gap"] = df["source_sampling_gap"]
    df["all_variables_missing"] = (
        df["temperature_c"].isna()
        & df["relative_humidity_pct"].isna()
        & df["pressure_hpa"].isna()
    ).astype(int)
    if "missing_any" not in df.columns:
        df["missing_any"] = (
            df["temperature_c"].isna()
            | df["relative_humidity_pct"].isna()
            | df["pressure_hpa"].isna()
        ).astype(int)
    return df


def build_feature_frame(
    observations: pd.DataFrame,
    config: PipelineConfig | None = None,
) -> pd.DataFrame:
    config = config or PipelineConfig()
    df = observations.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    df = add_temporal_features(df, config)
    df = add_persistence_features(df, config)
    df = add_multivariate_features(df)
    df = add_time_features(df)
    df = add_data_quality_features(df, config)
    return df


def build_features(
    input_path=None,
    output_path=None,
    config: PipelineConfig | None = None,
) -> pd.DataFrame:
    config = config or PipelineConfig()
    ensure_dirs()
    observations = load_processed_observations(input_path or PROCESSED_OBSERVATIONS)
    df = build_feature_frame(observations, config)
    output = output_path or CLEAN_FEATURES
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    LOGGER.info("Saved %s rows / %s columns to %s", len(df), df.shape[1], output)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_features()
