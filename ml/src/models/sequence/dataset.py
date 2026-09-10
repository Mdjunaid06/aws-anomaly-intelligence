"""Leakage-safe station-local sequence construction for GRU training/inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

VARIABLES = ("temperature_c", "relative_humidity_pct", "pressure_hpa")


def _valid_group(group: pd.DataFrame, max_gap_hours: float) -> pd.DataFrame:
    group = group.sort_values("timestamp").copy()
    group = group.dropna(subset=list(VARIABLES))
    if group.empty:
        return group
    gaps = group["timestamp"].diff().dt.total_seconds().div(3600.0)
    group["_segment"] = gaps.gt(max_gap_hours).cumsum()
    return group


def build_sequences(
    observations: pd.DataFrame,
    sequence_length: int,
    scaler,
    *,
    end_time: pd.Timestamp | None = None,
    max_gap_hours: float = 48.0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, pd.Timestamp]]]:
    """Build preceding-window/next-observation pairs per station and segment."""
    frame = observations.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if end_time is not None:
        frame = frame[frame["timestamp"] < pd.Timestamp(end_time)]
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[tuple[str, pd.Timestamp]] = []
    for station_id, station in frame.groupby("station_id", sort=True):
        valid = _valid_group(station, max_gap_hours)
        for _, segment in valid.groupby("_segment", sort=True):
            values = segment[list(VARIABLES)].to_numpy(dtype=float)
            timestamps = segment["timestamp"].tolist()
            if len(values) <= sequence_length:
                continue
            scaled = scaler.transform(segment[list(VARIABLES)]).astype(np.float32)
            for end in range(sequence_length, len(scaled)):
                sequences.append(scaled[end - sequence_length : end])
                targets.append(scaled[end])
                metadata.append((str(station_id), pd.Timestamp(timestamps[end])))
    if not sequences:
        return np.empty((0, sequence_length, len(VARIABLES)), dtype=np.float32), np.empty((0, len(VARIABLES)), dtype=np.float32), []
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), metadata


def causal_windows(
    observations: pd.DataFrame,
    sequence_length: int,
    scaler,
    *,
    max_gap_hours: float = 48.0,
) -> tuple[np.ndarray, list[int]]:
    """Return windows ending immediately before each scoreable input row."""
    frame = observations.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    windows: list[np.ndarray] = []
    row_indices: list[int] = []
    for _, station in frame.groupby("station_id", sort=True):
        valid = _valid_group(station, max_gap_hours)
        for _, segment in valid.groupby("_segment", sort=True):
            values = scaler.transform(segment[list(VARIABLES)]).astype(np.float32)
            indexes = segment.index.to_list()
            for position in range(sequence_length, len(values)):
                windows.append(values[position - sequence_length : position])
                row_indices.append(indexes[position])
    if not windows:
        return np.empty((0, sequence_length, len(VARIABLES)), dtype=np.float32), []
    return np.asarray(windows, dtype=np.float32), row_indices
