from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.src.config import PipelineConfig
from ml.src.models.sequence.dataset import build_sequences, causal_windows
from ml.src.models.sequence.gru_detector import GRUDetector


def frame(stations=("A", "B"), count=30):
    rows = []
    for station in stations:
        for index, timestamp in enumerate(pd.date_range("2024-01-01", periods=count, freq="h")):
            rows.append(
                {
                    "station_id": station,
                    "timestamp": timestamp,
                    "temperature_c": 20.0 + index * 0.1,
                    "relative_humidity_pct": 50.0,
                    "pressure_hpa": 1000.0 + index * 0.2,
                }
            )
    return pd.DataFrame(rows).sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def test_sequences_are_station_local_and_chronological():
    observations = frame()
    scaler = StandardScaler().fit(observations[["temperature_c", "relative_humidity_pct", "pressure_hpa"]])
    sequences, targets, metadata = build_sequences(observations, 4, scaler)
    assert sequences.shape[1:] == (4, 3)
    assert targets.shape[1] == 3
    assert {station for station, _ in metadata} == {"A", "B"}
    for station in {station for station, _ in metadata}:
        station_times = [timestamp for current_station, timestamp in metadata if current_station == station]
        assert station_times == sorted(station_times)


def test_missing_values_do_not_create_sequences_or_cross_stations():
    observations = frame()
    observations.loc[5, "temperature_c"] = np.nan
    scaler = StandardScaler().fit(observations.dropna()[["temperature_c", "relative_humidity_pct", "pressure_hpa"]])
    sequences, _, metadata = build_sequences(observations, 4, scaler, max_gap_hours=48)
    assert len(sequences) > 0
    assert all(station in {"A", "B"} for station, _ in metadata)


def test_causal_windows_end_before_scored_timestamp():
    observations = frame(("A",), 30)
    scaler = StandardScaler().fit(observations[["temperature_c", "relative_humidity_pct", "pressure_hpa"]])
    windows, indexes = causal_windows(observations, 4, scaler)
    assert len(windows) == len(indexes)
    assert min(indexes) >= 4


def test_gru_forward_fit_and_score_contract():
    observations = frame(("A",), 36)
    config = PipelineConfig(gru_sequence_length=4, gru_epochs=1, gru_batch_size=8)
    detector = GRUDetector(config).fit(observations, end_time="2025-01-01")
    scores = detector.score(observations)
    assert np.isnan(scores[:4]).all()
    assert np.isfinite(scores[4:]).all()
    assert np.nanmin(scores) >= 0.0
    assert np.nanmax(scores) <= 1.0
