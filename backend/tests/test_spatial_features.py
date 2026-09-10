from __future__ import annotations

import pandas as pd

from ml.src.config import PipelineConfig
from ml.src.features.spatial_features import attach_spatial_features


def test_spatial_alignment_does_not_use_future_neighbor_observation():
    timestamps = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 02:00"])
    observations = pd.DataFrame(
        {
            "station_id": ["A", "A", "B"],
            "timestamp": [timestamps[0], timestamps[1], timestamps[2]],
            "temperature_c": [20.0, 21.0, 35.0],
            "relative_humidity_pct": [50.0, 51.0, 80.0],
            "pressure_hpa": [1000.0, 1001.0, 1020.0],
            "temperature_change": [None, 1.0, 10.0],
            "humidity_change": [None, 1.0, 30.0],
            "pressure_change": [None, 1.0, 20.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "station_id": ["A", "B"],
            "latitude": [18.5, 18.6],
            "longitude": [73.8, 73.9],
            "elevation_m": [500.0, 510.0],
        }
    )

    result = attach_spatial_features(
        observations,
        metadata=metadata,
        config=PipelineConfig(spatial_tolerance="3h"),
    )
    target = result[(result["station_id"] == "A") & (result["timestamp"] == timestamps[1])].iloc[0]

    assert target["temperature_usable_neighbors"] == 0
    assert target["temperature_insufficient_spatial"] == 1
