from __future__ import annotations

import pandas as pd

from ml.src.config import PipelineConfig
from ml.src.detectors.rule_qc import rule_qc_scores
from ml.src.detectors.temporal import temporal_scores


def _frame(humidity: list[float], temperature: list[float], pressure: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=len(humidity), freq="12h")
    return pd.DataFrame(
        {
            "station_id": "S1",
            "timestamp": timestamps,
            "temperature_c": temperature,
            "relative_humidity_pct": humidity,
            "pressure_hpa": pressure,
            "missing_any": 0,
            "source_sampling_gap": 0,
        }
    )


def test_rh_saturation_is_not_a_stuck_sensor():
    frame = _frame([100.0, 100.0, 100.0], [24.0, 25.0, 26.0], [1000.0, 1001.0, 1002.0])
    frame["gap_hours"] = 12.0
    frame["humidity_change"] = frame.relative_humidity_pct.diff()
    frame["temperature_change"] = frame.temperature_c.diff()
    frame["pressure_change"] = frame.pressure_hpa.diff()
    frame["humidity_rate"] = frame.humidity_change / frame.gap_hours
    frame["temperature_rate"] = frame.temperature_change / frame.gap_hours
    frame["pressure_rate"] = frame.pressure_change / frame.gap_hours
    frame["humidity_flatline_run"] = [0, 1, 2]
    frame["temperature_flatline_run"] = 0
    frame["pressure_flatline_run"] = 0
    frame["humidity_persistence_hours"] = [0.0, 12.0, 24.0]
    frame["temperature_persistence_hours"] = 0.0
    frame["pressure_persistence_hours"] = 0.0
    result = rule_qc_scores(frame, PipelineConfig())
    assert result["humidity_stuck_score"].max() == 0


def test_non_boundary_flatline_is_stuck_after_elapsed_duration():
    frame = _frame([77.0, 77.0, 77.0], [24.0, 25.0, 26.0], [1000.0, 1001.0, 1002.0])
    frame["gap_hours"] = 12.0
    frame["humidity_change"] = frame.relative_humidity_pct.diff()
    frame["temperature_change"] = frame.temperature_c.diff()
    frame["pressure_change"] = frame.pressure_hpa.diff()
    frame["humidity_rate"] = frame.humidity_change / frame.gap_hours
    frame["temperature_rate"] = frame.temperature_change / frame.gap_hours
    frame["pressure_rate"] = frame.pressure_change / frame.gap_hours
    frame["humidity_flatline_run"] = [0, 1, 2]
    frame["temperature_flatline_run"] = 0
    frame["pressure_flatline_run"] = 0
    frame["humidity_persistence_hours"] = [0.0, 12.0, 24.0]
    frame["temperature_persistence_hours"] = 0.0
    frame["pressure_persistence_hours"] = 0.0
    result = rule_qc_scores(frame, PipelineConfig())
    assert result["humidity_stuck_score"].iloc[-1] == 1


def test_drift_score_responds_to_causal_residual_trend():
    frame = _frame([50.0] * 6, [20.0, 20.0, 21.0, 23.0, 26.0, 30.0], [1000.0] * 6)
    frame["temperature_change"] = frame.temperature_c.diff()
    frame["humidity_change"] = frame.relative_humidity_pct.diff()
    frame["pressure_change"] = frame.pressure_hpa.diff()
    frame["gap_hours"] = 4.0
    frame["temperature_rate"] = frame.temperature_change / frame.gap_hours
    frame["humidity_rate"] = 0.0
    frame["pressure_rate"] = 0.0
    frame["temperature_residual"] = frame.temperature_c - 20.0
    frame["humidity_residual"] = 0.0
    frame["pressure_residual"] = 0.0
    frame["temperature_zscore"] = 0.0
    frame["humidity_zscore"] = 0.0
    frame["pressure_zscore"] = 0.0
    result = temporal_scores(frame)
    assert result["drift_score"].iloc[-1] > 0
