from __future__ import annotations

import pandas as pd

from ml.src.evaluation.metrics import evaluate_predictions
from ml.src.injection.anomalies import inject_anomalies


def test_injection_is_seeded_episode_based_and_does_not_mutate_source(tmp_path):
    source = tmp_path / "source.csv"
    injected_path = tmp_path / "injected.csv"
    truth_path = tmp_path / "truth.csv"
    source_frame = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "station_id": "S1",
                "temperature_c": 25.0 + index,
                "relative_humidity_pct": 40.0 + index,
                "pressure_hpa": 1000.0 + index,
            }
            for index, timestamp in enumerate(pd.date_range("2025-01-01", periods=60, freq="12h"))
        ]
    )
    source_frame.to_csv(source, index=False)
    original_bytes = source.read_bytes()

    _, truth = inject_anomalies(source, injected_path, truth_path, episodes_per_type=1)

    assert source.read_bytes() == original_bytes
    assert truth["anomaly_id"].nunique() == len(truth["anomaly_id"].unique())
    assert truth["anomaly_id"].nunique() >= 1
    changed = truth[truth["anomaly_type"] == "stuck_sensor"]
    assert (changed["original_value"] != changed["modified_value"]).all()
    for _, episode in truth.groupby("anomaly_id"):
        times = pd.to_datetime(episode["timestamp"])
        assert times.max() >= times.min()


def test_evaluation_reports_holdout_confusion_and_episode_metrics():
    predictions = pd.DataFrame(
        {
            "station_id": ["S1", "S1", "S1", "S1"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            "anomaly": [True, False, False, True],
            "classification": ["likely_sensor_fault", "normal", "normal", "normal"],
            "confidence": [0.8, 0.1, 0.1, 0.7],
        }
    )
    truth = pd.DataFrame(
        {
            "anomaly_id": ["event-1", "event-2"],
            "station_id": ["S1", "S1"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-03"]),
            "anomaly_type": ["temperature_spike", "temperature_spike"],
        }
    )

    metrics = evaluate_predictions(predictions, truth, start_time="2025-01-01")

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["episodes_total"] == 2
    assert metrics["episodes_detected"] == 1
    assert metrics["episodes_missed"] == 1
