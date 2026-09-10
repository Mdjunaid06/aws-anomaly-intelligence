"""Metrics computed from predictions and separately stored ground truth."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


TYPE_TO_STATE = {
    "temperature_spike": "likely_sensor_fault",
    "pressure_drop": "likely_sensor_fault",
    "humidity_spike": "likely_sensor_fault",
    "isolated_multivariate": "likely_sensor_fault",
    "sensor_drift": "likely_sensor_drift",
    "stuck_sensor": "likely_stuck_sensor",
    "missing_data": "likely_communication_fault",
    "communication_outage": "likely_communication_fault",
}


def evaluate_predictions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    start_time: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Evaluate row and episode detection on an explicit holdout window."""
    predictions = predictions.copy()
    ground_truth = ground_truth.copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    ground_truth["timestamp"] = pd.to_datetime(ground_truth["timestamp"])
    if start_time is not None:
        start = pd.Timestamp(start_time)
        predictions = predictions[predictions["timestamp"] >= start].copy()
        ground_truth = ground_truth[ground_truth["timestamp"] >= start].copy()
    truth_keys = ground_truth[["station_id", "timestamp"]].drop_duplicates()
    truth_keys["ground_truth_label"] = 1
    joined = predictions.merge(truth_keys, on=["station_id", "timestamp"], how="left")
    y_true = joined["ground_truth_label"].fillna(0).astype(int)
    y_pred = joined["anomaly"].astype(int)
    result: dict[str, object] = {
        "rows_evaluated": int(len(joined)),
        "ground_truth_label_rows": int(len(ground_truth)),
        "positive_rows": int(y_true.sum()),
        "predicted_positive_rows": int(y_pred.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "true_positive": int(((y_true == 1) & (y_pred == 1)).sum()),
        "false_positive": int(((y_true == 0) & (y_pred == 1)).sum()),
        "true_negative": int(((y_true == 0) & (y_pred == 0)).sum()),
        "false_negative": int(((y_true == 1) & (y_pred == 0)).sum()),
        "false_alarm_rate": float(((y_true == 0) & (y_pred == 1)).sum() / max((y_true == 0).sum(), 1)),
    }
    if "validity_status" in ground_truth.columns:
        result["validity_status_counts"] = ground_truth["validity_status"].value_counts().to_dict()
    episode_rows = joined.merge(
        ground_truth[["anomaly_id", "station_id", "timestamp", "anomaly_type"]],
        on=["station_id", "timestamp"], how="inner",
    )
    episodes = ground_truth[["anomaly_id", "anomaly_type"]].drop_duplicates()
    detected_ids = set(episode_rows.loc[episode_rows["anomaly"], "anomaly_id"])
    result["episodes_total"] = int(len(episodes))
    result["episodes_detected"] = int(len(detected_ids))
    result["episodes_missed"] = int(len(episodes) - len(detected_ids))
    result["episode_detection_rate"] = float(len(detected_ids) / max(len(episodes), 1))
    episode_latencies: list[float] = []
    for _, group in ground_truth.groupby("anomaly_id"):
        episode_predictions = joined.merge(
            group[["station_id", "timestamp"]],
            on=["station_id", "timestamp"],
            how="inner",
        )
        detected_times = episode_predictions.loc[episode_predictions["anomaly"], "timestamp"]
        if not detected_times.empty:
            start_column = "start_timestamp" if "start_timestamp" in group.columns else "timestamp"
            start = pd.Timestamp(group[start_column].iloc[0])
            episode_latencies.append(float((detected_times.min() - start).total_seconds() / 3600.0))
    result["mean_detection_latency_hours"] = (
        float(sum(episode_latencies) / len(episode_latencies)) if episode_latencies else None
    )
    labelled = joined.merge(
        ground_truth[["anomaly_id", "station_id", "timestamp", "anomaly_type"]],
        on=["station_id", "timestamp"], how="inner",
    )
    if not labelled.empty:
        expected = labelled["anomaly_type"].map(TYPE_TO_STATE)
        result["root_cause_accuracy"] = (
            float((labelled["classification"] == expected).mean())
            if labelled["classification"].notna().any()
            else None
        )
        episode_diagnoses = []
        for anomaly_id, group in labelled.groupby("anomaly_id"):
            expected_state = TYPE_TO_STATE.get(group["anomaly_type"].iloc[0])
            classifications = group["classification"].dropna()
            episode_diagnoses.append(bool(expected_state and expected_state in set(classifications)))
        result["episode_root_cause_accuracy"] = (
            float(sum(episode_diagnoses) / len(episode_diagnoses))
            if labelled["classification"].notna().any() and episode_diagnoses
            else None
        )
        result["by_anomaly_type"] = {
            anomaly_type: {
                "rows": int(len(group)),
                "detected": int(group["anomaly"].sum()),
                "episodes": int(group["anomaly_id"].nunique()),
                "detected_episodes": int(group.groupby("anomaly_id")["anomaly"].any().sum()),
                "mean_confidence": float(group["confidence"].mean()),
            }
            for anomaly_type, group in labelled.groupby("anomaly_type")
        }
    else:
        result["root_cause_accuracy"] = None
        result["episode_root_cause_accuracy"] = None
        result["by_anomaly_type"] = {}
    return result


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")