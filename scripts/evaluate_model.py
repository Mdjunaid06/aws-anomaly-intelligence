"""Score injected observations and write honest evaluation metrics."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ml.src.config import PipelineConfig
from ml.src.detectors.isolation_forest import IsolationForestDetector
from ml.src.detectors.multivariate import MahalanobisDetector
from ml.src.evaluation.metrics import evaluate_predictions, save_metrics
from ml.src.models.pipeline import score_observations
from ml.src.paths import GROUND_TRUTH_FILE, INJECTED_OBSERVATIONS, METRICS_DIR, MODEL_DIR
from ml.src.preprocessing import load_processed_observations


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	if not INJECTED_OBSERVATIONS.exists() or not GROUND_TRUTH_FILE.exists():
		raise FileNotFoundError("Run scripts/inject_anomalies.py before evaluation.")
	models = {
		"isolation_forest": IsolationForestDetector.load(MODEL_DIR),
		"mahalanobis": MahalanobisDetector.load(MODEL_DIR),
	}
	all_predictions = score_observations(
		load_processed_observations(INJECTED_OBSERVATIONS), models, PipelineConfig()
	)
	train_end = pd.Timestamp(PipelineConfig().train_end)
	train_predictions = all_predictions[all_predictions["timestamp"] < train_end]
	predictions = all_predictions[all_predictions["timestamp"] >= train_end].copy()
	METRICS_DIR.mkdir(parents=True, exist_ok=True)
	training_path = METRICS_DIR / "training_diagnostics.csv"
	prediction_path = METRICS_DIR / "injected_predictions.csv"
	train_predictions.to_csv(training_path, index=False)
	predictions.to_csv(prediction_path, index=False)
	ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
	metrics = evaluate_predictions(predictions, ground_truth, start_time=train_end)
	isolation_flags = pd.Series(
		models["isolation_forest"].flag(all_predictions),
		index=all_predictions.index,
	)
	baseline_masks = {
		"rules_only": predictions["rule_qc_score"] >= 1.0,
		"isolation_forest_only": isolation_flags.loc[predictions.index],
		"multivariate_only": predictions["multivariate_score"] >= 1.0,
		"temporal_only": predictions["temporal_score"] >= 0.5,
		"spatial_only": predictions["spatial_score"] >= 0.42,
		"temporal_spatial": (predictions["temporal_score"] >= 0.5)
		| (predictions["spatial_disagreement"].fillna(0.0) >= 0.5),
	}
	metrics["baselines"] = {}
	for name, mask in baseline_masks.items():
		baseline = predictions.copy()
		baseline["anomaly"] = pd.Series(mask, index=predictions.index).astype(bool)
		baseline["classification"] = pd.NA
		metrics["baselines"][name] = evaluate_predictions(
			baseline, ground_truth, start_time=train_end
		)
	clean_predictions = score_observations(
		load_processed_observations(), models, PipelineConfig()
	)
	clean_predictions = clean_predictions[clean_predictions["timestamp"] >= train_end].copy()
	clean_path = METRICS_DIR / "clean_holdout_predictions.csv"
	clean_predictions.to_csv(clean_path, index=False)
	clean_truth = pd.DataFrame(columns=["anomaly_id", "station_id", "timestamp", "anomaly_type"])
	metrics["clean_holdout"] = evaluate_predictions(clean_predictions, clean_truth, start_time=train_end)
	metrics["training_diagnostics"] = {
		"rows": int(len(train_predictions)),
		"predicted_anomalies": int(train_predictions["anomaly"].sum()),
	}
	metrics_path = METRICS_DIR / "evaluation.json"
	save_metrics(metrics, metrics_path)
	print(json.dumps(metrics, indent=2, default=str))
	print(f"Saved predictions: {prediction_path}")
	print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
	main()
