"""Train the anomaly detectors using only the pre-2025 real observations."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.src.config import PipelineConfig
from ml.src.features.build_features import build_features
from ml.src.models.pipeline import fit_models
from ml.src.paths import PROCESSED_OBSERVATIONS
from ml.src.preprocessing import load_processed_observations, prepare_observations


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	if not PROCESSED_OBSERVATIONS.exists():
		prepare_observations()
	config = PipelineConfig()
	observations = load_processed_observations()
	build_features(config=config)
	models = fit_models(observations, config)
	print(f"Training rows: {models['train_rows']}")
	print("Saved: ml/artifacts/models/isolation_forest.joblib")
	print("Saved: ml/artifacts/models/mahalanobis.joblib")
	print("Saved: ml/artifacts/models/gru_detector.joblib")


if __name__ == "__main__":
	main()
