"""Project-relative paths. Never use machine-specific absolute roots."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw" / "NOAA" / "Pune_stations"
PROCESSED_DIR = DATA_ROOT / "processed"
FEATURES_DIR = DATA_ROOT / "features"
INJECTED_DIR = DATA_ROOT / "injected"
GROUND_TRUTH_DIR = DATA_ROOT / "ground_truth"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
METRICS_DIR = ARTIFACT_DIR / "metrics"

PROCESSED_OBSERVATIONS = PROCESSED_DIR / "aws_observations_2024_2025.csv"
STATION_METADATA = PROCESSED_DIR / "station_metadata.csv"
CLEAN_FEATURES = FEATURES_DIR / "aws_features_2024_2025.csv"
INJECTED_OBSERVATIONS = INJECTED_DIR / "aws_observations_injected.csv"
INJECTED_FEATURES = FEATURES_DIR / "aws_features_injected.csv"
GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "injected_anomalies.csv"
GRU_MODEL_FILE = MODEL_DIR / "gru_detector.joblib"


def ensure_dirs() -> None:
    for path in (
        PROCESSED_DIR,
        FEATURES_DIR,
        INJECTED_DIR,
        GROUND_TRUTH_DIR,
        MODEL_DIR,
        METRICS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
