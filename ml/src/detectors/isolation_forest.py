"""Shared Isolation Forest on station-relative engineered features."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ml.src.config import PipelineConfig, RANDOM_SEED
from ml.src.paths import MODEL_DIR, ensure_dirs

FEATURE_COLUMNS = [
    "temperature_zscore",
    "humidity_zscore",
    "pressure_zscore",
    "temperature_rate",
    "humidity_rate",
    "pressure_rate",
    "temperature_residual",
    "humidity_residual",
    "pressure_residual",
    "temperature_flatline_run",
    "humidity_flatline_run",
    "pressure_flatline_run",
    "gap_hours",
    "variable_disagreement",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "elevation_m",
]


class IsolationForestDetector:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.config.isolation_estimators,
            contamination=self.config.isolation_contamination,
            random_state=self.config.random_seed,
            n_jobs=-1,
        )
        self.columns = FEATURE_COLUMNS
        self.train_score_low_: float | None = None
        self.train_score_high_: float | None = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing Isolation Forest features: {missing}")
        values = df[self.columns].replace([np.inf, -np.inf], np.nan)
        return values.to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        matrix = self.imputer.fit_transform(self._matrix(df))
        scaled = self.scaler.fit_transform(matrix)
        self.model.fit(scaled)
        raw = -self.model.decision_function(scaled)
        self.train_score_low_ = float(np.quantile(raw, 0.01))
        self.train_score_high_ = float(np.quantile(raw, 0.99))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self.imputer.transform(self._matrix(df))
        scaled = self.scaler.transform(matrix)
        raw = -self.model.decision_function(scaled)
        low = self.train_score_low_ if self.train_score_low_ is not None else float(np.min(raw))
        high = self.train_score_high_ if self.train_score_high_ is not None else float(np.max(raw))
        denom = max(high - low, 1e-9)
        return np.clip((raw - low) / denom, 0, 1)

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        """Use Isolation Forest's fitted contamination decision for a baseline."""
        matrix = self.imputer.transform(self._matrix(df))
        scaled = self.scaler.transform(matrix)
        return self.model.predict(scaled) == -1

    def save(self, directory: Path | None = None) -> Path:
        ensure_dirs()
        directory = directory or MODEL_DIR
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "imputer": self.imputer,
                "scaler": self.scaler,
                "columns": self.columns,
                "train_score_low": self.train_score_low_,
                "train_score_high": self.train_score_high_,
                "config": self.config,
            },
            directory / "isolation_forest.joblib",
        )
        (directory / "isolation_forest_columns.json").write_text(
            json.dumps(self.columns, indent=2),
            encoding="utf-8",
        )
        return directory / "isolation_forest.joblib"

    @classmethod
    def load(cls, directory: Path | None = None) -> "IsolationForestDetector":
        directory = directory or MODEL_DIR
        payload = joblib.load(directory / "isolation_forest.joblib")
        detector = cls(payload.get("config") or PipelineConfig(random_seed=RANDOM_SEED))
        detector.model = payload["model"]
        detector.imputer = payload["imputer"]
        detector.scaler = payload["scaler"]
        detector.columns = payload["columns"]
        detector.train_score_low_ = payload["train_score_low"]
        detector.train_score_high_ = payload["train_score_high"]
        return detector
