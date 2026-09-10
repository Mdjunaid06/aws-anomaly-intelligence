"""Multivariate consistency via Mahalanobis distance on causal z-scores."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.impute import SimpleImputer

from ml.src.config import PipelineConfig
from ml.src.paths import MODEL_DIR, ensure_dirs

COLUMNS = ["temperature_zscore", "humidity_zscore", "pressure_zscore"]


class MahalanobisDetector:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.imputer = SimpleImputer(strategy="median")
        self.model = EmpiricalCovariance()
        self.threshold_: float | None = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        values = df[COLUMNS].replace([np.inf, -np.inf], np.nan)
        return values.to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame) -> "MahalanobisDetector":
        matrix = self.imputer.fit_transform(self._matrix(df))
        self.model.fit(matrix)
        distances = self.model.mahalanobis(matrix)
        self.threshold_ = float(np.nanpercentile(distances, self.config.mahalanobis_flag_percentile))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self.imputer.transform(self._matrix(df))
        distances = self.model.mahalanobis(matrix)
        threshold = self.threshold_ if self.threshold_ is not None else float(np.nanpercentile(distances, 99))
        return np.clip(distances / max(threshold, 1e-9), 0, 1)

    def save(self, directory: Path | None = None) -> Path:
        ensure_dirs()
        directory = directory or MODEL_DIR
        joblib.dump(
            {
                "model": self.model,
                "imputer": self.imputer,
                "threshold": self.threshold_,
                "config": self.config,
            },
            directory / "mahalanobis.joblib",
        )
        return directory / "mahalanobis.joblib"

    @classmethod
    def load(cls, directory: Path | None = None) -> "MahalanobisDetector":
        directory = directory or MODEL_DIR
        payload = joblib.load(directory / "mahalanobis.joblib")
        detector = cls(payload.get("config") or PipelineConfig())
        detector.model = payload["model"]
        detector.imputer = payload["imputer"]
        detector.threshold_ = payload["threshold"]
        return detector
