"""Small next-observation GRU detector with causal station-local scoring."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from ml.src.config import PipelineConfig
from ml.src.models.sequence.dataset import VARIABLES, build_sequences, causal_windows
from ml.src.paths import MODEL_DIR, ensure_dirs


class _GRUPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(values)
        return self.output(hidden[-1])


class GRUDetector:
    """Predict the next T/RH/P vector and normalize prediction error to [0, 1]."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.sequence_length = self.config.gru_sequence_length
        self.hidden_size = self.config.gru_hidden_size
        self.num_layers = self.config.gru_num_layers
        self.epochs = self.config.gru_epochs
        self.batch_size = self.config.gru_batch_size
        self.learning_rate = self.config.gru_learning_rate
        self.max_gap_hours = self.config.communication_gap_hours
        self.scaler = StandardScaler()
        self.model = _GRUPredictor(3, self.hidden_size, self.num_layers, self.config.gru_dropout)
        self.threshold_: float | None = None
        self.error_scale_: float | None = None
        self.fitted_ = False

    def fit(self, observations: pd.DataFrame, *, end_time: str | pd.Timestamp) -> "GRUDetector":
        frame = observations.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        train = frame[frame["timestamp"] < pd.Timestamp(end_time)]
        values = train[list(VARIABLES)].dropna()
        if len(values) < self.sequence_length + 1:
            raise RuntimeError("Not enough clean observations to train the GRU.")
        self.scaler.fit(values)
        sequences, targets, _ = build_sequences(
            train,
            self.sequence_length,
            self.scaler,
            max_gap_hours=self.max_gap_hours,
        )
        if len(sequences) == 0:
            raise RuntimeError("No station-local GRU sequences were available.")
        torch.manual_seed(self.config.random_seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model.train()
        inputs = torch.from_numpy(sequences)
        expected = torch.from_numpy(targets)
        generator = torch.Generator().manual_seed(self.config.random_seed)
        for _ in range(self.epochs):
            order = torch.randperm(len(inputs), generator=generator)
            for batch in order.split(self.batch_size):
                optimizer.zero_grad()
                loss = loss_fn(self.model(inputs[batch]), expected[batch])
                loss.backward()
                optimizer.step()
        train_errors = self._errors(sequences, targets)
        percentile = self.config.gru_threshold_percentile / 100.0
        self.threshold_ = float(np.quantile(train_errors, percentile))
        self.error_scale_ = max(float(self.threshold_), 1e-9)
        self.fitted_ = True
        return self

    def _errors(self, sequences: np.ndarray, targets: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(torch.from_numpy(sequences)).numpy()
        return np.mean((prediction - targets) ** 2, axis=1)

    def score(self, observations: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("GRUDetector must be fitted or loaded before scoring.")
        frame = observations.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        scores = np.full(len(frame), np.nan, dtype=float)
        windows, row_indices = causal_windows(
            frame,
            self.sequence_length,
            self.scaler,
            max_gap_hours=self.max_gap_hours,
        )
        if len(windows):
            targets = frame.loc[row_indices, list(VARIABLES)].to_numpy(dtype=float)
            scaled_targets = self.scaler.transform(targets).astype(np.float32)
            errors = self._errors(windows, scaled_targets)
            scores[row_indices] = np.clip(errors / max(self.error_scale_ or 1.0, 1e-9), 0.0, 1.0)
        return scores

    def flag(self, observations: pd.DataFrame) -> np.ndarray:
        scores = self.score(observations)
        return np.nan_to_num(scores, nan=0.0) >= (self.threshold_ / max(self.error_scale_ or 1.0, 1e-9))

    def save(self, directory: Path | None = None) -> Path:
        ensure_dirs()
        directory = directory or MODEL_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "gru_detector.joblib"
        joblib.dump(
            {
                "state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "threshold": self.threshold_,
                "error_scale": self.error_scale_,
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "config": self.config,
                "variables": list(VARIABLES),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, directory: Path | None = None) -> "GRUDetector":
        payload = joblib.load((directory or MODEL_DIR) / "gru_detector.joblib")
        detector = cls(payload["config"])
        detector.sequence_length = payload["sequence_length"]
        detector.hidden_size = payload["hidden_size"]
        detector.num_layers = payload["num_layers"]
        detector.model = _GRUPredictor(3, detector.hidden_size, detector.num_layers, detector.config.gru_dropout)
        detector.model.load_state_dict(payload["state_dict"])
        detector.scaler = payload["scaler"]
        detector.threshold_ = payload["threshold"]
        detector.error_scale_ = payload["error_scale"]
        detector.fitted_ = True
        return detector
