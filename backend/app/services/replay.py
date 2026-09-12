"""Real historical replay through the production observation and ML path."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import pandas as pd

from ..core.config import settings
from ..core.database import SessionLocal
from ..models import AnomalyPrediction, Observation, SensorHealth
from ..schemas.replay import REPLAY_SPEEDS, ReplayStartRequest, ReplayStatus
from .batch import BatchProcessingService

LOGGER = logging.getLogger(__name__)


class ReplayService:
    """Controls one cancellable replay job per backend process."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._task: Optional[asyncio.Task] = None
        self._stop_requested = False
        self._status = ReplayStatus(state="stopped", running=False, paused=False)
        self._resume_event = asyncio.Event()
        self._resume_event.set()

    @property
    def source_path(self) -> Path:
        return settings.ml_data_dir / "features" / "aws_features_2024_2025.csv"

    def status(self) -> ReplayStatus:
        with self._lock:
            return self._status.model_copy(deep=True)

    def config(self) -> dict:
        frame = self._load_source(usecols=["timestamp", "station_id"])
        return {
            "source": str(self.source_path),
            "speeds": list(REPLAY_SPEEDS),
            "stations": sorted(frame["station_id"].astype(str).unique().tolist()),
            "min_timestamp": frame["timestamp"].min().to_pydatetime() if len(frame) else None,
            "max_timestamp": frame["timestamp"].max().to_pydatetime() if len(frame) else None,
        }

    async def start(self, request: ReplayStartRequest) -> ReplayStatus:
        with self._lock:
            if self._task and not self._task.done():
                raise RuntimeError("A replay is already running")
            frame = self._load_source()
            if request.station_id:
                frame = frame[frame["station_id"].astype(str) == request.station_id]
                if frame.empty:
                    raise ValueError(f"Unknown station: {request.station_id}")
            if request.start_time:
                frame = frame[frame["timestamp"] >= request.start_time]
            if request.end_time:
                frame = frame[frame["timestamp"] <= request.end_time]
            if frame.empty:
                raise ValueError("Replay range contains no observations")
            self._stop_requested = False
            self._resume_event = asyncio.Event()
            self._resume_event.set()
            self._status = ReplayStatus(
                state="running",
                running=True,
                paused=False,
                start_time=frame["timestamp"].min().to_pydatetime(),
                end_time=frame["timestamp"].max().to_pydatetime(),
                speed=request.speed,
            )
            self._task = asyncio.create_task(self._run(frame, request))
            return self._status.model_copy(deep=True)

    def pause(self) -> ReplayStatus:
        with self._lock:
            if not self._task or self._task.done() or self._status.state != "running":
                raise RuntimeError("Replay is not running")
            self._status.state = "paused"
            self._status.running = False
            self._status.paused = True
            self._resume_event.clear()
            return self._status.model_copy(deep=True)

    def resume(self) -> ReplayStatus:
        with self._lock:
            if not self._task or self._task.done() or self._status.state != "paused":
                raise RuntimeError("Replay is not paused")
            self._status.state = "running"
            self._status.running = True
            self._status.paused = False
            self._resume_event.set()
            return self._status.model_copy(deep=True)

    def stop(self) -> ReplayStatus:
        with self._lock:
            if not self._task or self._task.done():
                raise RuntimeError("Replay is not running")
            self._stop_requested = True
            self._resume_event.set()
            self._status.state = "stopped"
            self._status.running = False
            self._status.paused = False
            return self._status.model_copy(deep=True)

    def reset(self) -> ReplayStatus:
        with self._lock:
            if self._task and not self._task.done():
                raise RuntimeError("Stop the replay before resetting")
            self._status = ReplayStatus(state="stopped", running=False, paused=False)
            return self._status.model_copy(deep=True)

    def _load_source(self, usecols: Optional[list[str]] = None) -> pd.DataFrame:
        if not self.source_path.exists():
            raise FileNotFoundError(f"Replay source not found: {self.source_path}")
        frame = pd.read_csv(self.source_path, parse_dates=["timestamp"], usecols=usecols)
        return frame.sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    async def _run(self, frame: pd.DataFrame, request: ReplayStartRequest) -> None:
        previous_timestamp: Optional[pd.Timestamp] = None
        try:
            for row in frame.itertuples(index=False):
                await self._resume_event.wait()
                if self._stop_requested:
                    return
                timestamp = pd.Timestamp(row.timestamp)
                if previous_timestamp is not None:
                    delay = max((timestamp - previous_timestamp).total_seconds() / request.speed, 0.02)
                    await asyncio.sleep(min(delay, 2.0))
                previous_timestamp = timestamp
                result = await asyncio.to_thread(self._process_row, row, request.include_spatial)
                with self._lock:
                    self._status.current_timestamp = timestamp.to_pydatetime()
                    self._status.current_station = str(row.station_id)
                    self._status.processed_observations += 1
                    if result and result.get("is_anomaly"):
                        self._status.anomaly_count += 1
                    self._status.current_prediction = result
                    self._status.current_health = self._health_for(str(row.station_id))
                    self._status.latest_spatial_evidence = (result or {}).get("spatial_evidence")
            with self._lock:
                self._status.state = "completed"
                self._status.running = False
                self._status.paused = False
        except Exception as exc:
            LOGGER.exception("Replay failed")
            with self._lock:
                self._status.state = "stopped"
                self._status.running = False
                self._status.paused = False
                self._status.error = str(exc)

    def _process_row(self, row, include_spatial: bool) -> Optional[dict]:
        db = SessionLocal()
        try:
            timestamp = pd.Timestamp(row.timestamp).to_pydatetime()
            observation = db.query(Observation).filter(
                Observation.station_id == str(row.station_id),
                Observation.timestamp == timestamp,
            ).first()
            if observation is None:
                observation = Observation(
                    station_id=str(row.station_id),
                    timestamp=timestamp,
                    temperature_c=float(row.temperature_c) if pd.notna(row.temperature_c) else None,
                    pressure_hpa=float(row.pressure_hpa) if pd.notna(row.pressure_hpa) else None,
                    relative_humidity_pct=float(row.relative_humidity_pct) if pd.notna(row.relative_humidity_pct) else None,
                    data_source="replay",
                )
                db.add(observation)
                db.commit()
                db.refresh(observation)
            if observation.is_processed:
                prediction = db.query(AnomalyPrediction).filter_by(observation_id=observation.id).first()
                return self._prediction_dict(prediction) if prediction else None
            return BatchProcessingService.score_and_save_observation(db, observation.id, include_spatial)
        finally:
            db.close()

    def _health_for(self, station_id: str) -> Optional[dict]:
        db = SessionLocal()
        try:
            health = db.query(SensorHealth).filter_by(station_id=station_id).first()
            if not health:
                return None
            return {
                "station_id": health.station_id,
                "overall_health": health.overall_health,
                "health_score": health.health_score,
                "health_metrics": health.health_metrics,
                "last_observation_timestamp": health.last_observation_timestamp,
            }
        finally:
            db.close()

    @staticmethod
    def _prediction_dict(prediction: Optional[AnomalyPrediction]) -> Optional[dict]:
        if not prediction:
            return None
        return {
            "prediction_id": prediction.id,
            "observation_id": prediction.observation_id,
            "station_id": prediction.station_id,
            "timestamp": prediction.timestamp,
            "is_anomaly": prediction.is_anomaly,
            "confidence": prediction.confidence,
            "classification": prediction.classification,
            "root_cause": prediction.root_cause,
            "spatial_evidence": prediction.spatial_evidence,
        }


replay_service = ReplayService()
