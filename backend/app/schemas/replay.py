"""Replay API contracts."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


REPLAY_SPEEDS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


class ReplayStartRequest(BaseModel):
    station_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    speed: float = Field(1.0, description="Replay multiplier")
    include_spatial: bool = True

    @model_validator(mode="after")
    def validate_range_and_speed(self):
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must be before end_time")
        if self.speed not in REPLAY_SPEEDS:
            raise ValueError(f"speed must be one of {REPLAY_SPEEDS}")
        return self


class ReplayStatus(BaseModel):
    state: str
    running: bool
    paused: bool
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_timestamp: Optional[datetime] = None
    speed: float = 1.0
    processed_observations: int = 0
    anomaly_count: int = 0
    current_station: Optional[str] = None
    current_prediction: Optional[dict] = None
    current_health: Optional[dict] = None
    latest_spatial_evidence: Optional[dict] = None
    error: Optional[str] = None


class ReplayConfigResponse(BaseModel):
    source: str
    speeds: list[float]
    stations: list[str]
    min_timestamp: Optional[datetime] = None
    max_timestamp: Optional[datetime] = None
