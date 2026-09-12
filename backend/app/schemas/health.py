"""Pydantic schemas for sensor health API."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class SensorHealthResponse(BaseModel):
    """Sensor health status for a station."""

    station_id: str
    overall_health: str = Field(..., description="operational, degraded, or failed")
    health_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Individual component health
    temperature_health: Optional[float] = None
    pressure_health: Optional[float] = None
    humidity_health: Optional[float] = None

    # Fault indicators
    has_stuck_values: int
    has_drift: int
    has_communication_faults: int
    has_calibration_issues: int

    # Operational metrics
    last_observation_timestamp: Optional[datetime] = None
    days_without_data: int
    anomaly_rate_percent: Optional[float] = None
    recent_anomaly_count: int

    # Metadata
    health_metrics: Optional[dict[str, Any]] = None
    updated_at: datetime

    class Config:
        from_attributes = True


class SensorHealthListResponse(BaseModel):
    """List of sensor health status."""

    total: int = Field(..., description="Total number of stations")
    operational: int = Field(..., description="Number of operational stations")
    degraded: int = Field(..., description="Number of degraded stations")
    failed: int = Field(..., description="Number of failed stations")
    items: list[SensorHealthResponse]


class HealthAlertResponse(BaseModel):
    """Alert for unhealthy sensor."""

    station_id: str
    health_status: str
    severity: str = Field(..., description="info, warning, critical")
    message: str
    affected_variables: list[str]
    recommended_action: str
    detected_at: datetime
