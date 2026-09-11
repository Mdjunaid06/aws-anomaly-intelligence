"""Pydantic schemas for observation API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ObservationCreate(BaseModel):
    """Incoming observation data from AWS or other source."""

    station_id: str = Field(..., description="Weather station identifier")
    timestamp: datetime = Field(..., description="Observation timestamp (UTC)")
    temperature_c: Optional[float] = Field(None, description="Temperature in Celsius")
    pressure_hpa: Optional[float] = Field(None, description="Pressure in hectopascals")
    relative_humidity_pct: Optional[float] = Field(None, description="Relative humidity in percent")

    class Config:
        json_schema_extra = {
            "example": {
                "station_id": "INM00043064",
                "timestamp": "2025-01-15T12:00:00Z",
                "temperature_c": 24.5,
                "pressure_hpa": 1013.25,
                "relative_humidity_pct": 65.0,
            }
        }


class ObservationResponse(BaseModel):
    """Observation data from database."""

    id: int
    station_id: str
    timestamp: datetime
    temperature_c: Optional[float] = None
    pressure_hpa: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    data_source: str
    is_injected: int
    is_processed: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ObservationListResponse(BaseModel):
    """List of observations with pagination."""

    total: int = Field(..., description="Total number of observations")
    limit: int = Field(..., description="Records per page")
    offset: int = Field(..., description="Records offset")
    items: list[ObservationResponse]
