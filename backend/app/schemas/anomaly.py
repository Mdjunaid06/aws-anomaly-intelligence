"""Pydantic schemas for anomaly prediction API."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AnomalyEvidenceDetail(BaseModel):
    """Detailed evidence from ML detectors."""

    temporal_score: Optional[float] = None
    multivariate_score: Optional[float] = None
    isolation_score: Optional[float] = None
    rule_qc_score: Optional[float] = None
    data_quality_score: Optional[float] = None
    persistence_score: Optional[float] = None
    drift_score: Optional[float] = None
    stuck_score: Optional[float] = None
    communication_score: Optional[float] = None
    gru_score: Optional[float] = None
    spatial_score: Optional[float] = None


class AnomalyResponse(BaseModel):
    """Anomaly detection result for an observation."""

    id: int
    observation_id: int
    station_id: str
    timestamp: datetime

    # Decision
    is_anomaly: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    classification: str = Field(..., description="DecisionState classification")

    # Evidence scores
    evidence_detail: AnomalyEvidenceDetail

    # Spatial reasoning
    affected_stations: Optional[list[str]] = None
    supporting_stations: Optional[list[str]] = None
    contradicting_stations: Optional[list[str]] = None

    # Root cause
    root_cause: Optional[str] = None
    recommended_action: Optional[str] = None
    explanation_facts: Optional[list[str]] = None

    # Metadata
    model_version: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AnomalyListResponse(BaseModel):
    """List of anomalies with pagination."""

    total: int = Field(..., description="Total number of anomalies")
    limit: int = Field(..., description="Records per page")
    offset: int = Field(..., description="Records offset")
    anomaly_count: int = Field(..., description="Number of anomalies in this page")
    items: list[AnomalyResponse]


class AnomalyStatsResponse(BaseModel):
    """Anomaly statistics for a station or time period."""

    station_id: Optional[str] = None
    total_observations: int
    total_anomalies: int
    anomaly_rate: float = Field(..., ge=0.0, le=1.0)
    confidence_avg: float
    confidence_min: float
    confidence_max: float
    top_classifications: dict[str, int]
    most_common_root_cause: Optional[str] = None
