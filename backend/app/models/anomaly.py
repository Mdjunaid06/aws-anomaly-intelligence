"""Anomaly detection model - stores ML pipeline predictions."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Index, Integer, JSON, String, Text

from ..core.database import Base


class AnomalyPrediction(Base):
    """ML anomaly detection result for an observation."""

    __tablename__ = "anomaly_predictions"

    id = Column(Integer, primary_key=True, index=True)
    observation_id = Column(Integer, nullable=False, index=True)
    station_id = Column(String(32), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Final anomaly decision
    is_anomaly = Column(Integer, default=0, nullable=False)  # 0=normal, 1=anomaly
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    classification = Column(String(50), nullable=False)  # DecisionState enum

    # Individual detector scores
    temporal_score = Column(Float, nullable=True)
    multivariate_score = Column(Float, nullable=True)
    isolation_score = Column(Float, nullable=True)
    rule_qc_score = Column(Float, nullable=True)
    data_quality_score = Column(Float, nullable=True)
    persistence_score = Column(Float, nullable=True)
    drift_score = Column(Float, nullable=True)
    stuck_score = Column(Float, nullable=True)
    communication_score = Column(Float, nullable=True)
    gru_score = Column(Float, nullable=True)
    spatial_score = Column(Float, nullable=True)

    # Spatial reasoning
    spatial_evidence = Column(JSON, nullable=True)  # SpatialEvidence as dict
    affected_stations = Column(JSON, nullable=True)  # List of station IDs
    supporting_stations = Column(JSON, nullable=True)
    contradicting_stations = Column(JSON, nullable=True)

    # Root cause and action
    root_cause = Column(String(255), nullable=True)
    recommended_action = Column(Text, nullable=True)
    explanation_facts = Column(JSON, nullable=True)  # List of explanation strings

    # Full evidence for auditability
    evidence = Column(JSON, nullable=True)  # Complete evidence dict

    # Model version for tracking
    model_version = Column(String(50), nullable=True)
    pipeline_config_hash = Column(String(64), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_anomaly_predictions_station_timestamp", "station_id", "timestamp"),
        Index("idx_is_anomaly", "is_anomaly"),
    )

    class Config:
        """SQLAlchemy config."""

        orm_mode = True
