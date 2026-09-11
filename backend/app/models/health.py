"""Sensor health model - tracks station health status."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Index, Integer, JSON, String

from ..core.database import Base


class SensorHealth(Base):
    """Station sensor health status and degradation tracking."""

    __tablename__ = "sensor_health"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(String(32), nullable=False, unique=True, index=True)

    # Health status
    overall_health = Column(String(20), default="operational")  # operational, degraded, failed
    health_score = Column(Float, nullable=True)  # 0.0 to 1.0

    # Specific health indicators
    temperature_health = Column(Float, nullable=True)
    pressure_health = Column(Float, nullable=True)
    humidity_health = Column(Float, nullable=True)

    # Fault indicators
    has_stuck_values = Column(Integer, default=0)  # 0=no, 1=yes
    has_drift = Column(Integer, default=0)
    has_communication_faults = Column(Integer, default=0)
    has_calibration_issues = Column(Integer, default=0)

    # Metadata
    last_observation_timestamp = Column(DateTime, nullable=True)
    days_without_data = Column(Integer, default=0)
    anomaly_rate_percent = Column(Float, nullable=True)  # Recent anomaly rate
    recent_anomaly_count = Column(Integer, default=0)  # Last N observations

    # Detailed health metrics
    health_metrics = Column(JSON, nullable=True)  # Extended health data

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_overall_health", "overall_health"),
    )

    class Config:
        """SQLAlchemy config."""

        orm_mode = True
