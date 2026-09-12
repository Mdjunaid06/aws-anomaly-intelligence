"""Observation model - stores weather observation data."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Index, Integer, String

from ..core.database import Base


class Observation(Base):
    """AWS weather observation."""

    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(String(32), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Required meteorological variables
    temperature_c = Column(Float, nullable=True)
    pressure_hpa = Column(Float, nullable=True)
    relative_humidity_pct = Column(Float, nullable=True)

    # Metadata
    data_source = Column(String(50), default="AWS")  # AWS, NOAA, etc.
    is_injected = Column(Integer, default=0)  # 0=original, 1=injected for testing
    is_processed = Column(Integer, default=0)  # Whether it went through feature engineering

    # Timestamps for tracking
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_observations_station_timestamp", "station_id", "timestamp"),
    )

    class Config:
        """SQLAlchemy config."""

        orm_mode = True
