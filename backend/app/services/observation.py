"""Observation service - handles observation CRUD operations."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models import Observation
from ..schemas import ObservationCreate, ObservationResponse

LOGGER = logging.getLogger(__name__)


class ObservationService:
    """Service for managing weather observations."""

    @staticmethod
    def create(db: Session, observation: ObservationCreate) -> Observation:
        """Create a new observation."""
        db_obs = Observation(
            station_id=observation.station_id,
            timestamp=observation.timestamp,
            temperature_c=observation.temperature_c,
            pressure_hpa=observation.pressure_hpa,
            relative_humidity_pct=observation.relative_humidity_pct,
            data_source="API",
            is_injected=0,
            is_processed=0,
        )
        db.add(db_obs)
        db.commit()
        db.refresh(db_obs)
        LOGGER.info(
            "Created observation for station %s at %s",
            observation.station_id,
            observation.timestamp,
        )
        return db_obs

    @staticmethod
    def get_by_id(db: Session, obs_id: int) -> Optional[Observation]:
        """Get observation by ID."""
        return db.query(Observation).filter(Observation.id == obs_id).first()

    @staticmethod
    def get_by_station_and_time(
        db: Session,
        station_id: str,
        timestamp: datetime,
    ) -> Optional[Observation]:
        """Get observation by station and timestamp."""
        return db.query(Observation).filter(
            Observation.station_id == station_id,
            Observation.timestamp == timestamp,
        ).first()

    @staticmethod
    def list_by_station(
        db: Session,
        station_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[Observation]]:
        """List observations for a station with optional time range.
        
        Returns:
            Tuple of (total_count, observations)
        """
        query = db.query(Observation).filter(Observation.station_id == station_id)
        
        if start_time:
            query = query.filter(Observation.timestamp >= start_time)
        if end_time:
            query = query.filter(Observation.timestamp <= end_time)
        
        total = query.count()
        observations = query.order_by(
            Observation.timestamp.desc()
        ).limit(limit).offset(offset).all()
        
        return total, observations

    @staticmethod
    def list_recent(
        db: Session,
        days: int = 7,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[Observation]]:
        """List recent observations from last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = db.query(Observation).filter(Observation.timestamp >= cutoff)
        total = query.count()
        observations = query.order_by(
            Observation.timestamp.desc()
        ).limit(limit).offset(offset).all()
        return total, observations

    @staticmethod
    def mark_processed(db: Session, obs_id: int) -> Observation:
        """Mark observation as processed by ML pipeline."""
        obs = db.query(Observation).filter(Observation.id == obs_id).first()
        if obs:
            obs.is_processed = 1
            db.commit()
            db.refresh(obs)
        return obs

    @staticmethod
    def get_unprocessed_for_station(
        db: Session,
        station_id: str,
        limit: int = 1000,
    ) -> list[Observation]:
        """Get unprocessed observations for a station."""
        return db.query(Observation).filter(
            Observation.station_id == station_id,
            Observation.is_processed == 0,
        ).order_by(Observation.timestamp.asc()).limit(limit).all()

    @staticmethod
    def get_stations(db: Session) -> list[str]:
        """Get list of all unique station IDs."""
        results = db.query(Observation.station_id).distinct().all()
        return [row[0] for row in results]

    @staticmethod
    def get_latest_timestamp(db: Session, station_id: str) -> Optional[datetime]:
        """Get latest observation timestamp for a station."""
        result = db.query(Observation.timestamp).filter(
            Observation.station_id == station_id
        ).order_by(Observation.timestamp.desc()).first()
        return result[0] if result else None
