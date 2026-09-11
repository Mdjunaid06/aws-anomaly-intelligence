"""Health service - tracks sensor health status."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models import AnomalyPrediction, Observation, SensorHealth

LOGGER = logging.getLogger(__name__)


class HealthService:
    """Service for managing sensor health status."""

    @staticmethod
    def update_health(
        db: Session,
        station_id: str,
        overall_health: str,
        health_score: Optional[float] = None,
        temperature_health: Optional[float] = None,
        pressure_health: Optional[float] = None,
        humidity_health: Optional[float] = None,
        has_stuck_values: int = 0,
        has_drift: int = 0,
        has_communication_faults: int = 0,
        has_calibration_issues: int = 0,
    ) -> SensorHealth:
        """Update or create sensor health record."""
        health = db.query(SensorHealth).filter(
            SensorHealth.station_id == station_id
        ).first()
        
        if not health:
            health = SensorHealth(station_id=station_id)
            db.add(health)
        
        health.overall_health = overall_health
        health.health_score = health_score
        health.temperature_health = temperature_health
        health.pressure_health = pressure_health
        health.humidity_health = humidity_health
        health.has_stuck_values = has_stuck_values
        health.has_drift = has_drift
        health.has_communication_faults = has_communication_faults
        health.has_calibration_issues = has_calibration_issues
        
        db.commit()
        db.refresh(health)
        LOGGER.info("Updated health for station %s: %s", station_id, overall_health)
        return health

    @staticmethod
    def get_health(db: Session, station_id: str) -> Optional[SensorHealth]:
        """Get health status for a station."""
        return db.query(SensorHealth).filter(
            SensorHealth.station_id == station_id
        ).first()

    @staticmethod
    def list_all_health(db: Session) -> list[SensorHealth]:
        """List health status for all stations."""
        return db.query(SensorHealth).order_by(
            SensorHealth.overall_health,
            SensorHealth.station_id,
        ).all()

    @staticmethod
    def compute_health(
        db: Session,
        station_id: str,
        days: int = 30,
    ) -> dict:
        """Compute health metrics for a station based on recent data."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Get recent observations and predictions
        observations = db.query(Observation).filter(
            Observation.station_id == station_id,
            Observation.timestamp >= cutoff,
        ).all()
        
        predictions = db.query(AnomalyPrediction).filter(
            AnomalyPrediction.station_id == station_id,
            AnomalyPrediction.timestamp >= cutoff,
        ).all()
        
        if not observations:
            return {
                "health": "unknown",
                "health_score": 0.5,
                "anomaly_rate": 0.0,
                "days_without_data": 999,
                "issues": ["No recent observations"],
            }
        
        # Calculate basic metrics
        total_obs = len(observations)
        anomaly_count = len([p for p in predictions if p.is_anomaly == 1])
        anomaly_rate = anomaly_count / total_obs if total_obs > 0 else 0.0
        
        # Calculate missing data
        latest = max(o.timestamp for o in observations)
        days_since_last = (datetime.utcnow() - latest).days
        
        # Check for specific faults
        has_stuck = any(p.stuck_score and p.stuck_score > 0.7 for p in predictions)
        has_drift = any(p.drift_score and p.drift_score > 0.7 for p in predictions)
        has_communication = any(
            p.communication_score and p.communication_score > 0.7 
            for p in predictions
        )
        
        # Determine overall health
        issues = []
        if days_since_last > 1:
            issues.append(f"{days_since_last} days without data")
        if anomaly_rate > 0.2:
            issues.append(f"High anomaly rate: {anomaly_rate:.1%}")
        if has_stuck:
            issues.append("Stuck sensor values detected")
        if has_drift:
            issues.append("Sensor drift detected")
        if has_communication:
            issues.append("Communication faults detected")
        
        # Overall health score (0-1, higher is better)
        health_score = 1.0
        health_score -= min(0.3, anomaly_rate)  # Up to 30% penalty for anomalies
        health_score -= min(0.2, days_since_last / 30)  # Up to 20% penalty for stale data
        health_score -= (0.1 if has_stuck else 0)
        health_score -= (0.1 if has_drift else 0)
        health_score -= (0.1 if has_communication else 0)
        health_score = max(0.0, min(1.0, health_score))
        
        if health_score >= 0.8:
            health_status = "operational"
        elif health_score >= 0.5:
            health_status = "degraded"
        else:
            health_status = "failed"
        
        return {
            "health": health_status,
            "health_score": health_score,
            "anomaly_rate": anomaly_rate,
            "recent_anomaly_count": anomaly_count,
            "days_without_data": days_since_last,
            "has_stuck_values": 1 if has_stuck else 0,
            "has_drift": 1 if has_drift else 0,
            "has_communication_faults": 1 if has_communication else 0,
            "issues": issues,
        }

    @staticmethod
    def update_all_health(db: Session, days: int = 30) -> dict:
        """Update health for all stations."""
        # Get all stations
        stations = db.query(Observation.station_id).distinct().all()
        station_ids = [row[0] for row in stations]
        
        results = {}
        for station_id in station_ids:
            health_data = HealthService.compute_health(db, station_id, days=days)
            health = HealthService.update_health(
                db,
                station_id,
                overall_health=health_data["health"],
                health_score=health_data["health_score"],
                has_stuck_values=health_data.get("has_stuck_values", 0),
                has_drift=health_data.get("has_drift", 0),
                has_communication_faults=health_data.get("has_communication_faults", 0),
            )
            results[station_id] = health_data
            LOGGER.info("Updated health for %s: %s", station_id, health_data["health"])
        
        return results
