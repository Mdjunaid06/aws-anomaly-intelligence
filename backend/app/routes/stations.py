"""Station metadata and current status routes."""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import get_db
from ..models import AnomalyPrediction, Observation, SensorHealth

router = APIRouter(prefix="/stations", tags=["stations"])


def _metadata() -> pd.DataFrame:
    path = settings.ml_data_dir / "processed" / "station_metadata.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Station metadata is unavailable")
    return pd.read_csv(path)


@router.get("/")
def list_stations(db: Session = Depends(get_db)):
    metadata = _metadata()
    items = []
    for row in metadata.to_dict("records"):
        station_id = str(row["station_id"])
        health = db.query(SensorHealth).filter_by(station_id=station_id).first()
        latest = db.query(Observation).filter_by(station_id=station_id).order_by(Observation.timestamp.desc()).first()
        anomaly = db.query(AnomalyPrediction).filter_by(station_id=station_id).order_by(AnomalyPrediction.timestamp.desc()).first()
        items.append({
            "station_id": station_id,
            "name": row.get("station_name"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "elevation_m": row.get("elevation_m"),
            "health_state": health.overall_health if health else "unknown",
            "health_score": health.health_score if health else None,
            "latest_timestamp": latest.timestamp if latest else None,
            "latest_anomaly": bool(anomaly.is_anomaly) if anomaly else False,
        })
    return {"stations": items, "count": len(items)}


@router.get("/{station_id}")
def get_station(station_id: str, db: Session = Depends(get_db)):
    metadata = _metadata()
    row = metadata[metadata["station_id"].astype(str) == station_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Station not found")
    row = row.iloc[0].to_dict()
    health = db.query(SensorHealth).filter_by(station_id=station_id).first()
    latest = db.query(Observation).filter_by(station_id=station_id).order_by(Observation.timestamp.desc()).first()
    anomaly = db.query(AnomalyPrediction).filter_by(station_id=station_id).order_by(AnomalyPrediction.timestamp.desc()).first()
    return {
        "station_id": station_id,
        "name": row.get("station_name"),
        "latitude": row.get("latitude"),
        "longitude": row.get("longitude"),
        "elevation_m": row.get("elevation_m"),
        "health_state": health.overall_health if health else "unknown",
        "health_score": health.health_score if health else None,
        "latest_timestamp": latest.timestamp if latest else None,
        "latest_anomaly": bool(anomaly.is_anomaly) if anomaly else False,
    }


@router.get("/{station_id}/history")
def get_station_history(
    station_id: str,
    range: str = "24h",
    limit: int = 200,
    db: Session = Depends(get_db),
):
    """Retrieve historical observations and time-aligned anomaly flags for plotting."""
    from datetime import datetime, timedelta

    metadata = _metadata()
    row = metadata[metadata["station_id"].astype(str) == station_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Station not found")
    name = row.iloc[0].get("station_name", station_id)

    # Determine latest observation timestamp to anchor relative ranges
    latest_obs = db.query(Observation).filter_by(station_id=station_id).order_by(Observation.timestamp.desc()).first()
    if not latest_obs:
        return {
            "station_id": station_id,
            "name": name,
            "range": range,
            "total_points": 0,
            "anomaly_points": 0,
            "observations": [],
        }

    anchor = latest_obs.timestamp
    query = db.query(Observation).filter(Observation.station_id == station_id)
    if range == "6h":
        query = query.filter(Observation.timestamp >= anchor - timedelta(hours=6))
    elif range == "24h":
        query = query.filter(Observation.timestamp >= anchor - timedelta(hours=24))
    elif range == "7d":
        query = query.filter(Observation.timestamp >= anchor - timedelta(days=7))
    elif range == "30d":
        query = query.filter(Observation.timestamp >= anchor - timedelta(days=30))

    observations = query.order_by(Observation.timestamp.asc()).limit(limit).all()

    # Pre-fetch predictions for these observations
    obs_ids = [o.id for o in observations]
    predictions_map = {}
    if obs_ids:
        preds = db.query(AnomalyPrediction).filter(AnomalyPrediction.observation_id.in_(obs_ids)).all()
        for p in preds:
            predictions_map[p.observation_id] = p

    items = []
    anomaly_count = 0
    for obs in observations:
        pred = predictions_map.get(obs.id)
        is_anom = bool(pred.is_anomaly) if pred else False
        if is_anom:
            anomaly_count += 1
        items.append({
            "observation_id": obs.id,
            "prediction_id": pred.id if pred else None,
            "timestamp": obs.timestamp.isoformat(),
            "temperature_c": obs.temperature_c,
            "pressure_hpa": obs.pressure_hpa,
            "relative_humidity_pct": obs.relative_humidity_pct,
            "is_anomaly": is_anom,
            "classification": pred.classification if pred else ("normal" if not is_anom else "unknown"),
            "confidence": pred.confidence if pred else None,
            "root_cause": pred.root_cause if pred else None,
        })

    return {
        "station_id": station_id,
        "name": name,
        "range": range,
        "total_points": len(items),
        "anomaly_points": anomaly_count,
        "observations": items,
    }
