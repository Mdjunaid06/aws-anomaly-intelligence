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
    result = next((item for item in list_stations(db)["stations"] if item["station_id"] == station_id), None)
    if not result:
        raise HTTPException(status_code=404, detail="Station not found")
    return result
