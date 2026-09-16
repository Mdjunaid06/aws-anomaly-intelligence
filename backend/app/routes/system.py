"""System status and real-time operational health checks."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..core.config import settings
from ..core.database import get_db, engine
from ..models import AnomalyPrediction, Observation, SensorHealth
from ..services.replay import replay_service
from ..services.ml_engine import get_ml_engine

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status")
def get_system_status(db: Session = Depends(get_db)):
    """Comprehensive health and readiness status for judges and operational dashboard."""
    # 1. Database status
    db_connected = False
    total_obs = 0
    total_anomalies = 0
    total_health = 0
    dialect = "unknown"
    try:
        dialect = engine.dialect.name
        total_obs = db.query(func.count(Observation.id)).scalar() or 0
        total_anomalies = db.query(func.count(AnomalyPrediction.id)).filter(AnomalyPrediction.is_anomaly == 1).scalar() or 0
        total_health = db.query(func.count(SensorHealth.id)).scalar() or 0
        db_connected = True
    except Exception:
        db_connected = False

    # 2. ML pipeline status
    ml_status = "offline"
    models_loaded = []
    try:
        ml_engine = get_ml_engine()
        models_loaded = list(ml_engine.models.keys())
        ml_status = "online"
    except Exception:
        ml_status = "error"

    # 3. Replay status
    replay_info = replay_service.status()

    # 4. GenAI configuration status
    genai_configured = bool(settings.genai_enabled and settings.genai_api_key and settings.genai_api_key.strip())

    return {
        "status": "operational" if (db_connected and ml_status == "online") else "degraded",
        "service": settings.api_title,
        "version": settings.api_version,
        "environment": settings.environment,
        "database": {
            "connected": db_connected,
            "dialect": dialect,
            "total_observations": total_obs,
            "total_anomalies": total_anomalies,
            "monitored_stations": total_health,
        },
        "ml_pipeline": {
            "status": ml_status,
            "models_loaded": models_loaded,
            "models_dir": str(settings.ml_models_dir),
            "frozen": True,
        },
        "replay": {
            "state": replay_info.state,
            "running": replay_info.running,
            "paused": replay_info.paused,
            "speed": replay_info.speed,
            "processed": replay_info.processed_observations,
            "total": replay_info.total_observations,
            "progress_pct": replay_info.progress_pct,
            "current_station": replay_info.current_station,
            "current_timestamp": replay_info.current_timestamp,
        },
        "genai": {
            "configured": genai_configured,
            "enabled": settings.genai_enabled,
            "provider": settings.genai_provider if genai_configured else None,
            "model": settings.genai_model if genai_configured else None,
            "fallback_active": not genai_configured,
        },
    }
