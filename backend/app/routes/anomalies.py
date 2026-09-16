"""Anomaly API routes."""

from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import get_db
from ..models import AnomalyPrediction, Observation
from ..schemas import (
    AnomalyEvidenceDetail,
    AnomalyListResponse,
    AnomalyResponse,
    AnomalyStatsResponse,
)
from ..services import AnomalyService

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


def _metadata() -> pd.DataFrame:
    path = settings.ml_data_dir / "processed" / "station_metadata.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Station metadata is unavailable")
    return pd.read_csv(path)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    angle = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(min(1.0, angle)))


def _build_evidence_availability(pred: AnomalyPrediction) -> dict[str, dict[str, Any]]:
    """Describe detector availability with explanations matching SIH specifications."""
    is_spatial_insufficient = pred.classification == "insufficient_spatial_evidence"
    return {
        "temporal": {
            "available": pred.temporal_score is not None,
            "reason": None if pred.temporal_score is not None else "Historical baseline unavailable for this station window.",
        },
        "multivariate": {
            "available": pred.multivariate_score is not None,
            "reason": None if pred.multivariate_score is not None else "Requires simultaneous valid temperature, pressure, and humidity.",
        },
        "isolation": {
            "available": pred.isolation_score is not None,
            "reason": None if pred.isolation_score is not None else "Isolation forest input features missing.",
        },
        "spatial": {
            "available": pred.spatial_score is not None,
            "status": "insufficient" if is_spatial_insufficient else ("available" if pred.spatial_score is not None else "unavailable"),
            "reason": (
                "Not enough valid, time-aligned neighboring observations were available to make a reliable spatial comparison."
                if is_spatial_insufficient
                else (None if pred.spatial_score is not None else "Neighbor station data unavailable.")
            ),
        },
        "rule_qc": {
            "available": pred.rule_qc_score is not None,
            "reason": None if pred.rule_qc_score is not None else "Physical limits and rate-of-change check unavailable.",
        },
        "gru": {
            "available": pred.gru_score is not None,
            "reason": (
                None
                if pred.gru_score is not None
                else "Sequence evidence unavailable for this observation because sufficient valid sequence context was not available."
            ),
        },
        "persistence": {
            "available": pred.persistence_score is not None,
            "reason": None if pred.persistence_score is not None else "Persistence evaluation requires preceding consecutive observations.",
        },
        "drift": {
            "available": pred.drift_score is not None,
            "reason": None if pred.drift_score is not None else "Drift requires sufficient historical residual trend.",
        },
        "stuck": {
            "available": pred.stuck_score is not None,
            "reason": None if pred.stuck_score is not None else "Stuck-sensor detection requires repeated/flat behavior.",
        },
        "communication": {
            "available": pred.communication_score is not None,
            "reason": None if pred.communication_score is not None else "Communication evidence requires missing or irregular transmission behavior.",
        },
        "data_quality": {
            "available": pred.data_quality_score is not None,
            "reason": None if pred.data_quality_score is not None else "Data quality metric unavailable.",
        },
    }


def _to_anomaly_response(pred: AnomalyPrediction) -> AnomalyResponse:
    return AnomalyResponse(
        id=pred.id,
        observation_id=pred.observation_id,
        station_id=pred.station_id,
        timestamp=pred.timestamp,
        is_anomaly=pred.is_anomaly,
        confidence=pred.confidence,
        classification=pred.classification,
        evidence=pred.evidence,
        evidence_detail=AnomalyEvidenceDetail(
            temporal_score=pred.temporal_score,
            multivariate_score=pred.multivariate_score,
            isolation_score=pred.isolation_score,
            rule_qc_score=pred.rule_qc_score,
            data_quality_score=pred.data_quality_score,
            persistence_score=pred.persistence_score,
            drift_score=pred.drift_score,
            stuck_score=pred.stuck_score,
            communication_score=pred.communication_score,
            gru_score=pred.gru_score,
            spatial_score=pred.spatial_score,
        ),
        evidence_availability=_build_evidence_availability(pred),
        affected_stations=pred.affected_stations,
        supporting_stations=pred.supporting_stations,
        contradicting_stations=pred.contradicting_stations,
        root_cause=pred.root_cause,
        recommended_action=pred.recommended_action,
        explanation_facts=pred.explanation_facts,
        model_version=pred.model_version,
        created_at=pred.created_at,
    )


@router.get("/{pred_id}", response_model=AnomalyResponse)
def get_anomaly(
    pred_id: int,
    db: Session = Depends(get_db),
):
    """Get anomaly prediction by ID."""
    pred = AnomalyService.get_by_id(db, pred_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return _to_anomaly_response(pred)


@router.get("/{pred_id}/spatial-context")
def get_anomaly_spatial_context(
    pred_id: int,
    db: Session = Depends(get_db),
):
    """Retrieve target station observation and time-aligned neighboring station observations."""
    pred = AnomalyService.get_by_id(db, pred_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    target_obs = db.query(Observation).filter_by(id=pred.observation_id).first()
    metadata = _metadata()
    target_row = metadata[metadata["station_id"].astype(str) == pred.station_id]
    target_meta = target_row.iloc[0].to_dict() if not target_row.empty else {}

    t_lat = float(target_meta.get("latitude", 0.0))
    t_lon = float(target_meta.get("longitude", 0.0))

    window_start = pred.timestamp - timedelta(hours=3)
    window_end = pred.timestamp + timedelta(hours=3)

    supporting_set = set(pred.supporting_stations or [])
    contradicting_set = set(pred.contradicting_stations or [])
    affected_set = set(pred.affected_stations or [])

    neighbors = []
    for row in metadata.to_dict("records"):
        sid = str(row["station_id"])
        if sid == pred.station_id:
            continue
        dist = _haversine(t_lat, t_lon, float(row["latitude"]), float(row["longitude"]))
        if dist > 250.0:
            continue

        nearby_obs = (
            db.query(Observation)
            .filter(
                Observation.station_id == sid,
                Observation.timestamp >= window_start,
                Observation.timestamp <= window_end,
            )
            .order_by(Observation.timestamp.asc())
            .all()
        )

        closest_obs = None
        min_diff = timedelta(days=99)
        for cand in nearby_obs:
            diff = abs(cand.timestamp - pred.timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_obs = cand

        relation = "neutral"
        if sid in supporting_set:
            relation = "supporting"
        elif sid in contradicting_set:
            relation = "contradicting"
        elif sid in affected_set:
            relation = "affected"
        elif not closest_obs:
            relation = "unavailable"

        temp_delta = None
        pres_delta = None
        rh_delta = None
        if target_obs and closest_obs:
            if target_obs.temperature_c is not None and closest_obs.temperature_c is not None:
                temp_delta = round(float(target_obs.temperature_c - closest_obs.temperature_c), 2)
            if target_obs.pressure_hpa is not None and closest_obs.pressure_hpa is not None:
                pres_delta = round(float(target_obs.pressure_hpa - closest_obs.pressure_hpa), 2)
            if target_obs.relative_humidity_pct is not None and closest_obs.relative_humidity_pct is not None:
                rh_delta = round(float(target_obs.relative_humidity_pct - closest_obs.relative_humidity_pct), 1)

        neighbors.append({
            "station_id": sid,
            "name": row.get("station_name"),
            "distance_km": round(dist, 1),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "elevation_m": row.get("elevation_m"),
            "observation_timestamp": closest_obs.timestamp.isoformat() if closest_obs else None,
            "temperature_c": closest_obs.temperature_c if closest_obs else None,
            "pressure_hpa": closest_obs.pressure_hpa if closest_obs else None,
            "relative_humidity_pct": closest_obs.relative_humidity_pct if closest_obs else None,
            "delta_temperature_c": temp_delta,
            "delta_pressure_hpa": pres_delta,
            "delta_relative_humidity_pct": rh_delta,
            "relation": relation,
            "data_available": closest_obs is not None,
        })

    neighbors.sort(key=lambda x: x["distance_km"])
    usable_count = len([n for n in neighbors if n["data_available"]])
    is_insufficient = (pred.classification == "insufficient_spatial_evidence") or (usable_count < 2)

    return {
        "target": {
            "prediction_id": pred.id,
            "observation_id": pred.observation_id,
            "station_id": pred.station_id,
            "name": target_meta.get("station_name", pred.station_id),
            "timestamp": pred.timestamp.isoformat(),
            "temperature_c": target_obs.temperature_c if target_obs else None,
            "pressure_hpa": target_obs.pressure_hpa if target_obs else None,
            "relative_humidity_pct": target_obs.relative_humidity_pct if target_obs else None,
            "classification": pred.classification,
            "confidence": pred.confidence,
            "is_anomaly": pred.is_anomaly,
            "root_cause": pred.root_cause,
        },
        "spatial_status": "insufficient" if is_insufficient else "sufficient",
        "explanation": (
            "Not enough valid, time-aligned neighboring observations were available to make a reliable spatial comparison."
            if is_insufficient
            else f"Cross-station comparison against {usable_count} time-aligned neighbor stations within 250 km."
        ),
        "usable_neighbors_count": usable_count,
        "supporting_count": len(supporting_set),
        "contradicting_count": len(contradicting_set),
        "neighbors": neighbors,
    }


@router.get("/station/{station_id}", response_model=AnomalyListResponse)
def list_anomalies_by_station(
    station_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    anomalies_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List anomalies for a station."""
    total, predictions = AnomalyService.list_by_station(
        db,
        station_id,
        start_time=start_time,
        end_time=end_time,
        anomalies_only=anomalies_only,
        limit=limit,
        offset=offset,
    )
    items = [_to_anomaly_response(pred) for pred in predictions]
    anomaly_count = len([p for p in predictions if p.is_anomaly == 1])

    return AnomalyListResponse(
        total=total,
        limit=limit,
        offset=offset,
        anomaly_count=anomaly_count,
        items=items,
    )


@router.get("/", response_model=AnomalyListResponse)
def list_all_anomalies(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List all detected anomalies."""
    total, predictions = AnomalyService.list_anomalies(
        db,
        start_time=start_time,
        end_time=end_time,
        min_confidence=min_confidence,
        limit=limit,
        offset=offset,
    )
    items = [_to_anomaly_response(pred) for pred in predictions]
    anomaly_count = len([p for p in predictions if p.is_anomaly == 1])

    return AnomalyListResponse(
        total=total,
        limit=limit,
        offset=offset,
        anomaly_count=anomaly_count,
        items=items,
    )


@router.get("/stats/station/{station_id}", response_model=AnomalyStatsResponse)
def get_station_stats(
    station_id: str,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Get anomaly statistics for a station."""
    stats = AnomalyService.get_stats(db, station_id=station_id, days=days)
    return AnomalyStatsResponse(
        station_id=station_id,
        total_observations=stats["total_observations"],
        total_anomalies=stats["total_anomalies"],
        anomaly_rate=stats["anomaly_rate"],
        confidence_avg=stats["confidence_avg"],
        confidence_min=stats["confidence_min"],
        confidence_max=stats["confidence_max"],
        top_classifications=stats["top_classifications"],
        most_common_root_cause=stats["most_common_root_cause"],
    )


@router.get("/stats/all", response_model=AnomalyStatsResponse)
def get_all_stats(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Get anomaly statistics across all stations."""
    stats = AnomalyService.get_stats(db, station_id=None, days=days)
    return AnomalyStatsResponse(
        total_observations=stats["total_observations"],
        total_anomalies=stats["total_anomalies"],
        anomaly_rate=stats["anomaly_rate"],
        confidence_avg=stats["confidence_avg"],
        confidence_min=stats["confidence_min"],
        confidence_max=stats["confidence_max"],
        top_classifications=stats["top_classifications"],
        most_common_root_cause=stats["most_common_root_cause"],
    )
