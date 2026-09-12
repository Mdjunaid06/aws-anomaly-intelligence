"""Anomaly API routes."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..schemas import (
    AnomalyListResponse,
    AnomalyResponse,
    AnomalyStatsResponse,
)
from ..services import AnomalyService

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.get("/{pred_id}", response_model=AnomalyResponse)
def get_anomaly(
    pred_id: int,
    db: Session = Depends(get_db),
):
    """Get anomaly prediction by ID."""
    pred = AnomalyService.get_by_id(db, pred_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    from ..schemas import AnomalyEvidenceDetail
    response = AnomalyResponse(
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
        affected_stations=pred.affected_stations,
        supporting_stations=pred.supporting_stations,
        contradicting_stations=pred.contradicting_stations,
        root_cause=pred.root_cause,
        recommended_action=pred.recommended_action,
        explanation_facts=pred.explanation_facts,
        model_version=pred.model_version,
        created_at=pred.created_at,
    )
    return response


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
    
    from ..schemas import AnomalyEvidenceDetail
    items = []
    for pred in predictions:
        items.append(AnomalyResponse(
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
            affected_stations=pred.affected_stations,
            supporting_stations=pred.supporting_stations,
            contradicting_stations=pred.contradicting_stations,
            root_cause=pred.root_cause,
            recommended_action=pred.recommended_action,
            explanation_facts=pred.explanation_facts,
            model_version=pred.model_version,
            created_at=pred.created_at,
        ))
    
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
    
    from ..schemas import AnomalyEvidenceDetail
    items = []
    for pred in predictions:
        items.append(AnomalyResponse(
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
            affected_stations=pred.affected_stations,
            supporting_stations=pred.supporting_stations,
            contradicting_stations=pred.contradicting_stations,
            root_cause=pred.root_cause,
            recommended_action=pred.recommended_action,
            explanation_facts=pred.explanation_facts,
            model_version=pred.model_version,
            created_at=pred.created_at,
        ))
    
    return AnomalyListResponse(
        total=total,
        limit=limit,
        offset=offset,
        anomaly_count=total,
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
