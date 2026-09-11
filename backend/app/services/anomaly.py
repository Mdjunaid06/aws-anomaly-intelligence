"""Anomaly service - handles anomaly predictions and evidence storage."""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models import AnomalyPrediction, Observation
from .ml_engine import get_ml_engine

LOGGER = logging.getLogger(__name__)


class AnomalyService:
    """Service for managing anomaly predictions."""

    @staticmethod
    def create_prediction(
        db: Session,
        observation_id: int,
        station_id: str,
        timestamp: datetime,
        is_anomaly: int,
        confidence: float,
        classification: str,
        evidence_scores: dict,
        spatial_evidence: Optional[dict] = None,
        affected_stations: Optional[list[str]] = None,
        supporting_stations: Optional[list[str]] = None,
        contradicting_stations: Optional[list[str]] = None,
        root_cause: Optional[str] = None,
        recommended_action: Optional[str] = None,
        explanation_facts: Optional[list[str]] = None,
        model_version: Optional[str] = None,
    ) -> AnomalyPrediction:
        """Create a new anomaly prediction."""
        ml_engine = get_ml_engine()
        
        prediction = AnomalyPrediction(
            observation_id=observation_id,
            station_id=station_id,
            timestamp=timestamp,
            is_anomaly=is_anomaly,
            confidence=confidence,
            classification=classification,
            temporal_score=evidence_scores.get("temporal_score"),
            multivariate_score=evidence_scores.get("multivariate_score"),
            isolation_score=evidence_scores.get("isolation_score"),
            rule_qc_score=evidence_scores.get("rule_qc_score"),
            data_quality_score=evidence_scores.get("data_quality_score"),
            persistence_score=evidence_scores.get("persistence_score"),
            drift_score=evidence_scores.get("drift_score"),
            stuck_score=evidence_scores.get("stuck_score"),
            communication_score=evidence_scores.get("communication_score"),
            gru_score=evidence_scores.get("gru_score"),
            spatial_score=evidence_scores.get("spatial_score"),
            spatial_evidence=spatial_evidence,
            affected_stations=affected_stations,
            supporting_stations=supporting_stations,
            contradicting_stations=contradicting_stations,
            root_cause=root_cause,
            recommended_action=recommended_action,
            explanation_facts=explanation_facts,
            evidence=evidence_scores,
            model_version=model_version or ml_engine.get_model_version(),
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        LOGGER.info(
            "Created anomaly prediction for station %s at %s (anomaly=%s, confidence=%.2f)",
            station_id,
            timestamp,
            is_anomaly,
            confidence,
        )
        return prediction

    @staticmethod
    def get_by_id(db: Session, pred_id: int) -> Optional[AnomalyPrediction]:
        """Get prediction by ID."""
        return db.query(AnomalyPrediction).filter(
            AnomalyPrediction.id == pred_id
        ).first()

    @staticmethod
    def get_by_observation(
        db: Session,
        observation_id: int,
    ) -> Optional[AnomalyPrediction]:
        """Get prediction for an observation."""
        return db.query(AnomalyPrediction).filter(
            AnomalyPrediction.observation_id == observation_id
        ).first()

    @staticmethod
    def list_by_station(
        db: Session,
        station_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        anomalies_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[AnomalyPrediction]]:
        """List predictions for a station."""
        query = db.query(AnomalyPrediction).filter(
            AnomalyPrediction.station_id == station_id
        )
        
        if anomalies_only:
            query = query.filter(AnomalyPrediction.is_anomaly == 1)
        
        if start_time:
            query = query.filter(AnomalyPrediction.timestamp >= start_time)
        if end_time:
            query = query.filter(AnomalyPrediction.timestamp <= end_time)
        
        total = query.count()
        predictions = query.order_by(
            AnomalyPrediction.timestamp.desc()
        ).limit(limit).offset(offset).all()
        
        return total, predictions

    @staticmethod
    def list_anomalies(
        db: Session,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[AnomalyPrediction]]:
        """List all detected anomalies."""
        query = db.query(AnomalyPrediction).filter(
            AnomalyPrediction.is_anomaly == 1,
            AnomalyPrediction.confidence >= min_confidence,
        )
        
        if start_time:
            query = query.filter(AnomalyPrediction.timestamp >= start_time)
        if end_time:
            query = query.filter(AnomalyPrediction.timestamp <= end_time)
        
        total = query.count()
        predictions = query.order_by(
            AnomalyPrediction.confidence.desc(),
            AnomalyPrediction.timestamp.desc(),
        ).limit(limit).offset(offset).all()
        
        return total, predictions

    @staticmethod
    def get_stats(
        db: Session,
        station_id: Optional[str] = None,
        days: int = 30,
    ) -> dict:
        """Get anomaly statistics for a station or all stations."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(AnomalyPrediction).filter(
            AnomalyPrediction.timestamp >= cutoff
        )
        
        if station_id:
            query = query.filter(AnomalyPrediction.station_id == station_id)
        
        all_predictions = query.all()
        if not all_predictions:
            return {
                "total_observations": 0,
                "total_anomalies": 0,
                "anomaly_rate": 0.0,
                "confidence_avg": 0.0,
                "confidence_min": 0.0,
                "confidence_max": 0.0,
                "top_classifications": {},
                "most_common_root_cause": None,
            }
        
        total = len(all_predictions)
        anomalies = [p for p in all_predictions if p.is_anomaly == 1]
        anomaly_count = len(anomalies)
        
        confidences = [p.confidence for p in all_predictions]
        anomaly_confidences = [p.confidence for p in anomalies]
        
        classifications = {}
        for pred in anomalies:
            classifications[pred.classification] = classifications.get(pred.classification, 0) + 1
        
        root_causes = {}
        for pred in anomalies:
            if pred.root_cause:
                root_causes[pred.root_cause] = root_causes.get(pred.root_cause, 0) + 1
        
        most_common_root_cause = max(root_causes, key=root_causes.get) if root_causes else None
        
        return {
            "total_observations": total,
            "total_anomalies": anomaly_count,
            "anomaly_rate": anomaly_count / total if total > 0 else 0.0,
            "confidence_avg": sum(anomaly_confidences) / len(anomaly_confidences) if anomaly_confidences else 0.0,
            "confidence_min": min(anomaly_confidences) if anomaly_confidences else 0.0,
            "confidence_max": max(anomaly_confidences) if anomaly_confidences else 0.0,
            "top_classifications": classifications,
            "most_common_root_cause": most_common_root_cause,
        }
