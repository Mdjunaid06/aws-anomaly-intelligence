"""Batch processing service - scores observations and saves predictions."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..core.config import settings
from .ml_engine import get_ml_engine
from .anomaly import AnomalyService
from .observation import ObservationService

LOGGER = logging.getLogger(__name__)

# Cache for station metadata to avoid repeated file reads
_STATION_METADATA_CACHE: Optional[pd.DataFrame] = None


def _load_station_metadata() -> pd.DataFrame:
    """Load station metadata from CSV file.
    
    Raises:
        FileNotFoundError: If station metadata file does not exist
        ValueError: If metadata file is empty or malformed
    """
    global _STATION_METADATA_CACHE
    
    if _STATION_METADATA_CACHE is not None:
        return _STATION_METADATA_CACHE
    
    metadata_path = settings.ml_data_dir / "processed" / "station_metadata.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Station metadata file not found at: {metadata_path}\n"
            f"Expected path: data/processed/station_metadata.csv"
        )
    
    try:
        df = pd.read_csv(metadata_path)
        if df.empty:
            raise ValueError("Station metadata file is empty")
        
        # Ensure required columns exist
        required_cols = ["station_id", "station_name", "latitude", "longitude", "elevation_m"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Station metadata missing required columns: {missing}")
        
        _STATION_METADATA_CACHE = df
        LOGGER.info("Loaded station metadata: %d stations from %s", len(df), metadata_path)
        return df
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse station metadata CSV: {e}")


def _validate_observation_data(obs) -> tuple[bool, Optional[str]]:
    """Validate that observation has critical weather values.
    
    Args:
        obs: Observation model instance
        
    Returns:
        Tuple of (is_valid, skip_reason)
        - is_valid: True if observation can be scored
        - skip_reason: Human-readable reason if invalid, None if valid
    """
    missing_fields = []
    
    if obs.temperature_c is None:
        missing_fields.append("temperature_c")
    if obs.relative_humidity_pct is None:
        missing_fields.append("relative_humidity_pct")
    if obs.pressure_hpa is None:
        missing_fields.append("pressure_hpa")
    
    if missing_fields:
        reason = f"Missing critical weather data: {', '.join(missing_fields)}"
        return False, reason
    
    return True, None


def _enrich_observation_with_metadata(obs_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich observation DataFrame with station metadata.
    
    Args:
        obs_df: DataFrame with station_id, timestamp, and observation values
        
    Returns:
        DataFrame enriched with station_name, latitude, longitude, elevation_m
        
    Raises:
        ValueError: If station_id not found in metadata
    """
    metadata = _load_station_metadata()
    
    # Merge on station_id
    result = obs_df.merge(
        metadata[["station_id", "station_name", "latitude", "longitude", "elevation_m"]],
        on="station_id",
        how="left",
    )
    
    # Check if merge succeeded (no NaN in critical fields means metadata was found)
    missing_metadata = result[result["elevation_m"].isna()]
    if not missing_metadata.empty:
        missing_stations = missing_metadata["station_id"].unique().tolist()
        raise ValueError(
            f"Station metadata not found for: {missing_stations}\n"
            f"Available stations in metadata: {sorted(metadata['station_id'].unique().tolist())}"
        )
    
    return result


class BatchProcessingService:
    """Process observations in batches through ML pipeline and save predictions."""

    @staticmethod
    def score_and_save_observation(
        db: Session,
        observation_id: int,
        include_spatial: bool = True,
    ) -> Optional[dict]:
        """Score a single observation and save prediction to database.
        
        Args:
            db: Database session
            observation_id: ID of observation to score
            include_spatial: Whether to include spatial reasoning
            
        Returns:
            Prediction dict or None if observation not found
        """
        # Get observation
        obs = ObservationService.get_by_id(db, observation_id)
        if not obs:
            LOGGER.warning("Observation %d not found", observation_id)
            return None

        # Convert to DataFrame for ML pipeline
        obs_df = pd.DataFrame([{
            "station_id": obs.station_id,
            "timestamp": obs.timestamp,
            "temperature_c": obs.temperature_c,
            "pressure_hpa": obs.pressure_hpa,
            "relative_humidity_pct": obs.relative_humidity_pct,
        }])
        
        # Enrich with station metadata (station_name, latitude, longitude, elevation_m)
        obs_df = _enrich_observation_with_metadata(obs_df)

        try:
            ml_engine = get_ml_engine()
            
            # Score observation
            results_df = ml_engine.score_observation(obs_df, include_spatial=include_spatial)
            
            if results_df.empty:
                LOGGER.warning("No results from ML pipeline for observation %d", observation_id)
                return None
            
            result = results_df.iloc[0]
            
            # Extract evidence scores (match EvidenceInput dataclass fields)
            evidence_scores = {
                "temporal_score": float(result.get("temporal_score", 0.0)),
                "multivariate_score": float(result.get("multivariate_score", 0.0)),
                "isolation_score": float(result.get("isolation_score", 0.0)),
                "rule_qc_score": float(result.get("rule_qc_score", 0.0)),
                "data_quality_score": float(result.get("data_quality_score", 1.0)),
                "persistence_score": float(result.get("persistence_score", 0.0)),
                "drift_score": float(result.get("drift_score", 0.0)),
                "stuck_score": float(result.get("stuck_score", 0.0)),
                "communication_score": float(result.get("communication_score", 0.0)),
                "gru_score": result.get("gru_score"),
            }
            
            # Fuse evidence to get final decision
            fusion_result = ml_engine.fuse_evidence(evidence_scores, obs.station_id)
            
            # Create prediction record
            prediction = AnomalyService.create_prediction(
                db,
                observation_id=observation_id,
                station_id=obs.station_id,
                timestamp=obs.timestamp,
                is_anomaly=1 if fusion_result["anomaly"] else 0,
                confidence=fusion_result["confidence"],
                classification=fusion_result["classification"],
                evidence_scores=evidence_scores,
                spatial_evidence=fusion_result.get("evidence", {}).get("spatial"),
                affected_stations=fusion_result.get("affected_stations"),
                supporting_stations=fusion_result.get("supporting_stations"),
                contradicting_stations=fusion_result.get("contradicting_stations"),
                root_cause=fusion_result.get("root_cause"),
                recommended_action=fusion_result.get("recommended_action"),
                explanation_facts=fusion_result.get("explanation_facts"),
            )
            
            # Mark observation as processed
            ObservationService.mark_processed(db, observation_id)
            
            LOGGER.info(
                "Scored observation %d: anomaly=%s, confidence=%.2f",
                observation_id,
                fusion_result["anomaly"],
                fusion_result["confidence"],
            )
            
            return {
                "observation_id": observation_id,
                "prediction_id": prediction.id,
                "is_anomaly": prediction.is_anomaly,
                "confidence": prediction.confidence,
                "classification": prediction.classification,
            }
        
        except Exception as e:
            LOGGER.error("Failed to score observation %d: %s", observation_id, e)
            raise

    @staticmethod
    def score_station_batch(
        db: Session,
        station_id: str,
        limit: int = 1000,
        include_spatial: bool = True,
    ) -> dict:
        """Score all unprocessed observations for a station.
        
        Args:
            db: Database session
            station_id: Station to process
            limit: Maximum observations to process
            include_spatial: Whether to include spatial reasoning
            
        Returns:
            Summary of processing results
        """
        # Get unprocessed observations
        unprocessed = ObservationService.get_unprocessed_for_station(db, station_id, limit=limit)
        
        if not unprocessed:
            LOGGER.info("No unprocessed observations for station %s", station_id)
            return {"station_id": station_id, "processed": 0, "anomalies": 0, "skipped": 0, "failed": 0}
        
        LOGGER.info("Processing %d observations for station %s", len(unprocessed), station_id)
        
        processed = 0
        anomalies = 0
        skipped = 0
        failed = 0
        
        for obs in unprocessed:
            # Validate observation has critical weather data
            is_valid, skip_reason = _validate_observation_data(obs)
            if not is_valid:
                skipped += 1
                LOGGER.warning(
                    "Skipping observation %d for station %s: %s",
                    obs.id,
                    station_id,
                    skip_reason
                )
                continue
            
            try:
                result = BatchProcessingService.score_and_save_observation(
                    db, obs.id, include_spatial=include_spatial
                )
                if result:
                    processed += 1
                    if result["is_anomaly"]:
                        anomalies += 1
            except Exception as e:
                failed += 1
                LOGGER.exception("Failed to process observation %d: %s", obs.id, e)
        
        LOGGER.info(
            "Batch complete for %s: processed=%d, anomalies=%d, skipped=%d, failed=%d",
            station_id,
            processed,
            anomalies,
            skipped,
            failed,
        )
        
        return {
            "station_id": station_id,
            "processed": processed,
            "anomalies": anomalies,
            "skipped": skipped,
            "failed": failed,
        }

    @staticmethod
    def score_all_stations(
        db: Session,
        include_spatial: bool = True,
    ) -> dict:
        """Score all unprocessed observations across all stations.
        
        Returns:
            Summary of processing results by station
        """
        stations = ObservationService.get_stations(db)
        
        if not stations:
            LOGGER.info("No stations found in database")
            return {"total_stations": 0, "results": {}}
        
        LOGGER.info("Starting batch processing for %d stations", len(stations))
        
        results = {}
        total_processed = 0
        total_anomalies = 0
        
        for station_id in stations:
            try:
                result = BatchProcessingService.score_station_batch(
                    db, station_id, include_spatial=include_spatial
                )
                results[station_id] = result
                total_processed += result["processed"]
                total_anomalies += result["anomalies"]
            except Exception as e:
                LOGGER.error("Failed to process station %s: %s", station_id, e)
                results[station_id] = {"error": str(e)}
        
        LOGGER.info(
            "All stations processed: total=%d, anomalies=%d",
            total_processed,
            total_anomalies,
        )
        
        return {
            "total_stations": len(stations),
            "total_processed": total_processed,
            "total_anomalies": total_anomalies,
            "results_by_station": results,
        }
