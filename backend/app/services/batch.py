"""Batch processing service - scores observations and saves predictions."""

import logging
from pathlib import Path
from datetime import timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..models import Observation
from ..core.config import settings
from .ml_engine import get_ml_engine
from .anomaly import AnomalyService
from .health import HealthService
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


def _load_ml_context(db: Session, obs) -> pd.DataFrame:
    """Load causal station/network context required by the existing ML pipeline."""
    start_time = obs.timestamp - timedelta(days=30)
    observations = db.query(Observation).filter(
        Observation.timestamp >= start_time,
        Observation.timestamp <= obs.timestamp,
    ).order_by(Observation.timestamp.asc()).all()
    df = pd.DataFrame([
        {
            "station_id": item.station_id,
            "timestamp": item.timestamp,
            "temperature_c": item.temperature_c,
            "pressure_hpa": item.pressure_hpa,
            "relative_humidity_pct": item.relative_humidity_pct,
        }
        for item in observations
    ])
    if df.empty:
        return df

    # Ensure any context station passed to the frozen ML pipeline has at least one valid measurement
    # so that station-local window grouping never encounters an all-NaN empty group.
    valid_station_ids = set(
        df.dropna(subset=["temperature_c", "relative_humidity_pct", "pressure_hpa"])["station_id"].unique()
    )
    valid_station_ids.add(obs.station_id)
    return df[df["station_id"].isin(valid_station_ids)].reset_index(drop=True)


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

        is_valid, skip_reason = _validate_observation_data(obs)
        if not is_valid:
            LOGGER.warning("Observation %d invalid for scoring: %s", observation_id, skip_reason)
            return None

        # Load causal network context so spatial and temporal evidence remain available.
        obs_df = _load_ml_context(db, obs)
        
        # Enrich with station metadata (station_name, latitude, longitude, elevation_m)
        obs_df = _enrich_observation_with_metadata(obs_df)

        try:
            ml_engine = get_ml_engine()
            
            # Score observation
            results_df = ml_engine.score_observation(obs_df, include_spatial=include_spatial)
            
            if results_df.empty:
                LOGGER.warning("No results from ML pipeline for observation %d", observation_id)
                return None
            
            target_rows = results_df[
                (results_df["station_id"] == obs.station_id)
                & (pd.to_datetime(results_df["timestamp"]) == pd.Timestamp(obs.timestamp))
            ]
            if target_rows.empty:
                raise RuntimeError(
                    f"ML pipeline returned no result for observation {observation_id}"
                )
            result = target_rows.iloc[-1]
            
            # Persist the authoritative decision and evidence returned by the ML pipeline.
            evidence = result.get("evidence") or {}
            prediction = AnomalyService.create_prediction(
                db,
                observation_id=observation_id,
                station_id=obs.station_id,
                timestamp=obs.timestamp,
                is_anomaly=1 if bool(result["anomaly"]) else 0,
                confidence=float(result["confidence"]),
                classification=str(result["classification"]),
                evidence_scores=evidence,
                spatial_evidence=evidence.get("spatial") or evidence.get("spatial_by_variable"),
                affected_stations=result.get("affected_stations"),
                supporting_stations=result.get("supporting_stations"),
                contradicting_stations=result.get("contradicting_stations"),
                root_cause=result.get("root_cause"),
                recommended_action=result.get("recommended_action"),
                explanation_facts=result.get("explanation_facts"),
            )

            HealthService.persist_ml_health(
                db,
                station_id=obs.station_id,
                observation_timestamp=obs.timestamp,
                ml_result=result.to_dict(),
            )
            
            # Mark observation as processed
            ObservationService.mark_processed(db, observation_id)
            
            LOGGER.info(
                "Scored observation %d: anomaly=%s, confidence=%.2f",
                observation_id,
                result["anomaly"],
                result["confidence"],
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
