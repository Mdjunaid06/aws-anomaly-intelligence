#!/usr/bin/env python
"""End-to-end backend validation - test complete pipeline."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Set up path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.config import settings
from backend.app.core.database import SessionLocal, init_db
from backend.app.models import Observation, AnomalyPrediction, SensorHealth
from backend.app.schemas import ObservationCreate
from backend.app.services import (
    ObservationService,
    AnomalyService,
    HealthService,
)
from backend.app.services.batch import BatchProcessingService

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_database_init():
    """Test database initialization."""
    logger.info("=" * 60)
    logger.info("TEST 1: Database Initialization")
    logger.info("=" * 60)
    
    try:
        init_db()
        logger.info("✓ Database initialized successfully")
        logger.info(f"✓ Database URL: {settings.database_url}")
        return True
    except Exception as e:
        logger.error("✗ Database initialization failed: %s", e)
        return False


def test_observation_crud(db):
    """Test observation CRUD operations."""
    logger.info("=" * 60)
    logger.info("TEST 2: Observation CRUD Operations")
    logger.info("=" * 60)
    
    try:
        # Create observation
        obs_data = ObservationCreate(
            station_id="INM00043064",
            timestamp=datetime.utcnow(),
            temperature_c=25.5,
            pressure_hpa=1013.25,
            relative_humidity_pct=65.0,
        )
        obs = ObservationService.create(db, obs_data)
        logger.info(f"✓ Created observation {obs.id}")
        
        # Read observation
        retrieved = ObservationService.get_by_id(db, obs.id)
        assert retrieved is not None
        logger.info(f"✓ Retrieved observation {retrieved.id}")
        
        # List observations
        total, items = ObservationService.list_recent(db, days=7, limit=10)
        logger.info(f"✓ Listed {total} recent observations (showing {len(items)})")
        
        return True, obs.id
    except Exception as e:
        logger.error("✗ CRUD operations failed: %s", e)
        return False, None


def test_batch_scoring(db, obs_id):
    """Test ML scoring via batch service."""
    logger.info("=" * 60)
    logger.info("TEST 3: ML Scoring via Batch Service")
    logger.info("=" * 60)
    
    try:
        result = BatchProcessingService.score_and_save_observation(
            db,
            obs_id,
            include_spatial=False,  # Disable spatial for single observation
        )
        
        if result:
            logger.info(f"✓ Scored observation {obs_id}")
            logger.info(f"  - Anomaly: {result['is_anomaly']}")
            logger.info(f"  - Confidence: {result['confidence']:.2f}")
            logger.info(f"  - Classification: {result['classification']}")
            logger.info(f"  - Prediction ID: {result['prediction_id']}")
            return True, result["prediction_id"]
        else:
            logger.error("✗ Scoring returned no result")
            return False, None
    except Exception as e:
        logger.error("✗ ML scoring failed: %s", e)
        return False, None


def test_anomaly_queries(db):
    """Test anomaly prediction queries."""
    logger.info("=" * 60)
    logger.info("TEST 4: Anomaly Prediction Queries")
    logger.info("=" * 60)
    
    try:
        # List anomalies
        total, anomalies = AnomalyService.list_anomalies(db, limit=10)
        logger.info(f"✓ Found {total} anomalies")
        
        # Get stats
        stats = AnomalyService.get_stats(db, days=30)
        logger.info(f"✓ Statistics computed:")
        logger.info(f"  - Total observations: {stats['total_observations']}")
        logger.info(f"  - Total anomalies: {stats['total_anomalies']}")
        logger.info(f"  - Anomaly rate: {stats['anomaly_rate']:.2%}")
        logger.info(f"  - Avg confidence: {stats['confidence_avg']:.2f}")
        
        return True
    except Exception as e:
        logger.error("✗ Anomaly queries failed: %s", e)
        return False


def test_health_service(db):
    """Test sensor health service."""
    logger.info("=" * 60)
    logger.info("TEST 5: Sensor Health Service")
    logger.info("=" * 60)
    
    try:
        # Get stations
        stations = ObservationService.get_stations(db)
        logger.info(f"✓ Found {len(stations)} stations")
        
        if stations:
            station_id = stations[0]
            
            # Compute health
            health_data = HealthService.compute_health(db, station_id, days=30)
            logger.info(f"✓ Computed health for {station_id}:")
            logger.info(f"  - Health: {health_data['health']}")
            logger.info(f"  - Health score: {health_data['health_score']:.2f}")
            logger.info(f"  - Anomaly rate: {health_data['anomaly_rate']:.2%}")
            logger.info(f"  - Issues: {health_data['issues']}")
            
            # Update health
            health = HealthService.update_health(
                db,
                station_id,
                overall_health=health_data["health"],
                health_score=health_data["health_score"],
            )
            logger.info(f"✓ Updated health for {station_id}")
            
            # List all health
            all_health = HealthService.list_all_health(db)
            logger.info(f"✓ Retrieved health for {len(all_health)} stations")
        
        return True
    except Exception as e:
        logger.error("✗ Health service failed: %s", e)
        return False


def test_ml_engine():
    """Test ML engine initialization."""
    logger.info("=" * 60)
    logger.info("TEST 6: ML Engine Initialization")
    logger.info("=" * 60)
    
    try:
        from backend.app.services import get_ml_engine
        
        ml_engine = get_ml_engine()
        logger.info("✓ ML engine initialized")
        
        # Get model info
        version = ml_engine.get_model_version()
        logger.info(f"✓ Model version: {version}")
        
        logger.info(f"✓ Models loaded:")
        logger.info(f"  - Isolation Forest: {ml_engine.models.get('isolation_forest')}")
        logger.info(f"  - Mahalanobis: {ml_engine.models.get('mahalanobis')}")
        logger.info(f"  - GRU: {ml_engine.models.get('gru')}")
        
        return True
    except Exception as e:
        logger.error("✗ ML engine initialization failed: %s", e)
        return False


def main():
    """Run all validation tests."""
    logger.info("")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║ AWS Anomaly Intelligence Backend - End-to-End Validation ║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("")
    
    results = {}
    
    # Test 1: ML Engine
    results["ML Engine"] = test_ml_engine()
    
    # Test 2: Database
    results["Database Init"] = test_database_init()
    if not results["Database Init"]:
        logger.error("Database initialization failed, cannot continue")
        return False
    
    db = SessionLocal()
    
    try:
        # Test 3: CRUD
        crud_ok, obs_id = test_observation_crud(db)
        results["CRUD Operations"] = crud_ok
        
        if not crud_ok:
            logger.error("CRUD operations failed, cannot continue with scoring")
            return False
        
        # Test 4: Scoring
        scoring_ok, pred_id = test_batch_scoring(db, obs_id)
        results["ML Scoring"] = scoring_ok
        
        # Test 5: Queries
        results["Anomaly Queries"] = test_anomaly_queries(db)
        
        # Test 6: Health
        results["Health Service"] = test_health_service(db)
    
    finally:
        db.close()
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("")
        logger.info("╔" + "=" * 58 + "╗")
        logger.info("║                 ALL TESTS PASSED ✓                        ║")
        logger.info("╚" + "=" * 58 + "╝")
        return True
    else:
        logger.error("")
        logger.error("╔" + "=" * 58 + "╗")
        logger.error("║                SOME TESTS FAILED ✗                      ║")
        logger.error("╚" + "=" * 58 + "╝")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
