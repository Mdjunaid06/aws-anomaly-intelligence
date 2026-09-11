#!/usr/bin/env python
"""Diagnostic script to reproduce batch processing failure."""

import logging
import sys
from pathlib import Path
import traceback

# Set up path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.database import SessionLocal, init_db
from backend.app.services.batch import BatchProcessingService
from backend.app.services.observation import ObservationService

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Reproduce batch processing failure."""
    init_db()
    db = SessionLocal()
    
    try:
        station_id = "INM00043071"
        
        # Get unprocessed observations
        unprocessed = ObservationService.get_unprocessed_for_station(db, station_id, limit=10)
        logger.info(f"Found {len(unprocessed)} unprocessed observations for {station_id}")
        
        if not unprocessed:
            logger.warning("No unprocessed observations found")
            return
        
        # Try to process first one
        obs = unprocessed[0]
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing observation {obs.id}")
        logger.info(f"{'='*60}")
        logger.info(f"Station ID: {obs.station_id}")
        logger.info(f"Timestamp: {obs.timestamp}")
        logger.info(f"Temperature: {obs.temperature_c}")
        logger.info(f"Pressure: {obs.pressure_hpa}")
        logger.info(f"Humidity: {obs.relative_humidity_pct}")
        logger.info(f"Is processed: {obs.is_processed}")
        
        try:
            result = BatchProcessingService.score_and_save_observation(
                db,
                obs.id,
                include_spatial=True
            )
            logger.info(f"\n✓ SUCCESS: {result}")
        except Exception as e:
            logger.error(f"\n✗ FAILURE:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"\nFull traceback:")
            traceback.print_exc()
            
            # Also try with include_spatial=False to narrow down the issue
            logger.info(f"\n{'='*60}")
            logger.info("Retrying with include_spatial=False...")
            logger.info(f"{'='*60}")
            try:
                result = BatchProcessingService.score_and_save_observation(
                    db,
                    obs.id,
                    include_spatial=False
                )
                logger.info(f"\n✓ SUCCESS with include_spatial=False: {result}")
            except Exception as e2:
                logger.error(f"\n✗ Also fails with include_spatial=False:")
                logger.error(f"Error type: {type(e2).__name__}")
                logger.error(f"Error message: {str(e2)}")
                traceback.print_exc()
    
    finally:
        db.close()

if __name__ == "__main__":
    main()
