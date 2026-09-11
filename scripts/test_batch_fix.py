#!/usr/bin/env python
"""Test the batch processing with the fix."""

import logging
import sys
from pathlib import Path

# Set up path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.database import SessionLocal, init_db
from backend.app.services.batch import BatchProcessingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Test batch processing for INM00043071."""
    init_db()
    db = SessionLocal()
    
    try:
        station_id = "INM00043071"
        
        logger.info("="*60)
        logger.info(f"Testing batch processing for {station_id}")
        logger.info("="*60)
        
        result = BatchProcessingService.score_station_batch(
            db,
            station_id,
            limit=145,
            include_spatial=True
        )
        
        logger.info(f"\nRESULT:")
        logger.info(f"  Station: {result['station_id']}")
        logger.info(f"  Processed: {result['processed']}")
        logger.info(f"  Anomalies: {result['anomalies']}")
        logger.info(f"  Skipped (missing data): {result.get('skipped', 0)}")
        logger.info(f"  Failed (errors): {result.get('failed', 0)}")
        
        # Test a station with good data quality
        logger.info("\n" + "="*60)
        logger.info(f"Testing batch processing for INM00043111 (good data)")
        logger.info("="*60)
        
        result2 = BatchProcessingService.score_station_batch(
            db,
            "INM00043111",
            limit=50,
            include_spatial=True
        )
        
        logger.info(f"\nRESULT:")
        logger.info(f"  Station: {result2['station_id']}")
        logger.info(f"  Processed: {result2['processed']}")
        logger.info(f"  Anomalies: {result2['anomalies']}")
        logger.info(f"  Skipped (missing data): {result2.get('skipped', 0)}")
        logger.info(f"  Failed (errors): {result2.get('failed', 0)}")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
