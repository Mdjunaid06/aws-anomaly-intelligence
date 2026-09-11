#!/usr/bin/env python
"""Batch processing script - score observations and save predictions."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Set up path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.config import settings
from backend.app.core.database import SessionLocal
from backend.app.services.batch import BatchProcessingService
from backend.app.services.observation import ObservationService

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main batch processing entry point."""
    parser = argparse.ArgumentParser(
        description="Score observations and save ML predictions to database"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "station", "observation"],
        default="all",
        help="Processing mode",
    )
    parser.add_argument(
        "--station",
        type=str,
        help="Station ID (required for --mode station)",
    )
    parser.add_argument(
        "--observation-id",
        type=int,
        help="Observation ID (required for --mode observation)",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Disable spatial reasoning",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum observations to process",
    )
    
    args = parser.parse_args()
    
    db = SessionLocal()
    
    try:
        if args.mode == "observation":
            if not args.observation_id:
                logger.error("--observation-id required for --mode observation")
                sys.exit(1)
            
            logger.info("Processing observation %d", args.observation_id)
            result = BatchProcessingService.score_and_save_observation(
                db,
                args.observation_id,
                include_spatial=not args.no_spatial,
            )
            logger.info("Result: %s", result)
        
        elif args.mode == "station":
            if not args.station:
                logger.error("--station required for --mode station")
                sys.exit(1)
            
            logger.info("Processing station %s", args.station)
            result = BatchProcessingService.score_station_batch(
                db,
                args.station,
                limit=args.limit,
                include_spatial=not args.no_spatial,
            )
            logger.info("Result: %s", result)
        
        elif args.mode == "all":
            logger.info("Processing all stations")
            result = BatchProcessingService.score_all_stations(
                db,
                include_spatial=not args.no_spatial,
            )
            logger.info("Result: %s", result)
        
        logger.info("Batch processing complete")
    
    except Exception as e:
        logger.error("Batch processing failed: %s", e, exc_info=True)
        sys.exit(1)
    
    finally:
        db.close()


if __name__ == "__main__":
    main()
