#!/usr/bin/env python
"""Seed database with processed NOAA observations for backend testing.

This script loads the processed observations CSV into the SQLite database
without duplicating existing data.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Set up path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.config import settings
from backend.app.core.database import SessionLocal, init_db
from backend.app.models import Observation
from backend.app.services.observation import ObservationService
from backend.app.schemas import ObservationCreate

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def seed_observations(observations_csv: Path, limit: int = None, skip_existing: bool = True) -> dict:
    """Load observations from CSV into database.
    
    Args:
        observations_csv: Path to processed observations CSV
        limit: Maximum rows to import (None = all)
        skip_existing: Skip observations already in database
        
    Returns:
        Dictionary with import statistics
    """
    if not observations_csv.exists():
        raise FileNotFoundError(f"Observations CSV not found: {observations_csv}")
    
    logger.info("Loading observations from: %s", observations_csv)
    
    # Load CSV
    df = pd.read_csv(observations_csv)
    logger.info("Loaded %d rows from CSV", len(df))
    
    if limit:
        df = df.head(limit)
        logger.info("Limited to first %d rows", limit)
    
    # Select required columns
    required_cols = [
        "station_id",
        "timestamp",
        "temperature_c",
        "pressure_hpa",
        "relative_humidity_pct",
    ]
    
    df = df[required_cols].copy()
    
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Remove rows with missing station_id or timestamp
    df = df.dropna(subset=["station_id", "timestamp"])
    logger.info("After removing null station_id/timestamp: %d rows", len(df))
    
    # Convert NaN values in weather variables to None (NULL in DB)
    for col in ["temperature_c", "pressure_hpa", "relative_humidity_pct"]:
        df[col] = df[col].where(pd.notna(df[col]), None)
    
    # Open database session
    db = SessionLocal()
    
    try:
        init_db()  # Ensure tables exist
        
        inserted = 0
        skipped = 0
        errors = 0
        
        for idx, row in df.iterrows():
            try:
                station_id = row["station_id"]
                timestamp = row["timestamp"]
                
                # Check if already exists
                existing = ObservationService.get_by_station_and_time(db, station_id, timestamp)
                if existing:
                    if skip_existing:
                        skipped += 1
                        continue
                    else:
                        # Could update here if needed
                        skipped += 1
                        continue
                
                # Create new observation
                obs_create = ObservationCreate(
                    station_id=station_id,
                    timestamp=timestamp,
                    temperature_c=row["temperature_c"],
                    pressure_hpa=row["pressure_hpa"],
                    relative_humidity_pct=row["relative_humidity_pct"],
                )
                
                obs = ObservationService.create(db, obs_create)
                inserted += 1
                
                if (inserted + skipped) % 100 == 0:
                    logger.info("Progress: inserted=%d, skipped=%d, errors=%d", inserted, skipped, errors)
            
            except Exception as e:
                errors += 1
                logger.error("Error at row %d: %s", idx, e)
                if errors >= 10:  # Stop if too many errors
                    logger.error("Too many errors, stopping import")
                    break
        
        logger.info("=" * 60)
        logger.info("IMPORT COMPLETE")
        logger.info("=" * 60)
        logger.info("Inserted: %d", inserted)
        logger.info("Skipped:  %d", skipped)
        logger.info("Errors:   %d", errors)
        logger.info("Total:    %d", inserted + skipped + errors)
        
        return {
            "inserted": inserted,
            "skipped": skipped,
            "errors": errors,
            "total": inserted + skipped + errors,
        }
    
    finally:
        db.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed database with processed NOAA observations"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Path to observations CSV (default: data/processed/aws_observations_2024_2025.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum observations to import",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate station_id/timestamp combinations (default: skip)",
    )
    
    args = parser.parse_args()
    
    # Determine CSV path
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = settings.ml_data_dir / "processed" / "aws_observations_2024_2025.csv"
    
    logger.info("Database URL: %s", settings.database_url)
    logger.info("CSV path: %s", csv_path)
    
    try:
        result = seed_observations(
            csv_path,
            limit=args.limit,
            skip_existing=not args.allow_duplicates,
        )
        sys.exit(0)
    except Exception as e:
        logger.error("Import failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
