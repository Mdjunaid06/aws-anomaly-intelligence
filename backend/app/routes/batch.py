"""Batch processing API routes."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..services.batch import BatchProcessingService

router = APIRouter(prefix="/batch", tags=["batch"])


@router.post("/station/{station_id}")
def process_station(
    station_id: str,
    limit: int = Query(1000, ge=1, le=10000),
    include_spatial: bool = Query(True),
    db: Session = Depends(get_db),
):
    """Process all unprocessed observations for a station."""
    result = BatchProcessingService.score_station_batch(
        db,
        station_id,
        limit=limit,
        include_spatial=include_spatial,
    )
    return {
        "status": "completed",
        "result": result,
    }


@router.post("/all")
def process_all_stations(
    include_spatial: bool = Query(True),
    db: Session = Depends(get_db),
):
    """Process all unprocessed observations across all stations."""
    result = BatchProcessingService.score_all_stations(
        db,
        include_spatial=include_spatial,
    )
    return {
        "status": "completed",
        "result": result,
    }


@router.post("/observation/{observation_id}")
def process_observation(
    observation_id: int,
    include_spatial: bool = Query(True),
    db: Session = Depends(get_db),
):
    """Process a single observation."""
    result = BatchProcessingService.score_and_save_observation(
        db,
        observation_id,
        include_spatial=include_spatial,
    )
    return {
        "status": "completed",
        "result": result,
    }
