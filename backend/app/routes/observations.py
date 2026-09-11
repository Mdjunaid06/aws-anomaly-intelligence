"""Observation API routes."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..schemas import ObservationCreate, ObservationListResponse, ObservationResponse
from ..services import ObservationService

router = APIRouter(prefix="/observations", tags=["observations"])


@router.post("/", response_model=ObservationResponse, status_code=201)
def create_observation(
    observation: ObservationCreate,
    db: Session = Depends(get_db),
):
    """Create a new weather observation."""
    db_obs = ObservationService.create(db, observation)
    return db_obs


@router.get("/{obs_id}", response_model=ObservationResponse)
def get_observation(
    obs_id: int,
    db: Session = Depends(get_db),
):
    """Get observation by ID."""
    obs = ObservationService.get_by_id(db, obs_id)
    if not obs:
        raise HTTPException(status_code=404, detail="Observation not found")
    return obs


@router.get("/station/{station_id}", response_model=ObservationListResponse)
def list_observations_by_station(
    station_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List observations for a station."""
    total, observations = ObservationService.list_by_station(
        db,
        station_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )
    return ObservationListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=observations,
    )


@router.get("/", response_model=ObservationListResponse)
def list_recent_observations(
    days: int = Query(7, ge=1, le=365),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List recent observations."""
    total, observations = ObservationService.list_recent(
        db,
        days=days,
        limit=limit,
        offset=offset,
    )
    return ObservationListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=observations,
    )


@router.get("/stations/list", response_model=dict)
def get_stations(db: Session = Depends(get_db)):
    """Get list of all station IDs."""
    stations = ObservationService.get_stations(db)
    return {"stations": stations, "count": len(stations)}
