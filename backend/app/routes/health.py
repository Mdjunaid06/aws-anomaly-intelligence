"""Health API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..schemas import HealthAlertResponse, SensorHealthListResponse, SensorHealthResponse
from ..services import HealthService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/{station_id}", response_model=SensorHealthResponse)
def get_health(
    station_id: str,
    db: Session = Depends(get_db),
):
    """Get health status for a station."""
    health = HealthService.get_health(db, station_id)
    if not health:
        raise HTTPException(status_code=404, detail="Station health not found")
    return health


@router.get("/", response_model=SensorHealthListResponse)
def list_all_health(db: Session = Depends(get_db)):
    """List health status for all stations."""
    all_health = HealthService.list_all_health(db)
    
    operational = len([h for h in all_health if h.overall_health == "operational"])
    degraded = len([h for h in all_health if h.overall_health == "degraded"])
    failed = len([h for h in all_health if h.overall_health == "failed"])
    
    return SensorHealthListResponse(
        total=len(all_health),
        operational=operational,
        degraded=degraded,
        failed=failed,
        items=all_health,
    )


@router.post("/refresh", response_model=dict)
def refresh_health(
    db: Session = Depends(get_db),
):
    """Recompute and update health for all stations."""
    results = HealthService.update_all_health(db, days=30)
    
    operational = len([r for r in results.values() if r["health"] == "operational"])
    degraded = len([r for r in results.values() if r["health"] == "degraded"])
    failed = len([r for r in results.values() if r["health"] == "failed"])
    
    return {
        "status": "Health refreshed",
        "total_stations": len(results),
        "operational": operational,
        "degraded": degraded,
        "failed": failed,
        "details": results,
    }


@router.get("/alerts/", response_model=list[HealthAlertResponse])
def get_health_alerts(
    severity: Optional[str] = Query(None),
    station_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Get health alerts for problematic stations."""
    all_health = HealthService.list_all_health(db)
    
    alerts = []
    for health in all_health:
        if station_id and health.station_id != station_id:
            continue
        
        if health.overall_health in ["degraded", "failed"]:
            issues = []
            if health.has_stuck_values:
                issues.append("temperature")  # Could be any variable
            if health.has_drift:
                issues.append("calibration")
            if health.has_communication_faults:
                issues.append("communication")
            
            alert_severity = "critical" if health.overall_health == "failed" else "warning"
            
            if severity and alert_severity != severity:
                continue
            
            alert = HealthAlertResponse(
                station_id=health.station_id,
                health_status=health.overall_health,
                severity=alert_severity,
                message=f"Station {health.station_id} is {health.overall_health}",
                affected_variables=issues,
                recommended_action="Investigate sensor hardware and communication link",
                detected_at=health.updated_at,
            )
            alerts.append(alert)
    
    return alerts
