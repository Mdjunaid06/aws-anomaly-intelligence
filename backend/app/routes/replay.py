"""Replay control routes."""

from fastapi import APIRouter, HTTPException

from ..schemas.replay import ReplayConfigResponse, ReplayStartRequest, ReplayStatus
from ..services.replay import replay_service

router = APIRouter(prefix="/replay", tags=["replay"])


def _error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409 if isinstance(exc, RuntimeError) else 400, detail=str(exc))


@router.get("/config", response_model=ReplayConfigResponse)
def replay_config():
    try:
        return replay_service.config()
    except Exception as exc:
        raise _error(exc) from exc


@router.get("/status", response_model=ReplayStatus)
def replay_status():
    return replay_service.status()


@router.post("/start", response_model=ReplayStatus)
async def replay_start(request: ReplayStartRequest):
    try:
        return await replay_service.start(request)
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/pause", response_model=ReplayStatus)
def replay_pause():
    try:
        return replay_service.pause()
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/resume", response_model=ReplayStatus)
def replay_resume():
    try:
        return replay_service.resume()
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/stop", response_model=ReplayStatus)
def replay_stop():
    try:
        return replay_service.stop()
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/reset", response_model=ReplayStatus)
def replay_reset():
    try:
        return replay_service.reset()
    except Exception as exc:
        raise _error(exc) from exc
