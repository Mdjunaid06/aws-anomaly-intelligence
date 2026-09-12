"""Grounded explanation and operator assistant routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..schemas.explanation import AssistantRequest, AssistantResponse, ExplanationRequest, ExplanationResponse
from ..services.genai import genai_service

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/explain", response_model=ExplanationResponse)
def explain(request: ExplanationRequest, db: Session = Depends(get_db)):
    try:
        return genai_service.explain(db, request.prediction_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/chat", response_model=AssistantResponse)
def chat(request: AssistantRequest, db: Session = Depends(get_db)):
    return genai_service.assistant(db, request.question, request.prediction_id)
