"""Grounded explanation and operator assistant contracts."""

from typing import Optional

from pydantic import BaseModel, Field


class ExplanationRequest(BaseModel):
    prediction_id: int = Field(..., ge=1)


class ExplanationResponse(BaseModel):
    prediction_id: int
    provider: str
    fallback: bool
    summary: str
    explanation: str
    contributing_signals: list[str]
    root_cause: Optional[str] = None
    recommended_action: Optional[str] = None
    sensor_health: Optional[dict] = None


class AssistantRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    prediction_id: Optional[int] = Field(None, ge=1)


class AssistantResponse(BaseModel):
    provider: str
    fallback: bool
    answer: str
    context: dict
