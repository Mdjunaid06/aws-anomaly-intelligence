"""Evidence-grounded explanation service with deterministic fallback."""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.orm import Session

from ..core.config import settings
from ..models import AnomalyPrediction, SensorHealth

LOGGER = logging.getLogger(__name__)


class GenAIService:
    def explain(self, db: Session, prediction_id: int) -> dict:
        prediction = db.query(AnomalyPrediction).filter_by(id=prediction_id).first()
        if not prediction:
            raise LookupError("Prediction not found")
        health = db.query(SensorHealth).filter_by(station_id=prediction.station_id).first()
        context = self._context(prediction, health)
        if settings.genai_enabled and settings.genai_api_key:
            try:
                return self._openai_explanation(prediction_id, context)
            except Exception:
                LOGGER.exception("OpenAI explanation failed; using grounded fallback")
        return self._fallback(prediction_id, context)

    def assistant(self, db: Session, question: str, prediction_id: Optional[int] = None) -> dict:
        predictions = []
        if prediction_id:
            prediction = db.query(AnomalyPrediction).filter_by(id=prediction_id).first()
            if prediction:
                predictions.append(prediction)
        else:
            predictions = db.query(AnomalyPrediction).order_by(AnomalyPrediction.timestamp.desc()).limit(10).all()
        health_rows = db.query(SensorHealth).order_by(SensorHealth.health_score.asc()).limit(10).all()
        context = {
            "question": question,
            "predictions": [self._context(item, None) for item in predictions],
            "health": [
                {"station_id": h.station_id, "state": h.overall_health, "score": h.health_score}
                for h in health_rows
            ],
        }
        if settings.genai_enabled and settings.genai_api_key:
            try:
                return self._openai_assistant(context)
            except Exception:
                LOGGER.exception("OpenAI assistant failed; using grounded fallback")
        return {
            "provider": "template",
            "fallback": True,
            "answer": self._assistant_fallback(context),
            "context": context,
        }

    def _openai_client(self):
        from openai import OpenAI
        return OpenAI(api_key=settings.genai_api_key)

    def _openai_explanation(self, prediction_id: int, context: dict) -> dict:
        client = self._openai_client()
        response = client.chat.completions.create(
            model=settings.genai_model or "gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Explain only the supplied AWS ML evidence. Do not recalculate or invent facts. Distinguish observed facts, ML inference, and recommendation."},
                {"role": "user", "content": json.dumps(context, default=str)},
            ],
        )
        text = response.choices[0].message.content or "No explanation was returned."
        return {
            "prediction_id": prediction_id,
            "provider": settings.genai_provider or "openai",
            "fallback": False,
            "summary": text.splitlines()[0][:300],
            "explanation": text,
            "contributing_signals": self._signals(context),
            "root_cause": context.get("root_cause"),
            "recommended_action": context.get("recommended_action"),
            "sensor_health": context.get("sensor_health"),
        }

    def _openai_assistant(self, context: dict) -> dict:
        client = self._openai_client()
        response = client.chat.completions.create(
            model=settings.genai_model or "gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Answer the operator using only the supplied AWS anomaly records. Say when data is unavailable. Do not invent values or perform SQL."},
                {"role": "user", "content": json.dumps(context, default=str)},
            ],
        )
        return {"provider": settings.genai_provider or "openai", "fallback": False, "answer": response.choices[0].message.content or "No answer was returned.", "context": context}

    @staticmethod
    def _context(prediction: AnomalyPrediction, health: Optional[SensorHealth]) -> dict:
        return {
            "prediction_id": prediction.id,
            "station_id": prediction.station_id,
            "timestamp": prediction.timestamp,
            "anomaly": bool(prediction.is_anomaly),
            "confidence": prediction.confidence,
            "classification": prediction.classification,
            "root_cause": prediction.root_cause,
            "recommended_action": prediction.recommended_action,
            "evidence": prediction.evidence or {},
            "affected_stations": prediction.affected_stations or [],
            "supporting_stations": prediction.supporting_stations or [],
            "contradicting_stations": prediction.contradicting_stations or [],
            "sensor_health": ({"state": health.overall_health, "score": health.health_score, "metrics": health.health_metrics} if health else None),
        }

    @staticmethod
    def _signals(context: dict) -> list[str]:
        evidence = context.get("evidence") or {}
        return [f"{key}={value}" for key, value in evidence.items() if value not in (None, {}, [])]

    def _fallback(self, prediction_id: int, context: dict) -> dict:
        signals = self._signals(context)
        signal_text = "; ".join(signals[:5]) or "no populated detector evidence"
        summary = f"{context['station_id']} was classified as {context['classification']} with confidence {context['confidence']:.3f}."
        explanation = (
            f"Observed fact: the stored ML result for {context['station_id']} at {context['timestamp']} "
            f"has anomaly={context['anomaly']}. ML inference: {context['root_cause'] or 'no root cause was recorded'}. "
            f"Contributing evidence: {signal_text}. Recommendation: {context['recommended_action'] or 'no recommendation was recorded'}."
        )
        return {
            "prediction_id": prediction_id,
            "provider": "template",
            "fallback": True,
            "summary": summary,
            "explanation": explanation,
            "contributing_signals": signals,
            "root_cause": context.get("root_cause"),
            "recommended_action": context.get("recommended_action"),
            "sensor_health": context.get("sensor_health"),
        }

    @staticmethod
    def _assistant_fallback(context: dict) -> str:
        if context.get("question") and not context.get("predictions") and not context.get("health"):
            return "No matching stored anomaly or sensor-health data is available."
        if context.get("predictions"):
            latest = context["predictions"][0]
            return f"Latest stored result: station {latest['station_id']} is classified as {latest['classification']} with confidence {latest['confidence']:.3f}; root cause: {latest.get('root_cause') or 'not recorded'}."
        if context.get("health"):
            item = context["health"][0]
            return f"The station needing the most attention in the stored health records is {item['station_id']} with state {item['state']} and score {item['score']}."
        return "No stored backend data is available for this question."


genai_service = GenAIService()
