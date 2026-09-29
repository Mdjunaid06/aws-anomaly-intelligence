"""Evidence-grounded GenAI explanation service with deterministic fallback."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..core.config import settings
from ..models import AnomalyPrediction, SensorHealth

LOGGER = logging.getLogger(__name__)


class GenAIService:
    """Generate grounded explanations and operator answers."""

    # Keep Groq requests comfortably below the current free-tier TPM limit.
    MAX_PREDICTIONS_FOR_ASSISTANT = 3
    MAX_HEALTH_ROWS_FOR_ASSISTANT = 3
    MAX_EVIDENCE_ITEMS = 6
    MAX_STATION_ITEMS = 5
    MAX_VALUE_CHARS = 250

    def explain(self, db: Session, prediction_id: int) -> dict:
        """Explain one stored anomaly prediction."""

        prediction = (
            db.query(AnomalyPrediction)
            .filter_by(id=prediction_id)
            .first()
        )

        if not prediction:
            raise LookupError("Prediction not found")

        health = (
            db.query(SensorHealth)
            .filter_by(station_id=prediction.station_id)
            .first()
        )

        # Full context is retained for deterministic fallback/API response.
        context = self._context(prediction, health)

        provider = (settings.genai_provider or "").strip().lower()

        # Use GenAI only when explicitly enabled and configured.
        if settings.genai_enabled and settings.genai_api_key:
            if provider in {"openai", "xai", "groq"}:
                try:
                    return self._llm_explanation(
                        prediction_id,
                        context,
                    )
                except Exception:
                    LOGGER.exception(
                        "%s explanation failed; using grounded fallback",
                        provider.upper(),
                    )

        return self._fallback(prediction_id, context)

    def assistant(
        self,
        db: Session,
        question: str,
        prediction_id: Optional[int] = None,
    ) -> dict:
        """Answer operator questions using compact backend evidence."""

        predictions = []

        if prediction_id:
            prediction = (
                db.query(AnomalyPrediction)
                .filter_by(id=prediction_id)
                .first()
            )

            if prediction:
                predictions.append(prediction)

        else:
            # Only retrieve the most recent few predictions.
            predictions = (
                db.query(AnomalyPrediction)
                .order_by(AnomalyPrediction.timestamp.desc())
                .limit(self.MAX_PREDICTIONS_FOR_ASSISTANT)
                .all()
            )

        # Only send the lowest-health stations.
        health_rows = (
            db.query(SensorHealth)
            .order_by(SensorHealth.health_score.asc())
            .limit(self.MAX_HEALTH_ROWS_FOR_ASSISTANT)
            .all()
        )

        context = {
            "question": question,
            "predictions": [
                self._assistant_prediction_context(item)
                for item in predictions
            ],
            "health": [
                {
                    "station_id": h.station_id,
                    "state": h.overall_health,
                    "score": h.health_score,
                }
                for h in health_rows
            ],
        }

        provider = (settings.genai_provider or "").strip().lower()

        # Use GenAI only when explicitly enabled and configured.
        if settings.genai_enabled and settings.genai_api_key:
            if provider in {"openai", "xai", "groq"}:
                try:
                    return self._llm_assistant(context)
                except Exception:
                    LOGGER.exception(
                        "%s assistant failed; using grounded fallback",
                        provider.upper(),
                    )

        return {
            "provider": "template",
            "fallback": True,
            "answer": self._assistant_fallback(context),
            "context": context,
        }

    def _llm_client(self):
        """Create an OpenAI-compatible client."""

        from openai import OpenAI

        client_kwargs = {
            "api_key": settings.genai_api_key,
        }

        if settings.genai_base_url:
            client_kwargs["base_url"] = settings.genai_base_url

        return OpenAI(**client_kwargs)

    def _llm_explanation(
        self,
        prediction_id: int,
        context: dict,
    ) -> dict:
        """Generate an explanation using compacted evidence."""

        client = self._llm_client()

        # Compact the context before sending it to Groq.
        llm_context = self._compact_llm_context(context)

        response = client.chat.completions.create(
            model=settings.genai_model,
            temperature=0,
            reasoning_effort="low",
            max_completion_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strictly evidence-grounded AWS weather-station "
    "anomaly explanation assistant. "

    "Use ONLY facts explicitly present in the supplied ML evidence. "

    "STRICT RULES: "
    "1. Never invent station IDs. "
    "2. Never invent sensor values or measurements. "
    "3. Never invent anomaly scores or confidence values. "
    "4. Never invent root causes. "
    "5. Never invent recommendations. "
    "6. Never invent neighboring or affected stations. "
    "7. Never recalculate or modify anomaly scores. "
    "8. Never change, reinterpret, or override the stored ML "
    "classification. "
    "9. Clearly distinguish observed facts from ML inference "
    "and recommended action. "
    "10. If information is missing, say: "
    "\"The supplied data does not contain that information.\" "
    "11. Do not assume information that is not explicitly provided. "
    "12. Keep the explanation concise, factual, and operational. "

    "The ML system is the source of truth for the anomaly result. "
    "Your role is only to explain the supplied result."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        llm_context,
                        default=str,
                    ),
                },
            ],
        )

        text = (
            response.choices[0].message.content
            or "No explanation was returned."
        )

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

    def _llm_assistant(self, context: dict) -> dict:
        """Answer operator questions using compact backend evidence."""

        client = self._llm_client()

        response = client.chat.completions.create(
            model=settings.genai_model,
            temperature=0,
            reasoning_effort="low",
            max_completion_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strictly evidence-grounded AWS weather-station "
    "operator assistant. "

    "Use ONLY facts explicitly present in the supplied context. "

    "STRICT RULES: "
    "1. Never invent station IDs. "
    "2. Never invent anomaly values or classifications. "
    "3. Never invent health states or health scores. "
    "4. Never invent root causes or recommendations. "
    "5. Never claim that other stations have or do not have issues "
    "unless all stations are explicitly included in the supplied data. "
    "6. Never infer missing information. "
    "7. If information is missing, say: "
    "\"The supplied data does not contain that information.\" "
    "8. When summarizing stations, mention ONLY station IDs that "
    "appear in the supplied context. "
    "9. Preserve the stored ML classification exactly. "
    "10. Do not change, reinterpret, or override the ML result. "
    "11. Do not perform SQL or pretend to query the database. "
    "12. Keep the answer concise, factual, and operational. "

    "When producing tables or summaries, use only the records "
    "actually supplied in the context."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        context,
                        default=str,
                    ),
                },
            ],
        )

        return {
            "provider": settings.genai_provider or "openai",
            "fallback": False,
            "answer": (
                response.choices[0].message.content
                or "No answer was returned."
            ),
            "context": context,
        }

    @classmethod
    def _assistant_prediction_context(
        cls,
        prediction: AnomalyPrediction,
    ) -> dict:
        """
        Build a compact prediction context for the operator assistant.

        This intentionally does NOT send the complete prediction object.
        """

        return {
            "prediction_id": prediction.id,
            "station_id": prediction.station_id,
            "timestamp": prediction.timestamp,
            "anomaly": bool(prediction.is_anomaly),
            "confidence": prediction.confidence,
            "classification": prediction.classification,
            "root_cause": cls._short_text(
                prediction.root_cause
            ),
            "recommended_action": cls._short_text(
                prediction.recommended_action
            ),
            "evidence": cls._compact_evidence(
                prediction.evidence
            ),
            "affected_stations": cls._compact_list(
                prediction.affected_stations
            ),
            "supporting_stations": cls._compact_list(
                prediction.supporting_stations
            ),
            "contradicting_stations": cls._compact_list(
                prediction.contradicting_stations
            ),
        }

    @classmethod
    def _compact_llm_context(cls, context: dict) -> dict:
        """
        Reduce the size of a single-prediction explanation request.

        The database remains untouched. Only the data sent to the LLM
        is compacted.
        """

        compact = dict(context)

        compact["evidence"] = cls._compact_evidence(
            context.get("evidence")
        )

        compact["affected_stations"] = cls._compact_list(
            context.get("affected_stations")
        )

        compact["supporting_stations"] = cls._compact_list(
            context.get("supporting_stations")
        )

        compact["contradicting_stations"] = cls._compact_list(
            context.get("contradicting_stations")
        )

        sensor_health = context.get("sensor_health")

        if isinstance(sensor_health, dict):
            compact["sensor_health"] = {
                "state": sensor_health.get("state"),
                "score": sensor_health.get("score"),
                "metrics": cls._compact_evidence(
                    sensor_health.get("metrics")
                ),
            }

        compact["root_cause"] = cls._short_text(
            context.get("root_cause")
        )

        compact["recommended_action"] = cls._short_text(
            context.get("recommended_action")
        )

        return compact

    @classmethod
    def _compact_evidence(
        cls,
        evidence: Any,
    ) -> dict:
        """
        Keep only a small number of evidence fields.

        Large nested detector outputs are converted to short strings so
        they cannot consume the entire Groq request.
        """

        if not isinstance(evidence, dict):
            return {}

        compact = {}

        for key, value in list(
            evidence.items()
        )[: cls.MAX_EVIDENCE_ITEMS]:

            if value in (None, {}, []):
                continue

            compact[str(key)] = cls._short_text(value)

        return compact

    @classmethod
    def _compact_list(
        cls,
        values: Any,
    ) -> list:
        """Limit station/list evidence."""

        if not isinstance(values, (list, tuple)):
            return []

        return [
            cls._short_text(value)
            for value in values[: cls.MAX_STATION_ITEMS]
        ]

    @classmethod
    def _short_text(
        cls,
        value: Any,
    ) -> str:
        """Convert arbitrary evidence to a bounded string."""

        if value is None:
            return ""

        if isinstance(value, (dict, list, tuple)):
            try:
                text = json.dumps(
                    value,
                    default=str,
                )
            except Exception:
                text = str(value)
        else:
            text = str(value)

        if len(text) > cls.MAX_VALUE_CHARS:
            return text[: cls.MAX_VALUE_CHARS] + "..."

        return text

    @staticmethod
    def _context(
        prediction: AnomalyPrediction,
        health: Optional[SensorHealth],
    ) -> dict:
        """Build the complete internal prediction context."""

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
            "contradicting_stations": (
                prediction.contradicting_stations or []
            ),
            "sensor_health": (
                {
                    "state": health.overall_health,
                    "score": health.health_score,
                    "metrics": health.health_metrics,
                }
                if health
                else None
            ),
        }

    @staticmethod
    def _signals(context: dict) -> list[str]:
        """Convert evidence into readable signal strings."""

        evidence = context.get("evidence") or {}

        return [
            f"{key}={value}"
            for key, value in evidence.items()
            if value not in (None, {}, [])
        ]

    def _fallback(
        self,
        prediction_id: int,
        context: dict,
    ) -> dict:
        """Deterministic explanation when GenAI is unavailable."""

        signals = self._signals(context)

        signal_text = (
            "; ".join(signals[:5])
            or "no populated detector evidence"
        )

        summary = (
            f"{context['station_id']} was classified as "
            f"{context['classification']} with confidence "
            f"{context['confidence']:.3f}."
        )

        explanation = (
            f"Observed fact: the stored ML result for "
            f"{context['station_id']} at {context['timestamp']} "
            f"has anomaly={context['anomaly']}. "
            f"ML inference: "
            f"{context['root_cause'] or 'no root cause was recorded'}. "
            f"Contributing evidence: {signal_text}. "
            f"Recommendation: "
            f"{context['recommended_action'] or 'no recommendation was recorded'}."
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
        """Deterministic operator response when GenAI is unavailable."""

        if (
            context.get("question")
            and not context.get("predictions")
            and not context.get("health")
        ):
            return (
                "No matching stored anomaly or "
                "sensor-health data is available."
            )

        if context.get("predictions"):
            latest = context["predictions"][0]

            return (
                f"Latest stored result: station "
                f"{latest['station_id']} is classified as "
                f"{latest['classification']} with confidence "
                f"{latest['confidence']:.3f}; root cause: "
                f"{latest.get('root_cause') or 'not recorded'}."
            )

        if context.get("health"):
            item = context["health"][0]

            return (
                f"The station needing the most attention in the stored "
                f"health records is {item['station_id']} with state "
                f"{item['state']} and score {item['score']}."
            )

        return "No stored backend data is available for this question."


genai_service = GenAIService()