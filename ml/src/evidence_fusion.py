"""Configurable evidence fusion and root-cause decision states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .spatial_reasoning import SpatialEvidence


class DecisionState(StrEnum):
    NORMAL = "normal"
    ANOMALY = "anomaly"
    LIKELY_REGIONAL_EVENT = "likely_regional_event"
    LIKELY_LOCAL_EVENT = "likely_local_event"
    LIKELY_SENSOR_FAULT = "likely_sensor_fault"
    LIKELY_SENSOR_DRIFT = "likely_sensor_drift"
    LIKELY_STUCK_SENSOR = "likely_stuck_sensor"
    LIKELY_COMMUNICATION_FAULT = "likely_communication_fault"
    LIKELY_COMMON_MODE_DATA_FAULT = "likely_common_mode_data_fault"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_SPATIAL_EVIDENCE = "insufficient_spatial_evidence"


@dataclass(frozen=True)
class FusionConfig:
    """Centralized thresholds and weights for validation-time tuning."""

    temporal_weight: float = 0.20
    multivariate_weight: float = 0.18
    isolation_weight: float = 0.18
    spatial_weight: float = 0.18
    rule_qc_weight: float = 0.14
    data_quality_weight: float = 0.06
    persistence_weight: float = 0.06
    drift_weight: float = 0.10
    gru_weight: float = 0.10
    anomaly_threshold: float = 0.55
    regional_threshold: float = 0.60
    localized_threshold: float = 0.42
    common_mode_threshold: float = 0.90
    strong_fault_threshold: float = 0.68
    insufficient_spatial_threshold: float = 0.50
    regional_min_neighbors: int = 3
    localized_min_neighbors: int = 2


@dataclass(frozen=True)
class EvidenceInput:
    temporal_score: float = 0.0
    multivariate_score: float = 0.0
    isolation_score: float = 0.0
    rule_qc_score: float = 0.0
    data_quality_score: float = 1.0
    persistence_score: float = 0.0
    spatial: SpatialEvidence | None = None
    spatial_by_variable: dict[str, SpatialEvidence] | None = None
    drift_score: float = 0.0
    stuck_score: float = 0.0
    communication_score: float = 0.0
    gru_score: float | None = None


@dataclass(frozen=True)
class FusedDecision:
    anomaly: bool
    classification: DecisionState
    confidence: float
    evidence: dict[str, Any]
    affected_stations: tuple[str, ...]
    supporting_stations: tuple[str, ...]
    contradicting_stations: tuple[str, ...]
    root_cause: str
    recommended_action: str
    explanation_facts: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "anomaly": self.anomaly,
            "classification": self.classification.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "affected_stations": list(self.affected_stations),
            "supporting_stations": list(self.supporting_stations),
            "contradicting_stations": list(self.contradicting_stations),
            "root_cause": self.root_cause,
            "recommended_action": self.recommended_action,
            "explanation_facts": list(self.explanation_facts),
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _weighted_score(evidence: EvidenceInput, config: FusionConfig) -> float:
    spatial_score = evidence.spatial.spatial_score if evidence.spatial else 0.0
    weights = [
        config.temporal_weight,
        config.multivariate_weight,
        config.isolation_weight,
        config.spatial_weight,
        config.rule_qc_weight,
        config.data_quality_weight,
        config.persistence_weight,
        config.drift_weight,
    ]
    values = [
        evidence.temporal_score,
        evidence.multivariate_score,
        evidence.isolation_score,
        spatial_score,
        evidence.rule_qc_score,
        evidence.data_quality_score,
        evidence.persistence_score,
        evidence.drift_score,
    ]
    if evidence.gru_score is not None:
        weights.append(config.gru_weight)
        values.append(evidence.gru_score)
    weight_total = sum(weights)
    return _clamp(sum(weight * _clamp(value) for weight, value in zip(weights, values)) / weight_total)


def fuse_evidence(
    evidence: EvidenceInput,
    *,
    target_station_id: str,
    config: FusionConfig | None = None,
) -> FusedDecision:
    """Fuse numerical evidence into an explicit, non-binary decision state."""

    config = config or FusionConfig()
    spatial = evidence.spatial
    spatial_score = spatial.spatial_score if spatial else 0.0
    common_mode_risk = spatial.common_mode_risk if spatial else 0.0
    confidence = _weighted_score(evidence, config)
    affected = (target_station_id,)
    supporting = spatial.agreeing_stations if spatial else ()
    contradicting = spatial.contradicting_stations if spatial else ()
    facts: list[str] = []

    spatial_event_support = bool(
        spatial
        and (
            spatial_score >= config.regional_threshold
            or (
                spatial_score >= config.localized_threshold
                and spatial.geographically_coherent
            )
        )
    )
    strong_nonspatial_support = (
        evidence.temporal_score >= 0.8 and evidence.multivariate_score >= 0.5
    )
    drift_support = (
        evidence.drift_score >= config.strong_fault_threshold
        and (
            evidence.multivariate_score >= 0.35
            or evidence.isolation_score >= 0.50
        )
    )
    enough_evidence = confidence >= config.anomaly_threshold or strong_nonspatial_support or drift_support
    if not enough_evidence:
        if spatial is None or spatial.insufficient_coverage:
            classification = DecisionState.INSUFFICIENT_SPATIAL_EVIDENCE
            root_cause = "insufficient valid neighboring observations"
            action = "collect more station observations before regional attribution"
            facts.append("spatial coverage is insufficient for a strong comparison")
        elif confidence >= 0.30:
            classification = DecisionState.INCONCLUSIVE
            root_cause = "conflicting or incomplete evidence"
            action = "continue monitoring and investigate supporting evidence"
            facts.append("diagnostic evidence is present but overall anomaly support is insufficient")
        else:
            classification = DecisionState.NORMAL
            root_cause = "no significant anomaly evidence"
            action = "no immediate action"
    elif common_mode_risk >= config.common_mode_threshold:
        classification = DecisionState.LIKELY_COMMON_MODE_DATA_FAULT
        root_cause = "suspiciously duplicated or propagated station data"
        action = "inspect ingestion, source messages, and last-known-value propagation"
        facts.append("multiple stations contain suspiciously identical values or sequences")
    elif evidence.communication_score >= config.strong_fault_threshold:
        classification = DecisionState.LIKELY_COMMUNICATION_FAULT
        root_cause = "missing or interrupted station communication"
        action = "inspect station connectivity and transmission history"
        facts.append("communication-gap evidence is strong")
    elif evidence.stuck_score >= config.strong_fault_threshold:
        classification = DecisionState.LIKELY_STUCK_SENSOR
        root_cause = "flatline or stuck sensor behavior"
        action = "inspect sensor output and calibration"
        facts.append("the station has persistent unchanged observations")
    elif evidence.drift_score >= config.strong_fault_threshold:
        classification = DecisionState.LIKELY_SENSOR_DRIFT
        root_cause = "gradual sensor drift"
        action = "schedule sensor review or calibration"
        facts.append("the station shows a persistent gradual deviation")
    elif (
        spatial
        and evidence.temporal_score >= config.anomaly_threshold
        and evidence.multivariate_score >= config.anomaly_threshold
        and spatial.magnitude_similarity_score < 0.85
    ):
        classification = DecisionState.LIKELY_SENSOR_FAULT
        root_cause = "localized station or sensor fault"
        action = "inspect the station and compare recent sensor history"
        facts.append("the target change is inconsistent with neighboring magnitudes")
    elif spatial and spatial_score >= config.regional_threshold and len(supporting) >= config.regional_min_neighbors:
        classification = DecisionState.LIKELY_REGIONAL_EVENT
        root_cause = "spatially coherent meteorological change"
        action = "monitor the regional event and continue observing stations"
        facts.append("multiple reliable nearby stations changed coherently")
    elif spatial and spatial_score >= config.localized_threshold and len(supporting) >= config.localized_min_neighbors and spatial.geographically_coherent:
        classification = DecisionState.LIKELY_LOCAL_EVENT
        root_cause = "localized or sub-regional meteorological change"
        action = "monitor the localized event and verify nearby coverage"
        facts.append("a geographically coherent subset changed together")
    elif evidence.temporal_score >= 0.8 and evidence.multivariate_score >= 0.5 and spatial_score < config.localized_threshold:
        classification = DecisionState.LIKELY_SENSOR_FAULT
        root_cause = "localized station or sensor fault"
        action = "inspect the station and compare recent sensor history"
        facts.append("temporal and non-spatial evidence outweigh spatial support")
    else:
        classification = DecisionState.ANOMALY
        root_cause = "anomalous observation with unresolved cause"
        action = "review the observation and collect more context"
        facts.append("multiple evidence sources indicate unusual behavior")

    if spatial:
        facts.append(f"usable neighbors: {spatial.usable_neighbors}/{spatial.total_neighbors}")
        facts.append(f"weighted spatial evidence: {spatial.spatial_score:.3f}")
        if spatial.common_mode_risk > 0:
            facts.append(f"common-mode risk: {spatial.common_mode_risk:.3f}")

    anomaly = classification not in {
        DecisionState.NORMAL,
        DecisionState.INSUFFICIENT_SPATIAL_EVIDENCE,
        DecisionState.INCONCLUSIVE,
    }
    return FusedDecision(
        anomaly=anomaly,
        classification=classification,
        confidence=confidence,
        evidence={
            "temporal_score": evidence.temporal_score,
            "multivariate_score": evidence.multivariate_score,
            "isolation_score": evidence.isolation_score,
            "gru_score": evidence.gru_score,
            "spatial_score": spatial_score,
            "rule_qc_score": evidence.rule_qc_score,
            "data_quality_score": evidence.data_quality_score,
            "common_mode_risk": common_mode_risk,
            "station_reliability": spatial.station_reliability if spatial else 0.0,
            "persistence_score": evidence.persistence_score,
            "spatial_by_variable": {
                name: value.as_dict()
                for name, value in (evidence.spatial_by_variable or {}).items()
            },
        },
        affected_stations=affected,
        supporting_stations=supporting,
        contradicting_stations=contradicting,
        root_cause=root_cause,
        recommended_action=action,
        explanation_facts=tuple(facts),
    )
