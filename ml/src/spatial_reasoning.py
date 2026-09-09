"""Weighted spatial evidence for multi-station anomaly reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from collections import Counter
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SpatialConfig:
    """Tunable spatial thresholds and decay parameters."""

    max_neighbor_distance_km: float = 250.0
    distance_scale_km: float = 100.0
    temporal_tolerance_minutes: float = 15.0
    duration_tolerance_minutes: float = 30.0
    magnitude_scale: float = 5.0
    regional_score_threshold: float = 0.72
    localized_score_threshold: float = 0.42
    minimum_regional_neighbors: int = 3
    minimum_localized_neighbors: int = 2
    insufficient_coverage_ratio: float = 0.5
    common_mode_threshold: float = 0.9


@dataclass(frozen=True)
class StationSignal:
    """A station's comparable change and reliability context."""

    station_id: str
    baseline_value: float | None
    current_value: float | None
    distance_km: float | None = None
    onset_minutes: float | None = None
    duration_minutes: float | None = None
    reliability: float = 1.0
    data_quality: float = 1.0
    elevation_m: float | None = None
    sequence: tuple[float | None, ...] = field(default_factory=tuple)

    @property
    def change(self) -> float | None:
        if self.baseline_value is None or self.current_value is None:
            return None
        return self.current_value - self.baseline_value


@dataclass(frozen=True)
class SpatialEvidence:
    """Continuous spatial evidence and its supporting diagnostics."""

    spatial_score: float
    agreement_score: float
    coverage_score: float
    temporal_alignment_score: float
    elevation_context_score: float
    magnitude_similarity_score: float
    direction_similarity_score: float
    station_reliability: float
    common_mode_risk: float
    total_neighbors: int
    usable_neighbors: int
    missing_neighbors: int
    agreeing_stations: tuple[str, ...]
    contradicting_stations: tuple[str, ...]
    geographically_coherent: bool
    insufficient_coverage: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "spatial_score": self.spatial_score,
            "agreement_score": self.agreement_score,
            "coverage_score": self.coverage_score,
            "temporal_alignment_score": self.temporal_alignment_score,
            "elevation_context_score": self.elevation_context_score,
            "magnitude_similarity_score": self.magnitude_similarity_score,
            "direction_similarity_score": self.direction_similarity_score,
            "station_reliability": self.station_reliability,
            "common_mode_risk": self.common_mode_risk,
            "total_neighbors": self.total_neighbors,
            "usable_neighbors": self.usable_neighbors,
            "missing_neighbors": self.missing_neighbors,
            "agreeing_stations": list(self.agreeing_stations),
            "contradicting_stations": list(self.contradicting_stations),
            "geographically_coherent": self.geographically_coherent,
            "insufficient_coverage": self.insufficient_coverage,
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _direction_similarity(target_change: float, neighbor_change: float) -> float:
    if target_change == 0 or neighbor_change == 0:
        return 1.0 if target_change == neighbor_change else 0.0
    return 1.0 if target_change * neighbor_change > 0 else 0.0


def _usable(signal: StationSignal, config: SpatialConfig) -> bool:
    return (
        signal.change is not None
        and signal.data_quality > 0
        and signal.reliability > 0
        and (signal.distance_km is None or signal.distance_km <= config.max_neighbor_distance_km)
    )


def _common_mode_risk(signals: Sequence[StationSignal]) -> float:
    usable = [signal for signal in signals if signal.change is not None]
    if len(usable) < 2:
        return 0.0

    values = [(signal.current_value, signal.baseline_value) for signal in usable]
    duplicate_ratio = max(Counter(values).values()) / len(values)
    duplicate_ratio = duplicate_ratio if duplicate_ratio > 0.5 else 0.0

    sequences = [signal.sequence for signal in usable if signal.sequence]
    sequence_ratio = 0.0
    if len(sequences) >= 2:
        sequence_ratio = max(Counter(sequences).values()) / len(sequences)
        sequence_ratio = sequence_ratio if sequence_ratio > 0.5 else 0.0

    # A single identical observation can be a real regional value. Repeated
    # identical sequences are stronger evidence of propagation or duplication.
    return _clamp(sequence_ratio if sequences else 0.0)


def calculate_spatial_evidence(
    target: StationSignal,
    neighbors: Iterable[StationSignal],
    *,
    total_neighbors: int | None = None,
    config: SpatialConfig | None = None,
) -> SpatialEvidence:
    """Calculate weighted spatial agreement without hard-coded vote classes."""

    config = config or SpatialConfig()
    neighbor_list = list(neighbors)
    total = max(total_neighbors if total_neighbors is not None else len(neighbor_list), len(neighbor_list))
    usable = [signal for signal in neighbor_list if _usable(signal, config)]
    missing = max(0, total - len(usable))
    target_change = target.change

    if target_change is None or not usable:
        return SpatialEvidence(
            spatial_score=0.0,
            agreement_score=0.0,
            coverage_score=0.0,
            temporal_alignment_score=0.0,
            elevation_context_score=0.0,
            magnitude_similarity_score=0.0,
            direction_similarity_score=0.0,
            station_reliability=0.0,
            common_mode_risk=_common_mode_risk([target, *neighbor_list]),
            total_neighbors=total,
            usable_neighbors=len(usable),
            missing_neighbors=missing,
            agreeing_stations=(),
            contradicting_stations=(),
            geographically_coherent=False,
            insufficient_coverage=True,
        )

    weighted_agreement = 0.0
    weight_total = 0.0
    agreeing: list[str] = []
    contradicting: list[str] = []
    agreeing_distances: list[float] = []

    temporal_values: list[float] = []
    elevation_values: list[float] = []
    magnitude_values: list[float] = []
    direction_values: list[float] = []

    for signal in usable:
        change = signal.change
        assert change is not None
        distance_weight = exp(-(signal.distance_km or 0.0) / config.distance_scale_km)
        reliability_weight = _clamp(signal.reliability) * _clamp(signal.data_quality)
        weight = distance_weight * reliability_weight
        direction = _direction_similarity(target_change, change)
        magnitude = exp(-abs(abs(target_change) - abs(change)) / max(config.magnitude_scale, 1e-9))
        if target.onset_minutes is None or signal.onset_minutes is None:
            onset_alignment = 1.0
        else:
            onset_alignment = exp(-abs(target.onset_minutes - signal.onset_minutes) / config.temporal_tolerance_minutes)
        if target.duration_minutes is None or signal.duration_minutes is None:
            duration_alignment = 1.0
        else:
            duration_alignment = exp(
                -abs(target.duration_minutes - signal.duration_minutes) / config.duration_tolerance_minutes
            )
        temporal = onset_alignment * duration_alignment
        if target.elevation_m is None or signal.elevation_m is None:
            elevation_context = 1.0
        else:
            elevation_context = exp(abs(target.elevation_m - signal.elevation_m) / 1000.0 * -1.0)
        contribution = direction * magnitude * temporal
        contribution *= elevation_context

        weighted_agreement += weight * contribution
        weight_total += weight
        temporal_values.append(temporal)
        elevation_values.append(elevation_context)
        magnitude_values.append(magnitude)
        direction_values.append(direction)

        if direction > 0 and magnitude >= 0.5 and temporal >= 0.5:
            agreeing.append(signal.station_id)
            if signal.distance_km is not None:
                agreeing_distances.append(signal.distance_km)
        else:
            contradicting.append(signal.station_id)

    agreement_score = _clamp(weighted_agreement / weight_total) if weight_total else 0.0
    coverage_score = _clamp(len(usable) / max(total, 1))
    spatial_score = _clamp(agreement_score * coverage_score)
    geographically_coherent = len(agreeing_distances) >= 2 and max(agreeing_distances) <= config.max_neighbor_distance_km / 2
    insufficient = len(usable) == 0 or coverage_score < config.insufficient_coverage_ratio

    return SpatialEvidence(
        spatial_score=spatial_score,
        agreement_score=agreement_score,
        coverage_score=coverage_score,
        temporal_alignment_score=sum(temporal_values) / len(temporal_values),
        elevation_context_score=sum(elevation_values) / len(elevation_values),
        magnitude_similarity_score=sum(magnitude_values) / len(magnitude_values),
        direction_similarity_score=sum(direction_values) / len(direction_values),
        station_reliability=sum(signal.reliability for signal in usable) / len(usable),
        common_mode_risk=_common_mode_risk([target, *neighbor_list]),
        total_neighbors=total,
        usable_neighbors=len(usable),
        missing_neighbors=missing,
        agreeing_stations=tuple(agreeing),
        contradicting_stations=tuple(contradicting),
        geographically_coherent=geographically_coherent,
        insufficient_coverage=insufficient,
    )
