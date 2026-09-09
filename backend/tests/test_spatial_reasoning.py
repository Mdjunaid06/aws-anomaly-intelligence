from __future__ import annotations

import pytest

from ml.src.evidence_fusion import DecisionState, EvidenceInput, FusionConfig, fuse_evidence
from ml.src.spatial_reasoning import SpatialConfig, StationSignal, calculate_spatial_evidence


def signal(
    station_id: str,
    baseline: float,
    current: float,
    *,
    distance: float = 10.0,
    onset: float = 0.0,
    reliability: float = 1.0,
    quality: float = 1.0,
    sequence: tuple[float | None, ...] = (),
) -> StationSignal:
    return StationSignal(
        station_id=station_id,
        baseline_value=baseline,
        current_value=current,
        distance_km=distance,
        onset_minutes=onset,
        reliability=reliability,
        data_quality=quality,
        sequence=sequence,
    )


def regional_case(neighbor_count: int = 5):
    target = signal("A", 34, 40, onset=0)
    neighbors = [
        signal("B", 33, 40.2, distance=12, onset=0),
        signal("C", 35, 39.5, distance=18, onset=5),
        signal("D", 34, 40.0, distance=25, onset=0),
        signal("E", 34, 39.7, distance=35, onset=5),
        signal("F", 34, 40.1, distance=45, onset=0),
    ][:neighbor_count]
    return target, neighbors


def test_five_of_five_coherent_change_is_strong_regional_evidence():
    target, neighbors = regional_case()
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=5)
    decision = fuse_evidence(
        EvidenceInput(temporal_score=0.75, multivariate_score=0.15, spatial=evidence),
        target_station_id="A",
    )
    assert evidence.spatial_score > 0.7
    assert decision.classification is DecisionState.LIKELY_REGIONAL_EVENT


def test_four_of_five_with_one_faulty_station_remains_regional():
    target, neighbors = regional_case()
    neighbors[-1] = signal("F", 34, 34, distance=45, onset=0)
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=5)
    decision = fuse_evidence(
        EvidenceInput(temporal_score=0.8, multivariate_score=0.1, spatial=evidence),
        target_station_id="A",
    )
    assert evidence.agreeing_stations == ("B", "C", "D", "E")
    assert decision.classification is DecisionState.LIKELY_REGIONAL_EVENT


def test_three_of_five_is_not_a_hard_majority_decision():
    target, neighbors = regional_case()
    neighbors[2] = signal("D", 34, 34, distance=25, onset=0)
    neighbors[3] = signal("E", 34, 34, distance=35, onset=0)
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=5)
    assert 0.0 < evidence.spatial_score < 0.72
    assert evidence.usable_neighbors == 5


def test_two_nearby_coherent_stations_can_be_localized_event():
    target = signal("A", 34, 40, onset=0)
    neighbors = [
        signal("B", 33, 39, distance=5, onset=0),
        signal("C", 35, 41, distance=8, onset=5),
        signal("D", 34, 34, distance=100, onset=0),
        signal("E", 34, 34, distance=120, onset=0),
    ]
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=4)
    decision = fuse_evidence(
        EvidenceInput(temporal_score=0.3, multivariate_score=0.1, spatial=evidence),
        target_station_id="A",
    )
    assert evidence.geographically_coherent
    assert decision.classification is DecisionState.LIKELY_LOCAL_EVENT


def test_two_distant_stations_are_inconclusive():
    target = signal("A", 34, 40, onset=0)
    neighbors = [
        signal("B", 33, 39, distance=180, onset=0),
        signal("C", 35, 41, distance=200, onset=5),
        signal("D", 34, 34, distance=10, onset=0),
        signal("E", 34, 34, distance=15, onset=0),
    ]
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=4)
    assert not evidence.geographically_coherent
    assert evidence.spatial_score < 0.42


def test_isolated_spike_with_normal_neighbors_is_sensor_fault():
    target = signal("A", 34, 40, onset=0)
    neighbors = [signal(str(index), 34, 34, distance=index * 5) for index in range(1, 6)]
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=5)
    decision = fuse_evidence(
        EvidenceInput(temporal_score=0.95, multivariate_score=0.9, rule_qc_score=0.9, spatial=evidence),
        target_station_id="A",
    )
    assert evidence.spatial_score == 0.0
    assert decision.classification is DecisionState.LIKELY_SENSOR_FAULT


def test_zero_neighbors_returns_insufficient_spatial_evidence():
    evidence = calculate_spatial_evidence(signal("A", 34, 35), [], total_neighbors=5)
    assert evidence.insufficient_coverage
    assert evidence.missing_neighbors == 5


def test_missing_neighbors_are_not_treated_as_disagreement():
    target = signal("A", 34, 40)
    evidence = calculate_spatial_evidence(target, [signal("B", 33, 39)], total_neighbors=5)
    assert evidence.usable_neighbors == 1
    assert evidence.missing_neighbors == 4
    assert evidence.insufficient_coverage


def test_poor_reliability_reduces_spatial_influence():
    target = signal("A", 34, 40)
    neighbors = [
        signal("B", 33, 40, reliability=1.0),
        signal("C", 35, 40, reliability=0.9),
        signal("D", 34, 34, reliability=0.2),
    ]
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=3)
    assert evidence.station_reliability == pytest.approx(0.7)
    assert evidence.spatial_score > 0.3


def test_identical_values_are_common_mode_data_fault():
    target = signal("A", 30, 40, sequence=(40, 40, 40))
    neighbors = [
        signal(station, 30, 40, sequence=(40, 40, 40))
        for station in ("B", "C", "D", "E")
    ]
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=4)
    decision = fuse_evidence(EvidenceInput(spatial=evidence), target_station_id="A")
    assert evidence.common_mode_risk == 1.0
    assert decision.classification is DecisionState.LIKELY_COMMON_MODE_DATA_FAULT


def test_flatline_and_drift_states_are_explicit():
    stuck = fuse_evidence(EvidenceInput(stuck_score=0.9), target_station_id="A")
    drift = fuse_evidence(EvidenceInput(drift_score=0.9), target_station_id="A")
    assert stuck.classification is DecisionState.LIKELY_STUCK_SENSOR
    assert drift.classification is DecisionState.LIKELY_SENSOR_DRIFT


def test_communication_gap_state_is_explicit():
    decision = fuse_evidence(EvidenceInput(communication_score=0.9), target_station_id="A")
    assert decision.classification is DecisionState.LIKELY_COMMUNICATION_FAULT


def test_intermittent_evidence_remains_inconclusive_when_signals_conflict():
    target = signal("A", 34, 40)
    neighbors = [signal("B", 34, 34), signal("C", 34, 34)]
    spatial = calculate_spatial_evidence(target, neighbors, total_neighbors=2)
    decision = fuse_evidence(
        EvidenceInput(
            temporal_score=0.7,
            multivariate_score=0.4,
            rule_qc_score=0.4,
            persistence_score=0.2,
            spatial=spatial,
        ),
        target_station_id="A",
        config=FusionConfig(anomaly_threshold=0.65),
    )
    assert decision.classification is DecisionState.INCONCLUSIVE


@pytest.mark.parametrize(
    ("target_change", "neighbor_change", "expected"),
    [
        (6.0, 5.0, DecisionState.LIKELY_SENSOR_FAULT),
        (6.0, 6.0, DecisionState.LIKELY_REGIONAL_EVENT),
    ],
)
def test_univariate_and_multivariate_context_change_classification(target_change, neighbor_change, expected):
    target = signal("A", 34, 34 + target_change)
    neighbors = [signal(str(index), 34, 34 + neighbor_change) for index in range(1, 5)]
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=4)
    multivariate_score = 0.9 if expected is DecisionState.LIKELY_SENSOR_FAULT else 0.1
    decision = fuse_evidence(
        EvidenceInput(temporal_score=0.9, multivariate_score=multivariate_score, spatial=evidence),
        target_station_id="A",
    )
    assert decision.classification is expected


def test_slight_timing_difference_preserves_regional_evidence():
    target = signal("A", 34, 40, onset=0)
    neighbors = [signal(str(index), 34, 40, onset=5) for index in range(1, 5)]
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=4)
    assert evidence.temporal_alignment_score > 0.7
    assert evidence.spatial_score > 0.7


def test_decision_exposes_structured_support_and_contradictions():
    target, neighbors = regional_case()
    evidence = calculate_spatial_evidence(target, neighbors, total_neighbors=5)
    decision = fuse_evidence(EvidenceInput(spatial=evidence), target_station_id="A")
    result = decision.as_dict()
    assert result["supporting_stations"]
    assert "confidence" in result
    assert "evidence" in result
