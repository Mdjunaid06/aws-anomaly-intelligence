from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.database import Base, get_db
from backend.app.main import app
from backend.app.models import AnomalyPrediction, Observation, SensorHealth
from backend.app.services import batch as batch_service
from backend.app.services import anomaly as anomaly_service
from backend.app.services.batch import BatchProcessingService, _enrich_observation_with_metadata
from backend.app.services.ml_engine import get_ml_engine
from ml.src.config import PipelineConfig
from ml.src.models.pipeline import score_observations
from ml.src.preprocessing import load_processed_observations


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session: Session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_backend_import_and_anomaly_serialization(client: TestClient, db_session: Session):
    db_session.add(
        AnomalyPrediction(
            observation_id=1,
            station_id="S1",
            timestamp=datetime(2025, 1, 1),
            is_anomaly=1,
            confidence=0.8,
            classification="likely_sensor_fault",
            evidence={"temporal_score": 0.9, "spatial_by_variable": {}},
            temporal_score=0.9,
            spatial_score=0.0,
            root_cause="sensor_fault",
            explanation_facts=["temporal deviation"],
        )
    )
    db_session.commit()

    response = client.get("/anomalies/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["anomaly_count"] == 1
    assert payload["items"][0]["root_cause"] == "sensor_fault"


def test_missing_observation_returns_404(client: TestClient):
    response = client.post("/batch/observation/999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Observation not found"


def test_authoritative_ml_result_and_spatial_evidence_are_persisted(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
):
    observation = Observation(
        station_id="INM00043064",
        timestamp=datetime(2024, 1, 1, 3),
        temperature_c=20.0,
        pressure_hpa=1000.0,
        relative_humidity_pct=50.0,
    )
    db_session.add(observation)
    db_session.commit()
    db_session.refresh(observation)

    authoritative = pd.DataFrame(
        [
            {
                "station_id": "INM00043064",
                "timestamp": datetime(2024, 1, 1, 3),
                "anomaly": True,
                "confidence": 0.91,
                "classification": "likely_sensor_fault",
                "root_cause": "sensor_fault",
                "recommended_action": "inspect sensor",
                "explanation_facts": ["spatial disagreement"],
                "temporal_score": 0.8,
                "multivariate_score": 0.7,
                "isolation_score": 0.6,
                "rule_qc_score": 1.0,
                "data_quality_score": 1.0,
                "persistence_score": 0.2,
                "drift_score": 0.1,
                "stuck_score": 0.0,
                "communication_score": 0.0,
                "gru_score": 0.5,
                "spatial_score": 0.9,
                "affected_stations": ["INM00043064"],
                "supporting_stations": ["INM00043069"],
                "contradicting_stations": ["INM00043071"],
                "sensor_anomaly_rate": 0.09,
                "sensor_health_score": 0.91,
                "sensor_health_state": "healthy",
                "maintenance_recommendation": "no immediate maintenance action",
                "evidence": {
                    "temporal_score": 0.8,
                    "spatial_score": 0.9,
                    "spatial_by_variable": {"temperature": {"score": 0.9}},
                    "temporal_nan": np.nan,
                },
            }
        ]
    )

    class FakeMLEngine:
        def score_observation(self, observation_df, include_spatial=True):
            return authoritative

        def get_model_version(self):
            return "test-model"

    fake_engine = FakeMLEngine()
    monkeypatch.setattr(batch_service, "get_ml_engine", lambda: fake_engine)
    monkeypatch.setattr(anomaly_service, "get_ml_engine", lambda: fake_engine)

    result = BatchProcessingService.score_and_save_observation(db_session, observation.id)
    stored = db_session.query(AnomalyPrediction).one()

    assert result["is_anomaly"] == 1
    assert stored.confidence == 0.91
    assert stored.root_cause == "sensor_fault"
    assert stored.spatial_score == 0.9
    assert stored.supporting_stations == ["INM00043069"]
    assert stored.contradicting_stations == ["INM00043071"]
    assert stored.spatial_evidence == {"temperature": {"score": 0.9}}
    assert stored.evidence["temporal_nan"] is None
    health = db_session.query(SensorHealth).one()
    assert health.station_id == observation.station_id
    assert health.health_score == 0.91
    assert health.overall_health == "healthy"
    assert health.last_observation_timestamp == observation.timestamp
    assert health.health_metrics["maintenance_recommendation"] == "no immediate maintenance action"
    health_response = client.get(f"/health/{observation.station_id}")
    assert health_response.status_code == 200
    assert health_response.json()["health_score"] == 0.91
    assert health_response.json()["overall_health"] == "healthy"
    assert observation.is_processed == 1


def test_anomaly_api_exposes_complete_evidence(client: TestClient, db_session: Session):
    prediction = AnomalyPrediction(
        observation_id=1,
        station_id="S1",
        timestamp=datetime(2025, 1, 1),
        is_anomaly=1,
        confidence=0.8,
        classification="likely_sensor_fault",
        evidence={
            "temporal_score": 0.9,
            "spatial_by_variable": {"temperature": {"spatial_score": 0.8}},
            "gru_score": 0.4,
        },
        temporal_score=0.9,
        spatial_score=0.8,
    )
    db_session.add(prediction)
    db_session.commit()

    response = client.get("/anomalies/1")

    assert response.status_code == 200
    assert response.json()["evidence"] == prediction.evidence


def test_ml_engine_has_no_backend_fusion_entry_point():
    from backend.app.services.ml_engine import MLEngine

    assert not hasattr(MLEngine, "fuse_evidence")


def test_ml_health_updates_only_for_newer_observations(db_session: Session):
    older = {
        "sensor_health_score": 0.8,
        "sensor_health_state": "watch",
        "sensor_anomaly_rate": 0.2,
        "maintenance_recommendation": "monitor recent behavior",
    }
    newer = {
        "sensor_health_score": 0.6,
        "sensor_health_state": "degraded",
        "sensor_anomaly_rate": 0.4,
        "maintenance_recommendation": "schedule calibration or engineering review",
    }

    from backend.app.services.health import HealthService

    HealthService.persist_ml_health(db_session, "S1", datetime(2025, 1, 2), newer)
    HealthService.persist_ml_health(db_session, "S1", datetime(2025, 1, 1), older)

    health = db_session.query(SensorHealth).one()
    assert health.last_observation_timestamp == datetime(2025, 1, 2)
    assert health.health_score == 0.6
    assert health.overall_health == "degraded"


def test_direct_ml_and_backend_adapter_have_matching_authoritative_results():
    observations = load_processed_observations().head(64)
    observations = _enrich_observation_with_metadata(
        observations[[
            "station_id",
            "timestamp",
            "temperature_c",
            "pressure_hpa",
            "relative_humidity_pct",
        ]]
    )
    engine = get_ml_engine()

    direct = score_observations(
        observations,
        engine.models,
        PipelineConfig(),
        include_spatial=False,
    )
    adapter = engine.score_batch(observations, include_spatial=False)

    fields = [
        "station_id",
        "timestamp",
        "temperature_c",
        "pressure_hpa",
        "relative_humidity_pct",
        "anomaly",
        "confidence",
        "classification",
        "root_cause",
        "temporal_score",
        "multivariate_score",
        "isolation_score",
        "rule_qc_score",
        "spatial_score",
        "gru_score",
    ]
    pd.testing.assert_frame_equal(
        direct[fields].reset_index(drop=True),
        adapter[fields].reset_index(drop=True),
        check_dtype=False,
    )