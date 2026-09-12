from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.database import Base
from backend.app.models import AnomalyPrediction, SensorHealth
from backend.app.schemas.replay import ReplayStartRequest
from backend.app.services.genai import GenAIService
from backend.app.services.replay import ReplayService


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_replay_request_validates_speed_and_time_range():
    with pytest.raises(ValueError):
        ReplayStartRequest(speed=3)
    with pytest.raises(ValueError):
        ReplayStartRequest(start_time=datetime(2025, 1, 2), end_time=datetime(2025, 1, 1))


def test_replay_start_runs_in_event_loop():
    service = ReplayService()
    service.reset()

    async def run_start():
        return await service.start(ReplayStartRequest(
            speed=5,
            start_time=datetime(2024, 1, 1, 3, 0),
            end_time=datetime(2024, 1, 1, 4, 0),
            include_spatial=True,
        ))

    status = asyncio.run(run_start())
    assert status.running is True
    assert status.state == "running"
    service.reset()


def test_replay_config_reads_real_source():
    config = ReplayService().config()
    assert config["source"].endswith("aws_features_2024_2025.csv")
    assert config["stations"]
    assert 5.0 in config["speeds"]


def test_genai_fallback_uses_stored_evidence(db_session: Session):
    db_session.add(AnomalyPrediction(
        id=10,
        observation_id=1,
        station_id="S1",
        timestamp=datetime(2025, 1, 1),
        is_anomaly=1,
        confidence=0.91,
        classification="likely_sensor_fault",
        root_cause="localized station or sensor fault",
        recommended_action="inspect the station",
        evidence={"temporal_score": 0.9, "spatial_score": 0.2},
    ))
    db_session.add(SensorHealth(
        station_id="S1",
        overall_health="degraded",
        health_score=0.61,
        health_metrics={"maintenance_recommendation": "schedule review"},
    ))
    db_session.commit()

    result = GenAIService().explain(db_session, 10)

    assert result["fallback"] is True
    assert result["provider"] == "template"
    assert "temporal_score=0.9" in result["explanation"]
    assert result["sensor_health"]["state"] == "degraded"
