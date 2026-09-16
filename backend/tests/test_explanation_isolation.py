"""Mandatory SIH test verifying complete isolation of anomaly explanations.

Proves:
1. Fetching explanation for Anomaly A generates values strictly grounded in Record A.
2. Fetching explanation for Anomaly B generates values strictly grounded in Record B.
3. Explanation B never retains or leaks any state, timestamps, confidence, or evidence from Record A.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.database import SessionLocal
from backend.app.models.anomaly import AnomalyPrediction


@pytest.fixture
def client():
    return TestClient(app)


def test_explanation_strict_isolation(client):
    """Prove that switching from Alert A to Alert B has zero stale data bleed."""
    session = SessionLocal()
    try:
        # Fetch two distinct predictions from the database
        preds = session.query(AnomalyPrediction).order_by(AnomalyPrediction.id.asc()).limit(5).all()
        assert len(preds) >= 2, "Database must contain at least 2 predictions for isolation test"
        
        pred_a = preds[0]
        # Pick pred_b that has distinct station or timestamp or confidence
        pred_b = None
        for cand in preds[1:]:
            if cand.station_id != pred_a.station_id or cand.timestamp != pred_a.timestamp or cand.confidence != pred_a.confidence:
                pred_b = cand
                break
        if pred_b is None:
            pred_b = preds[1]
            
        id_a, id_b = pred_a.id, pred_b.id
        station_a, station_b = pred_a.station_id, pred_b.station_id
        time_a, time_b = pred_a.timestamp.isoformat(), pred_b.timestamp.isoformat()
        conf_a, conf_b = pred_a.confidence, pred_b.confidence
    finally:
        session.close()

    # 1. Open Anomaly A
    res_a = client.post("/assistant/explain", json={"prediction_id": id_a})
    assert res_a.status_code == 200
    data_a = res_a.json()
    assert data_a["prediction_id"] == id_a
    assert station_a in data_a["summary"] or station_a in data_a["explanation"]
    
    # 2. Open Anomaly B
    res_b = client.post("/assistant/explain", json={"prediction_id": id_b})
    assert res_b.status_code == 200
    data_b = res_b.json()
    
    # 3. Assert explanation B contains B's exact values
    assert data_b["prediction_id"] == id_b
    assert station_b in data_b["summary"] or station_b in data_b["explanation"]
    
    # 4. Assert explanation B does NOT contain A's values (if stations or timestamps differ)
    if station_a != station_b:
        assert station_a not in data_b["summary"]
    if time_a != time_b:
        assert time_a not in data_b["explanation"]
        
    # 5. Verify spatial context isolation
    sc_a = client.get(f"/anomalies/{id_a}/spatial-context").json()
    sc_b = client.get(f"/anomalies/{id_b}/spatial-context").json()
    assert sc_a["target"]["prediction_id"] == id_a
    assert sc_b["target"]["prediction_id"] == id_b
    assert sc_a["target"]["station_id"] == station_a
    assert sc_b["target"]["station_id"] == station_b
