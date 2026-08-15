from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_and_analyze_endpoints():
    with TestClient(create_app(Settings(default_sample_seconds=120))) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        response = client.post("/analyze", json={"sample_id":"synthetic-default"})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["timestamps"]) == 120
        assert len(payload["fused_state"]) == 120


def test_rejects_mismatched_signal_lengths():
    with TestClient(create_app()) as client:
        response = client.post("/analyze", json={"signals":{"timestamps":[0,1],"heart_rate":[70],"spo2":[98,98],"respiration_rate":[15,15],"temperature":[36.8,36.8]}})
        assert response.status_code == 422
