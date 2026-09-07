from fastapi.testclient import TestClient

from src.api import app


def test_health_endpoint_reports_a_known_state():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ready", "model_missing"}
