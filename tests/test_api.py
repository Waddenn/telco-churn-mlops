import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import api

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((PROJECT_ROOT / "example_customer.json").read_text())


@pytest.fixture(autouse=True)
def clear_model_cache():
    api.get_model.cache_clear()
    yield
    api.get_model.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def test_health_reports_ready_before_loading(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_loaded": False}


def test_predict_returns_typed_probability(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {0, 1}
    assert body["label"] in {"Yes", "No"}
    assert 0 <= body["churn_probability"] <= 1


def test_predict_accepts_missing_total_charges(client):
    customer = {**EXAMPLE, "tenure": 0, "TotalCharges": None}
    assert client.post("/predict", json=customer).status_code == 200


def test_batch_predictions_preserve_cardinality(client):
    response = client.post("/predict/batch", json=[EXAMPLE, EXAMPLE])
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("Contract", "Weekly"), ("InternetService", "Satellite")],
)
def test_invalid_categories_return_422(client, field, invalid_value):
    response = client.post("/predict", json={**EXAMPLE, field: invalid_value})
    assert response.status_code == 422


def test_extra_fields_return_422(client):
    response = client.post("/predict", json={**EXAMPLE, "customerID": "secret-id"})
    assert response.status_code == 422


def test_missing_model_returns_503(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 503
    assert "Model not found" in response.json()["detail"]
