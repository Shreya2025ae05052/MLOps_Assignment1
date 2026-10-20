"""Tests for the FastAPI app."""
import pytest
from fastapi.testclient import TestClient

from api import main

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


@pytest.fixture
def client(trained_model):
    main.model = trained_model  # use the small test model
    return TestClient(main.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence(client):
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["confidence"] <= 1


def test_predict_rejects_missing_fields(client):
    response = client.post("/predict", json={"age": 63})
    assert response.status_code == 422


def test_metrics_endpoint(client):
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
