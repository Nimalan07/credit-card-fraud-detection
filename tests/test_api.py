from fastapi.testclient import TestClient
from src.deployment.app import app

client = TestClient(app)

def test_read_health():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert "status" in json_data
    assert "model_loaded" in json_data

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "finguard_api_requests_total" in response.text
