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

def test_predict_endpoint():
    dummy_input = {
        "TransactionAmt": 59.0,
        "TransactionDT": 86400.0,
        "ProductCD": "W",
        "card1": 13926.0,
        "card2": 327.0,
        "card3": 150.0,
        "card4": "discover",
        "card5": 142.0,
        "card6": "credit",
        "addr1": 315.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "DeviceInfo": "Windows"
    }
    response = client.post("/predict", json=dummy_input)
    assert response.status_code == 200
    json_data = response.json()
    assert "is_fraud" in json_data
    assert "probability" in json_data
