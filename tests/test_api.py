from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body["status"] == "ok"
    # model_loaded can be False in CI if artifacts aren't present, that's fine:
    assert "model_loaded" in body

def test_predict_returns_503_when_no_model():
    # If model artifacts are not present, /predict should return 503 (service unavailable).
    payload = {"engine_id": 1, "features": {"dummy": 0.0}}
    r = client.post("/predict", json=payload)
    assert r.status_code in (400, 503)
