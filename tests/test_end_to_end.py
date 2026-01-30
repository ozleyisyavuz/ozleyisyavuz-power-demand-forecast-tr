from fastapi.testclient import TestClient

from renewable_generation_forecast.app.main import app
from renewable_generation_forecast.data.make_dataset import main as make_data
from renewable_generation_forecast.models.train import main as train_models


def test_end_to_end_pipeline_and_api():
    
    make_data()

    
    train_models()

  
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    payload = {
        "timestamp": "2024-06-15T12:00:00",
        "wind_speed_mps": 7.2,
        "ghi_wm2": 650.0,
        "temperature_c": 27.0,
    }
    r2 = client.post("/predict", json=payload)
    assert r2.status_code == 200
    body = r2.json()

    assert body["p10_mw"] <= body["p50_mw"] <= body["p90_mw"]
