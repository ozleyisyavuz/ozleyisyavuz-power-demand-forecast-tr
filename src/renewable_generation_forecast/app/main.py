from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Renewable Generation Forecast TR", version="0.1.0")


class PredictRequest(BaseModel):
    timestamp: datetime
    wind_speed_mps: float = Field(..., ge=0, description="Wind speed (m/s)")
    ghi_wm2: float = Field(..., ge=0, description="Global Horizontal Irradiance (W/m^2)")
    temperature_c: float = Field(..., description="Temperature (C)")


class PredictResponse(BaseModel):
    p10_mw: float
    p50_mw: float
    p90_mw: float


def _model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", "models"))


def _load(q: int):
    path = _model_dir() / f"q{q:02d}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model yok: {path}. Önce eğit: python -m renewable_generation_forecast.models.train")
    return joblib.load(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        m10 = _load(10)
        m50 = _load(50)
        m90 = _load(90)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    ts = req.timestamp
    X = pd.DataFrame(
        [
            {
                "hour": ts.hour,
                "dayofweek": ts.weekday(),
                "month": ts.month,
                "is_weekend": 1 if ts.weekday() >= 5 else 0,
                "wind_speed_mps": req.wind_speed_mps,
                "ghi_wm2": req.ghi_wm2,
                "temperature_c": req.temperature_c,
            }
        ]
    )

    p10 = float(m10.predict(X)[0])
    p50 = float(m50.predict(X)[0])
    p90 = float(m90.predict(X)[0])

    return PredictResponse(p10_mw=p10, p50_mw=p50, p90_mw=p90)
