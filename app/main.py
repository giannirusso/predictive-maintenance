from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

app = FastAPI(title="Predictive Maintenance API", version="0.1.0")

# We will classify: "failure within N cycles"
DEFAULT_HORIZON = 30


class PredictRequest(BaseModel):
    engine_id: int = Field(..., description="Engine identifier (C-MAPSS unit number)")
    features: Dict[str, float] = Field(..., description="Precomputed features for the latest window")


class PredictResponse(BaseModel):
    engine_id: int
    horizon_cycles: int
    failure_risk: float
    will_fail_within_horizon: bool
    model_version: Optional[str] = "baseline"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # TODO: replace with real model inference after training pipeline is added
    risk = 0.25
    return PredictResponse(
        engine_id=payload.engine_id,
        horizon_cycles=DEFAULT_HORIZON,
        failure_risk=risk,
        will_fail_within_horizon=risk >= 0.5,
        model_version="stub",
    )
