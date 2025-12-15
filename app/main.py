from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from pathlib import Path
import json
import joblib
import numpy as np

app = FastAPI(title="Predictive Maintenance API", version="0.2.0")

DEFAULT_HORIZON = 30

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_columns.json"

model = None
feature_cols: List[str] = []


class PredictRequest(BaseModel):
    engine_id: int = Field(..., description="Engine identifier (C-MAPSS unit number)")
    features: Dict[str, float] = Field(..., description="Precomputed features for the latest window")


class PredictResponse(BaseModel):
    engine_id: int
    horizon_cycles: int
    failure_risk: float
    will_fail_within_horizon: bool
    model_version: Optional[str] = "baseline"


@app.on_event("startup")
def load_artifacts():
    global model, feature_cols

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = None

    if FEATURE_COLS_PATH.exists():
        feature_cols = json.loads(FEATURE_COLS_PATH.read_text())
    else:
        feature_cols = []


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "num_features_expected": len(feature_cols),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None or not feature_cols:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train the model first and ensure artifacts/ are present.",
        )

    missing = [c for c in feature_cols if c not in payload.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing[:10]}{'...' if len(missing) > 10 else ''}",
        )

    x = np.array([[payload.features[c] for c in feature_cols]], dtype=float)

    prob = float(model.predict_proba(x)[:, 1][0])
    return PredictResponse(
        engine_id=payload.engine_id,
        horizon_cycles=DEFAULT_HORIZON,
        failure_risk=prob,
        will_fail_within_horizon=prob >= 0.5,
        model_version="logreg_v1",
    )
