"""
api.py — FastAPI REST API for the SMR Digital Twin surrogate models.

Endpoints:
  GET  /health           — liveness check
  GET  /config           — model metadata and input ranges
  POST /predict          — point prediction
  POST /predict/uncertainty — prediction with ±1σ intervals
  POST /predict/batch    — batch prediction (CSV-style)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import get_config, predict, predict_with_uncertainty

app = FastAPI(
    title="SMR Digital Twin API",
    description=(
        "Surrogate model API for a Steam Methane Reforming reactor. "
        "Predicts CH₄ conversion, H₂ yield, CO₂ selectivity, and carbon deposition "
        "from operating conditions."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_startup_time = time.time()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    X1: float = Field(..., description="Temperature (°C)", example=700.0)
    X2: float = Field(..., description="Contact time W/F°CH4 (g_cat·h/mol)", example=31.3)

    @field_validator("X1")
    @classmethod
    def validate_X1(cls, v):
        cfg = get_config()
        lo = cfg["inputs"]["X1"]["min"]
        hi = cfg["inputs"]["X1"]["max"]
        if not (lo <= v <= hi):
            raise ValueError(f"X1 must be between {lo} and {hi} °C, got {v}")
        return v

    @field_validator("X2")
    @classmethod
    def validate_X2(cls, v):
        cfg = get_config()
        lo = cfg["inputs"]["X2"]["min"]
        hi = cfg["inputs"]["X2"]["max"]
        if not (lo <= v <= hi):
            raise ValueError(f"X2 must be between {lo} and {hi} g_cat·h/mol, got {v}")
        return v


class BatchItem(BaseModel):
    X1: float
    X2: float


class PredictResponse(BaseModel):
    X1: float
    X2: float
    Y1: float = Field(..., description="CH4 conversion (%)")
    Y2: float = Field(..., description="H2 yield (%)")
    Y3: float = Field(..., description="CO2 selectivity (%)")
    Y4: float = Field(..., description="Carbon deposition (%)")


class PredictWithUncertaintyResponse(BaseModel):
    X1: float
    X2: float
    Y1_mean: float
    Y1_std: float
    Y2_mean: float
    Y2_std: float
    Y3_mean: float
    Y3_std: float
    Y4_mean: float
    Y4_std: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — returns model version and uptime."""
    cfg = get_config()
    return {
        "status": "ok",
        "model_type": cfg["model_type"],
        "version": cfg["version"],
        "uptime_s": round(time.time() - _startup_time, 1),
    }


@app.get("/config")
def config():
    """Return full model metadata including input/output ranges and units."""
    return get_config()


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    """
    Point prediction for a single operating condition.

    Cascade order: X1,X2 → Y1 → Y2 → Y3, Y4
    """
    try:
        result = predict(req.X1, req.X2)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"X1": req.X1, "X2": req.X2, **result}


@app.post("/predict/uncertainty", response_model=PredictWithUncertaintyResponse)
def predict_uncertainty_endpoint(req: PredictRequest):
    """
    Prediction with ±1σ confidence intervals estimated from the Random Forest ensemble.
    """
    try:
        result = predict_with_uncertainty(req.X1, req.X2)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "X1": req.X1,
        "X2": req.X2,
        "Y1_mean": result["Y1"]["mean"],
        "Y1_std":  result["Y1"]["std"],
        "Y2_mean": result["Y2"]["mean"],
        "Y2_std":  result["Y2"]["std"],
        "Y3_mean": result["Y3"]["mean"],
        "Y3_std":  result["Y3"]["std"],
        "Y4_mean": result["Y4"]["mean"],
        "Y4_std":  result["Y4"]["std"],
    }


@app.post("/predict/batch")
def predict_batch_endpoint(items: list[BatchItem]):
    """Batch predictions — returns a list of prediction dicts."""
    if len(items) > 10_000:
        raise HTTPException(status_code=400, detail="Batch size limit is 10,000 rows.")
    results = []
    for item in items:
        try:
            p = predict(item.X1, item.X2)
            results.append({"X1": item.X1, "X2": item.X2, **p})
        except ValueError as exc:
            results.append({"X1": item.X1, "X2": item.X2, "error": str(exc)})
    return results


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
