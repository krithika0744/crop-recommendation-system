from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
# __file__ -> .../crop_recommendation/app.py
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # repo root (where static/ lives)

MODEL_PATH = BASE_DIR / "model.pkl"
CROP_INFO_PATH = BASE_DIR / "crop_info.json"
STATIC_PATH = PROJECT_ROOT / "static"

# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------
app = FastAPI(title="Crop Recommendation API", version="2.0.0")

# Static (images, etc.)
if STATIC_PATH.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

# CORS (open for now; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Load artifacts (lazy so import doesn't crash if files absent at build)
# ------------------------------------------------------------------
_model = None
_crop_info = {}


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        with MODEL_PATH.open("rb") as f:
            _model = pickle.load(f)
    return _model


def _load_crop_info():
    global _crop_info
    if not _crop_info:
        if CROP_INFO_PATH.is_file():
            with CROP_INFO_PATH.open("r", encoding="utf-8") as f:
                _crop_info = json.load(f)
        else:
            _crop_info = {}
    return _crop_info


# ------------------------------------------------------------------
# Request model (Pydantic v2)
# ------------------------------------------------------------------
class CropInput(BaseModel):
    N: float = Field(..., description="Nitrogen content in soil")
    P: float = Field(..., description="Phosphorus content in soil")
    K: float = Field(..., description="Potassium content in soil")
    temperature: float = Field(..., description="Temperature in °C")
    humidity: float = Field(..., description="Relative humidity %")
    ph: float = Field(..., ge=0, le=14, description="Soil pH (0-14)")
    rainfall: float = Field(..., description="Rainfall (mm)")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 20.8,
                "humidity": 82.0,
                "ph": 6.5,
                "rainfall": 203.0,
            }
        },
    )


# ------------------------------------------------------------------
# Health/root
# ------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Crop Recommendation API",
        "predict_endpoint": "/predict",
    }


# ------------------------------------------------------------------
# Predict endpoint
# ------------------------------------------------------------------
@app.post("/predict")
def predict(data: CropInput):
    """
    Return top 3 crop recommendations with metadata.
    """
    model = _load_model()
    crop_info = _load_crop_info()

    # scikit-learn expects 2D array
    X = [[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall,
    ]]

    try:
        probabilities: List[float] = model.predict_proba(X)[0]
        classes: List[str] = list(model.classes_)
    except Exception as exc:  # noqa: BLE001
        # fall back to raw predict if proba unavailable
        try:
            pred = model.predict(X)[0]
        except Exception:  # noqa: BLE001
            return {"error": f"Model inference failed: {exc!s}"}
        info = crop_info.get(pred, {})
        return {
            "predictions": [{
                "crop": pred,
                "match": "N/A",
                "image_url": info.get("image_url", ""),
                "expected_yield": info.get("expected_yield", "N/A"),
                "market_price": info.get("market_price", "N/A"),
                "season": info.get("best_season", "N/A"),
                "advantages": info.get("advantages", []),
                "considerations": info.get("considerations", []),
                "name_hi": info.get("name_hi", ""),
                "name_te": info.get("name_te", ""),
            }]
        }

    # top 3
    top_idx = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]

    preds = []
    for i in top_idx:
        crop_name = classes[i]
        prob = probabilities[i]
        match_percent = round(float(prob) * 100, 2)
        info = crop_info.get(crop_name, {})

        preds.append({
            "crop": crop_name,
            "match": f"{match_percent}%",
            "probability": prob,
            "image_url": info.get("image_url", ""),
            "expected_yield": info.get("expected_yield", "N/A"),
            "market_price": info.get("market_price", "N/A"),
            "season": info.get("best_season", "N/A"),
            "advantages": info.get("advantages", []),
            "considerations": info.get("considerations", []),
            "name_hi": info.get("name_hi", ""),
            "name_te": info.get("name_te", ""),
        })

    return {"predictions": preds}
