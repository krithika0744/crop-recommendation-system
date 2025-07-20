from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_PATH = BASE_DIR / "model.pkl"
CROP_INFO_PATH = BASE_DIR / "crop_info.json"
STATIC_PATH = PROJECT_ROOT / "static"
TEMPLATES_PATH = PROJECT_ROOT / "templates"

# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------
app = FastAPI(title="Crop Recommendation API", version="2.0.0")

if STATIC_PATH.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_PATH))

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Global variables
# -------------------------------------------------------
_model = None
_crop_info = {}


@app.on_event("startup")
def load_artifacts():
    """Load model and crop info once during startup."""
    global _model, _crop_info
    if MODEL_PATH.is_file():
        with MODEL_PATH.open("rb") as f:
            _model = pickle.load(f)
    else:
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if CROP_INFO_PATH.is_file():
        with CROP_INFO_PATH.open("r", encoding="utf-8") as f:
            _crop_info = json.load(f)
    else:
        _crop_info = {}


# -------------------------------------------------------
# Pydantic Schema
# -------------------------------------------------------
class CropInput(BaseModel):
    N: float = Field(..., description="Nitrogen content in soil")
    P: float = Field(..., description="Phosphorus content in soil")
    K: float = Field(..., description="Potassium content in soil")
    temperature: float = Field(..., description="Temperature (°C)")
    humidity: float = Field(..., description="Humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
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


# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/results", response_class=HTMLResponse)
def results(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})


@app.get("/results.html")
def redirect_results():
    """Redirect /results.html to /results (fix 404 issue)."""
    return RedirectResponse(url="/results")


@app.get("/health")
def health():
    return {"status": "ok", "message": "Crop Recommendation API"}


@app.post("/predict")
def predict(data: CropInput):
    if not _model:
        return {"error": "Model not loaded"}
    X = [[
        data.N, data.P, data.K,
        data.temperature, data.humidity,
        data.ph, data.rainfall
    ]]

    try:
        probabilities = _model.predict_proba(X)[0]
        classes = _model.classes_
    except Exception:
        pred = _model.predict(X)[0]
        info = _crop_info.get(pred, {})
        return {"predictions": [format_prediction(pred, 0.0, info)]}

    top_idx = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]
    predictions = [
        format_prediction(classes[i], probabilities[i], _crop_info.get(classes[i], {}))
        for i in top_idx
    ]
    return {"predictions": predictions}


def format_prediction(crop_name, prob, info):
    return {
        "crop": crop_name,
        "match": f"{round(prob * 100, 2)}%",
        "probability": prob,
        "image_url": info.get("image_url", ""),
        "expected_yield": info.get("expected_yield", {"en": "N/A", "hi": "N/A", "te": "N/A"}),
        "market_price": info.get("market_price", "N/A"),
        "season": info.get("best_season", {"en": "N/A", "hi": "N/A", "te": "N/A"}),
        "advantages": info.get("advantages", {"en": [], "hi": [], "te": []}),
        "considerations": info.get("considerations", {"en": [], "hi": [], "te": []}),
        "name_hi": info.get("name_hi", ""),
        "name_te": info.get("name_te", "")
    }
