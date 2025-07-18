from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import json
import os

# Create FastAPI app
app = FastAPI()

# Mount the static folder (for serving images)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static")), name="static")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to model and crop_info.json
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
CROP_INFO_PATH = os.path.join(BASE_DIR, "crop_info.json")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load crop info
with open(CROP_INFO_PATH, "r", encoding="utf-8") as f:
    crop_info = json.load(f)

# Input schema
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Predict endpoint
@app.post("/predict")
def predict(data: CropInput):
    input_data = [[
        data.N, data.P, data.K,
        data.temperature, data.humidity,
        data.ph, data.rainfall
    ]]

    try:
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_

        # Top 3 crops
        top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]
        predictions = []

        for i in top_indices:
            crop_name = classes[i]
            match_percent = round(probabilities[i] * 100, 2)
            info = crop_info.get(crop_name, {})

            predictions.append({
                "crop": crop_name,
                "match": f"{match_percent}%",
                "image_url": info.get("image_url", ""),
                "expected_yield": info.get("expected_yield", "N/A"),
                "market_price": info.get("market_price", "N/A"),
                "season": info.get("best_season", "N/A"),
                "advantages": info.get("advantages", []),
                "considerations": info.get("considerations", []),
                "name_hi": info.get("name_hi", ""),
                "name_te": info.get("name_te", "")
            })

        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}
