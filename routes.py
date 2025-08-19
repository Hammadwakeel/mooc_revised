from fastapi import APIRouter, HTTPException
import numpy as np

from schemas import SinglePredictionRequest, BatchPredictionRequest, PredictionResponse
from model_loader import get_model, get_scaler, get_label_encoder
from utils import predict_from_array

router = APIRouter()

@router.get("/", tags=["health"])
def health_check():
    model = get_model()
    return {"status": "ok", "model_loaded": model is not None}

@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_single(payload: SinglePredictionRequest):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = np.array(payload.features, dtype=float).reshape(1, -1)
        results = predict_from_array(X, model, get_scaler(), get_label_encoder())
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/batch", tags=["prediction"])
def predict_batch(payload: BatchPredictionRequest):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = np.array(payload.features, dtype=float)
        results = predict_from_array(X, model, get_scaler(), get_label_encoder())
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
