"""REST API serving the sentiment model.

Run:  uvicorn sociallistening.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .preprocess import clean_text

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/sentiment.joblib"))

app = FastAPI(title="Social Listening NLP API", version=__version__)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100, examples=[["Loving the new update, great job!"]])


class Prediction(BaseModel):
    text: str
    sentiment: str
    positive_probability: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503, detail="Model not found. Run `python -m sociallistening.train` first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists(), "version": __version__}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = get_model()
    cleaned = [clean_text(t) for t in request.texts]
    probabilities = model.predict_proba(cleaned)[:, 1]
    return PredictResponse(
        predictions=[
            Prediction(
                text=text,
                sentiment="positive" if p >= 0.5 else "negative",
                positive_probability=round(float(p), 4),
            )
            for text, p in zip(request.texts, probabilities)
        ]
    )
