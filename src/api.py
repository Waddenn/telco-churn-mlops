"""FastAPI inference service backed by the exact fitted sklearn pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Telco Churn Classifier", version="1.0.0")


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))


@lru_cache(maxsize=1)
def get_model():
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


@app.get("/health")
def health() -> dict[str, str]:
    status = "ready" if model_path().exists() else "model_missing"
    return {"status": status}


@app.post("/predict")
def predict(customer: Customer) -> dict[str, float | int]:
    try:
        model = get_model()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503, detail=f"Model not found: {error}"
        ) from error
    frame = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(model.predict(frame)[0])
    return {"prediction": prediction, "churn_probability": probability}
