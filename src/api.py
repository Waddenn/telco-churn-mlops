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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class Customer(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]}
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class Prediction(BaseModel):
    prediction: int = Field(ge=0, le=1)
    churn_probability: float = Field(ge=0, le=1)
    label: YesNo


class BatchPrediction(BaseModel):
    predictions: list[Prediction]


def model_path() -> Path:
    path = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_model():
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


@app.get("/health")
def health() -> dict[str, str | bool]:
    status = "ready" if model_path().exists() else "model_missing"
    return {"status": status, "model_loaded": get_model.cache_info().currsize > 0}


def _predict(customers: list[Customer]) -> list[Prediction]:
    try:
        model = get_model()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503, detail=f"Model not found: {error}"
        ) from error
    frame = pd.DataFrame([customer.model_dump() for customer in customers])
    probabilities = model.predict_proba(frame)[:, 1]
    predictions = model.predict(frame)
    return [
        Prediction(
            prediction=int(prediction),
            churn_probability=float(probability),
            label="Yes" if prediction else "No",
        )
        for prediction, probability in zip(predictions, probabilities, strict=True)
    ]


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return _predict([customer])[0]


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(customers: list[Customer]) -> BatchPrediction:
    return BatchPrediction(predictions=_predict(customers))
