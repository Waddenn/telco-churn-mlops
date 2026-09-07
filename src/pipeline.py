"""Build the preprocessing and classification pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    model_type: str = "logreg",
    *,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """Return a leakage-safe preprocessing and model pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )

    models: dict[str, Any] = {
        "logreg": LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if model_type not in models:
        raise ValueError(
            f"Unsupported model type {model_type!r}; choose one of {sorted(models)}"
        )

    return Pipeline(
        steps=[("preprocessor", preprocessor), ("model", models[model_type])]
    )
