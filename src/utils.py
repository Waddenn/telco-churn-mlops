"""Shared configuration, data, metric, and plotting helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load YAML and return it with the project root inferred from configs/."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config, config_path.parent.parent


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_dataset(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Load and validate the IBM Telco dataset without silently dropping rows."""
    data_cfg = config["data"]
    data_path = project_path(project_root, data_cfg["csv_path"])
    frame = pd.read_csv(data_path)

    # IBM encodes eleven missing TotalCharges values as whitespace.
    if "TotalCharges" in frame:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")

    required = {
        data_cfg["target"],
        *config["features"]["numeric"],
        *config["features"]["categorical"],
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    target = data_cfg["target"]
    if not pd.api.types.is_numeric_dtype(frame[target]):
        mapped = frame[target].map({"No": 0, "Yes": 1})
        if mapped.isna().any():
            unknown = sorted(frame.loc[mapped.isna(), target].astype(str).unique())
            raise ValueError(f"Unsupported target values: {unknown}")
        frame[target] = mapped.astype("int8")
    return frame


def feature_columns(config: dict[str, Any]) -> list[str]:
    return config["features"]["numeric"] + config["features"]["categorical"]


def create_split(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the one reproducible stratified holdout used by train and evaluate."""
    target = config["data"]["target"]
    return train_test_split(
        frame[feature_columns(config)],
        frame[target],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=frame[target],
    )


def save_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict[str, Any],
    project_root: Path,
) -> None:
    destination = project_path(project_root, config["data"]["processed_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    target = config["data"]["target"]
    x_train.assign(**{target: y_train}).to_csv(destination / "train.csv", index=False)
    x_test.assign(**{target: y_test}).to_csv(destination / "test.csv", index=False)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }


def save_evaluation_artifacts(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    x_test: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write publication-ready evaluation plots and row-level predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    matrix_path = output_dir / "confusion_matrix.png"
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, predictions),
        display_labels=["No churn", "Churn"],
    ).plot(cmap="Blues", values_format="d")
    plt.title("Holdout confusion matrix")
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=160)
    plt.close()
    generated.append(matrix_path)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    roc_path = output_dir / "roc_curve.png"
    plt.figure()
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"AUC = {roc_auc_score(y_true, probabilities):.3f}",
    )
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve — holdout set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()
    generated.append(roc_path)

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    pr_path = output_dir / "precision_recall_curve.png"
    plt.figure()
    plt.plot(
        recall,
        precision,
        label=f"AP = {average_precision_score(y_true, probabilities):.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–recall curve — holdout set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path, dpi=160)
    plt.close()
    generated.append(pr_path)

    predictions_path = output_dir / "predictions.csv"
    prediction_frame = x_test.reset_index().rename(columns={"index": "source_row"})
    prediction_frame["actual_churn"] = np.asarray(y_true)
    prediction_frame["predicted_churn"] = predictions
    prediction_frame["churn_probability"] = probabilities
    prediction_frame["is_error"] = (
        prediction_frame["actual_churn"] != prediction_frame["predicted_churn"]
    )
    prediction_frame.to_csv(predictions_path, index=False)
    generated.append(predictions_path)
    return generated


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
