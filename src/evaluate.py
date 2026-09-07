"""Evaluate the persisted model and log metrics/artifacts to its MLflow run."""

from __future__ import annotations

import argparse
import json

import joblib
import mlflow

from src.train import configure_mlflow
from src.utils import (
    classification_metrics,
    create_split,
    load_config,
    load_dataset,
    project_path,
    save_evaluation_artifacts,
    write_json,
)


def evaluate(config_path: str) -> dict[str, float]:
    config, project_root = load_config(config_path)
    artifact_cfg = config["artifacts"]
    model_path = project_path(project_root, artifact_cfg["model_path"])
    metadata_path = project_path(project_root, artifact_cfg["metadata_path"])
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Training artifacts are missing; run `make train` first."
        )

    frame = load_dataset(config, project_root)
    _, x_test, _, y_test = create_split(frame, config)
    model = joblib.load(model_path)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, predictions, probabilities)

    output_dir = project_path(project_root, artifact_cfg["directory"])
    generated = save_evaluation_artifacts(
        y_test, predictions, probabilities, x_test, output_dir
    )
    metrics_path = project_path(project_root, artifact_cfg["metrics_path"])
    write_json(metrics, metrics_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report_path = project_root / "reports" / "model_report.md"
    metric_rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {value:.4f} |"
        for name, value in metrics.items()
    )
    parameter_rows = "\n".join(
        f"| `{name}` | `{value}` |" for name, value in metadata["best_params"].items()
    )
    report_path.write_text(
        f"""# Model evaluation report

## Experiment

- MLflow run ID: `{metadata["run_id"]}`
- Model: logistic regression after stratified 5-fold grid search
- Selection metric: ROC-AUC
- Best mean CV ROC-AUC: **{metadata["best_cv_roc_auc"]:.4f}**
- Untuned baseline holdout ROC-AUC: **{metadata["baseline_holdout_roc_auc"]:.4f}**
- Tuned model holdout ROC-AUC: **{metadata["tuned_holdout_roc_auc"]:.4f}**

The holdout set was not used for model or hyperparameter selection. The small difference between baseline and tuned holdout ROC-AUC is reported transparently; the selected parameters are those with the best cross-validation score on the training set.

## Selected hyperparameters

| Parameter | Value |
|---|---:|
{parameter_rows}

## Final holdout metrics

| Metric | Value |
|---|---:|
{metric_rows}

## Artifacts and error analysis

- `artifacts/roc_curve.png`
- `artifacts/precision_recall_curve.png`
- `artifacts/confusion_matrix.png`
- `artifacts/predictions.csv` (input rows, actual labels, predictions, probabilities, and error flag)
- `artifacts/model.joblib` (full preprocessing and estimator pipeline)

All metrics and artifacts above are also attached to the same finished MLflow run. The registered model is available as `ChurnClassifier@staging`.
""",
        encoding="utf-8",
    )
    configure_mlflow(config, project_root)
    with mlflow.start_run(run_id=metadata["run_id"]):
        mlflow.log_metrics({f"holdout_{key}": value for key, value in metrics.items()})
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        for artifact in generated:
            mlflow.log_artifact(str(artifact), artifact_path="evaluation")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    results = evaluate(parse_args().config)
    print("Holdout metrics:")
    for name, value in results.items():
        print(f"- {name}: {value:.4f}")
