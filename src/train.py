"""Train, tune, track, persist, and register the churn classifier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.pipeline import build_pipeline
from src.utils import (
    create_split,
    load_config,
    load_dataset,
    project_path,
    save_split,
    sha256,
    write_json,
)


def configure_mlflow(config: dict, project_root: Path) -> None:
    mlflow_cfg = config["mlflow"]
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
    # Keep a relative SQLite URI anchored to the project, regardless of caller cwd.
    if tracking_uri.startswith("sqlite:///./") or tracking_uri == "sqlite:///mlflow.db":
        database_name = tracking_uri.removeprefix("sqlite:///").removeprefix("./")
        tracking_uri = f"sqlite:///{project_root / database_name}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", mlflow_cfg["experiment_name"])
    mlflow.set_experiment(experiment)


def train(config_path: str) -> dict:
    config, project_root = load_config(config_path)
    frame = load_dataset(config, project_root)
    x_train, x_test, y_train, y_test = create_split(frame, config)
    save_split(x_train, x_test, y_train, y_test, config, project_root)

    pipeline = build_pipeline(
        config["features"]["numeric"],
        config["features"]["categorical"],
        config["model"]["type"],
        random_state=config["model"]["random_state"],
        max_iter=config["model"].get("max_iter", 1000),
    )
    cv = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        shuffle=config["cv"]["shuffle"],
        random_state=config["data"]["random_state"],
    )
    search = GridSearchCV(
        pipeline,
        param_grid=config["model"]["params"],
        cv=cv,
        scoring=config["cv"]["scoring"],
        n_jobs=config["cv"]["n_jobs"],
        refit=True,
        return_train_score=True,
    )

    configure_mlflow(config, project_root)
    mlflow.sklearn.autolog(log_models=False, silent=True, max_tuning_runs=None)
    artifact_cfg = config["artifacts"]
    model_path = project_path(project_root, artifact_cfg["model_path"])
    metadata_path = project_path(project_root, artifact_cfg["metadata_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"{config['model']['type']}-grid-search") as run:
        mlflow.log_params(
            {
                "dataset_rows": len(frame),
                "training_rows": len(x_train),
                "holdout_rows": len(x_test),
                "numeric_feature_count": len(config["features"]["numeric"]),
                "categorical_feature_count": len(config["features"]["categorical"]),
                "data_sha256": sha256(
                    project_path(project_root, config["data"]["csv_path"])
                ),
            }
        )

        baseline = clone(pipeline).fit(x_train, y_train)
        baseline_auc = roc_auc_score(y_test, baseline.predict_proba(x_test)[:, 1])
        mlflow.log_metric("baseline_holdout_roc_auc", baseline_auc)

        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        holdout_auc = roc_auc_score(y_test, best_model.predict_proba(x_test)[:, 1])
        mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
        mlflow.log_metric("tuned_holdout_roc_auc", float(holdout_auc))
        mlflow.log_params(
            {f"selected_{key}": value for key, value in search.best_params_.items()}
        )

        joblib.dump(best_model, model_path)
        mlflow.log_artifact(
            str(Path(config_path).resolve()), artifact_path="configuration"
        )
        mlflow.log_artifact(str(model_path), artifact_path="joblib")
        eda_report = project_root / "reports" / "eda_report.md"
        if eda_report.exists():
            mlflow.log_artifacts(str(eda_report.parent), artifact_path="analysis")

        signature = mlflow.models.infer_signature(x_train, best_model.predict(x_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            signature=signature,
            input_example=x_train.head(3),
            registered_model_name=(
                config["mlflow"]["registered_model_name"]
                if config["mlflow"].get("register_model")
                else None
            ),
        )
        if (
            config["mlflow"].get("register_model")
            and model_info.registered_model_version
        ):
            MlflowClient().set_registered_model_alias(
                config["mlflow"]["registered_model_name"],
                config["mlflow"].get("model_alias", "staging"),
                model_info.registered_model_version,
            )

        metadata = {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "registered_model_version": model_info.registered_model_version,
            "best_cv_roc_auc": float(search.best_score_),
            "baseline_holdout_roc_auc": float(baseline_auc),
            "tuned_holdout_roc_auc": float(holdout_auc),
            "best_params": search.best_params_,
            "model_path": str(model_path.relative_to(project_root)),
        }
        write_json(metadata, metadata_path)
        mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args().config)
    print(f"Run ID: {result['run_id']}")
    print(f"Best CV ROC-AUC: {result['best_cv_roc_auc']:.4f}")
    print(f"Tuned holdout ROC-AUC: {result['tuned_holdout_roc_auc']:.4f}")
    print(f"Best parameters: {result['best_params']}")
