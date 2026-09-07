from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

from src.utils import (
    classification_metrics,
    create_split,
    load_config,
    load_dataset,
    save_evaluation_artifacts,
    sha256,
)
from src.verify_data import verify


def test_dataset_schema_and_target():
    config, project_root = load_config(CONFIG_PATH)
    frame = load_dataset(config, project_root)
    assert len(frame) == 7043
    assert set(frame[config["data"]["target"]].unique()) == {0, 1}
    assert frame[config["data"]["id_column"]].is_unique


def test_split_is_reproducible_and_stratified():
    config, project_root = load_config(CONFIG_PATH)
    frame = load_dataset(config, project_root)
    first = create_split(frame, config)
    second = create_split(frame, config)
    assert first[0].index.equals(second[0].index)
    assert abs(first[3].mean() - frame[config["data"]["target"]].mean()) < 0.01


def test_versioned_dataset_hash_and_shape():
    result = verify(str(CONFIG_PATH))
    assert result["rows"] == 7043
    assert result["columns"] == 21
    assert result["sha256"] == sha256(PROJECT_ROOT / "data" / "raw.csv")


def test_total_charges_whitespace_is_parsed_as_missing():
    config, project_root = load_config(CONFIG_PATH)
    frame = load_dataset(config, project_root)
    assert frame["TotalCharges"].isna().sum() == 11


def test_config_feature_groups_are_disjoint():
    config, _ = load_config(CONFIG_PATH)
    numeric = set(config["features"]["numeric"])
    categorical = set(config["features"]["categorical"])
    assert numeric.isdisjoint(categorical)


def test_perfect_classification_metrics():
    metrics = classification_metrics(
        pd.Series([0, 0, 1, 1]),
        predictions=pd.Series([0, 0, 1, 1]).to_numpy(),
        probabilities=pd.Series([0.1, 0.2, 0.8, 0.9]).to_numpy(),
    )
    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
        "average_precision": 1.0,
    }


def test_evaluation_artifacts_are_written(tmp_path):
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    probabilities = pd.Series([0.1, 0.4, 0.6, 0.9, 0.7, 0.3]).to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    features = pd.DataFrame({"feature": range(len(y_true))})
    outputs = save_evaluation_artifacts(
        y_true, predictions, probabilities, features, tmp_path
    )
    assert {path.name for path in outputs} == {
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "predictions.csv",
    }
    assert all(path.stat().st_size > 0 for path in outputs)


def test_missing_required_column_is_rejected(tmp_path):
    config, project_root = load_config(CONFIG_PATH)
    broken = pd.read_csv(project_root / config["data"]["csv_path"]).drop(
        columns=["Contract"]
    )
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)
    config["data"]["csv_path"] = str(path)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(config, project_root)
