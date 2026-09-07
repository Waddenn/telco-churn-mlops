"""Run validated batch inference from a JSON object or JSON-lines file."""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from src.utils import feature_columns, load_config, project_path


def predict(config_path: str, input_path: str) -> list[dict]:
    config, project_root = load_config(config_path)
    model = joblib.load(project_path(project_root, config["artifacts"]["model_path"]))
    input_file = project_path(project_root, input_path)
    if input_file.suffix == ".jsonl":
        rows = [
            json.loads(line) for line in input_file.read_text().splitlines() if line
        ]
    else:
        payload = json.loads(input_file.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
    frame = pd.DataFrame(rows)
    missing = sorted(set(feature_columns(config)).difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing features: {missing}")
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    probabilities = model.predict_proba(frame)[:, 1]
    predictions = model.predict(frame)
    return [
        {"prediction": int(prediction), "churn_probability": float(probability)}
        for prediction, probability in zip(predictions, probabilities, strict=True)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True, help="JSON or JSONL input file")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(predict(arguments.config, arguments.input), indent=2))
