"""Verify the versioned dataset before training."""

from __future__ import annotations

import argparse

from src.utils import load_config, load_dataset, project_path, sha256


def verify(config_path: str) -> dict[str, int | str]:
    config, project_root = load_config(config_path)
    data_config = config["data"]
    data_path = project_path(project_root, data_config["csv_path"])
    digest = sha256(data_path)
    if digest != data_config["sha256"]:
        raise ValueError(
            f"Dataset SHA-256 mismatch: expected {data_config['sha256']}, got {digest}"
        )

    frame = load_dataset(config, project_root)
    actual_shape = (len(frame), len(frame.columns))
    expected_shape = (
        data_config["expected_rows"],
        data_config["expected_columns"],
    )
    if actual_shape != expected_shape:
        raise ValueError(
            f"Dataset shape mismatch: expected {expected_shape}, got {actual_shape}"
        )
    return {"rows": actual_shape[0], "columns": actual_shape[1], "sha256": digest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    result = verify(parse_args().config)
    print(
        f"Dataset verified: {result['rows']} rows x {result['columns']} columns, "
        f"SHA-256 {result['sha256']}"
    )
