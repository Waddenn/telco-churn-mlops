from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

from src.utils import create_split, load_config, load_dataset


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
