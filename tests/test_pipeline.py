import numpy as np
import pandas as pd
import pytest

from src.pipeline import build_pipeline


def test_build_pipeline_has_expected_steps():
    pipeline = build_pipeline(["numeric"], ["category"], "logreg")
    assert list(pipeline.named_steps) == ["preprocessor", "model"]


def test_pipeline_handles_missing_and_unseen_values():
    training = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "category": ["a", "b", "a", None, "b", "a"],
        }
    )
    target = [0, 1, 0, 1, 1, 0]
    pipeline = build_pipeline(["numeric"], ["category"], "logreg").fit(training, target)
    result = pipeline.predict(pd.DataFrame({"numeric": [np.nan], "category": ["new"]}))
    assert result.shape == (1,)


def test_unsupported_model_is_explicit():
    with pytest.raises(ValueError, match="Unsupported model type"):
        build_pipeline(["numeric"], ["category"], "neural_network")
