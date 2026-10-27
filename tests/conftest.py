"""Shared fixtures. Tests use synthetic data, so they never touch the network."""

import pytest

from src.config import load_config


@pytest.fixture
def cfg(tmp_path):
    config = load_config()
    config["data"]["source"] = "synthetic"
    config["data"]["allow_synthetic_fallback"] = False
    config["models"]["isolation_forest"]["n_estimators"] = 25
    config["models"]["random_forest"]["n_estimators"] = 25
    config["explain"]["sample_size"] = 60
    config["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    config["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    return config
