import numpy as np
import pandas as pd
import pytest

from src import data_preprocessing as dp


def test_to_binary_label_maps_attacks_to_one():
    labels = pd.Series(["normal", "neptune", "normal", "smurf"])
    assert dp.to_binary_label(labels).tolist() == [0, 1, 0, 1]


def test_encode_categoricals_handles_categories_only_in_test():
    train = pd.DataFrame({"protocol_type": ["tcp", "udp"], "x": [1, 2]})
    test = pd.DataFrame({"protocol_type": ["icmp", "tcp"], "x": [3, 4]})

    enc_train, enc_test = dp.encode_categoricals(train, test, columns=["protocol_type"])

    assert enc_train["protocol_type"].dtype.kind == "i"
    # The same category must get the same code in both frames.
    assert enc_train["protocol_type"].iloc[0] == enc_test["protocol_type"].iloc[1]
    # Inputs are not mutated.
    assert train["protocol_type"].tolist() == ["tcp", "udp"]


def test_prepare_data_shapes_and_scaling(cfg):
    ds = dp.prepare_data(cfg)

    assert ds.X_train.shape[1] == ds.X_test.shape[1] == len(ds.feature_names)
    assert len(ds.X_train) == len(ds.y_train)
    assert set(np.unique(ds.y_train)) == {0, 1}
    # The scaler is fit on the training split only.
    assert np.allclose(ds.X_train.mean(axis=0), 0, atol=1e-6)
    assert ds.source == "synthetic-telemetry"


def test_prepare_data_is_deterministic(cfg):
    first, second = dp.prepare_data(cfg), dp.prepare_data(cfg)
    assert np.array_equal(first.X_test, second.X_test)


def test_falls_back_to_synthetic_when_download_fails(cfg, monkeypatch):
    def boom(_cfg):
        raise OSError("network down")

    monkeypatch.setattr(dp, "load_nsl_kdd", boom)
    cfg["data"]["source"] = "nsl-kdd"
    cfg["data"]["allow_synthetic_fallback"] = True

    assert dp.load_and_encode(cfg).source == "synthetic-telemetry"


def test_download_failure_raises_when_fallback_disabled(cfg, monkeypatch):
    def boom(_cfg):
        raise OSError("network down")

    monkeypatch.setattr(dp, "load_nsl_kdd", boom)
    cfg["data"]["source"] = "nsl-kdd"
    cfg["data"]["allow_synthetic_fallback"] = False

    with pytest.raises(OSError):
        dp.load_and_encode(cfg)
