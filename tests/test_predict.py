import pandas as pd
import pytest

from src.predict import predict_telemetry
from src.train_models import train

NORMAL = {"Temperature_C": 22.0, "Power_Load_kW": 150.0, "Failed_Logins": 0, "Packet_Rate": 500.0}
ATTACK = {"Temperature_C": 38.5, "Power_Load_kW": 220.0, "Failed_Logins": 10, "Packet_Rate": 2500.0}


@pytest.fixture
def trained(cfg):
    train(cfg)
    return cfg


def test_predict_telemetry_separates_normal_from_attack(trained):
    normal, attack = predict_telemetry([NORMAL, ATTACK], trained)

    assert normal["prediction"] == "benign"
    assert attack["prediction"] == "malicious"
    assert attack["isolation_forest"]["score"] > normal["isolation_forest"]["score"]
    assert attack["random_forest"]["probability"] > normal["random_forest"]["probability"]


def test_predict_telemetry_accepts_dict_series_and_dataframe(trained):
    for record in (ATTACK, pd.Series(ATTACK), pd.DataFrame([ATTACK, NORMAL])):
        results = predict_telemetry(record, trained)
        assert results and results[0]["prediction"] == "malicious"
    assert len(predict_telemetry(pd.DataFrame([ATTACK, NORMAL]), trained)) == 2


def test_predict_telemetry_reports_top_drivers(trained):
    result = predict_telemetry(ATTACK, trained, top_k=2)[0]

    assert len(result["top_drivers"]) == 2
    assert {"feature", "value", "deviation_sigma", "importance"} <= set(result["top_drivers"][0])
    assert result["top_drivers"][0]["feature"] in NORMAL


def test_predict_telemetry_ignores_extra_columns(trained):
    result = predict_telemetry({**ATTACK, "label": 1, "note": "x"}, trained)[0]
    assert result["prediction"] == "malicious"


def test_predict_telemetry_strategies_follow_the_flags(trained):
    for strategy in ("isolation_forest", "random_forest", "either_flags", "both_flag"):
        result = predict_telemetry(ATTACK, trained, strategy=strategy)[0]
        assert result["strategy"] == strategy
    with pytest.raises(ValueError, match="Unknown strategy"):
        predict_telemetry(ATTACK, trained, strategy="vote")


def test_predict_telemetry_rejects_records_with_missing_features(trained):
    with pytest.raises(ValueError, match="missing"):
        predict_telemetry({"Temperature_C": 22.0}, trained)
