import numpy as np
import pytest

from src.thresholds import (
    STRATEGIES,
    evaluate_strategies,
    flag_connections,
    optimize_threshold,
    strategy_metrics,
)


def test_optimize_threshold_meets_precision_floor_with_full_recall():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    result = optimize_threshold(y, scores, min_precision=0.95)

    assert result["precision"] >= 0.95
    assert result["recall"] == 1.0
    assert 0.4 < result["threshold"] <= 0.6
    assert result["method"].startswith("max_recall_at_precision")


def test_optimize_threshold_prefers_higher_recall_among_eligible_thresholds():
    # One attack (0.35) scores below two benign rows (0.4, 0.5), so full recall costs precision.
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.35, 0.6, 0.7, 0.8, 0.9])

    strict = optimize_threshold(y, scores, min_precision=1.0)
    loose = optimize_threshold(y, scores, min_precision=0.7)

    assert (strict["recall"], strict["precision"]) == (0.8, 1.0)
    assert loose["recall"] == 1.0 and loose["precision"] >= 0.7
    assert loose["threshold"] < strict["threshold"]


def test_optimize_threshold_falls_back_to_fbeta_when_floor_is_unreachable():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = rng.random(200)  # uninformative scores cannot reach 99.9% precision

    result = optimize_threshold(y, scores, min_precision=0.999)

    assert result["method"].startswith("max_f")
    assert 0.0 <= result["threshold"] <= 1.0


def test_flag_connections_combines_models_with_or_and_and():
    iso_score = np.array([0.9, 0.9, 0.1, 0.1])
    rf_proba = np.array([0.9, 0.1, 0.9, 0.1])
    flags = flag_connections(iso_score, rf_proba, {"isolation_forest": 0.5, "random_forest": 0.5})

    assert flags["either_flags"].tolist() == [True, True, True, False]
    assert flags["both_flag"].tolist() == [True, False, False, False]
    assert set(flags) == set(STRATEGIES)


def test_hybrid_strategies_bracket_the_single_models():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    iso_score = y * 0.3 + rng.random(500) * 0.7
    rf_proba = y * 0.4 + rng.random(500) * 0.6
    metrics = evaluate_strategies(
        y, iso_score, rf_proba, {"isolation_forest": 0.5, "random_forest": 0.5}
    )

    for single in ("isolation_forest", "random_forest"):
        assert metrics["either_flags"]["recall"] >= metrics[single]["recall"]
        assert metrics["both_flag"]["false_alarms"] <= metrics[single]["false_alarms"]
        assert metrics["both_flag"]["missed_attacks"] >= metrics[single]["missed_attacks"]


def test_strategy_metrics_counts_and_rates():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    flagged = np.array([1, 1, 0, 1, 0, 0, 0, 0])

    m = strategy_metrics(y, flagged)

    assert (m["true_positives"], m["missed_attacks"], m["false_alarms"]) == (2, 1, 1)
    assert m["true_negatives"] == 4
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["false_alarm_rate"] == pytest.approx(1 / 5)
