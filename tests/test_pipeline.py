import json

import numpy as np
import pytest

from src.config import resolve_path
from src.data_preprocessing import prepare_data
from src.evaluate import evaluate
from src.explain import explain
from src.train_models import (
    ISO_FILE,
    META_FILE,
    RF_FILE,
    SCALER_FILE,
    fit_models,
    load_artifacts,
    train,
)


def test_train_saves_loadable_artifacts(cfg):
    metadata = train(cfg)
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])

    for name in (ISO_FILE, RF_FILE, SCALER_FILE, META_FILE):
        assert (artifacts_dir / name).exists()

    loaded = load_artifacts(artifacts_dir)
    assert loaded["metadata"]["feature_names"] == metadata["feature_names"]
    assert loaded["rf"].n_features_in_ == len(metadata["feature_names"])


def test_load_artifacts_fails_clearly_before_training(cfg):
    with pytest.raises(FileNotFoundError, match="src.train_models"):
        load_artifacts(resolve_path(cfg["paths"]["artifacts_dir"]))


def test_isolation_forest_scores_attacks_as_more_anomalous(cfg):
    ds = prepare_data(cfg)
    iso, _ = fit_models(ds.X_train, ds.y_train, cfg)

    score = -iso.score_samples(ds.X_test)
    y = ds.y_test.to_numpy()
    assert score[y == 1].mean() > score[y == 0].mean()


def test_isolation_forest_contamination_comes_from_training_labels(cfg):
    ds = prepare_data(cfg)
    iso, _ = fit_models(ds.X_train, ds.y_train, cfg)
    assert iso.contamination == pytest.approx(float(np.clip(ds.y_train.mean(), 0.01, 0.5)))


def test_evaluate_reports_metrics_and_writes_outputs(cfg):
    train(cfg)
    metrics = evaluate(cfg)
    outputs_dir = resolve_path(cfg["paths"]["outputs_dir"])

    for model in ("isolation_forest", "random_forest"):
        for key in ("precision", "recall", "f1", "accuracy", "roc_auc", "confusion_matrix"):
            assert key in metrics[model]
    # The synthetic attacks are well separated, so a working pipeline scores high.
    assert metrics["random_forest"]["recall"] > 0.9
    assert metrics["isolation_forest"]["recall"] > 0.9

    for name in ("metrics.json", "classification_report.txt", "results.png"):
        assert (outputs_dir / name).exists()
    assert json.loads((outputs_dir / "metrics.json").read_text())["n_test"] == metrics["n_test"]


def test_explain_writes_alert_explanations(cfg):
    pytest.importorskip("shap")
    train(cfg)

    alerts = explain(cfg)
    outputs_dir = resolve_path(cfg["paths"]["outputs_dir"])

    assert alerts is not None and not alerts.empty
    assert {"model", "true_label", "feature_1", "shap_1"} <= set(alerts.columns)
    for name in ("shap_summary_rf.png", "shap_summary_iso.png", "alert_explanations.csv"):
        assert (outputs_dir / name).exists()
