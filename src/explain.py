"""Explain alerts with SHAP (Explainable AI).

Writes to outputs/:
  shap_summary_rf.png     what drives the Random Forest toward 'malicious'
  shap_summary_iso.png    what makes traffic look anomalous to the Isolation Forest
  alert_explanations.csv  top reasons for each alert in a sample of the test set

Run:  python -m src.explain
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import load_config, resolve_path, setup_logging  # noqa: E402
from .data_preprocessing import load_and_encode  # noqa: E402
from .train_models import load_artifacts  # noqa: E402

logger = logging.getLogger(__name__)


def _malicious_shap(explainer, X: np.ndarray) -> np.ndarray:
    """SHAP values pushing toward 'malicious', shape (rows, features)."""
    values = np.asarray(explainer(X).values)
    if values.ndim == 3:  # classifier output: (rows, features, classes)
        values = values[..., 1]
    return values


def explain(cfg: dict) -> pd.DataFrame | None:
    """Write SHAP plots and per-alert explanations. Returns None if SHAP is missing."""
    try:
        import shap
    except ImportError:
        logger.warning("'shap' is not installed; skipping explanations (pip install shap).")
        return None

    explain_cfg = cfg["explain"]
    top_k = explain_cfg["top_k"]
    artifacts = load_artifacts(resolve_path(cfg["paths"]["artifacts_dir"]))
    iso, rf, scaler = artifacts["iso"], artifacts["rf"], artifacts["scaler"]
    feature_names = np.array(artifacts["metadata"]["feature_names"])

    raw = load_and_encode(cfg)
    X_test = scaler.transform(raw.X_test)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(explain_cfg["sample_size"], len(X_test)), replace=False)
    Xs, X_raw = X_test[idx], raw.X_test.iloc[idx]
    y_sample = raw.y_test.to_numpy()[idx]
    logger.info("Computing SHAP values for %d test rows...", len(idx))

    rf_sv = _malicious_shap(shap.TreeExplainer(rf), Xs)
    # The Isolation Forest score is higher for NORMAL rows, so negate it: positive
    # contributions then push toward "anomalous".
    iso_sv = -_malicious_shap(shap.TreeExplainer(iso), Xs)
    # Explain the alerts the system actually raises, at the tuned thresholds.
    thresholds = artifacts["metadata"]["thresholds"]
    iso_alert = -iso.score_samples(Xs) > thresholds["isolation_forest"]
    rf_alert = rf.predict_proba(Xs)[:, 1] >= thresholds["random_forest"]

    outputs_dir = resolve_path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for name, sv, title in (
        ("rf", rf_sv, "Random Forest - what pushes toward 'malicious'"),
        ("iso", iso_sv, "Isolation Forest - what makes traffic anomalous"),
    ):
        plt.figure()
        shap.summary_plot(sv, X_raw, feature_names=list(feature_names), max_display=12, show=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outputs_dir / f"shap_summary_{name}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    rows = []
    for model, sv, alerts in (
        ("IsolationForest", iso_sv, iso_alert),
        ("RandomForest", rf_sv, rf_alert),
    ):
        for i in np.flatnonzero(alerts):
            top = np.argsort(sv[i])[::-1][:top_k]
            rows.append(
                {
                    "test_row": int(idx[i]),
                    "model": model,
                    "true_label": "malicious" if y_sample[i] == 1 else "benign",
                    **{f"feature_{k + 1}": feature_names[j] for k, j in enumerate(top)},
                    **{f"value_{k + 1}": X_raw.iloc[i, j] for k, j in enumerate(top)},
                    **{f"shap_{k + 1}": round(float(sv[i, j]), 4) for k, j in enumerate(top)},
                }
            )
    alerts_df = pd.DataFrame(rows)
    alerts_df.to_csv(outputs_dir / "alert_explanations.csv", index=False)
    logger.info("Saved SHAP plots and %d alert explanations to %s", len(alerts_df), outputs_dir)
    return alerts_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    args = parser.parse_args(argv)
    setup_logging()
    explain(load_config(args.config))


if __name__ == "__main__":
    main()
