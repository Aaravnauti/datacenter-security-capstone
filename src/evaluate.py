"""Evaluate the saved models on the held-out test set.

Reports precision, recall, F1, accuracy and ROC-AUC for the malicious class,
not accuracy alone, because accuracy is misleading on imbalanced security data.

Writes to outputs/: metrics.json, classification_report.txt and results.png.

Run:  python -m src.evaluate
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")  # headless-safe (CI, Docker, Colab)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from .config import load_config, resolve_path, setup_logging  # noqa: E402
from .data_preprocessing import load_and_encode  # noqa: E402
from .train_models import load_artifacts  # noqa: E402

logger = logging.getLogger(__name__)

CLASS_NAMES = ["benign", "malicious"]


def compute_metrics(y_true, y_pred, score) -> dict:
    """Metrics for the malicious class. `score` is higher for more suspicious rows."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, score))
    except ValueError:  # only one class present
        auc = None
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": None if auc is None else round(auc, 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def save_plots(y_true, iso_pred, rf_pred, iso_score, rf_proba, importances, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, pred, title in (
        (axes[0, 0], iso_pred, "Isolation Forest - confusion matrix"),
        (axes[0, 1], rf_pred, "Random Forest - confusion matrix"),
    ):
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, pred, labels=[0, 1]), display_labels=CLASS_NAMES
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)

    ax = axes[1, 0]
    for score, name in ((iso_score, "Isolation Forest"), (rf_proba, "Random Forest")):
        try:
            fpr, tpr, _ = roc_curve(y_true, score)
            ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, score):.3f})")
        except ValueError:
            continue
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(loc="lower right")

    importances.nlargest(10).sort_values().plot.barh(ax=axes[1, 1], color="#2a6f97")
    axes[1, 1].set_title("Random Forest - top 10 features")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(cfg: dict) -> dict:
    """Load saved models, score the test split, write reports, return the metrics."""
    artifacts = load_artifacts(resolve_path(cfg["paths"]["artifacts_dir"]))
    iso, rf, scaler = artifacts["iso"], artifacts["rf"], artifacts["scaler"]

    raw = load_and_encode(cfg)
    X_test = scaler.transform(raw.X_test)
    y_test = raw.y_test.to_numpy()

    iso_score = -iso.score_samples(X_test)           # higher = more anomalous
    iso_pred = (iso.predict(X_test) == -1).astype(int)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)

    metrics = {
        "source": raw.source,
        "n_test": int(len(y_test)),
        "isolation_forest": compute_metrics(y_test, iso_pred, iso_score),
        "random_forest": compute_metrics(y_test, rf_pred, rf_proba),
    }

    outputs_dir = resolve_path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = "\n".join(
        f"=== {name} ===\n"
        + classification_report(y_test, pred, target_names=CLASS_NAMES, zero_division=0)
        for name, pred in (("Isolation Forest", iso_pred), ("Random Forest", rf_pred))
    )
    (outputs_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)

    importances = pd.Series(rf.feature_importances_, index=artifacts["metadata"]["feature_names"])
    save_plots(
        y_test, iso_pred, rf_pred, iso_score, rf_proba, importances, outputs_dir / "results.png"
    )
    logger.info("Saved metrics and plots to %s", outputs_dir)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    args = parser.parse_args(argv)
    setup_logging()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
