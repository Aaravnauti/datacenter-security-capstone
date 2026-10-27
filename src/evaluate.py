"""Evaluate the saved models and the four detection strategies on the test set.

Strategies: Isolation Forest alone, Random Forest alone, "either flags" (maximizes
coverage) and "both flag" (minimizes false alarms), each at the tuned thresholds
and, for comparison, at the models' default thresholds.

Reports recall, precision, F1, accuracy, false alarms and missed attacks for the
malicious class. Accuracy alone is misleading on imbalanced security data.

Writes to outputs/: metrics.json, strategy_comparison.csv, feature_drivers.csv,
classification_report.txt, results.png and confusion_matrices.png.

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
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import load_config, resolve_path, setup_logging  # noqa: E402
from .data_preprocessing import load_and_encode  # noqa: E402
from .feature_analysis import flagged_drivers, global_importances  # noqa: E402
from .thresholds import STRATEGIES, evaluate_strategies, flag_connections  # noqa: E402
from .train_models import load_artifacts  # noqa: E402

logger = logging.getLogger(__name__)

CLASS_NAMES = ["benign", "malicious"]
LABELS = {
    "isolation_forest": "Isolation forest",
    "random_forest": "Random forest",
    "either_flags": "Either flags",
    "both_flag": "Both flag",
}


def comparison_table(strategies: dict[str, dict]) -> pd.DataFrame:
    """One row per strategy, matching the dashboard's comparison table."""
    return pd.DataFrame(
        [
            {
                "Strategy": LABELS[name],
                "Recall": m["recall"],
                "Precision": m["precision"],
                "F1": m["f1"],
                "Accuracy": m["accuracy"],
                "False alarms": m["false_alarms"],
                "Missed attacks": m["missed_attacks"],
            }
            for name, m in strategies.items()
        ]
    )


def save_results_plot(y, iso_score, rf_proba, strategies, importances, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Strategy comparison
    ax = axes[0, 0]
    table = comparison_table(strategies).set_index("Strategy")[["Recall", "Precision", "F1"]]
    table.plot.bar(ax=ax, rot=15, ylim=(0, 1.2))
    ax.set_title("Detection strategies (tuned thresholds)")
    ax.legend(loc="upper center", ncol=3)

    # Precision-recall curves with the chosen operating points
    ax = axes[0, 1]
    for score, key in ((iso_score, "isolation_forest"), (rf_proba, "random_forest")):
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, label=LABELS[key])
        point = strategies[key]
        ax.scatter([point["recall"]], [point["precision"]], zorder=3)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (dots = chosen thresholds)")
    ax.legend(loc="lower left")

    # ROC
    ax = axes[1, 0]
    for score, key in ((iso_score, "isolation_forest"), (rf_proba, "random_forest")):
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, label=f"{LABELS[key]} (AUC {roc_auc_score(y, score):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(loc="lower right")

    importances.nlargest(10).sort_values().plot.barh(ax=axes[1, 1], color="#2a6f97")
    axes[1, 1].set_title("Random Forest - top 10 features")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_matrices(y, flags, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, name in zip(axes.ravel(), STRATEGIES, strict=True):
        ConfusionMatrixDisplay(
            confusion_matrix(y, flags[name], labels=[0, 1]), display_labels=CLASS_NAMES
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(LABELS[name])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(cfg: dict) -> dict:
    """Load saved models, score the test split, write reports, return the metrics."""
    artifacts = load_artifacts(resolve_path(cfg["paths"]["artifacts_dir"]))
    iso, rf, scaler, meta = (artifacts[k] for k in ("iso", "rf", "scaler", "metadata"))
    thresholds, defaults = meta["thresholds"], meta["default_thresholds"]

    raw = load_and_encode(cfg)
    X_test = scaler.transform(raw.X_test)
    y = raw.y_test.to_numpy()

    iso_score = -iso.score_samples(X_test)       # higher = more anomalous
    rf_proba = rf.predict_proba(X_test)[:, 1]

    strategies = evaluate_strategies(y, iso_score, rf_proba, thresholds)
    baseline = evaluate_strategies(y, iso_score, rf_proba, defaults)
    flags = flag_connections(iso_score, rf_proba, thresholds)

    metrics = {
        "source": raw.source,
        "n_test": int(len(y)),
        "thresholds": {
            "mode": thresholds["mode"],
            "isolation_forest": round(thresholds["isolation_forest"], 4),
            "random_forest": round(thresholds["random_forest"], 4),
        },
        "roc_auc": {
            "isolation_forest": round(float(roc_auc_score(y, iso_score)), 4),
            "random_forest": round(float(roc_auc_score(y, rf_proba)), 4),
        },
        "strategies": strategies,
        "baseline_default_thresholds": baseline,
    }

    outputs_dir = resolve_path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    table = comparison_table(strategies)
    table.to_csv(outputs_dir / "strategy_comparison.csv", index=False)

    report = "\n".join(
        f"=== {LABELS[name]} ===\n"
        + classification_report(y, flags[name], target_names=CLASS_NAMES, zero_division=0)
        for name in STRATEGIES
    )
    (outputs_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print(
        f"\nThresholds ({thresholds['mode']}): "
        f"Isolation Forest > {thresholds['isolation_forest']:.3f}, "
        f"Random Forest >= {thresholds['random_forest']:.3f}"
    )
    print("\nStrategy comparison at tuned thresholds:")
    print(table.to_string(index=False))
    print("\nFor reference, the same strategies at default thresholds:")
    print(comparison_table(baseline).to_string(index=False))

    # Feature analysis: what the model relies on and what makes flagged traffic unusual.
    importances = global_importances(rf, meta["feature_names"])
    drivers = flagged_drivers(X_test, flags["either_flags"], importances)
    drivers.to_csv(outputs_dir / "feature_drivers.csv", index=False)
    print("\nTop features by Random Forest importance:")
    print(importances.nlargest(10).round(4).to_string())
    if not drivers.empty:
        print("\nTop drivers of flagged traffic (importance x shift from unflagged traffic):")
        print(drivers.head(8).round(3).to_string(index=False))

    save_results_plot(
        y, iso_score, rf_proba, strategies, importances, outputs_dir / "results.png"
    )
    save_confusion_matrices(y, flags, outputs_dir / "confusion_matrices.png")
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
