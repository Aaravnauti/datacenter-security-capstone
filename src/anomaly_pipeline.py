"""
AI Defense Layer - Data Center Anomaly Detection & Threat Classification
========================================================================

Trains two complementary models on network/telemetry data:

  * Isolation Forest (unsupervised) - learns the BENIGN baseline only, so it can
    flag unseen / zero-day anomalies such as spoofed sensor telemetry.
  * Random Forest   (supervised)   - classifies known attack signatures.

Evaluation uses precision / recall / F1 / ROC-AUC because security datasets are
imbalanced and plain accuracy is misleading (a model that calls everything
"benign" can still score 90%+ accuracy while catching zero attacks).

Data source:
  1. NSL-KDD benchmark from a public mirror (preferred - real, citable data).
  2. Synthetic data-center telemetry (offline fallback).

Run:
    python anomaly_pipeline.py
Outputs (confusion matrix + feature importances) are written to ../outputs/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe (works in Colab, servers, CI)
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NSL_KDD_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"

NSL_KDD_TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data():
    """Return (X_train, y_train, X_test, y_test, source). y: 0=benign, 1=malicious.

    NSL-KDD ships a separate KDDTest+ file with attack types absent from
    KDDTrain+, so train-on-Train / test-on-Test measures real generalization.
    """
    try:
        print(f"Attempting to load NSL-KDD from {NSL_KDD_URL} ...")
        train = pd.read_csv(NSL_KDD_URL, names=NSL_KDD_COLUMNS).drop(columns=["difficulty"])
        test = pd.read_csv(NSL_KDD_TEST_URL, names=NSL_KDD_COLUMNS).drop(columns=["difficulty"])
        y_train = (train["label"] != "normal").astype(int)
        y_test = (test["label"] != "normal").astype(int)
        # Fit encoders on both files so categories only seen in test don't break.
        for col in CATEGORICAL:
            enc = LabelEncoder().fit(pd.concat([train[col], test[col]]))
            train[col] = enc.transform(train[col])
            test[col] = enc.transform(test[col])
        X_train = train.drop(columns=["label"])
        X_test = test.drop(columns=["label"])
        print(f"  Loaded NSL-KDD: train {len(train):,} rows, test {len(test):,} rows, "
              f"{X_train.shape[1]} features.")
        return X_train, y_train, X_test, y_test, "NSL-KDD (train=KDDTrain+, test=KDDTest+)"
    except Exception as exc:  # offline / URL moved -> fall back
        print(f"  NSL-KDD unavailable ({exc!r}). Falling back to synthetic data.")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from generate_synthetic_data import generate
        df = generate(n_samples=2000, attack_fraction=0.1)
        y = df["label"].astype(int)
        X = df.drop(columns=["label"])
        print(f"  Generated synthetic telemetry: {len(df):,} rows, {X.shape[1]} features.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        return X_train, y_train, X_test, y_test, "synthetic-telemetry"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def run_isolation_forest(X_train_s, y_train, X_test_s, y_test) -> np.ndarray:
    print("\n=== Isolation Forest (unsupervised, benign-only training) ===")
    benign_train = X_train_s[y_train.values == 0]
    contamination = float(np.clip(y_train.mean(), 0.01, 0.5))
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(benign_train)
    # IsolationForest: -1 = anomaly -> map to 1 (malicious)
    pred = np.where(iso.predict(X_test_s) == -1, 1, 0)
    print(classification_report(y_test, pred, target_names=["benign", "malicious"]))
    return iso, pred


def run_random_forest(X_train_s, y_train, X_test_s, y_test, feature_names):
    print("\n=== Random Forest (supervised) ===")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)[:, 1]
    print(classification_report(y_test, pred, target_names=["benign", "malicious"]))
    try:
        print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    except ValueError:
        pass
    return clf, pred


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save_plots(clf, y_test, rf_pred, feature_names):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, rf_pred),
        display_labels=["benign", "malicious"],
    ).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Random Forest - Confusion Matrix")

    importances = pd.Series(clf.feature_importances_, index=feature_names)
    importances.nlargest(10).sort_values().plot.barh(ax=axes[1], color="#2a6f97")
    axes[1].set_title("Top 10 features driving detection")

    fig.tight_layout()
    out = OUTPUT_DIR / "results.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plots to {out}")


# --------------------------------------------------------------------------- #
# Explainable AI (SHAP)
# --------------------------------------------------------------------------- #
SHAP_SAMPLE = 500   # rows explained; TreeExplainer cost grows with rows x trees
TOP_K = 3           # features reported per alert


def _malicious_shap(explainer, X):
    """SHAP values pushing toward the 'malicious' class, shape (rows, features)."""
    values = np.asarray(explainer(X).values)
    if values.ndim == 3:            # classifier: (rows, features, classes)
        values = values[..., 1]
    return values


def explain_alerts(iso, clf, X_test, X_test_s, y_test, iso_pred, rf_pred, feature_names):
    """Explain why alerts fired: global SHAP plots + per-alert top features.

    Writes:
      outputs/shap_summary_rf.png   - global view of what drives the Random Forest
      outputs/shap_summary_iso.png  - global view of what drives the Isolation Forest
      outputs/alert_explanations.csv - top features for each alert in the sample
    """
    try:
        import shap
    except ImportError:
        print("\n[SHAP] 'shap' not installed - skipping explanations (pip install shap).")
        return

    print(f"\n=== Explainable AI (SHAP, {SHAP_SAMPLE} sampled test rows) ===")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test_s), size=min(SHAP_SAMPLE, len(X_test_s)), replace=False)
    Xs = X_test_s[idx]
    X_raw = X_test.iloc[idx]       # unscaled values, readable in plots and reports

    rf_sv = _malicious_shap(shap.TreeExplainer(clf), Xs)
    # Isolation Forest SHAP explains the anomaly score; higher = more normal,
    # so negate it so positive contributions push toward "anomalous".
    iso_sv = -_malicious_shap(shap.TreeExplainer(iso), Xs)

    for name, sv, title in [("rf", rf_sv, "Random Forest - what pushes toward 'malicious'"),
                            ("iso", iso_sv, "Isolation Forest - what makes traffic anomalous")]:
        plt.figure()
        shap.summary_plot(sv, X_raw, feature_names=feature_names,
                          max_display=12, show=False)
        plt.title(title)
        plt.tight_layout()
        out = OUTPUT_DIR / f"shap_summary_{name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved {out}")

    # Per-alert explanations: the "why" an analyst sees next to each alert.
    rows = []
    names = np.array(feature_names)
    for model, sv, preds in [("IsolationForest", iso_sv, iso_pred[idx]),
                             ("RandomForest", rf_sv, rf_pred[idx])]:
        for i in np.flatnonzero(preds == 1):
            top = np.argsort(sv[i])[::-1][:TOP_K]
            rows.append({
                "test_row": int(idx[i]),
                "model": model,
                "true_label": "malicious" if y_test.iloc[idx[i]] == 1 else "benign",
                **{f"feature_{k + 1}": names[j] for k, j in enumerate(top)},
                **{f"value_{k + 1}": X_raw.iloc[i, j] for k, j in enumerate(top)},
                **{f"shap_{k + 1}": round(float(sv[i, j]), 4) for k, j in enumerate(top)},
            })
    alerts = pd.DataFrame(rows)
    out = OUTPUT_DIR / "alert_explanations.csv"
    alerts.to_csv(out, index=False)
    print(f"  Saved {out}  ({len(alerts)} alert explanations)")

    if not alerts.empty:
        print("\n  Example alert explanations:")
        for _, r in alerts.groupby("model").head(2).iterrows():
            reasons = ", ".join(f"{r[f'feature_{k}']}={r[f'value_{k}']} "
                                f"({r[f'shap_{k}']:+.3f})" for k in range(1, TOP_K + 1))
            print(f"    [{r['model']}] row {r['test_row']} (truth: {r['true_label']}): {reasons}")


# --------------------------------------------------------------------------- #
def main():
    X_train, y_train, X_test, y_test, source = load_data()
    feature_names = list(X_train.columns)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    iso, iso_pred = run_isolation_forest(X_train_s, y_train, X_test_s, y_test)
    clf, rf_pred = run_random_forest(
        X_train_s, y_train, X_test_s, y_test, feature_names
    )
    save_plots(clf, y_test, rf_pred, feature_names)
    explain_alerts(iso, clf, X_test, X_test_s, y_test, iso_pred, rf_pred, feature_names)

    print(f"\nDone. Data source: {source}.")


if __name__ == "__main__":
    main()
