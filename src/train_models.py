"""Train the Isolation Forest and Random Forest, tune thresholds, save with joblib.

Isolation Forest: unsupervised, fit on benign traffic only, so it can flag
anomalies it has never seen (zero-day behaviour, spoofed telemetry).
Random Forest: supervised and class-weighted, classifies known attack signatures.

Alert thresholds are tuned from precision-recall curves on a validation split of
the TRAINING data (never the test set), then both models are refit on all of the
training data.

Artifacts written to artifacts/: isolation_forest.joblib, random_forest.joblib,
scaler.joblib and metadata.json (feature names, category maps, thresholds).

Run:  python -m src.train_models
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

from .config import load_config, resolve_path, setup_logging
from .data_preprocessing import prepare_data
from .thresholds import optimize_threshold

logger = logging.getLogger(__name__)

ISO_FILE = "isolation_forest.joblib"
RF_FILE = "random_forest.joblib"
SCALER_FILE = "scaler.joblib"
META_FILE = "metadata.json"


def build_isolation_forest(cfg: dict, y_train: pd.Series) -> IsolationForest:
    params = dict(cfg["models"]["isolation_forest"])
    if params.get("contamination") == "from_train":
        params["contamination"] = float(np.clip(y_train.mean(), 0.01, 0.5))
    return IsolationForest(**params)


def fit_models(
    X_train: np.ndarray, y_train: pd.Series, cfg: dict
) -> tuple[IsolationForest, RandomForestClassifier]:
    """Fit both models. The Isolation Forest only ever sees benign rows."""
    iso = build_isolation_forest(cfg, y_train)
    iso.fit(X_train[y_train.to_numpy() == 0])
    rf = RandomForestClassifier(**cfg["models"]["random_forest"])
    rf.fit(X_train, y_train)
    return iso, rf


def fit_and_tune(
    X_train: np.ndarray, y_train: pd.Series, cfg: dict
) -> tuple[IsolationForest, RandomForestClassifier, dict]:
    """Fit both models and choose their alert thresholds.

    auto:   hold out part of the training data, fit on the rest, and pick each
            threshold from the precision-recall curve on the held-out part. The
            final models are then refit on all training data.
    manual: use the fixed thresholds from the config.
    """
    th_cfg = cfg["thresholds"]
    if th_cfg["mode"] == "manual":
        iso, rf = fit_models(X_train, y_train, cfg)
        return iso, rf, {
            "mode": "manual",
            "isolation_forest": float(th_cfg["manual"]["isolation_forest"]),
            "random_forest": float(th_cfg["manual"]["random_forest"]),
        }

    seed = cfg["models"]["random_forest"].get("random_state", 42)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=th_cfg["validation_size"], stratify=y_train, random_state=seed
    )
    iso_val, rf_val = fit_models(X_fit, y_fit, cfg)
    tuning = {
        "isolation_forest": optimize_threshold(
            y_val, -iso_val.score_samples(X_val), th_cfg["min_precision"], th_cfg["beta"]
        ),
        "random_forest": optimize_threshold(
            y_val, rf_val.predict_proba(X_val)[:, 1], th_cfg["min_precision"], th_cfg["beta"]
        ),
    }
    for name, result in tuning.items():
        logger.info(
            "Tuned %s threshold = %.3f (validation precision %.3f, recall %.3f)",
            name, result["threshold"], result["precision"], result["recall"],
        )

    iso, rf = fit_models(X_train, y_train, cfg)
    return iso, rf, {
        "mode": "auto",
        "isolation_forest": tuning["isolation_forest"]["threshold"],
        "random_forest": tuning["random_forest"]["threshold"],
        "tuning": tuning,
    }


def save_artifacts(
    iso: IsolationForest,
    rf: RandomForestClassifier,
    scaler,
    metadata: dict,
    artifacts_dir: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, artifacts_dir / ISO_FILE)
    joblib.dump(rf, artifacts_dir / RF_FILE)
    joblib.dump(scaler, artifacts_dir / SCALER_FILE)
    (artifacts_dir / META_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_artifacts(artifacts_dir: Path) -> dict:
    """Return {'iso', 'rf', 'scaler', 'metadata'} or raise if training has not run."""
    if not (artifacts_dir / ISO_FILE).exists():
        raise FileNotFoundError(
            f"No trained models in {artifacts_dir}. Run `python -m src.train_models` first."
        )
    return {
        "iso": joblib.load(artifacts_dir / ISO_FILE),
        "rf": joblib.load(artifacts_dir / RF_FILE),
        "scaler": joblib.load(artifacts_dir / SCALER_FILE),
        "metadata": json.loads((artifacts_dir / META_FILE).read_text(encoding="utf-8")),
    }


def train(cfg: dict) -> dict:
    """Prepare data, fit and tune both models, save artifacts, return the metadata."""
    ds = prepare_data(cfg)
    logger.info("Training Isolation Forest (benign only) and class-weighted Random Forest...")
    iso, rf, thresholds = fit_and_tune(ds.X_train, ds.y_train, cfg)
    metadata = {
        "source": ds.source,
        "n_train": int(len(ds.y_train)),
        "n_test": int(len(ds.y_test)),
        "feature_names": ds.feature_names,
        "categories": ds.categories,
        # Mean of benign training rows in scaled space: the reference for explanations.
        "benign_mean_scaled": ds.X_train[ds.y_train.to_numpy() == 0].mean(axis=0).tolist(),
        "thresholds": thresholds,
        # Model defaults, kept so evaluation can show the before/after of tuning.
        "default_thresholds": {"isolation_forest": float(-iso.offset_), "random_forest": 0.5},
    }
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    save_artifacts(iso, rf, ds.scaler, metadata, artifacts_dir)
    logger.info("Saved models to %s", artifacts_dir)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    args = parser.parse_args(argv)
    setup_logging()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
