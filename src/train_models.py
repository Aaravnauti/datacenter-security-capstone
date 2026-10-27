"""Train the Isolation Forest and Random Forest and save them with joblib.

Isolation Forest: unsupervised, fit on benign traffic only, so it can flag
anomalies it has never seen (zero-day behaviour, spoofed telemetry).
Random Forest: supervised, classifies known attack signatures.

Artifacts written to artifacts/: isolation_forest.joblib, random_forest.joblib,
scaler.joblib and metadata.json.

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

from .config import load_config, resolve_path, setup_logging
from .data_preprocessing import prepare_data

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
    """Prepare data, fit both models, save artifacts, and return the metadata."""
    ds = prepare_data(cfg)
    logger.info("Training Isolation Forest (benign only) and Random Forest...")
    iso, rf = fit_models(ds.X_train, ds.y_train, cfg)
    metadata = {
        "source": ds.source,
        "n_train": int(len(ds.y_train)),
        "n_test": int(len(ds.y_test)),
        "feature_names": ds.feature_names,
        "iso_threshold": float(-iso.offset_),
        "iso_contamination": float(iso.contamination),
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
