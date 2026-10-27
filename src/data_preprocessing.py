"""Data loading, encoding, scaling and splitting.

The primary source is the NSL-KDD benchmark. Training and testing use the two
official files (KDDTrain+ and KDDTest+). KDDTest+ contains attack types that do
not appear in KDDTrain+, so scores measure generalization to unseen attacks.
A synthetic data-center telemetry generator is the offline fallback and the
data source for tests.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import resolve_path
from .generate_synthetic_data import generate

logger = logging.getLogger(__name__)

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
BENIGN_LABEL = "normal"


@dataclass
class RawData:
    """Encoded but unscaled data. Labels: 0 = benign, 1 = malicious."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    source: str


@dataclass
class Dataset:
    """Scaled arrays ready for modelling, plus the fitted scaler."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    scaler: StandardScaler
    source: str


def to_binary_label(labels: pd.Series) -> pd.Series:
    """Map attack names to 0 (benign) or 1 (malicious)."""
    return (labels != BENIGN_LABEL).astype(int)


def encode_categoricals(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str] = CATEGORICAL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encode columns with encoders fit on train and test together.

    Fitting on both avoids errors for categories that only appear in the test
    file. It uses category names only, never labels or numeric values.
    """
    train, test = train.copy(), test.copy()
    for col in columns:
        encoder = LabelEncoder().fit(pd.concat([train[col], test[col]]))
        train[col] = encoder.transform(train[col])
        test[col] = encoder.transform(test[col])
    return train, test


def _download(url: str, cache_dir: Path) -> Path:
    """Download a file once and reuse the cached copy afterwards."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / unquote(url.rsplit("/", 1)[-1])
    if not dest.exists():
        logger.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        partial = dest.with_suffix(dest.suffix + ".part")
        partial.write_bytes(payload)
        partial.replace(dest)
    return dest


def load_nsl_kdd(cfg: dict) -> RawData:
    data_cfg = cfg["data"]
    cache_dir = resolve_path(data_cfg["cache_dir"])
    frames = []
    for key in ("train_url", "test_url"):
        path = _download(data_cfg[key], cache_dir)
        frames.append(pd.read_csv(path, names=NSL_KDD_COLUMNS).drop(columns=["difficulty"]))
    train, test = frames
    y_train, y_test = to_binary_label(train["label"]), to_binary_label(test["label"])
    X_train, X_test = encode_categoricals(
        train.drop(columns=["label"]), test.drop(columns=["label"])
    )
    logger.info(
        "Loaded NSL-KDD: train %s rows, test %s rows, %s features.",
        f"{len(X_train):,}", f"{len(X_test):,}", X_train.shape[1],
    )
    return RawData(X_train, y_train, X_test, y_test, "NSL-KDD (train=KDDTrain+, test=KDDTest+)")


def load_synthetic(cfg: dict) -> RawData:
    syn = cfg["data"]["synthetic"]
    df = generate(syn["n_samples"], syn["attack_fraction"], syn["seed"])
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=syn["seed"]
    )
    logger.info("Generated synthetic telemetry: %s rows.", f"{len(df):,}")
    return RawData(X_train, y_train, X_test, y_test, "synthetic-telemetry")


def load_and_encode(cfg: dict) -> RawData:
    """Load the configured data source, falling back to synthetic data if allowed."""
    data_cfg = cfg["data"]
    if data_cfg["source"] == "synthetic":
        return load_synthetic(cfg)
    try:
        return load_nsl_kdd(cfg)
    except Exception as exc:
        if not data_cfg.get("allow_synthetic_fallback", False):
            raise
        logger.warning("NSL-KDD unavailable (%r). Falling back to synthetic data.", exc)
        return load_synthetic(cfg)


def prepare_data(cfg: dict) -> Dataset:
    """Load, encode and scale. The scaler is fit on the training split only."""
    raw = load_and_encode(cfg)
    scaler = StandardScaler().fit(raw.X_train)
    return Dataset(
        X_train=scaler.transform(raw.X_train),
        X_test=scaler.transform(raw.X_test),
        y_train=raw.y_train,
        y_test=raw.y_test,
        feature_names=list(raw.X_train.columns),
        scaler=scaler,
        source=raw.source,
    )
