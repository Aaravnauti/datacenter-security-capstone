"""Inference: classify raw network/telemetry records with the saved models.

    from src.predict import predict_telemetry
    predict_telemetry({"duration": 0, "protocol_type": "tcp", ...})

Loads the Isolation Forest, Random Forest, scaler and category maps from
artifacts/, so run `python -m src.train_models` first.

CLI:
    python -m src.predict --demo
    python -m src.predict --input records.json     (a JSON object or list of objects)
    python -m src.predict --input records.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve_path, setup_logging
from .data_preprocessing import apply_category_maps
from .feature_analysis import global_importances, record_drivers
from .thresholds import STRATEGIES, flag_connections
from .train_models import load_artifacts

logger = logging.getLogger(__name__)

# Illustrative NSL-KDD-style connection records for --demo (not rows from the dataset).
_BASE_RECORD = {
    "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
    "src_bytes": 232, "dst_bytes": 8153, "land": 0, "wrong_fragment": 0, "urgent": 0,
    "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
    "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
    "is_guest_login": 0, "count": 5, "srv_count": 5, "serror_rate": 0.0,
    "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
    "dst_host_count": 30, "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.03,
    "dst_host_srv_diff_host_rate": 0.04, "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0,
}
DEMO_RECORDS = {
    "normal web request": _BASE_RECORD,
    "SYN flood against a monitoring host": {
        **_BASE_RECORD, "service": "private", "flag": "S0", "src_bytes": 0, "dst_bytes": 0,
        "logged_in": 0, "count": 123, "srv_count": 6, "serror_rate": 1.0,
        "srv_serror_rate": 1.0, "same_srv_rate": 0.05, "diff_srv_rate": 0.07,
        "dst_host_count": 255, "dst_host_srv_count": 26, "dst_host_same_srv_rate": 0.1,
        "dst_host_diff_srv_rate": 0.05, "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 1.0,
        "dst_host_srv_serror_rate": 1.0,
    },
}


def _to_frame(new_data_point) -> pd.DataFrame:
    if isinstance(new_data_point, pd.DataFrame):
        return new_data_point.reset_index(drop=True)
    if isinstance(new_data_point, pd.Series):
        return new_data_point.to_frame().T.reset_index(drop=True)
    if isinstance(new_data_point, dict):
        return pd.DataFrame([new_data_point])
    return pd.DataFrame(list(new_data_point))


def predict_telemetry(
    new_data_point,
    cfg: dict | None = None,
    strategy: str | None = None,
    artifacts: dict | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Predict whether raw records are benign or malicious.

    Args:
        new_data_point: one record (dict or Series) or many (list of dicts or DataFrame),
            with the raw feature columns, categorical values as text (e.g. "tcp").
            Extra columns such as a label are ignored.
        cfg: loaded config; defaults to config/config.yaml.
        strategy: isolation_forest, random_forest, either_flags or both_flag.
            Defaults to inference.strategy in the config.
        artifacts: pre-loaded artifacts, to avoid reloading from disk in a loop.
        top_k: number of unusual features to report per record.

    Returns:
        One dict per record with the verdict, both model scores against their
        thresholds, and the top unusual features (importance x deviation from the
        benign baseline).
    """
    cfg = cfg or load_config()
    strategy = strategy or cfg.get("inference", {}).get("strategy", "either_flags")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose one of {STRATEGIES}.")

    art = artifacts or load_artifacts(resolve_path(cfg["paths"]["artifacts_dir"]))
    iso, rf, scaler, meta = art["iso"], art["rf"], art["scaler"], art["metadata"]
    features = meta["feature_names"]

    frame = _to_frame(new_data_point)
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError(f"Record is missing {len(missing)} required feature(s): {missing}")
    raw = frame[features]

    # Report categories the model has never seen. They are encoded as -1.
    unknown = [
        [
            f"{col}={value}"
            for col, names in meta["categories"].items()
            if (value := str(row[col])) not in names
        ]
        for _, row in raw.iterrows()
    ]

    encoded = apply_category_maps(raw, meta["categories"])
    scaled = scaler.transform(encoded[features])
    iso_score = -iso.score_samples(scaled)
    rf_proba = rf.predict_proba(scaled)[:, 1]
    thresholds = meta["thresholds"]
    flags = flag_connections(iso_score, rf_proba, thresholds)

    importances = global_importances(rf, features)
    baseline = np.asarray(meta["benign_mean_scaled"])

    results = []
    for i in range(len(raw)):
        results.append(
            {
                "prediction": "malicious" if flags[strategy][i] else "benign",
                "strategy": strategy,
                "isolation_forest": {
                    "score": round(float(iso_score[i]), 4),
                    "threshold": round(thresholds["isolation_forest"], 4),
                    "flagged": bool(flags["isolation_forest"][i]),
                },
                "random_forest": {
                    "probability": round(float(rf_proba[i]), 4),
                    "threshold": round(thresholds["random_forest"], 4),
                    "flagged": bool(flags["random_forest"][i]),
                },
                "top_drivers": record_drivers(scaled[i], raw.iloc[i], baseline, importances, top_k),
                "unknown_categories": unknown[i],
            }
        )
    return results


def _print_result(name: str, result: dict) -> None:
    iso, rf = result["isolation_forest"], result["random_forest"]
    print(f"\n[{name}] -> {result['prediction'].upper()}  (strategy: {result['strategy']})")
    print(f"  Isolation Forest: score {iso['score']:.3f} vs threshold {iso['threshold']:.3f}"
          f" -> {'flag' if iso['flagged'] else 'ok'}")
    print(f"  Random Forest:    P(malicious) {rf['probability']:.3f} vs threshold "
          f"{rf['threshold']:.3f} -> {'flag' if rf['flagged'] else 'ok'}")
    print("  Most unusual features (importance x deviation from benign baseline):")
    for d in result["top_drivers"]:
        print(f"    {d['feature']} = {d['value']}  ({d['deviation_sigma']:+.1f} sigma, "
              f"importance {d['importance']:.3f})")
    if result["unknown_categories"]:
        print(f"  Unseen categories: {', '.join(result['unknown_categories'])}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    parser.add_argument("--strategy", choices=STRATEGIES, default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", help="Score two illustrative records")
    group.add_argument("--input", help="JSON (object or list) or CSV file of raw records")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)

    if args.demo:
        names, records = list(DEMO_RECORDS), list(DEMO_RECORDS.values())
    else:
        path = Path(args.input)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.DataFrame([payload] if isinstance(payload, dict) else payload)
        names, records = [f"record {i + 1}" for i in range(len(frame))], frame

    for name, result in zip(names, predict_telemetry(records, cfg, args.strategy), strict=True):
        _print_result(name, result)


if __name__ == "__main__":
    main()
