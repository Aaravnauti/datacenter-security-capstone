"""Decision-threshold tuning and hybrid detection strategies.

Both models produce a continuous score (Isolation Forest: anomaly score, higher is
more anomalous; Random Forest: probability of 'malicious'). A threshold turns the
score into an alert. The threshold sets the balance between false alarms and
missed attacks, so it is tuned from precision-recall curves on a validation split
rather than left at the default.

Four strategies are compared:
    isolation_forest  alert when the Isolation Forest flags the connection
    random_forest     alert when the Random Forest flags the connection
    either_flags      alert when EITHER model flags it (maximizes coverage)
    both_flag         alert only when BOTH models agree (minimizes false alarms)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

STRATEGIES = ("isolation_forest", "random_forest", "either_flags", "both_flag")


def optimize_threshold(
    y_true, scores, min_precision: float = 0.90, beta: float = 1.0
) -> dict:
    """Pick the alert threshold from the precision-recall curve.

    Primary rule: the lowest-recall-loss threshold whose precision is at least
    `min_precision`, which caps the false-alarm rate. Among thresholds with the same
    recall it takes the highest one, so it raises the fewest alerts. If no threshold
    reaches `min_precision`, it falls back to the best F-beta score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more entry than thresholds; drop the final (1, 0) point.
    precision, recall = precision[:-1], recall[:-1]

    eligible = np.flatnonzero(precision >= min_precision)
    if eligible.size:
        best_recall = recall[eligible].max()
        idx = eligible[recall[eligible] == best_recall][-1]
        method = f"max_recall_at_precision>={min_precision:g}"
    else:
        denom = beta**2 * precision + recall
        fbeta = np.divide(
            (1 + beta**2) * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0
        )
        idx = int(fbeta.argmax())
        method = f"max_f{beta:g}"

    return {
        "threshold": float(thresholds[idx]),
        "precision": round(float(precision[idx]), 4),
        "recall": round(float(recall[idx]), 4),
        "method": method,
    }


def flag_connections(iso_score, rf_proba, thresholds: dict) -> dict[str, np.ndarray]:
    """Boolean alert arrays for each strategy at the given thresholds."""
    iso_flag = np.asarray(iso_score) > thresholds["isolation_forest"]
    rf_flag = np.asarray(rf_proba) >= thresholds["random_forest"]
    return {
        "isolation_forest": iso_flag,
        "random_forest": rf_flag,
        "either_flags": iso_flag | rf_flag,
        "both_flag": iso_flag & rf_flag,
    }


def strategy_metrics(y_true, flagged) -> dict:
    """Detection metrics for the malicious class, with false alarms and misses as counts."""
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(flagged).astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(y), 4),
        "false_alarms": fp,
        "missed_attacks": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
    }


def evaluate_strategies(y_true, iso_score, rf_proba, thresholds: dict) -> dict[str, dict]:
    """Metrics for all four strategies at the given thresholds."""
    flags = flag_connections(iso_score, rf_proba, thresholds)
    return {name: strategy_metrics(y_true, flags[name]) for name in STRATEGIES}
