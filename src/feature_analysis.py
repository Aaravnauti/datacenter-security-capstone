"""Lightweight root-cause analysis for flagged connections.

Complements SHAP (src/explain.py), which is more rigorous but slower. Two views:

  Global  - Random Forest feature importances, plus how far flagged traffic
            shifts from unflagged traffic on each feature (in standard deviations).
  Local   - for a single record, importance x deviation from the benign baseline,
            so a feature ranks high when the model relies on it AND the record is
            unusual on it. Cheap enough to run on every prediction.

These are heuristics for triage. Use SHAP for exact per-prediction attribution.
Categorical columns (protocol_type, service, flag) are integer-coded, so their sigma
values only indicate "an unusual category", not a direction or magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def global_importances(rf, feature_names: list[str]) -> pd.Series:
    """Random Forest feature importances, in feature (column) order.

    Keep this order: the functions below index scaled data columns by position.
    Use `.nlargest(k)` when you want the top features for display.
    """
    return pd.Series(rf.feature_importances_, index=feature_names)


def flagged_drivers(
    X_scaled: np.ndarray, flagged: np.ndarray, importances: pd.Series
) -> pd.DataFrame:
    """Which features differ most between flagged and unflagged traffic.

    `importances` must be in the same column order as `X_scaled`. `shift_sigma` is
    the mean difference in standard deviations (features are standardized), a
    standardized effect size. `driver_score` multiplies it by the model's importance
    for that feature, so both must be large to rank high.
    """
    flagged = np.asarray(flagged).astype(bool)
    if not flagged.any() or flagged.all():
        return pd.DataFrame(columns=["feature", "importance", "shift_sigma", "driver_score"])
    shift = X_scaled[flagged].mean(axis=0) - X_scaled[~flagged].mean(axis=0)
    table = pd.DataFrame(
        {
            "feature": importances.index,
            "importance": importances.to_numpy(),
            "shift_sigma": shift,
        }
    )
    table["driver_score"] = table["importance"] * table["shift_sigma"].abs()
    return table.sort_values("driver_score", ascending=False).reset_index(drop=True)


def record_drivers(
    scaled_row: np.ndarray,
    raw_row: pd.Series,
    baseline: np.ndarray,
    importances: pd.Series,
    top_k: int = 3,
) -> list[dict]:
    """Top features explaining why one record looks unusual.

    `importances` and `baseline` must be in the same column order as `scaled_row`.
    """
    deviation = scaled_row - baseline
    score = importances.to_numpy() * np.abs(deviation)
    top = np.argsort(score)[::-1][:top_k]
    return [
        {
            "feature": importances.index[j],
            "value": raw_row[importances.index[j]],
            "deviation_sigma": round(float(deviation[j]), 2),
            "importance": round(float(importances.iloc[j]), 4),
        }
        for j in top
    ]
