"""
Data center cyber-physical defense - interactive dashboard.

Run:
    streamlit run app.py

Trains the Isolation Forest + Random Forest from src/anomaly_pipeline.py once
(cached) and lets you explore detection trade-offs, SHAP explanations, the
kill chain and the enterprise roadmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from anomaly_pipeline import load_data  # noqa: E402

OUTPUTS = ROOT / "outputs"

st.set_page_config(
    page_title="Data center defense",
    page_icon=":material/shield:",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Data and models (cached: trained once per server session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training both models on NSL-KDD (about 30 seconds the first time)...")
def train_models() -> dict:
    X_train, y_train, X_test, y_test, source = load_data()
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    iso = IsolationForest(
        contamination=float(np.clip(y_train.mean(), 0.01, 0.5)),
        random_state=42,
        n_jobs=-1,
    ).fit(Xtr[y_train.values == 0])

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_train)

    return {
        "source": source,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "y": y_test.to_numpy(),
        "iso_score": -iso.score_samples(Xte),   # higher = more anomalous
        "iso_default": float(-iso.offset_),     # the pipeline's default threshold
        "rf_proba": clf.predict_proba(Xte)[:, 1],
        "importances": pd.Series(clf.feature_importances_, index=X_train.columns),
    }


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / len(y),
    }


# --------------------------------------------------------------------------- #
# Header + sidebar (render before the slow work)
# --------------------------------------------------------------------------- #
st.title("Data center cyber-physical defense", icon=":material/shield:")
st.caption(
    "AI anomaly detection for the IT/OT boundary: monitoring tools, IPMI/BMC, "
    "and UPS/HVAC telemetry. ICS3120 + VAPT capstone."
)

data = train_models()

with st.sidebar:
    st.header("Detector settings", icon=":material/tune:")
    lo, hi = float(data["iso_score"].min()), float(data["iso_score"].max())
    iso_t = st.slider(
        "Isolation forest threshold",
        min_value=round(lo, 3),
        max_value=round(hi, 3),
        value=round(data["iso_default"], 3),
        step=0.005,
        help="Alert when the anomaly score is above this. Higher means fewer, "
        "more confident alerts (fewer false alarms, more missed attacks).",
    )
    rf_t = st.slider(
        "Random forest threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Alert when the predicted probability of 'malicious' reaches this.",
    )
    st.caption(
        f"Data: {data['source']}  \n"
        f"{data['n_train']:,} training rows, {data['n_test']:,} test rows, "
        f"{data['n_features']} features."
    )
    st.caption(
        "All offensive testing is done only in an isolated lab you own and are "
        "authorized to test."
    )

y = data["y"]
iso_pred = data["iso_score"] > iso_t
rf_pred = data["rf_proba"] >= rf_t
strategies = {
    "Isolation forest": iso_pred,
    "Random forest": rf_pred,
    "Either flags": iso_pred | rf_pred,
    "Both flag": iso_pred & rf_pred,
}

tab_detect, tab_explain, tab_chain, tab_road = st.tabs(
    [
        ":material/query_stats: Detection",
        ":material/psychology: Explainability",
        ":material/account_tree: Kill chain and findings",
        ":material/map: Enterprise roadmap",
    ]
)

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
with tab_detect:
    choice = st.segmented_control(
        "Detection strategy",
        list(strategies),
        default="Isolation forest",
        help="'Either flags' catches the most attacks. 'Both flag' gives the fewest "
        "false alarms.",
    ) or "Isolation forest"
    m = evaluate(y, strategies[choice].astype(int))

    with st.container(horizontal=True):
        st.metric("Attacks caught (recall)", f"{m['recall']:.1%}", border=True)
        st.metric("Alert precision", f"{m['precision']:.1%}", border=True)
        st.metric("F1 score", f"{m['f1']:.2f}", border=True)
        st.metric("Accuracy", f"{m['accuracy']:.1%}", border=True)

    left, right = st.columns([1, 1.4])
    with left:
        with st.container(border=True):
            st.subheader("What happened to the traffic")
            st.caption(f"{len(y):,} unseen test connections, strategy: {choice.lower()}")
            with st.container(horizontal=True):
                st.metric("Attacks caught", f"{m['tp']:,}", border=True)
                st.metric("Attacks missed", f"{m['fn']:,}", border=True)
            with st.container(horizontal=True):
                st.metric("False alarms", f"{m['fp']:,}", border=True)
                st.metric("Correctly ignored", f"{m['tn']:,}", border=True)
            st.caption(
                "In a data center every false alarm costs analyst time, and an "
                "automated reaction to a false thermal alert can cause an outage."
            )

    with right:
        with st.container(border=True):
            st.subheader("Isolation forest threshold trade-off")
            ts = np.linspace(
                np.quantile(data["iso_score"], 0.02), np.quantile(data["iso_score"], 0.98), 60
            )
            rows = []
            for t in ts:
                r = evaluate(y, (data["iso_score"] > t).astype(int))
                rows.append({"Threshold": t, "Precision": r["precision"], "Recall": r["recall"]})
            st.line_chart(pd.DataFrame(rows), x="Threshold", y=["Precision", "Recall"])
            st.caption(
                f"Current threshold: {iso_t:.3f}. Raising it removes false alarms but "
                "starts to miss attacks. Move the slider in the sidebar."
            )

    with st.container(border=True):
        st.subheader("Strategy comparison")
        rows = []
        for name, pred in strategies.items():
            r = evaluate(y, pred.astype(int))
            rows.append(
                {
                    "Strategy": name,
                    "Recall": r["recall"],
                    "Precision": r["precision"],
                    "F1": r["f1"],
                    "Accuracy": r["accuracy"],
                    "False alarms": r["fp"],
                    "Missed attacks": r["fn"],
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "Recall": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                "Precision": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                "F1": st.column_config.NumberColumn(format="%.2f"),
                "Accuracy": st.column_config.NumberColumn(format="percent"),
                "False alarms": st.column_config.NumberColumn(format="%d"),
                "Missed attacks": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption(
            "The two models fail in opposite ways: the Isolation Forest catches unseen "
            "attacks but raises false alarms, while the Random Forest is precise but "
            "misses novel attack types."
        )

    with st.container(border=True):
        st.subheader("Features the Random Forest relies on")
        top = data["importances"].nlargest(10).rename_axis("Feature").reset_index(name="Importance")
        st.bar_chart(top, x="Feature", y="Importance", horizontal=True, sort="-Importance")

# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
with tab_explain:
    st.caption(
        "SHAP shows which features pushed each model toward raising an alert, so an "
        "analyst sees why, not just that. Generated by `python src/anomaly_pipeline.py`."
    )
    iso_png, rf_png = OUTPUTS / "shap_summary_iso.png", OUTPUTS / "shap_summary_rf.png"
    csv_path = OUTPUTS / "alert_explanations.csv"

    if not (iso_png.exists() and rf_png.exists() and csv_path.exists()):
        st.warning(
            "SHAP outputs not found. Run `python src/anomaly_pipeline.py` first.",
            icon=":material/warning:",
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.subheader("Isolation forest")
                st.image(str(iso_png), caption="What makes traffic look anomalous", width="stretch")
        with c2:
            with st.container(border=True):
                st.subheader("Random forest")
                st.image(str(rf_png), caption="What pushes toward 'malicious'", width="stretch")

        with st.container(border=True):
            st.subheader("Why each alert fired")
            alerts = pd.read_csv(csv_path)
            f1c, f2c = st.columns(2)
            with f1c:
                model_pick = st.segmented_control(
                    "Model", ["All", "IsolationForest", "RandomForest"], default="All"
                ) or "All"
            with f2c:
                truth_pick = st.segmented_control(
                    "Ground truth", ["All", "malicious", "benign"], default="All",
                    help="'benign' rows are false alarms.",
                ) or "All"
            view = alerts
            if model_pick != "All":
                view = view[view["model"] == model_pick]
            if truth_pick != "All":
                view = view[view["true_label"] == truth_pick]

            reasons = view.apply(
                lambda r: ", ".join(f"{r[f'feature_{k}']}={r[f'value_{k}']:g}" for k in (1, 2, 3)),
                axis=1,
            )
            table = pd.DataFrame(
                {
                    "Model": view["model"],
                    "Ground truth": view["true_label"],
                    "Top reasons": reasons,
                    "Strongest SHAP": view["shap_1"],
                }
            )
            st.dataframe(
                table,
                hide_index=True,
                column_config={"Strongest SHAP": st.column_config.NumberColumn(format="%.2f")},
            )
            st.caption(f"{len(view):,} alerts from a 500-row sample of the test set.")

# --------------------------------------------------------------------------- #
# Kill chain and findings
# --------------------------------------------------------------------------- #
with tab_chain:
    st.caption(
        "Defensive reconstruction mapped to MITRE ATT&CK for ICS. Each stage names "
        "the signal the AI layer watches for."
    )
    stages = [
        (
            ":material/travel_explore: 1. Initial access",
            "T1190",
            "Exploit an exposed monitoring tool (Cacti-class RCE, CVE-2022-46169).",
            "Spike in failed logins and request rate on the monitoring segment.",
        ),
        (
            ":material/key: 2. Credential harvest",
            "TA0006 / TA0008",
            "Read plaintext secrets, pivot over SSH toward NUT/UPS or IPMI hosts.",
            "Unusual internal connections, new source-to-destination pairs.",
        ),
        (
            ":material/thermostat: 3. Manipulation",
            "T0831",
            "Spoof UPS battery telemetry or change HVAC cooling setpoints.",
            "Telemetry drifting outside the learned benign baseline.",
        ),
        (
            ":material/bolt: 4. Physical impact",
            "T0879",
            "Thermal shutdown, emergency power cutoff, hardware damage.",
            "Alert raised at stages 1 to 3, before this point.",
        ),
    ]
    for col, (title, ttp, what, signal) in zip(st.columns(4, border=True), stages):
        with col:
            st.markdown(f"**{title}**")
            st.badge(ttp, color="blue")
            st.write(what)
            st.caption(f":material/radar: {signal}")

    with st.container(border=True):
        st.subheader("Vulnerability findings (CVSS v3.1)")
        findings = pd.DataFrame(
            [
                ("F-01", "Unauthenticated RCE in monitoring tool", "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8, "Critical"),
                ("F-02", "Plaintext credentials in config files", "AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:L/A:N", 6.0, "Medium"),
                ("F-03", "IPMI/BMC reachable, weak credentials", "AV:A/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 9.6, "Critical"),
                ("F-04", "Unauthenticated UPS/HVAC control writes", "AV:A/AC:L/PR:N/UI:N/S:C/C:L/I:H/A:H", 9.3, "Critical"),
                ("F-05", "No IT/OT network segmentation", "AV:A/AC:L/PR:L/UI:N/S:C/C:L/I:H/A:H", 8.5, "High"),
            ],
            columns=["ID", "Finding", "Vector", "CVSS", "Severity"],
        )
        st.dataframe(
            findings,
            hide_index=True,
            column_config={
                "CVSS": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
                "Vector": st.column_config.TextColumn(width="large"),
            },
        )
        st.caption(
            "Illustrative lab scores. The OT findings reach Critical because Scope is "
            "Changed (IT to OT crossover) with high availability impact, which is the "
            "quantitative case for segmentation."
        )

# --------------------------------------------------------------------------- #
# Enterprise roadmap
# --------------------------------------------------------------------------- #
with tab_road:
    st.caption("How this prototype scales into a production capability.")
    phases = [
        (
            ":material/foundation: Phase 1: Foundation",
            "0 to 3 months",
            "Segment IT, OT DMZ, facility and controller zones. Move IPMI/BMC out of "
            "band. Fix findings F-01 to F-05. Deploy log, NetFlow, SNMPv3 and passive "
            "BACnet collectors.",
        ),
        (
            ":material/visibility: Phase 2: Shadow mode",
            "3 to 6 months",
            "Containerize inference, register models in MLflow, retrain on the "
            "facility's own baseline. Score without alerting to tune thresholds. Add "
            "SHAP to every event.",
        ),
        (
            ":material/automation: Phase 3: Response",
            "6 to 9 months",
            "Go live with SOAR playbooks for enrichment and IT-side containment. "
            "Humans approve every OT action. Add sensor cross-checks and mTLS "
            "between zones.",
        ),
        (
            ":material/verified: Phase 4: Assurance",
            "9 to 12 months",
            "Drift monitoring and governed retraining. Purple-team replays of this "
            "kill chain. Align with NIST SP 800-82, ISO 27001 and ISO 42001.",
        ),
    ]
    for col, (title, when, text) in zip(st.columns(4, border=True), phases):
        with col:
            st.markdown(f"**{title}**")
            st.badge(when, color="violet")
            st.write(text)

    with st.container(border=True):
        st.subheader("Success measures")
        with st.container(horizontal=True):
            st.metric("Time to detect lateral movement", "< 15 min", border=True)
            st.metric("Automated physical actions", "0", border=True,
                      help="Every physical action stays human-approved.")
            st.metric("Alert-to-model traceability", "100%", border=True,
                      help="Every alert links to a model version, data and explanation.")
        st.caption(
            "Limitations: NSL-KDD is a dated network benchmark with no real BMS/UPS "
            "telemetry. Re-establish performance on facility data before relying on it."
        )
