"""
Data center cyber-physical defense - interactive dashboard.

Run:
    streamlit run app.py

Loads the models saved by `python -m src.pipeline` (or trains them once, cached)
and lets you explore detection trade-offs, SHAP explanations, the kill chain
and the enterprise roadmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import load_config, resolve_path  # noqa: E402
from src.data_preprocessing import prepare_data  # noqa: E402
from src.thresholds import STRATEGIES, flag_connections, strategy_metrics  # noqa: E402
from src.train_models import fit_and_tune, load_artifacts  # noqa: E402

CFG = load_config()
OUTPUTS = resolve_path(CFG["paths"]["outputs_dir"])
LABELS = {
    "isolation_forest": "Isolation forest",
    "random_forest": "Random forest",
    "either_flags": "Either flags",
    "both_flag": "Both flag",
}

st.set_page_config(
    page_title="Data center defense",
    page_icon=":material/shield:",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Data and models (cached: trained once per server session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models (trains on the first run, about a minute)...")
def load_models() -> dict:
    ds = prepare_data(CFG)

    # Reuse models saved by `python -m src.pipeline` when they match this data.
    try:
        art = load_artifacts(resolve_path(CFG["paths"]["artifacts_dir"]))
        meta = art["metadata"]
        reuse = meta["source"] == ds.source and meta["feature_names"] == ds.feature_names
    except FileNotFoundError:
        reuse = False

    if reuse:
        iso, rf, thresholds = art["iso"], art["rf"], meta["thresholds"]
        origin = "saved models in artifacts/"
    else:
        iso, rf, thresholds = fit_and_tune(ds.X_train, ds.y_train, CFG)
        origin = "trained in this session"

    return {
        "source": ds.source,
        "origin": origin,
        "n_train": len(ds.y_train),
        "n_test": len(ds.y_test),
        "n_features": len(ds.feature_names),
        "y": ds.y_test.to_numpy(),
        "iso_score": -iso.score_samples(ds.X_test),   # higher = more anomalous
        "rf_proba": rf.predict_proba(ds.X_test)[:, 1],
        "thresholds": thresholds,                     # tuned on a training-data validation split
        "defaults": {"isolation_forest": float(-iso.offset_), "random_forest": 0.5},
        "importances": pd.Series(rf.feature_importances_, index=ds.feature_names),
    }


def metrics_for(flagged: np.ndarray) -> dict:
    return strategy_metrics(data["y"], flagged)


# --------------------------------------------------------------------------- #
# Header + sidebar (render before the slow work)
# --------------------------------------------------------------------------- #
st.title("Data center cyber-physical defense", icon=":material/shield:")
st.caption(
    "AI anomaly detection for the IT/OT boundary: monitoring tools, IPMI/BMC, "
    "and UPS/HVAC telemetry. ICS3120 + VAPT capstone."
)

data = load_models()
tuned = data["thresholds"]


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


with st.sidebar:
    st.header("Detector settings", icon=":material/tune:")
    lo, hi = round(float(data["iso_score"].min()), 3), round(float(data["iso_score"].max()), 3)
    iso_t = st.slider(
        "Isolation forest threshold",
        min_value=lo,
        max_value=hi,
        value=round(_clamp(tuned["isolation_forest"], lo, hi), 3),
        step=0.005,
        help="Alert when the anomaly score is above this. Higher means fewer, "
        "more confident alerts (fewer false alarms, more missed attacks).",
    )
    rf_t = st.slider(
        "Random forest threshold",
        min_value=0.005,
        max_value=0.95,
        value=round(_clamp(tuned["random_forest"], 0.005, 0.95), 3),
        step=0.005,
        help="Alert when the predicted probability of 'malicious' reaches this. "
        "Lower means more alerts and fewer missed attacks.",
    )
    st.caption(
        f"Sliders start at the {tuned['mode']}-tuned thresholds: "
        f"{tuned['isolation_forest']:.3f} and {tuned['random_forest']:.3f}."
    )
    st.caption(
        f"Data: {data['source']}  \n"
        f"{data['n_train']:,} training rows, {data['n_test']:,} test rows, "
        f"{data['n_features']} features.  \n"
        f"Models: {data['origin']}."
    )
    st.caption(
        "All offensive testing is done only in an isolated lab you own and are "
        "authorized to test."
    )

y = data["y"]
current = {"isolation_forest": iso_t, "random_forest": rf_t}
flags = flag_connections(data["iso_score"], data["rf_proba"], current)
strategies = {LABELS[name]: flags[name] for name in STRATEGIES}

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
        default="Either flags",
        help="'Either flags' alerts when either model flags a connection (most "
        "coverage). 'Both flag' alerts only when both agree (fewest false alarms).",
    ) or "Either flags"
    m = metrics_for(strategies[choice])

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
                st.metric("Attacks caught", f"{m['true_positives']:,}", border=True)
                st.metric("Attacks missed", f"{m['missed_attacks']:,}", border=True)
            with st.container(horizontal=True):
                st.metric("False alarms", f"{m['false_alarms']:,}", border=True)
                st.metric("Correctly ignored", f"{m['true_negatives']:,}", border=True)
            st.caption(
                "In a data center every false alarm costs analyst time, and an "
                "automated reaction to a false thermal alert can cause an outage."
            )

    with right:
        with st.container(border=True):
            st.subheader("Threshold trade-off")
            curve_model = st.segmented_control(
                "Model", ["Random forest", "Isolation forest"], default="Random forest"
            ) or "Random forest"
            is_rf = curve_model == "Random forest"
            scores = data["rf_proba"] if is_rf else data["iso_score"]
            ts = np.linspace(
                np.quantile(scores, 0.02) if not is_rf else 0.005,
                np.quantile(scores, 0.98) if not is_rf else 0.95,
                60,
            )
            rows = []
            for t in ts:
                r = metrics_for(scores > t)
                rows.append({"Threshold": t, "Precision": r["precision"], "Recall": r["recall"]})
            st.line_chart(pd.DataFrame(rows), x="Threshold", y=["Precision", "Recall"])
            st.caption(
                f"Current threshold: {rf_t if is_rf else iso_t:.3f}. "
                + (
                    "Lowering it catches more attacks but raises more false alarms."
                    if is_rf
                    else "Raising it removes false alarms but starts to miss attacks."
                )
                + " Move the slider in the sidebar."
            )

    with st.container(border=True):
        st.subheader("Strategy comparison")
        basis = st.segmented_control(
            "Thresholds",
            ["Current sliders", "Model defaults"],
            default="Current sliders",
            help="Model defaults are the untuned thresholds: Isolation Forest at its "
            "contamination-based cutoff and Random Forest at 0.50.",
        ) or "Current sliders"
        shown = flags if basis == "Current sliders" else flag_connections(
            data["iso_score"], data["rf_proba"], data["defaults"]
        )
        rows = []
        for key in STRATEGIES:
            r = metrics_for(shown[key])
            rows.append(
                {
                    "Strategy": LABELS[key],
                    "Recall": r["recall"],
                    "Precision": r["precision"],
                    "F1": r["f1"],
                    "Accuracy": r["accuracy"],
                    "False alarms": r["false_alarms"],
                    "Missed attacks": r["missed_attacks"],
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
            "At default thresholds the models fail in opposite ways: the Isolation Forest "
            "catches unseen attacks but raises false alarms, while the Random Forest is "
            "precise but misses novel attack types. Tuning the thresholds on a validation "
            "split moves the Random Forest to a much higher recall."
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
        "analyst sees why, not just that. Generated by `python -m src.explain`."
    )
    iso_png, rf_png = OUTPUTS / "shap_summary_iso.png", OUTPUTS / "shap_summary_rf.png"
    csv_path = OUTPUTS / "alert_explanations.csv"

    if not (iso_png.exists() and rf_png.exists() and csv_path.exists()):
        st.warning(
            "SHAP outputs not found. Run `python -m src.pipeline` first.",
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
    for col, (title, ttp, what, signal) in zip(st.columns(4, border=True), stages, strict=True):
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
                ("F-02", "Plaintext credentials in config files", "AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:L/A:N", 6.1, "Medium"),
                ("F-03", "IPMI/BMC reachable, weak credentials", "AV:A/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 9.6, "Critical"),
                ("F-04", "Unauthenticated UPS/HVAC control writes", "AV:A/AC:L/PR:N/UI:N/S:C/C:L/I:H/A:H", 9.6, "Critical"),
                ("F-05", "No IT/OT network segmentation", "AV:A/AC:L/PR:L/UI:N/S:C/C:L/I:H/A:H", 8.9, "High"),
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
    for col, (title, when, text) in zip(st.columns(4, border=True), phases, strict=True):
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
