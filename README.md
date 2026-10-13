# Data Center Cyber-Physical Security Capstone

**Combining VAPT (Vulnerability Assessment & Penetration Testing) with an AI Anomaly-Detection Defense Layer.**

Sector: **Data Center** — the convergence of IT infrastructure, monitoring tools, and
OT / Building Management Systems (BMS) such as HVAC and UPS power units.

Courses: *Artificial Intelligence for Cyber Security* (ICS3120) + *VAPT* (ICS3102).

---

## What this project is

A data center depends on both IT systems (servers, monitoring dashboards) and
Operational Technology (UPS, PDUs, HVAC). An attacker who breaches an exposed
monitoring tool can pivot toward the facility-management network and manipulate
*telemetry* — spoofing UPS battery data or shifting HVAC setpoints — to cause a
thermal shutdown without ever touching the production OS.

This project has two halves that meet at that telemetry boundary:

1. **VAPT (offense, documented in a controlled lab):** reconnaissance, vulnerability
   scoring (CVSS), and a kill chain mapped to MITRE ATT&CK — see [`docs/`](docs/).
2. **AI defense layer (this repo's code):** an anomaly-detection + threat-classification
   pipeline that flags unauthorized telemetry manipulation and intrusion patterns
   that signature-based firewalls miss — see [`src/`](src/).

> **Scope & ethics.** All offensive work is performed only against an isolated lab you
> own and are authorized to test. This repository contains **no weaponized exploit
> code** — the VAPT material is detection/scoring/remediation documentation, which is
> what earns marks and what a real blue team produces.

---

## Repository layout

```
datacenter-security-capstone/
├── README.md
├── requirements.txt
├── app.py                         # interactive Streamlit dashboard
├── .streamlit/config.toml         # dashboard theme
├── src/
│   ├── anomaly_pipeline.py        # main AI defense layer (run this)
│   └── generate_synthetic_data.py # offline fallback dataset generator
├── docs/
│   ├── kill_chain.md              # 4-stage kill chain mapped to MITRE ATT&CK for ICS
│   ├── cvss_findings.md           # vulnerability findings + CVSS v3.1 vectors
│   ├── remediation_playbook.md    # defensive controls & segmentation
│   ├── case_study.md              # long-form documented case study
│   └── enterprise_architecture.md # SIEM/SOAR, MLOps/XAI, Zero Trust, compliance, roadmap
└── outputs/                       # plots & reports written here at runtime
```

---

## Quick start

```bash
pip install -r requirements.txt
python src/anomaly_pipeline.py
```

The pipeline:

1. Loads the **NSL-KDD** benchmark dataset from a public mirror (real, citable data).
   If offline, it falls back to the synthetic data-center telemetry generator.
2. Preprocesses (encodes categoricals, scales features, stratified split).
3. Trains an **Isolation Forest** on *benign-only* traffic (unsupervised, for unseen /
   zero-day anomalies) and a **Random Forest** classifier (supervised, for known
   attack signatures).
4. Evaluates with **precision / recall / F1 / ROC-AUC** (not just accuracy — which is
   misleading on imbalanced security data).
5. Saves a confusion matrix and a feature-importance plot to `outputs/`.
6. Explains alerts with **SHAP** (Explainable AI): global summary plots for both
   models (`shap_summary_rf.png`, `shap_summary_iso.png`) and the top 3 reasons behind
   each alert in a 500-row sample (`alert_explanations.csv`). If `shap` isn't
   installed, this step is skipped and the rest still runs.

### Interactive dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501 with four tabs: **Detection** (live threshold sliders and
a strategy comparison), **Explainability** (SHAP plots and per-alert reasons),
**Kill chain and findings** (ATT&CK stages and the CVSS table), and **Enterprise
roadmap**. The first load trains both models (about 30 seconds). Run
`python src/anomaly_pipeline.py` once beforehand so the SHAP outputs exist.

### Google Colab

Open a new notebook and run:

```python
!pip install -q scikit-learn pandas matplotlib
!wget -q https://raw.githubusercontent.com/<your-repo>/src/anomaly_pipeline.py
%run anomaly_pipeline.py
```

Or paste the contents of [`src/anomaly_pipeline.py`](src/anomaly_pipeline.py) into a cell.

---

## How the two halves connect

| Kill-chain stage | Technique (MITRE ATT&CK-style) | What the AI layer watches for |
|---|---|---|
| Initial Access | Exploit public-facing monitoring app (e.g. Cacti, CVE-2022-46169) | Spike in failed logins / anomalous request rate |
| Credential Harvest & Lateral Movement | Pivot toward facility mgmt (NUT / IPMI) | Unusual internal connection patterns |
| Impact | Manipulated UPS / HVAC telemetry | Sensor values outside the learned baseline |
| Defense | — | Real-time flag before physical thermal shutdown |

Full narrative in [`docs/case_study.md`](docs/case_study.md).
