# Enterprise-Grade Upgrade Architecture

This document describes how the academic prototype in this repository (an Isolation
Forest + Random Forest defense layer, evaluated on NSL-KDD) would scale into a
production security capability for a data center. It covers five pillars:

1. SIEM & SOAR integration
2. MLOps, false-positive reduction & Explainable AI
3. Zero Trust & IT/OT microsegmentation
4. Regulatory & framework mapping
5. Enterprise roadmap

**Results the design builds on** (train = KDDTrain+, test = KDDTest+, malicious class):

| Model | Role | Default threshold: recall / precision / false alarms | Tuned threshold: recall / precision / false alarms |
|---|---|---|---|
| Isolation Forest | Novelty detection, no attack labels needed | 0.95 / 0.79 / 3,174 | 0.62 / 0.98 / 195 |
| Random Forest | Known-signature classification, class-weighted | 0.62 / 0.97 / 278 | 0.98 / 0.91 / 1,262 |

Thresholds are tuned on a validation split of the training data, never the test set. The
numbers show that **the operating threshold, not the model, sets the false-alarm cost**.
At default thresholds the models fail in opposite ways: the Isolation Forest catches unseen
attacks but raises many false alarms, and the Random Forest is precise but misses novel
attacks. Tuning moves each along its own precision-recall curve. The architecture below is
built around that trade-off: a detector *raises* an alert at a chosen operating point, and
later stages *confirm, explain and triage* it.

---

## Pillar 1 — Enterprise SIEM & SOAR Integration

### 1.1 From terminal script to pipeline

| Prototype | Production |
|---|---|
| `python -m src.pipeline` on a laptop | Containerized inference service behind a message bus |
| Static CSV / benchmark file | Streaming telemetry from live collectors |
| Batch metrics printed to the terminal | Per-event anomaly score written back to the SIEM |
| Plots in `outputs/` | SIEM dashboards + SOAR case records |

### 1.2 Telemetry sources

| Source | Protocol / port | What it gives the model | Collector |
|---|---|---|---|
| Server, hypervisor, Cacti host logs | Syslog (UDP 514, TLS 6514) | Auth failures, process execution, web requests | Elastic Agent / Splunk Universal Forwarder |
| Network flows | NetFlow v9 / IPFIX (UDP 2055 / 4739) | Connection counts, byte/packet rates, new src→dst pairs | Filebeat NetFlow module / Splunk Stream |
| UPS, PDU, switches | SNMP v3 polling (UDP 161) + traps (UDP 162) | Battery charge, load kW, input voltage, port state | Telegraf / SNMP poller |
| HVAC / BMS controllers | BACnet/IP (UDP 47808), Modbus/TCP (502) | Setpoints, supply/return temperature, write commands | Passive tap + Zeek with CISA ICSNPP parsers |
| IPMI / BMC | Redfish API / IPMI SEL | Out-of-band logins, firmware events, power actions | Redfish poller → Syslog |

Two design rules for OT sources:
- **Collect passively.** Use SPAN ports or network TAPs on BMS segments rather than
  actively polling fragile controllers.
- **Use SNMPv3 only** (authPriv). SNMPv1/v2c community strings travel in cleartext.

### 1.3 Reference pipeline

```
 OT / IT sources            Ingest & normalize           Detect                 Respond
┌────────────────┐      ┌──────────────────────┐    ┌──────────────────┐   ┌──────────────────┐
│ Syslog         │─────▶│ Collectors           │    │ ML inference svc │   │ SOAR             │
│ NetFlow/IPFIX  │─────▶│ (Elastic Agent /     │───▶│ (FastAPI + model │──▶│ (Splunk SOAR /   │
│ SNMPv3         │─────▶│  UF / Telegraf/Zeek) │    │  from registry)  │   │  Elastic / Tines │
│ BACnet/Modbus  │─────▶│         │            │    │  IF score + RF   │   │  / Shuffle)      │
│ Redfish/IPMI   │─────▶│         ▼            │    │  class + SHAP    │   │        │         │
└────────────────┘      │ Kafka topic: raw     │    └────────┬─────────┘   │        ▼         │
                        │         │            │             │             │ Ticket / page /  │
                        │         ▼            │             ▼             │ containment      │
                        │ Normalize to ECS/CIM │    SIEM index: ml-alerts  └──────────────────┘
                        │ → SIEM index         │    (score, label, top features)
                        └──────────────────────┘
```

- **Normalization:** map fields to Elastic Common Schema (ECS) or the Splunk Common
  Information Model (CIM). Detection rules and dashboards can then be reused across sources.
- **Feature engineering:** run as a stream job (Kafka Streams / Flink, or SIEM
  transforms) that builds the same features the model was trained on. Examples are
  failed logins per 60 s window, flows per source per minute, and temperature delta
  against a rolling baseline.
- **Inference:** a stateless container loads the model version from the registry,
  scores each window, and writes `{score, predicted_class, top_shap_features, model_version}`
  back to an `ml-alerts` index.

### 1.4 SOAR playbook — "Anomalous telemetry on facility segment"

| Step | Action | Automated? |
|---|---|---|
| 1 | Trigger: IF score above threshold on a UPS/HVAC asset | Yes |
| 2 | Enrich: asset owner, criticality and maintenance window (CMDB); recent change tickets; SHAP top features | Yes |
| 3 | Suppress if inside an approved maintenance window or a duplicate of an open case | Yes |
| 4 | Cross-check: compare the reported value against an independent sensor (e.g. BMS temperature vs. rack inlet probes, UPS load vs. PDU sum) | Yes |
| 5 | Mismatch → classify as **probable telemetry spoofing**, raise severity | Yes |
| 6 | IT-side containment: disable the compromised account, isolate the Cacti/monitoring host through EDR or NAC | Yes, with approval gate |
| 7 | OT-side action: put the controller in read-only mode or block write commands at the OT firewall | **Human approval required** |
| 8 | Physical action (load shedding, failover) | **Never automated**. Facilities engineer decides |
| 9 | Close: case record, timeline, and analyst verdict fed back as a label | Yes |

**Design principle:** automation may contain the *IT* foothold freely. Any action that
changes the *physical* state of the facility stays human-in-the-loop. An automated
response to a false positive must not itself cause the outage you are trying to prevent.

---

## Pillar 2 — MLOps, False-Positive Reduction & Explainable AI

### 2.1 Why false positives matter here
At its default threshold the Isolation Forest flags about a third of benign traffic (3,174
false alarms in 9,711 benign connections), and the tuned Random Forest raises 1,262. Either
is too noisy to page a person on its own. In a data center,
alert fatigue leads to ignored alerts, and an automated reaction to a false thermal alarm
can trigger an unnecessary failover. The goal is to keep the detector's high recall
while removing noise *downstream* of it.

### 2.2 Layered false-positive reduction

| Layer | Technique | Effect |
|---|---|---|
| Threshold | Tune the IF decision threshold on a validation set to a target alert budget (e.g. ≤ 5 pages/shift), using a precision–recall curve instead of `contamination=0.1` | Trades a little recall for much better precision |
| Two-stage scoring | IF raises a candidate. RF + rule checks confirm or downgrade it | Combines IF's label-free coverage with RF's precision |
| Physical consistency | Accept a temperature/power anomaly only if an independent sensor disagrees or agrees, depending on the hypothesis | Separates spoofed telemetry from real thermal events |
| Context suppression | Maintenance windows, change tickets, known batch jobs, deduplication over a time window | Removes expected anomalies |
| Risk-based alerting | Accumulate a risk score per asset; page only when it crosses a threshold | One page per incident instead of one per event |
| Analyst feedback | Analyst verdicts become labels for retraining | Precision improves over time |

### 2.3 Explainable AI

Every alert should carry the reason it fired.

- **SHAP (TreeExplainer)** works on both tree ensembles in this project, the Random
  Forest and the Isolation Forest. For each alert, store the top 3–5 features and their
  contributions.
  - Example: `num_failed_logins +0.41, srv_count +0.22, dst_host_serror_rate +0.17`
    → "credential brute force against the monitoring host".
- **LIME** is a model-agnostic alternative, useful if the model is later replaced by an
  autoencoder or gradient-boosted model.
- **Global explanations:** a SHAP summary plot per model version is attached to each
  release so reviewers can see what the model relies on overall.
- **Analyst UI:** show the explanation in the SIEM alert view and the SOAR case. The
  analyst should not have to open a notebook.

**Implemented in the prototype:** `explain()` in
[`src/explain.py`](../src/explain.py) runs SHAP TreeExplainer on
both models and writes:
- `outputs/shap_summary_iso.png` and `outputs/shap_summary_rf.png`: global views of
  what drives each model.
- `outputs/alert_explanations.csv`: the top 3 features, their values and SHAP
  contributions for every alert in a 500-row test sample.

Example output from the Isolation Forest on a true attack:
`dst_host_srv_serror_rate=1.0 (+1.68), dst_host_serror_rate=1.0 (+1.50), serror_rate=1.0 (+1.49)`.
Every connection ending in a SYN error is the signature of a SYN flood or port scan,
which an analyst can read directly from the explanation.

### 2.4 MLOps lifecycle

| Stage | Tooling (examples) | Control |
|---|---|---|
| Data versioning | DVC / lakeFS | Every model traces back to the exact training data |
| Experiment tracking | MLflow | Parameters, metrics and SHAP plots per run |
| Model registry | MLflow Model Registry | Staging → Production promotion needs sign-off |
| Validation gate | CI job | Block promotion if recall on a held-out attack set drops, or if the false-positive rate exceeds budget |
| Deployment | Container + Kubernetes | Shadow mode first (score, don't alert), then canary |
| Monitoring | Evidently / custom jobs | Drift and performance dashboards |
| Rollback | Registry version pin | One-step revert to the last good model |

### 2.5 Drift management
- **Data drift:** compare live feature distributions with training using the Population
  Stability Index (PSI) or the Kolmogorov–Smirnov test. Data centers have legitimate
  drift such as seasonal cooling load, new racks, and firmware changes.
- **Concept drift:** track precision from analyst verdicts. A falling precision with
  stable inputs means the definition of "normal" has changed.
- **Retraining policy:** scheduled (e.g. monthly) plus triggered (PSI > 0.2 on a key
  feature). Always use a champion/challenger comparison before promotion.
- **Poisoning guard:** retrain the benign baseline only on windows with no open
  incidents. An attacker who is already inside could otherwise teach the model that
  their activity is normal.

---

## Pillar 3 — Zero Trust & IT/OT Microsegmentation

### 3.1 Target zone model (Purdue / IEC 62443 zones & conduits)

```
 Level 4-5  Enterprise IT ─────────── Internet
                 │  (firewall, no direct path to OT)
 Level 3.5  ┌───────────────── OT DMZ ──────────────────┐
            │ Jump host + PAM (MFA, session recording)  │
            │ Historian replica / telemetry broker      │
            │ SIEM collectors (receive-only)            │
            └───────────────────────────────────────────┘
                 │  conduit: allow-listed, mTLS, logged
 Level 3    Facility management zone: BMS server, NUT/UPS mgmt, DCIM
                 │  conduit: protocol-aware firewall (BACnet/Modbus DPI)
 Level 1-2  Controllers: HVAC controllers, UPS NMCs, PDUs, CRAC units

 Out-of-band   Separate IPMI/BMC management network → reachable only via PAM jump host
```

Rules that block the kill chain in [`kill_chain.md`](kill_chain.md):

| Kill-chain step | Control that breaks it |
|---|---|
| Cacti host reachable from the internet | Monitoring stays in IT. Reachable only through the identity-aware proxy |
| Cacti host SSHes to the NUT/UPS server | No IT→Level 3 route. All admin access goes through the PAM jump host with MFA |
| Harvested `.env` credentials reused | Short-lived, per-session credentials issued by PAM. Static secrets in a vault |
| Write to UPS/HVAC setpoints | OT firewall allows BACnet `WriteProperty` / Modbus function codes 5/6/15/16 only from the BMS server |
| IPMI/BMC reachable from server VLANs | Dedicated out-of-band network. No route from production |

### 3.2 Software-defined perimeter (SDP)
- Management UIs (Cacti, DCIM, BMS web consoles) are published only through an
  identity-aware proxy or SDP broker, such as OpenZiti, Cloudflare Access or Zscaler ZPA.
  The underlying ports are not routable from user VLANs.
- Each access is authorized per request on identity, device posture and context.
  Network location grants no trust.
- Policy is deny-by-default. Allow rules name an identity, a service and an action.

### 3.3 Mutual TLS
- Workload identity via SPIFFE/SPIRE (or an internal CA such as step-ca) gives every
  collector, broker and inference service a short-lived X.509 certificate.
- All conduits between zones (collectors → Kafka, Kafka → inference, inference → SIEM,
  BMS server → telemetry broker) require **mTLS**. A connection without a valid client
  certificate is rejected before any application data is exchanged.
- **Legacy OT reality:** most UPS network cards and HVAC controllers cannot do mTLS.
  Place a **protocol gateway** next to them. The gateway speaks BACnet/Modbus on a
  short, isolated wire and mTLS upstream. Where telemetry only needs to leave the OT
  zone, use a **unidirectional gateway (data diode)** so nothing can be sent back in.
- Certificate rotation is automated (lifetimes in hours, not years) and revocation is
  monitored.

### 3.4 Microsegmentation enforcement
- **East-west inside IT:** host-based policy (e.g. eBPF/Cilium, Illumio, or
  hypervisor-level distributed firewall) so a compromised Cacti host can reach only
  the services it needs.
- **OT:** zone firewalls with industrial deep packet inspection. Alert on any BACnet/Modbus
  write from an unexpected source. These alerts also feed the ML pipeline as features.

---

## Pillar 4 — Regulatory & Framework Mapping

Control identifiers below refer to NIST SP 800-82 Rev. 3 (which applies the
SP 800-53 Rev. 5 control catalog with an OT overlay), ISO/IEC 27001:2022 Annex A, and
ISO/IEC 42001:2023. Verify exact clause wording against the published standards before
citing them in the final report.

### 4.1 VAPT findings → controls

| Finding ([`cvss_findings.md`](cvss_findings.md)) | NIST SP 800-82 / 800-53 | ISO/IEC 27001:2022 Annex A |
|---|---|---|
| F-01 Unauthenticated RCE in monitoring tool | SI-2 Flaw Remediation, RA-5 Vulnerability Monitoring, CM-7 Least Functionality | 8.8 Management of technical vulnerabilities, 8.9 Configuration management |
| F-02 Plaintext credentials in config files | IA-5 Authenticator Management, SC-28 Protection of Information at Rest | 5.17 Authentication information, 8.24 Use of cryptography |
| F-03 IPMI/BMC reachable, weak credentials | AC-17 Remote Access, IA-2 Identification & Authentication (MFA), AC-6 Least Privilege | 8.2 Privileged access rights, 8.5 Secure authentication |
| F-04 Unauthenticated telemetry/control writes | SC-8 Transmission Confidentiality & Integrity, SI-7 Integrity, AC-3 Access Enforcement | 8.20 Networks security, 8.24 Use of cryptography |
| F-05 No IT/OT segmentation | SC-7 Boundary Protection, AC-4 Information Flow Enforcement (800-82 defense-in-depth architecture) | 8.22 Segregation of networks, 8.20 Networks security |

### 4.2 AI defense layer → controls

| Mechanism | NIST SP 800-82 / 800-53 | ISO/IEC 27001:2022 | ISO/IEC 42001:2023 |
|---|---|---|---|
| Telemetry + log collection into SIEM | AU-2 Event Logging, AU-6 Audit Review | 8.15 Logging | — |
| Anomaly detection on OT/IT telemetry | SI-4 System Monitoring | 8.16 Monitoring activities | A.6 AI system life cycle (operation and monitoring) |
| SOAR playbooks, human approval for OT actions | IR-4 Incident Handling | 5.24–5.26 Incident management | A.9 Use of AI systems (human oversight) |
| Training data provenance, poisoning guard | SI-7 Integrity | 8.32 Change management | A.7 Data for AI systems (data quality, provenance) |
| Validation gate, champion/challenger | CA-2 Control Assessments | 8.29 Security testing in development and acceptance | A.6 AI system life cycle (verification and validation) |
| SHAP explanations on every alert | — | — | A.8 Information for interested parties (transparency) |
| Impact of false positives on facility operations | RA-3 Risk Assessment | 6.1 Risk assessment (clause) | Clause 8.4 AI system impact assessment, A.5 Assessing impacts |
| Drift monitoring and retraining policy | CA-7 Continuous Monitoring | 8.16 Monitoring activities | Clause 9.1 Monitoring, measurement, analysis and evaluation |

**Why ISO/IEC 42001 matters here:** the AI layer influences decisions about physical
infrastructure. ISO/IEC 42001 requires the organization to assess the AI system's
impact, keep humans accountable for its outputs, and monitor its performance over
time. The human-in-the-loop playbook, SHAP explanations and drift controls above are
the evidence for that.

---

## Pillar 5 — Enterprise Roadmap

*The section below is written to be appended to the report's conclusion.*

### Enterprise Roadmap

This project reconstructs how a data center's IT/OT boundary can be crossed from a
vulnerable monitoring application, and builds and evaluates an AI defense layer for the
resulting behaviour. Evaluated against previously unseen attack types (NSL-KDD KDDTest+),
a class-weighted Random Forest with a tuned alert threshold detected 98% of malicious
connections at 91% precision, up from 62% recall at the default threshold. The unsupervised
Isolation Forest detected 95% of attacks at its default threshold without using any attack
labels, which makes it the fallback when labeled attack data is unavailable. Because the
operating threshold sets the false-alarm cost, a production system should choose it from
facility data and an agreed alert budget, and add context enrichment and human review
before any automated response. Moving from prototype to production is planned in four
phases.

**Phase 1 — Foundation (0–3 months).** Segment the network into IT, OT DMZ, facility
management and controller zones following the Purdue model and IEC 62443 zones and
conduits. Move IPMI/BMC to an out-of-band network. Remediate findings F-01 to F-05,
prioritizing the Critical items. Deploy SIEM collectors for Syslog, NetFlow, SNMPv3 and
passive BACnet/Modbus monitoring, and establish a clean telemetry baseline.

**Phase 2 — Detection in shadow mode (3–6 months).** Containerize the inference service
and register models in MLflow. Retrain the Isolation Forest on the facility's own
baseline telemetry instead of a public benchmark. Run it in shadow mode, scoring
without alerting, to calibrate thresholds against a defined alert budget. Add SHAP
explanations to every scored event.

**Phase 3 — Response integration (6–9 months).** Promote the model to live alerting with
SOAR playbooks for enrichment, suppression and IT-side containment. Keep every OT and
physical action behind human approval. Introduce physical-consistency checks between
independent sensors to separate spoofed telemetry from genuine thermal or power events.
Deploy mTLS between zones and identity-aware access to all management interfaces.

**Phase 4 — Assurance and continuous improvement (9–12 months and ongoing).** Operate
drift monitoring and a governed retraining cycle fed by analyst verdicts. Run regular
purple-team exercises that replay this project's kill chain to verify detection and
response. Align the program with NIST SP 800-82, ISO/IEC 27001 and ISO/IEC 42001, with
documented AI impact assessments and model governance as audit evidence.

**Success measures.** Mean time to detect IT→OT lateral movement under 15 minutes. A
false-positive rate within the agreed alert budget. Zero automated physical actions
triggered by unconfirmed alerts. Full traceability from every alert to its model
version, training data and explanation.

**Limitations to carry forward.** NSL-KDD is a dated network benchmark and does not
contain real BMS or UPS telemetry. Production performance must be re-established on
facility data. Tree-based detectors also do not model time. Sequence models such as
LSTM autoencoders, and digital-twin simulation of cooling and power behavior, are the
natural next research steps.
