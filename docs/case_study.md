# Case Study — Cyber-Physical Compromise of a Data Center Facility

*Long-form documented case study for the capstone report. Embed your own artefacts
(terminal output, CVSS calculator screenshots, the `outputs/results.png` plots)
under the relevant sections.*

---

## 1. Executive Summary
A data center's environmental and power controls (UPS, PDU, HVAC) are increasingly
network-reachable through Building Management Systems and remote-management
interfaces. This case study reconstructs — in an authorized lab — how an attacker
could breach an exposed monitoring tool, pivot to facility-management hosts, and
manipulate telemetry to force a thermal shutdown. It then demonstrates an AI
anomaly-detection layer that flags the manipulation before physical impact, catching
behavior that signature-based firewalls miss.

## 2. Sector & Threat Landscape
Modern data centers fuse IT (servers, monitoring dashboards) with Operational
Technology (UPS, PDUs, HVAC) via BMS and IPMI/BMC. Many facility networks sit
adjacent to IT tooling and rely on legacy protocols (BACnet, Modbus) with weak
authentication. A breach of an internet-facing monitoring app can therefore become
a *physical* incident. Industry research (e.g. Claroty/Team82 disclosures on UPS
network cards and HVAC controllers; Eaton OT-security whitepapers) documents this
exact IT→OT crossover risk.

## 3. Lab Environment & Authorization
- Isolated virtual lab, owned and authorized for testing (no production systems).
- Components: a vulnerable monitoring VM, a simulated facility-management host, and
  a telemetry simulator feeding the AI layer.
- Rules of engagement: destructive OT actions are *simulated* (telemetry spoofing in
  the simulator), never executed against real hardware.

## 4. Attack Reconstruction (mapped to MITRE ATT&CK)
Follow the four stages in [`kill_chain.md`](kill_chain.md):
1. Initial Access — exploit the monitoring app (Cacti-class RCE, CVE-2022-46169).
2. Credential Harvest & Lateral Movement — extract secrets, pivot to facility hosts.
3. Subsystem Manipulation — spoof UPS battery data / shift HVAC setpoints.
4. Physical Impact — thermal shutdown / power cutoff (simulated).
*Artefacts:* include scan output and the CVSS findings table from
[`cvss_findings.md`](cvss_findings.md).

## 5. AI Detection Layer
- **Dataset:** NSL-KDD benchmark (real, citable) with a synthetic data-center
  telemetry fallback. Features include failed-login counts, byte/packet rates, and
  connection statistics — analogues of the monitoring-segment signals above.
- **Models:** Isolation Forest (unsupervised, trained on benign-only traffic to catch
  unseen/zero-day manipulation) + Random Forest (supervised signature classifier).
- **Metrics:** precision, recall, F1, ROC-AUC, confusion matrix, feature importances
  (see `outputs/results.png`). Discuss the false-positive/false-negative trade-off
  explicitly — a false thermal-shutdown alert has real operational cost.
- **Mapping:** each kill-chain stage names the signal the model keys on.

## 6. Findings & CVSS
Summarize [`cvss_findings.md`](cvss_findings.md). Highlight that OT findings reach the
Critical band because **CVSS Scope = Changed** (IT→OT crossover) combined with high
Availability impact — the quantitative case for segmentation.

## 7. Remediation & Defense Playbook
Summarize [`remediation_playbook.md`](remediation_playbook.md): segmentation, MFA on
management interfaces, secrets management, patching, signed telemetry, and continuous
AI monitoring as defense-in-depth.

## 8. Conclusion & Limitations
- The AI layer detects behavioral deviations but needs a clean training window and
  periodic retraining; concept drift (seasonal load changes) can raise false positives.
- Isolation Forest flags *that* something is anomalous, not *what* — pair with the
  supervised classifier and human triage.
- Future work: autoencoder/LSTM for temporal telemetry, streaming inference, and
  integration with SOAR for automated containment.
- Production scaling (SIEM/SOAR, XAI, Zero Trust, compliance mapping, phased roadmap):
  see [`enterprise_architecture.md`](enterprise_architecture.md). Its "Enterprise
  Roadmap" section can be appended here.

## 9. References (collect inline citations in your report)
- MITRE ATT&CK for ICS.
- NIST SP 800-82 (Guide to ICS Security).
- OWASP Testing Guide / OWASP Top 10.
- CVE-2022-46169 (Cacti) advisory.
- Claroty/Team82 data-center BMS/OT threat research.
- NSL-KDD dataset documentation.
