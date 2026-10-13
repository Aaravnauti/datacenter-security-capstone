# Kill Chain — Data Center Cyber-Physical Attack

Mapped to **MITRE ATT&CK for ICS** / Enterprise tactics. This is a *defensive*
reconstruction: for each adversary stage it names the detection opportunity the
AI layer is designed to catch. All testing is confined to an isolated, authorized
lab (deliberately-vulnerable VMs you own).

---

## Stage 1 — Initial Access
**Tactic:** Initial Access (TA0001) / Exploit Public-Facing Application (T1190)

- Attacker finds an exposed infrastructure-monitoring web app (e.g. Cacti) or a
  remote-management portal reachable from the IT network.
- Example reference vulnerability: **Cacti CVE-2022-46169** — unauthenticated
  command injection in a monitoring tool. (Used here only to frame the scenario;
  exploitation is performed against a lab instance.)

**Detection signal → AI layer:** sudden rise in failed-login counts and an
anomalous request/packet rate on the monitoring segment.

---

## Stage 2 — Credential Harvest & Lateral Movement
**Tactic:** Credential Access (TA0006) / Lateral Movement (TA0008)

- From the foothold, the attacker reads configuration/environment files holding
  plaintext credentials or tokens, then pivots over SSH toward facility-management
  hosts (e.g. a NUT / UPS management server, or an IPMI/BMC interface).

**Detection signal → AI layer:** unusual internal connection patterns — new
source→destination pairs, off-hours sessions, connection-rate deviations.

---

## Stage 3 — Subsystem Manipulation (Impact setup)
**Tactic:** Impair Process Control (TA0106, ICS) / Manipulation of Control (T0831)

- With management access, the attacker targets power/environmental controls:
  falsifying UPS battery telemetry, or altering HVAC cooling setpoints.

**Detection signal → AI layer:** telemetry values drifting outside the learned
benign distribution (temperature climbing, power-load readings inconsistent with
actual draw) — exactly what the Isolation Forest baseline is trained to flag.

---

## Stage 4 — Physical Impact
**Tactic:** Impact (TA0040) / Damage to Property (T0879, ICS)

- Spoofed stability data or raised thermal thresholds trigger automatic thermal
  shutdowns, emergency power cutoffs, or hardware damage — without touching the
  production servers' OS.

**Defensive outcome:** the AI layer raises an alert at Stage 1–3, before Stage 4
physical disruption can occur. See [`remediation_playbook.md`](remediation_playbook.md).
