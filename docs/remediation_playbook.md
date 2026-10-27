# Remediation & Defense Playbook

Translates the findings in [`cvss_findings.md`](cvss_findings.md) into actionable
controls. Ordered by impact-to-effort.

## 1. Network segmentation (addresses F-05, contains F-01→F-04)
- Isolate OT / BMS subnets (UPS, PDU, HVAC, IPMI/BMC) from the IT and
  monitoring networks with firewalls / VLANs.
- Allow only an explicit management-station allowlist to reach OT control ports.
- Deny OT→Internet egress by default.

## 2. Harden management interfaces (F-03)
- Enforce MFA on IPMI/BMC and facility-management portals.
- Remove default credentials; rotate on a schedule.
- Place management interfaces on a dedicated out-of-band network.

## 3. Secure secrets (F-02)
- Remove plaintext credentials from config/`.env` files; use a secrets manager.
- Restrict file permissions; audit for committed secrets.

## 4. Patch & reduce attack surface (F-01)
- Patch monitoring platforms promptly; subscribe to CVE feeds (e.g. Cacti).
- Remove internet exposure of monitoring dashboards; front with VPN/SSO.

## 5. Authenticated, signed telemetry (F-04)
- Require authentication + integrity protection on control writes to UPS/HVAC.
- Reject unsigned setpoint changes; log all control-plane writes.

## 6. Continuous AI anomaly monitoring (the detection layer)
- Deploy the [detection pipeline](../src/pipeline.py) against live
  telemetry + network logs.
- Train the Isolation Forest baseline on a known-clean window; alert on drift.
- Tune the decision threshold for an acceptable false-positive rate — in a data
  center a false thermal-shutdown alert is costly, so **recall vs. precision is a
  deliberate operational trade-off**, documented and reviewed.
- Feed alerts to the SOC / on-call with the top contributing features (from the
  feature-importance output) so responders can triage quickly.

## Defense-in-depth summary
No single control is sufficient. Segmentation limits blast radius, hardening and
patching shrink the attack surface, signed telemetry protects integrity, and the
AI layer provides behavioral detection for the manipulations that signature-based
tools miss.
