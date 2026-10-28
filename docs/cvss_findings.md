# Vulnerability Findings & CVSS v3.1 Scoring

Scenario findings for the data center threat model. The vectors are illustrative of the
vulnerability classes and have not been measured against a lab; re-derive them from your
own lab configuration. Vector strings follow the CVSS v3.1 base metric format, and the
base scores were computed with the official CVSS v3.1 equations.

| ID | Finding | CVSS v3.1 Vector | Base Score | Severity |
|----|---------|------------------|-----------|----------|
| F-01 | Unauthenticated RCE in monitoring tool (Cacti-class, CVE-2022-46169) | `AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H` | 9.8 | Critical |
| F-02 | Plaintext credentials in config/environment files | `AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:L/A:N` | 6.1 | Medium |
| F-03 | Management interface (IPMI/BMC) reachable from IT segment, weak/default creds | `AV:A/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H` | 9.6 | Critical |
| F-04 | UPS/HVAC telemetry accepts unauthenticated/unsigned control writes | `AV:A/AC:L/PR:N/UI:N/S:C/C:L/I:H/A:H` | 9.6 | Critical |
| F-05 | No network segmentation between IT monitoring and OT/BMS subnets | `AV:A/AC:L/PR:L/UI:N/S:C/C:L/I:H/A:H` | 8.9 | High |

**Severity bands (CVSS v3.1):** None 0.0 · Low 0.1–3.9 · Medium 4.0–6.9 ·
High 7.0–8.9 · Critical 9.0–10.0.

## How to reproduce a vector string
Use the FIRST CVSS v3.1 calculator (`first.org/cvss/calculator/3.1`) and record:
- **AV** Attack Vector (N network / A adjacent / L local / P physical)
- **AC** Attack Complexity, **PR** Privileges Required, **UI** User Interaction
- **S** Scope (U unchanged / C changed — set **C** when a breach of one component
  impacts others, e.g. IT→OT crossover: this is why F-03/F-04/F-05 score high)
- **C/I/A** Confidentiality / Integrity / Availability impact

Note how **Scope: Changed** + high **Availability** impact drives the OT findings
into the Critical band — that is the quantitative argument for IT/OT segmentation.
