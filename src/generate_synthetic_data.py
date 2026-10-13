"""
Synthetic data-center telemetry generator (offline fallback).

Produces a labelled CSV of simulated sensor + network metrics so the pipeline
can run with no internet access. Mirrors the kinds of signals a real data-center
anomaly detector would see:

    Temperature_C   - inlet/aisle temperature from HVAC sensors
    Power_Load_kW   - UPS / PDU reported load
    Failed_Logins   - failed auth attempts on management interfaces
    Packet_Rate     - network packets/sec on the monitoring segment

Attack rows model overheating (HVAC tamper), power spikes (UPS spoofing),
brute-force auth, and exfiltration/DoS packet floods.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def generate(n_samples: int = 2000, attack_fraction: float = 0.1,
             seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_attack = int(n_samples * attack_fraction)
    n_normal = n_samples - n_attack

    normal = pd.DataFrame({
        "Temperature_C": rng.normal(22.0, 1.5, n_normal),
        "Power_Load_kW": rng.normal(150.0, 10.0, n_normal),
        "Failed_Logins": rng.choice([0, 1], size=n_normal, p=[0.95, 0.05]),
        "Packet_Rate":   rng.normal(500, 50, n_normal),
        "label":         0,  # benign
    })

    attack = pd.DataFrame({
        "Temperature_C": rng.normal(38.5, 3.0, n_attack),   # overheating
        "Power_Load_kW": rng.normal(220.0, 25.0, n_attack),  # power spike / spoof
        "Failed_Logins": rng.choice([5, 10, 15], size=n_attack),  # brute force
        "Packet_Rate":   rng.normal(2500, 300, n_attack),    # DoS / exfiltration
        "label":         1,  # malicious
    })

    df = (pd.concat([normal, attack])
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True))
    return df


if __name__ == "__main__":
    out = generate()
    out.to_csv("datacenter_telemetry_dataset.csv", index=False)
    print(f"Wrote datacenter_telemetry_dataset.csv  ({len(out)} rows)")
    print(out.groupby("label").size().rename({0: "benign", 1: "malicious"}))
