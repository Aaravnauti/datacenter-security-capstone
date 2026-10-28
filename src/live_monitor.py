"""Live alert replay: stream connections through the models and raise alerts.

Recorded connections (the held-out test set) are replayed one at a time, paced like
live traffic. Each one is scored by the saved Isolation Forest and Random Forest, and
an alert is raised, with the reasons, whenever the chosen strategy flags it.

This is a replay of recorded traffic, not a packet capture. In a real deployment a
flow collector would feed the same scoring step. Because the replayed connections are
labeled, the monitor can also report which alerts were real and which were false
alarms; a live system would not know that.

Writes to outputs/: live_alerts.jsonl (one JSON object per alert) and
live_summary.json.

Run:  python -m src.live_monitor
      python -m src.live_monitor --count 500 --attack-rate 0.1 --rate 25
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve_path, setup_logging
from .data_preprocessing import BENIGN_LABEL, load_replay_frame
from .predict import predict_telemetry
from .thresholds import STRATEGIES
from .train_models import load_artifacts

logger = logging.getLogger(__name__)

ALERTS_FILE = "live_alerts.jsonl"
SUMMARY_FILE = "live_summary.json"


def build_stream(
    frame: pd.DataFrame, count: int, attack_fraction: float, seed: int
) -> pd.DataFrame:
    """Sample a traffic mix with the requested share of attacks, in random order.

    The test set is about 57% attacks, far more than real traffic. Sampling a lower
    share gives a stream that looks like a network where attacks are rare.
    """
    rng = np.random.default_rng(seed)
    is_attack = (frame["attack_name"] != BENIGN_LABEL).to_numpy()
    n_attack = int(min(round(count * attack_fraction), is_attack.sum()))
    n_benign = int(min(count - n_attack, (~is_attack).sum()))
    attacks = rng.choice(np.flatnonzero(is_attack), n_attack, replace=False)
    benign = rng.choice(np.flatnonzero(~is_attack), n_benign, replace=False)
    order = rng.permutation(np.concatenate([attacks, benign]))
    return frame.iloc[order].reset_index(drop=True)


def _severity(result: dict) -> str:
    """High when both models agree, medium when only one flags the connection."""
    both = result["isolation_forest"]["flagged"] and result["random_forest"]["flagged"]
    return "high" if both else "medium"


def monitor(
    cfg: dict,
    count: int = 300,
    attack_fraction: float = 0.10,
    rate: float = 15.0,
    seed: int = 7,
    strategy: str | None = None,
    batch: int = 10,
    sleep=time.sleep,
) -> Iterator[dict]:
    """Replay connections and yield one event per connection.

    Args:
        count: number of connections to replay.
        attack_fraction: share of attacks in the replayed traffic.
        rate: connections per second. Zero or less disables pacing.
        strategy: isolation_forest, random_forest, either_flags or both_flag.
        batch: connections scored together. Pacing still releases them one by one.
        sleep: pacing function, replaceable for tests.

    Each event holds the connection's sequence number, replay time, verdict, whether
    it raised an alert, severity, the ground truth (replay only), the full scoring
    result, and running statistics.
    """
    art = load_artifacts(resolve_path(cfg["paths"]["artifacts_dir"]))
    synthetic = str(art["metadata"]["source"]).startswith("synthetic")
    frame = build_stream(load_replay_frame(cfg, synthetic), count, attack_fraction, seed)
    features = frame.drop(columns="attack_name")

    started = datetime.now()
    stats = {
        "processed": 0, "alerts": 0, "real_alerts": 0, "false_alarms": 0,
        "missed_attacks": 0, "attacks_seen": 0, "benign_seen": 0,
    }
    for lo in range(0, len(frame), batch):
        results = predict_telemetry(features.iloc[lo : lo + batch], cfg, strategy, art)
        for offset, result in enumerate(results):
            seq = lo + offset + 1
            attack_name = str(frame["attack_name"].iloc[lo + offset])
            is_attack = attack_name != BENIGN_LABEL
            alert = result["prediction"] == "malicious"

            stats["processed"] += 1
            stats["attacks_seen" if is_attack else "benign_seen"] += 1
            if alert:
                stats["alerts"] += 1
                stats["real_alerts" if is_attack else "false_alarms"] += 1
            elif is_attack:
                stats["missed_attacks"] += 1

            if rate > 0:
                sleep(1.0 / rate)
            yield {
                "seq": seq,
                "time": (started + timedelta(seconds=seq / rate if rate > 0 else 0)).isoformat(
                    timespec="seconds"
                ),
                "alert": alert,
                "severity": _severity(result) if alert else None,
                "attack_name": attack_name,
                "is_attack": is_attack,
                "result": result,
                "stats": dict(stats),
            }


def alert_record(event: dict) -> dict:
    """The compact JSON object written to live_alerts.jsonl for one alert."""
    r = event["result"]
    return {
        "seq": event["seq"],
        "time": event["time"],
        "severity": event["severity"],
        "strategy": r["strategy"],
        "isolation_forest_score": r["isolation_forest"]["score"],
        "random_forest_probability": r["random_forest"]["probability"],
        "reasons": r["top_drivers"],
        "replay_truth": {"is_attack": event["is_attack"], "attack_name": event["attack_name"]},
    }


def _jsonable(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def reasons_text(event: dict, k: int = 3) -> str:
    return ", ".join(
        f"{d['feature']}={d['value']}" for d in event["result"]["top_drivers"][:k]
    )


def summarize(stats: dict) -> dict:
    """Final statistics with recall and precision of the replayed alerts."""
    real, fa, miss = stats["real_alerts"], stats["false_alarms"], stats["missed_attacks"]
    return {
        **stats,
        "recall": round(real / (real + miss), 4) if real + miss else None,
        "precision": round(real / (real + fa), 4) if real + fa else None,
    }


def run(cfg: dict, quiet: bool = False, **kwargs) -> dict:
    """Run the monitor, print alerts as they happen, and write the output files."""
    outputs_dir = resolve_path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    alerts_path: Path = outputs_dir / ALERTS_FILE
    stats: dict = {}

    with alerts_path.open("w", encoding="utf-8") as sink:
        for event in monitor(cfg, **kwargs):
            stats = event["stats"]
            if not event["alert"]:
                continue
            sink.write(json.dumps(alert_record(event), default=_jsonable) + "\n")
            sink.flush()
            if not quiet:
                verdict = "REAL ATTACK" if event["is_attack"] else "FALSE ALARM"
                scores = event["result"]
                print(
                    f"[{event['time'][11:]}] ALERT #{stats['alerts']:<3} "
                    f"{event['severity'].upper():<6} conn {event['seq']:<4} "
                    f"IF {scores['isolation_forest']['score']:.3f} / "
                    f"RF {scores['random_forest']['probability']:.3f}  "
                    f"why: {reasons_text(event)}  "
                    f"[replay check: {verdict}"
                    f"{'' if not event['is_attack'] else ' - ' + event['attack_name']}]",
                    flush=True,
                )

    summary = summarize(stats)
    (outputs_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    parser.add_argument("--count", type=int, default=300, help="Connections to replay")
    parser.add_argument(
        "--attack-rate", type=float, default=0.10, help="Share of attacks in the traffic (0-1)"
    )
    parser.add_argument(
        "--rate", type=float, default=15.0, help="Connections per second (0 = as fast as possible)"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the traffic mix")
    parser.add_argument("--strategy", choices=STRATEGIES, default=None)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    print(
        f"Replaying {args.count} connections ({args.attack_rate:.0%} attacks) at "
        f"{args.rate:g}/s. Alerts appear below as they are raised.\n",
        flush=True,
    )
    s = run(
        cfg, count=args.count, attack_fraction=args.attack_rate, rate=args.rate,
        seed=args.seed, strategy=args.strategy,
    )
    print(
        f"\nDone. {s['processed']} connections ({s['attacks_seen']} attacks, "
        f"{s['benign_seen']} benign) -> {s['alerts']} alerts.\n"
        f"  Real attacks caught: {s['real_alerts']} of {s['attacks_seen']}"
        f" (recall {s['recall']})\n"
        f"  False alarms:        {s['false_alarms']} (alert precision {s['precision']})\n"
        f"  Missed attacks:      {s['missed_attacks']}\n"
        f"Alerts saved to outputs/{ALERTS_FILE}"
    )


if __name__ == "__main__":
    main()
