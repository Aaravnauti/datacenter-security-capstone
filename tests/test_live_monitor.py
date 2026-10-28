import json

import pandas as pd
import pytest

from src.config import resolve_path
from src.live_monitor import (
    ALERTS_FILE,
    SUMMARY_FILE,
    build_stream,
    monitor,
    run,
    summarize,
)
from src.train_models import train


@pytest.fixture
def trained(cfg):
    train(cfg)
    return cfg


def test_build_stream_honours_attack_share_and_is_deterministic():
    frame = pd.DataFrame(
        {"x": range(1000), "attack_name": ["normal"] * 600 + ["attack"] * 400}
    )

    stream = build_stream(frame, count=100, attack_fraction=0.2, seed=3)

    assert len(stream) == 100
    assert (stream["attack_name"] != "normal").sum() == 20
    assert stream.equals(build_stream(frame, count=100, attack_fraction=0.2, seed=3))
    assert not stream.equals(build_stream(frame, count=100, attack_fraction=0.2, seed=4))


def test_build_stream_never_asks_for_more_attacks_than_exist():
    frame = pd.DataFrame({"x": range(10), "attack_name": ["normal"] * 8 + ["attack"] * 2})
    stream = build_stream(frame, count=10, attack_fraction=0.9, seed=1)
    assert (stream["attack_name"] != "normal").sum() == 2


def test_monitor_yields_one_event_per_connection_with_running_stats(trained):
    events = list(monitor(trained, count=60, attack_fraction=0.3, rate=0, seed=5))

    assert [e["seq"] for e in events] == list(range(1, 61))
    final = events[-1]["stats"]
    assert final["processed"] == 60
    assert final["attacks_seen"] + final["benign_seen"] == 60
    # Every alert is either a real attack or a false alarm; every attack is caught or missed.
    assert final["alerts"] == final["real_alerts"] + final["false_alarms"]
    assert final["attacks_seen"] == final["real_alerts"] + final["missed_attacks"]
    # The running counters never go backwards.
    alerts = [e["stats"]["alerts"] for e in events]
    assert alerts == sorted(alerts)


def test_alert_events_carry_severity_and_reasons(trained):
    events = list(monitor(trained, count=60, attack_fraction=0.3, rate=0, seed=5))
    alerts = [e for e in events if e["alert"]]

    assert alerts, "the synthetic attacks should raise at least one alert"
    for event in alerts:
        assert event["severity"] in {"high", "medium"}
        assert event["result"]["top_drivers"]
    assert all(e["severity"] is None for e in events if not e["alert"])


def test_monitor_catches_the_synthetic_attacks(trained):
    final = list(monitor(trained, count=100, attack_fraction=0.2, rate=0, seed=2))[-1]["stats"]
    assert final["real_alerts"] / final["attacks_seen"] > 0.9


def test_monitor_paces_one_sleep_per_connection(trained):
    pauses = []
    list(monitor(trained, count=20, rate=10.0, seed=1, sleep=pauses.append))
    assert pauses == [pytest.approx(0.1)] * 20


def test_strategy_changes_which_connections_alert(trained):
    def alerts(strategy):
        events = monitor(trained, count=80, attack_fraction=0.25, rate=0, seed=9, strategy=strategy)
        return {e["seq"] for e in events if e["alert"]}

    either, both = alerts("either_flags"), alerts("both_flag")
    assert both <= either


def test_run_writes_alert_log_and_summary(trained):
    summary = run(trained, quiet=True, count=60, attack_fraction=0.3, rate=0, seed=5)
    outputs = resolve_path(trained["paths"]["outputs_dir"])

    lines = (outputs / ALERTS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == summary["alerts"]
    first = json.loads(lines[0])
    assert {"seq", "time", "severity", "reasons", "replay_truth"} <= set(first)
    assert json.loads((outputs / SUMMARY_FILE).read_text())["processed"] == 60


def test_summarize_reports_recall_and_precision():
    stats = {
        "processed": 100, "alerts": 12, "real_alerts": 8, "false_alarms": 4,
        "missed_attacks": 2, "attacks_seen": 10, "benign_seen": 90,
    }
    result = summarize(stats)
    assert result["recall"] == 0.8
    assert result["precision"] == pytest.approx(0.6667, abs=1e-4)


def test_summarize_handles_streams_with_no_alerts_or_attacks():
    empty = {
        "processed": 5, "alerts": 0, "real_alerts": 0, "false_alarms": 0,
        "missed_attacks": 0, "attacks_seen": 0, "benign_seen": 5,
    }
    assert summarize(empty)["recall"] is None
    assert summarize(empty)["precision"] is None
