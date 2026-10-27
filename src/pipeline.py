"""Run the full pipeline: train -> evaluate -> explain.

Run:  python -m src.pipeline
"""

from __future__ import annotations

import argparse

from .config import load_config, setup_logging
from .evaluate import evaluate
from .explain import explain
from .train_models import train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    parser.add_argument("--skip-explain", action="store_true", help="Skip the SHAP step")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    metadata = train(cfg)
    evaluate(cfg)
    if cfg["explain"]["enabled"] and not args.skip_explain:
        explain(cfg)
    print(f"\nDone. Data source: {metadata['source']}.")


if __name__ == "__main__":
    main()
