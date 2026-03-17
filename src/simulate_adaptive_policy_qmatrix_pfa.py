from __future__ import annotations

import argparse
from pathlib import Path

from policy_suite_common import run_policy_suite
from qmatrix_common import load_json


DEFAULT_CONFIG_PATH = Path("config/phase1_adaptive_policy_model2_qmatrix_pfa.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline adaptive policy suite for explicit Q-matrix PFA/R-PFA models.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_policy_suite(load_json(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
