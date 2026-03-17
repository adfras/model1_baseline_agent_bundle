from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from model1_common import ensure_parent, load_json


DEFAULT_CONFIG_PATH = Path("config/model3_vs_model1_track_b_bootstrap.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap a paired student-level model comparison.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def student_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for student_id, group in df.groupby("student_id", sort=True):
        y_true = group["correct"].to_numpy(dtype="int8")
        prob = group["predicted_probability"].to_numpy(dtype="float64")
        rows.append(
            {
                "student_id": student_id,
                "n_rows": int(len(group)),
                "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
                "brier_score": float(np.mean((prob - y_true) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    draws = np.empty(n_bootstrap, dtype="float64")
    for index in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        draws[index] = float(values[sample_idx].mean())
    return (
        float(np.quantile(draws, 0.025)),
        float(np.mean(draws)),
        float(np.quantile(draws, 0.975)),
    )


def compare_window(
    baseline_df: pd.DataFrame,
    challenger_df: pd.DataFrame,
    *,
    attempt_start: int | None,
    attempt_end: int | None,
    n_bootstrap: int,
    random_seed: int,
) -> dict[str, float | int | str | None]:
    if attempt_start is None or attempt_end is None:
        baseline_window = baseline_df.copy()
        challenger_window = challenger_df.copy()
        label = "overall"
    else:
        baseline_window = baseline_df.loc[
            baseline_df["trial_index_within_student"].between(attempt_start, attempt_end)
        ].copy()
        challenger_window = challenger_df.loc[
            challenger_df["trial_index_within_student"].between(attempt_start, attempt_end)
        ].copy()
        label = f"attempts_{attempt_start}_{attempt_end}"

    baseline_student = student_metrics(baseline_window)
    challenger_student = student_metrics(challenger_window)
    merged = baseline_student.merge(
        challenger_student,
        on="student_id",
        suffixes=("_baseline", "_challenger"),
        how="inner",
    )
    merged["delta_log_loss"] = merged["log_loss_challenger"] - merged["log_loss_baseline"]
    merged["delta_brier_score"] = merged["brier_score_challenger"] - merged["brier_score_baseline"]

    rng = np.random.default_rng(random_seed)
    log_loss_ci_low, log_loss_boot_mean, log_loss_ci_high = bootstrap_mean(
        merged["delta_log_loss"].to_numpy(dtype="float64"),
        n_bootstrap,
        rng,
    )
    rng = np.random.default_rng(random_seed + 1)
    brier_ci_low, brier_boot_mean, brier_ci_high = bootstrap_mean(
        merged["delta_brier_score"].to_numpy(dtype="float64"),
        n_bootstrap,
        rng,
    )

    return {
        "window": label,
        "attempt_start": attempt_start,
        "attempt_end": attempt_end,
        "n_students": int(merged["student_id"].nunique()),
        "n_rows_baseline": int(len(baseline_window)),
        "n_rows_challenger": int(len(challenger_window)),
        "delta_log_loss_mean": float(merged["delta_log_loss"].mean()),
        "delta_log_loss_bootstrap_mean": log_loss_boot_mean,
        "delta_log_loss_ci_low": log_loss_ci_low,
        "delta_log_loss_ci_high": log_loss_ci_high,
        "delta_brier_mean": float(merged["delta_brier_score"].mean()),
        "delta_brier_bootstrap_mean": brier_boot_mean,
        "delta_brier_ci_low": brier_ci_low,
        "delta_brier_ci_high": brier_ci_high,
        "challenger_better_log_loss_share": float((merged["delta_log_loss"] < 0).mean()),
        "challenger_better_brier_share": float((merged["delta_brier_score"] < 0).mean()),
    }


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    baseline_df = pd.read_csv(config["baseline_row_predictions_path"])
    challenger_df = pd.read_csv(config["challenger_row_predictions_path"])

    merged_keys = ["attempt_id", "student_id", "item_id", "trial_index_within_student", "correct"]
    merged = baseline_df[merged_keys].merge(
        challenger_df[merged_keys],
        on=merged_keys,
        how="inner",
    )
    if len(merged) != len(baseline_df) or len(merged) != len(challenger_df):
        raise ValueError("Baseline and challenger row prediction files are not aligned on held-out attempts.")

    windows = [[None, None]]
    windows.extend([list(window) for window in config.get("attempt_windows", [])])
    rows = []
    for offset, (start, stop) in enumerate(windows):
        rows.append(
            compare_window(
                baseline_df,
                challenger_df,
                attempt_start=start,
                attempt_end=stop,
                n_bootstrap=int(config.get("n_bootstrap", 2000)),
                random_seed=int(config.get("random_seed", 20260330)) + offset * 10,
            )
        )

    result_table = pd.DataFrame(rows)
    output_csv_path = Path(config["output_csv_path"])
    ensure_parent(output_csv_path)
    result_table.to_csv(output_csv_path, index=False)

    summary = {
        "baseline_row_predictions_path": config["baseline_row_predictions_path"],
        "challenger_row_predictions_path": config["challenger_row_predictions_path"],
        "output_csv_path": str(output_csv_path),
        "n_bootstrap": int(config.get("n_bootstrap", 2000)),
        "random_seed": int(config.get("random_seed", 20260330)),
    }
    write_json(Path(config["output_summary_path"]), summary)

    print(f"Saved bootstrap comparison to {output_csv_path}")
    print(f"Saved comparison summary to {config['output_summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
