from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from policy_suite_common import write_json
from qmatrix_common import load_json


DEFAULT_CONFIG_PATH = Path("config/phase1_policy_alignment_calibration.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Model 2 vs Model 3 calibration on logged actual-next items in policy-relevant contexts."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def load_numeric_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in frame.columns:
        if column in {"student_id", "actual_item_id", "track", "model_name", "history_mode", "policy_name"}:
            continue
        try:
            frame[column] = pd.to_numeric(frame[column])
        except (TypeError, ValueError):
            pass
    if "student_id" in frame.columns:
        frame["student_id"] = frame["student_id"].astype(str)
    if "actual_item_id" in frame.columns:
        frame["actual_item_id"] = frame["actual_item_id"].astype(str)
    return frame


def student_average(values: pd.Series, student_ids: pd.Series) -> float:
    grouped = pd.DataFrame({"student_id": student_ids, "value": values}).groupby("student_id", sort=False)["value"].mean()
    return float(grouped.mean()) if len(grouped) else float("nan")


def fit_calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=np.float64)
    z = logit(p)

    def objective(params: np.ndarray) -> float:
        intercept, slope = params
        mu = expit(intercept + slope * z)
        ll = y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)
        return -float(np.sum(ll))

    result = minimize(objective, x0=np.array([0.0, 1.0], dtype=np.float64), method="BFGS")
    intercept, slope = result.x
    return float(intercept), float(slope)


def summarize_context(frame: pd.DataFrame, *, probability_column: str) -> dict[str, float | int]:
    y = frame["actual_correct"].to_numpy(dtype=np.float64)
    p = np.clip(frame[probability_column].to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
    intercept, slope = fit_calibration(y, p)
    return {
        "n": int(len(frame)),
        "students": int(frame["student_id"].nunique()),
        "mean_probability": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    router_summary = load_json(Path(config["router_summary_path"]))
    thresholds = router_summary["selected_thresholds"]

    low_proficiency_threshold = float(thresholds["low_proficiency_threshold"])
    recent_failure_threshold = float(thresholds["recent_failure_threshold"])

    friction_rules = {
        "current": {
            "hint": float(config["current_hint_rate_threshold"]),
            "selection_change": float(config["current_selection_change_rate_threshold"]),
            "duration": float(config["current_duration_inflation_threshold"]),
        },
        "stricter": {
            "hint": float(config["stricter_hint_rate_threshold"]),
            "selection_change": float(config["stricter_selection_change_rate_threshold"]),
            "duration": float(config["stricter_duration_inflation_threshold"]),
        },
    }
    friction_rule = friction_rules[str(thresholds["friction_rule_name"])]

    base_rows = load_numeric_csv(Path(config["router_base_rows_path"]))
    model2_rows = load_numeric_csv(Path(config["model2_policy_rows_path"]))
    model3_rows = load_numeric_csv(Path(config["model3_policy_rows_path"]))

    model2_rows = model2_rows.loc[
        model2_rows["policy_name"] == str(config["reference_policy_name"]),
        ["student_id", "attempt_id", "actual_next_probability"],
    ].rename(columns={"actual_next_probability": "model2_actual_next_probability"})
    model3_rows = model3_rows.loc[
        model3_rows["policy_name"] == str(config["reference_policy_name"]),
        ["student_id", "attempt_id", "actual_next_probability"],
    ].rename(columns={"actual_next_probability": "model3_actual_next_probability"})

    rows = (
        base_rows.merge(model2_rows, on=["student_id", "attempt_id"], how="inner", validate="one_to_one")
        .merge(model3_rows, on=["student_id", "attempt_id"], how="inner", validate="one_to_one")
        .copy()
    )

    high_friction = (
        (rows["recent_hint_rate"] >= friction_rule["hint"])
        | (rows["recent_selection_change_rate"] >= friction_rule["selection_change"])
        | (rows["response_time_inflation"] >= friction_rule["duration"])
    )
    confidence_trigger = (
        (rows["eval_step"] <= int(thresholds["early_step_cutoff"]))
        | (rows["balanced_reference_probability"] <= low_proficiency_threshold)
        | (rows["recent_failure_total"] >= recent_failure_threshold)
        | high_friction
    )
    review_due = rows["due_review_available"] == 1
    balanced_default = (~review_due) & (~confidence_trigger)

    contexts = {
        "all_rows": np.ones(len(rows), dtype=bool),
        "early_steps_1_5": rows["eval_step"] <= 5,
        "confidence_trigger_context": confidence_trigger,
        "balanced_default_context": balanced_default,
        "review_due_context": review_due,
        "low_predicted_proficiency": rows["balanced_reference_probability"] <= low_proficiency_threshold,
        "high_recent_failure": rows["recent_failure_total"] >= recent_failure_threshold,
        "high_friction": high_friction,
    }

    summary: dict[str, dict] = {
        "reference_policy_name": str(config["reference_policy_name"]),
        "selected_router_thresholds": thresholds,
        "contexts": {},
    }
    comparison_rows: list[dict[str, object]] = []
    for context_name, mask in contexts.items():
        subset = rows.loc[mask].copy()
        if len(subset) < int(config.get("min_rows_per_context", 50)):
            continue
        model2_summary = summarize_context(subset, probability_column="model2_actual_next_probability")
        model3_summary = summarize_context(subset, probability_column="model3_actual_next_probability")
        delta = {
            "brier": float(model3_summary["brier"] - model2_summary["brier"]),
            "log_loss": float(model3_summary["log_loss"] - model2_summary["log_loss"]),
            "calibration_slope": float(model3_summary["calibration_slope"] - model2_summary["calibration_slope"]),
            "calibration_slope_distance_to_1": float(
                abs(model3_summary["calibration_slope"] - 1.0) - abs(model2_summary["calibration_slope"] - 1.0)
            ),
        }
        summary["contexts"][context_name] = {
            "model2": model2_summary,
            "model3": model3_summary,
            "delta_model3_minus_model2": delta,
        }
        comparison_rows.append(
            {
                "context_name": context_name,
                "n": int(model2_summary["n"]),
                "students": int(model2_summary["students"]),
                "model2_brier": float(model2_summary["brier"]),
                "model3_brier": float(model3_summary["brier"]),
                "delta_brier_model3_minus_model2": delta["brier"],
                "model2_log_loss": float(model2_summary["log_loss"]),
                "model3_log_loss": float(model3_summary["log_loss"]),
                "delta_log_loss_model3_minus_model2": delta["log_loss"],
                "model2_calibration_slope": float(model2_summary["calibration_slope"]),
                "model3_calibration_slope": float(model3_summary["calibration_slope"]),
                "delta_calibration_slope_model3_minus_model2": delta["calibration_slope"],
                "delta_slope_distance_to_1_model3_minus_model2": delta["calibration_slope_distance_to_1"],
            }
        )

    summary_output_path = Path(config["summary_output_path"])
    write_json(summary_output_path, summary)

    comparison_output_path = Path(config["comparison_output_path"])
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(comparison_output_path, index=False)

    print(f"Saved policy-alignment calibration summary to {summary_output_path}")
    print(f"Saved policy-alignment calibration comparison table to {comparison_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
