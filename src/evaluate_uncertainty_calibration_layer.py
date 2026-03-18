from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from policy_suite_common import write_json
from qmatrix_common import load_json


DEFAULT_CONFIG_PATH = Path("config/phase1_uncertainty_calibration_layer.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Model 2 calibration against a Model 2 + Model 3 uncertainty calibration layer."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def stable_student_bucket(student_id: str, *, modulo: int = 100) -> int:
    digest = hashlib.sha1(student_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def load_numeric_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in frame.columns:
        if column in {"student_id", "actual_item_id", "policy_name", "selected_policy_name", "route_reason", "track", "model_name", "history_mode"}:
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


def fit_logistic_calibrator(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2_penalty: float,
) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def objective(beta: np.ndarray) -> float:
        linear = X @ beta
        p = expit(linear)
        ll = y * np.log(np.clip(p, 1e-6, 1.0 - 1e-6)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-6, 1.0 - 1e-6))
        penalty = l2_penalty * float(np.sum(beta[1:] ** 2))
        return -float(np.sum(ll)) + penalty

    result = minimize(objective, x0=np.zeros(X.shape[1], dtype=np.float64), method="BFGS")
    beta = result.x.astype(np.float64)
    return beta, float(result.fun)


def fit_calibration_summary(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=np.float64)
    z = logit(p)
    X = np.column_stack([np.ones(len(z), dtype=np.float64), z])
    beta, _ = fit_logistic_calibrator(X, y, l2_penalty=0.0)
    return float(beta[0]), float(beta[1])


def summarize_predictions(frame: pd.DataFrame, probability_column: str) -> dict[str, float | int]:
    y = frame["actual_correct"].to_numpy(dtype=np.float64)
    p = np.clip(frame[probability_column].to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
    intercept, slope = fit_calibration_summary(y, p)
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


def build_context_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    early = frame["eval_step"].to_numpy(dtype=np.float64) <= 5
    low_proficiency = frame["lower_predicted_proficiency"].to_numpy(dtype=np.float64) == 1
    high_failure = frame["high_recent_failure_context"].to_numpy(dtype=np.float64) == 1
    high_friction = frame["high_friction_context"].to_numpy(dtype=np.float64) == 1
    review_due = frame["due_review_available"].to_numpy(dtype=np.float64) == 1
    confidence_trigger = early | low_proficiency | high_failure | high_friction
    balanced_default = (~review_due) & (~confidence_trigger)
    return {
        "all_rows": np.ones(len(frame), dtype=bool),
        "early_steps_1_5": early,
        "confidence_trigger_context": confidence_trigger,
        "balanced_default_context": balanced_default,
        "review_due_context": review_due,
        "low_predicted_proficiency": low_proficiency,
        "high_recent_failure": high_failure,
        "high_friction": high_friction,
    }


def build_band_masks(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "harder_band_0p55_0p65": (probabilities >= 0.55) & (probabilities <= 0.65),
        "balanced_band_0p65_0p80": (probabilities >= 0.65) & (probabilities <= 0.80),
        "confidence_band_0p80_0p90": (probabilities >= 0.80) & (probabilities <= 0.90),
    }


def band_summary(frame: pd.DataFrame, probability_column: str) -> dict[str, dict[str, float | int]]:
    probabilities = frame[probability_column].to_numpy(dtype=np.float64)
    masks = build_band_masks(probabilities)
    summaries: dict[str, dict[str, float | int]] = {}
    for band_name, mask in masks.items():
        subset = frame.loc[mask].copy()
        if subset.empty:
            summaries[band_name] = {"n": 0}
            continue
        summaries[band_name] = {
            "n": int(len(subset)),
            "students": int(subset["student_id"].nunique()),
            "mean_probability": float(subset[probability_column].mean()),
            "observed_rate": float(subset["actual_correct"].mean()),
            "absolute_alignment_gap": float(abs(subset[probability_column].mean() - subset["actual_correct"].mean())),
        }
    return summaries


def standardize(series: pd.Series, *, mean_value: float, std_value: float) -> np.ndarray:
    return ((series.to_numpy(dtype=np.float64) - mean_value) / std_value).astype(np.float64)


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    rows = load_numeric_csv(Path(config["hybrid_rows_path"]))
    if rows["attempt_id"].duplicated().any():
        raise ValueError("Expected one row per attempt_id in the hybrid rows file.")

    student_ids = sorted(rows["student_id"].unique())
    calibration_share = float(config.get("calibration_student_share", 0.5))
    calibration_cutoff = int(round(calibration_share * 100.0))
    calibration_students = {
        student_id for student_id in student_ids if stable_student_bucket(student_id, modulo=100) < calibration_cutoff
    }
    rows["student_split"] = np.where(rows["student_id"].isin(calibration_students), "calibration", "evaluation")

    calibration_rows = rows.loc[rows["student_split"] == "calibration"].copy()
    evaluation_rows = rows.loc[rows["student_split"] == "evaluation"].copy()

    probability = np.clip(rows["actual_next_probability"].to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
    rows["model2_logit"] = logit(probability)
    calibration_rows["model2_logit"] = rows.loc[calibration_rows.index, "model2_logit"].to_numpy(dtype=np.float64)
    evaluation_rows["model2_logit"] = rows.loc[evaluation_rows.index, "model2_logit"].to_numpy(dtype=np.float64)

    uncertainty_mean = float(calibration_rows["uncertainty_sd"].mean())
    uncertainty_std = float(calibration_rows["uncertainty_sd"].std(ddof=0))
    uncertainty_std = max(uncertainty_std, 1e-6)
    rows["uncertainty_sd_z"] = standardize(rows["uncertainty_sd"], mean_value=uncertainty_mean, std_value=uncertainty_std)
    calibration_rows["uncertainty_sd_z"] = rows.loc[calibration_rows.index, "uncertainty_sd_z"].to_numpy(dtype=np.float64)
    evaluation_rows["uncertainty_sd_z"] = rows.loc[evaluation_rows.index, "uncertainty_sd_z"].to_numpy(dtype=np.float64)

    uncertainty_quantiles = calibration_rows["uncertainty_sd_z"].quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy(dtype=np.float64)
    uncertainty_low_cut = float(uncertainty_quantiles[0])
    uncertainty_high_cut = float(uncertainty_quantiles[1])
    for frame in (rows, calibration_rows, evaluation_rows):
        frame["uncertainty_band_low"] = (frame["uncertainty_sd_z"] <= uncertainty_low_cut).astype(np.int64)
        frame["uncertainty_band_mid"] = (
            (frame["uncertainty_sd_z"] > uncertainty_low_cut) & (frame["uncertainty_sd_z"] <= uncertainty_high_cut)
        ).astype(np.int64)
        frame["uncertainty_band_high"] = (frame["uncertainty_sd_z"] > uncertainty_high_cut).astype(np.int64)

    rows["early_steps_1_5"] = (rows["eval_step"] <= 5).astype(np.int64)
    calibration_rows["early_steps_1_5"] = rows.loc[calibration_rows.index, "early_steps_1_5"].to_numpy(dtype=np.int64)
    evaluation_rows["early_steps_1_5"] = rows.loc[evaluation_rows.index, "early_steps_1_5"].to_numpy(dtype=np.int64)

    context_columns = [
        "early_steps_1_5",
        "due_review_available",
        "high_recent_failure_context",
        "high_friction_context",
        "lower_predicted_proficiency",
    ]

    def design_matrix(frame: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, list[str]]:
        intercept = np.ones(len(frame), dtype=np.float64)
        base_logit = frame["model2_logit"].to_numpy(dtype=np.float64)
        columns = [intercept, base_logit]
        names = ["intercept", "model2_logit"]

        if feature_set in {"context", "uncertainty"}:
            for column in context_columns:
                columns.append(frame[column].to_numpy(dtype=np.float64))
                names.append(column)

        if feature_set == "uncertainty":
            low_band = frame["uncertainty_band_low"].to_numpy(dtype=np.float64)
            mid_band = frame["uncertainty_band_mid"].to_numpy(dtype=np.float64)
            high_band = frame["uncertainty_band_high"].to_numpy(dtype=np.float64)
            columns.append(mid_band)
            names.append("uncertainty_band_mid")
            columns.append(high_band)
            names.append("uncertainty_band_high")
            columns.append(base_logit * low_band)
            names.append("model2_logit_x_uncertainty_band_low")
            columns.append(base_logit * mid_band)
            names.append("model2_logit_x_uncertainty_band_mid")
            columns.append(base_logit * high_band)
            names.append("model2_logit_x_uncertainty_band_high")

        X = np.column_stack(columns).astype(np.float64)
        return X, names

    y_cal = calibration_rows["actual_correct"].to_numpy(dtype=np.float64)
    y_eval = evaluation_rows["actual_correct"].to_numpy(dtype=np.float64)
    l2_penalty = float(config.get("l2_penalty", 0.25))

    method_specs = {
        "model2_raw": None,
        "model2_platt": "platt",
        "model2_context_calibrated": "context",
        "model2_plus_model3_uncertainty": "uncertainty",
    }

    evaluation_predictions: dict[str, np.ndarray] = {
        "model2_raw": np.clip(evaluation_rows["actual_next_probability"].to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
    }
    coefficient_tables: dict[str, dict[str, float]] = {}

    for method_name, feature_set in method_specs.items():
        if feature_set is None:
            continue
        X_cal, feature_names = design_matrix(calibration_rows, feature_set)
        beta, objective_value = fit_logistic_calibrator(X_cal, y_cal, l2_penalty=l2_penalty)
        X_eval, _ = design_matrix(evaluation_rows, feature_set)
        evaluation_predictions[method_name] = np.clip(expit(X_eval @ beta), 1e-6, 1.0 - 1e-6)
        coefficient_tables[method_name] = {name: float(value) for name, value in zip(feature_names, beta)}
        coefficient_tables[method_name]["objective_value"] = objective_value

    evaluation_frame = evaluation_rows.loc[:, [
        "student_id",
        "attempt_id",
        "actual_item_id",
        "actual_correct",
        "eval_step",
        "due_review_available",
        "high_recent_failure_context",
        "high_friction_context",
        "lower_predicted_proficiency",
        "uncertainty_sd",
    ]].copy()
    for method_name, probabilities in evaluation_predictions.items():
        evaluation_frame[method_name] = probabilities

    context_masks = build_context_masks(evaluation_frame)
    summary: dict[str, object] = {
        "calibration_student_share": calibration_share,
        "calibration_students": int(len(calibration_students)),
        "evaluation_students": int(evaluation_rows["student_id"].nunique()),
        "calibration_rows": int(len(calibration_rows)),
        "evaluation_rows": int(len(evaluation_rows)),
        "uncertainty_standardization": {
            "mean": uncertainty_mean,
            "std": uncertainty_std,
        },
        "uncertainty_band_thresholds": {
            "low_cut": uncertainty_low_cut,
            "high_cut": uncertainty_high_cut,
        },
        "methods": {},
        "contexts": {},
        "bands": {},
    }
    comparison_rows: list[dict[str, object]] = []

    for method_name in method_specs:
        summary["methods"][method_name] = summarize_predictions(evaluation_frame, method_name)
        summary["bands"][method_name] = band_summary(evaluation_frame, method_name)
        if method_name in coefficient_tables:
            summary["methods"][method_name]["coefficients"] = coefficient_tables[method_name]

    for context_name, mask in context_masks.items():
        subset = evaluation_frame.loc[mask].copy()
        if len(subset) < int(config.get("min_rows_per_context", 50)):
            continue
        summary["contexts"][context_name] = {}
        for method_name in method_specs:
            summary["contexts"][context_name][method_name] = summarize_predictions(subset, method_name)
        comparison_rows.append(
            {
                "context_name": context_name,
                "n": int(len(subset)),
                "model2_raw_log_loss": float(summary["contexts"][context_name]["model2_raw"]["log_loss"]),
                "model2_platt_log_loss": float(summary["contexts"][context_name]["model2_platt"]["log_loss"]),
                "context_calibrated_log_loss": float(summary["contexts"][context_name]["model2_context_calibrated"]["log_loss"]),
                "uncertainty_calibrated_log_loss": float(summary["contexts"][context_name]["model2_plus_model3_uncertainty"]["log_loss"]),
                "delta_uncertainty_minus_context_log_loss": float(
                    summary["contexts"][context_name]["model2_plus_model3_uncertainty"]["log_loss"]
                    - summary["contexts"][context_name]["model2_context_calibrated"]["log_loss"]
                ),
                "model2_raw_brier": float(summary["contexts"][context_name]["model2_raw"]["brier"]),
                "model2_platt_brier": float(summary["contexts"][context_name]["model2_platt"]["brier"]),
                "context_calibrated_brier": float(summary["contexts"][context_name]["model2_context_calibrated"]["brier"]),
                "uncertainty_calibrated_brier": float(summary["contexts"][context_name]["model2_plus_model3_uncertainty"]["brier"]),
                "delta_uncertainty_minus_context_brier": float(
                    summary["contexts"][context_name]["model2_plus_model3_uncertainty"]["brier"]
                    - summary["contexts"][context_name]["model2_context_calibrated"]["brier"]
                ),
                "model2_raw_calibration_slope": float(summary["contexts"][context_name]["model2_raw"]["calibration_slope"]),
                "model2_platt_calibration_slope": float(summary["contexts"][context_name]["model2_platt"]["calibration_slope"]),
                "context_calibrated_slope": float(summary["contexts"][context_name]["model2_context_calibrated"]["calibration_slope"]),
                "uncertainty_calibrated_slope": float(summary["contexts"][context_name]["model2_plus_model3_uncertainty"]["calibration_slope"]),
                "delta_uncertainty_minus_context_slope_distance_to_1": float(
                    abs(summary["contexts"][context_name]["model2_plus_model3_uncertainty"]["calibration_slope"] - 1.0)
                    - abs(summary["contexts"][context_name]["model2_context_calibrated"]["calibration_slope"] - 1.0)
                ),
            }
        )

    evaluation_output_path = Path(config["evaluation_rows_output_path"])
    evaluation_output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_frame.to_csv(evaluation_output_path, index=False)

    summary_output_path = Path(config["summary_output_path"])
    write_json(summary_output_path, summary)

    comparison_output_path = Path(config["comparison_output_path"])
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(comparison_output_path, index=False)

    print(f"Saved uncertainty calibration summary to {summary_output_path}")
    print(f"Saved uncertainty calibration comparison to {comparison_output_path}")
    print(f"Saved evaluation rows to {evaluation_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
