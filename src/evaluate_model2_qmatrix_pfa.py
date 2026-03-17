from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from kc_history_common import resolve_history_value_columns
from qmatrix_common import QMatrixContext, ensure_parent, load_json, load_trials
from qmatrix_pfa_common import (
    build_feature_matrix,
    load_attempt_kc_long_pfa,
    prepare_attempt_kc_long_for_history,
)


DEFAULT_CONFIG_PATH = Path("config/phase1_multikc_qmatrix_pfa_model2_evaluate.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate explicit Q-matrix PFA Model 2.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def calibration_intercept_slope(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    x = logit(prob)

    def objective(params: np.ndarray) -> float:
        linear = params[0] + params[1] * x
        pred = expit(linear)
        pred = np.clip(pred, 1e-9, 1 - 1e-9)
        return -np.sum(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred))

    def gradient(params: np.ndarray) -> np.ndarray:
        linear = params[0] + params[1] * x
        pred = expit(linear)
        error = pred - y_true
        return np.array([error.sum(), np.sum(error * x)], dtype=float)

    result = minimize(objective, x0=np.array([0.0, 1.0]), jac=gradient, method="BFGS")
    if result.success and np.all(np.isfinite(result.x)):
        return float(result.x[0]), float(result.x[1])

    fallback = minimize(objective, x0=np.array([0.0, 1.0]), method="L-BFGS-B")
    if fallback.success and np.all(np.isfinite(fallback.x)):
        return float(fallback.x[0]), float(fallback.x[1])

    fallback = minimize(objective, x0=np.array([0.0, 1.0]), method="Powell")
    if fallback.success and np.all(np.isfinite(fallback.x)):
        return float(fallback.x[0]), float(fallback.x[1])

    return float("nan"), float("nan")


def overall_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    metrics = {
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "brier_score": float(np.mean((prob - y_true) ** 2)),
        "accuracy": float(accuracy_score(y_true, prob >= 0.5)),
    }
    metrics["auc"] = float(roc_auc_score(y_true, prob)) if np.unique(y_true).size > 1 else float("nan")
    intercept, slope = calibration_intercept_slope(y_true, prob)
    metrics["calibration_intercept"] = intercept
    metrics["calibration_slope"] = slope
    return metrics


def calibration_table(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"correct": y_true, "predicted_probability": prob})
    df["bin"] = pd.qcut(df["predicted_probability"], q=n_bins, duplicates="drop")
    table = (
        df.groupby("bin", observed=True)
        .agg(
            count=("correct", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_rate=("correct", "mean"),
        )
        .reset_index()
    )
    table["bin"] = table["bin"].astype("string")
    return table


def calibration_plot(table: pd.DataFrame, output_path: Path, title: str) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.plot(table["mean_predicted_probability"], table["observed_rate"], marker="o", linewidth=1.5, color="#1f77b4")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed correctness rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def learner_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for student_id, group in df.groupby("student_id", sort=True):
        metrics = overall_metrics(group["correct"].to_numpy(), group["predicted_probability"].to_numpy())
        rows.append({"student_id": student_id, "n_rows": int(len(group)), **metrics})
    return pd.DataFrame(rows)


def load_context(posterior_npz: np.lib.npyio.NpzFile) -> QMatrixContext:
    student_levels = [str(value) for value in posterior_npz["student_levels"].tolist()]
    item_levels = [str(value) for value in posterior_npz["item_levels"].tolist()]
    kc_levels = [str(value) for value in posterior_npz["kc_levels"].tolist()]
    return QMatrixContext(
        student_levels=student_levels,
        item_levels=item_levels,
        kc_levels=kc_levels,
        student_lookup={value: index for index, value in enumerate(student_levels)},
        item_lookup={value: index for index, value in enumerate(item_levels)},
        kc_lookup={value: index for index, value in enumerate(kc_levels)},
    )


def predict_probabilities(
    posterior: np.lib.npyio.NpzFile,
    context: QMatrixContext,
    eval_df: pd.DataFrame,
    attempt_kc_long: pd.DataFrame,
    *,
    batch_size: int,
    success_value_column: str,
    failure_value_column: str,
) -> np.ndarray:
    x_kc_base = build_feature_matrix(eval_df, attempt_kc_long, context, value_column="kc_base_indicator")
    x_kc_success = build_feature_matrix(eval_df, attempt_kc_long, context, value_column=success_value_column)
    x_kc_failure = build_feature_matrix(eval_df, attempt_kc_long, context, value_column=failure_value_column)
    x_kc_practice = build_feature_matrix(eval_df, attempt_kc_long, context, value_column="kc_practice_component")
    practice_total = x_kc_practice.sum(axis=1).astype("float64")

    intercept = posterior["Intercept"].astype("float64")
    student_intercept = posterior["student_intercept"].astype("float64")
    student_slope = posterior["student_slope"].astype("float64")
    item_effect = posterior["item_effect"].astype("float64")
    kc_intercept = posterior["kc_intercept"].astype("float64")
    kc_success = posterior["kc_success"].astype("float64")
    kc_failure = posterior["kc_failure"].astype("float64")

    student_idx = eval_df["student_id"].map(context.student_lookup).to_numpy(dtype="int64")
    item_idx = eval_df["item_id"].map(context.item_lookup).to_numpy(dtype="int64")

    n_rows = len(eval_df)
    probabilities = np.empty(n_rows, dtype="float64")
    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        student_term = student_intercept[:, student_idx[start:stop]].T
        slope_term = student_slope[:, student_idx[start:stop]].T * practice_total[start:stop, None]
        item_term = item_effect[:, item_idx[start:stop]].T
        kc_base_term = x_kc_base[start:stop].astype("float64") @ kc_intercept.T
        kc_success_term = x_kc_success[start:stop].astype("float64") @ kc_success.T
        kc_failure_term = x_kc_failure[start:stop].astype("float64") @ kc_failure.T
        linear = intercept[None, :] + student_term + slope_term + item_term + kc_base_term + kc_success_term + kc_failure_term
        probabilities[start:stop] = expit(linear).mean(axis=1)
        print(f"Predicted rows {start + 1}-{stop} of {n_rows}")

    return np.clip(probabilities, 1e-6, 1 - 1e-6)


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    trials = load_trials(Path(config["processed_trials_path"]))
    posterior = np.load(Path(config["posterior_draws_path"]), allow_pickle=True)
    context = load_context(posterior)
    history_mode = str(config.get("history_mode", posterior["history_mode"].reshape(-1)[0] if "history_mode" in posterior.files else "pfa")).lower()
    decay_alpha = float(config.get("decay_alpha", np.asarray(posterior["decay_alpha"]).reshape(-1)[0] if "decay_alpha" in posterior.files else 1.0))
    due_review_hours = float(config.get("due_review_hours", 48.0))
    attempt_kc_long = prepare_attempt_kc_long_for_history(
        load_attempt_kc_long_pfa(Path(config["attempt_kc_long_path"])),
        history_mode=history_mode,
        decay_alpha=decay_alpha,
        due_review_hours=due_review_hours,
    )
    success_value_column, failure_value_column = resolve_history_value_columns(history_mode)

    eval_df = trials.loc[trials["split"] == str(config.get("evaluate_split", "test"))].copy()
    if bool(config.get("primary_eval_only", True)):
        eval_df = eval_df.loc[eval_df["primary_eval_eligible"] == 1].copy()
    eval_df = eval_df.reset_index(drop=True)

    probabilities = predict_probabilities(
        posterior,
        context,
        eval_df,
        attempt_kc_long,
        batch_size=int(config.get("prediction_batch_size", 1000)),
        success_value_column=success_value_column,
        failure_value_column=failure_value_column,
    )

    eval_df = eval_df.copy()
    eval_df["predicted_probability"] = probabilities
    eval_df["track"] = str(config.get("track_name", "phase1_multikc_qmatrix_pfa"))
    eval_df["model_name"] = f"model2_qmatrix_{history_mode}"

    metrics = overall_metrics(eval_df["correct"].to_numpy(), probabilities)
    overall_path = Path(config["overall_metrics_path"])
    ensure_parent(overall_path)
    pd.DataFrame([metrics]).to_csv(overall_path, index=False)

    learner_path = Path(config["learner_metrics_path"])
    ensure_parent(learner_path)
    learner_metrics(eval_df).to_csv(learner_path, index=False)

    cal_table = calibration_table(eval_df["correct"].to_numpy(), probabilities)
    cal_table_path = Path(config["calibration_table_path"])
    ensure_parent(cal_table_path)
    cal_table.to_csv(cal_table_path, index=False)
    calibration_plot(cal_table, Path(config["calibration_figure_path"]), "Explicit Q-matrix PFA Model 2 Calibration")

    row_predictions_path = Path(config["row_predictions_path"])
    ensure_parent(row_predictions_path)
    eval_df[
        [
            "attempt_id",
            "student_id",
            "item_id",
            "correct",
            "trial_index_within_student",
            "practice_feature",
            "kc_count",
            "predicted_probability",
            "track",
            "model_name",
        ]
    ].to_csv(row_predictions_path, index=False)

    summary = {
        "evaluation_rows": int(len(eval_df)),
        "evaluation_students": int(eval_df["student_id"].nunique()),
        "evaluate_split": str(config.get("evaluate_split", "test")),
        "evaluation_mode": "track_a",
        "history_mode": history_mode,
        "decay_alpha": decay_alpha,
        "primary_eval_only": bool(config.get("primary_eval_only", True)),
        "seen_item_rows": int(len(eval_df)),
        "new_item_rows": 0,
        "metrics": metrics,
        "overall_metrics_path": str(overall_path),
        "learner_metrics_path": str(learner_path),
        "calibration_table_path": str(cal_table_path),
        "calibration_figure_path": str(config["calibration_figure_path"]),
        "row_predictions_path": str(row_predictions_path),
        "track": str(config.get("track_name", "phase1_multikc_qmatrix_pfa")),
    }
    write_json(Path(config["evaluation_summary_path"]), summary)

    print(f"Saved overall metrics to {overall_path}")
    print(f"Saved learner metrics to {learner_path}")
    print(f"Saved calibration outputs to {cal_table_path} and {config['calibration_figure_path']}")
    print(f"Saved row-level predictions to {row_predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
