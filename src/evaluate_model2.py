from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from model1_common import ensure_parent, load_json, load_trials, prepend_compiler_to_path
from model2_common import build_model


DEFAULT_CONFIG_PATH = Path("config/model2_evaluate.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the held-out performance of Model 2.")
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
    if not result.success:
        return float("nan"), float("nan")
    return float(result.x[0]), float(result.x[1])


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


def learner_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for student_id, group in df.groupby("student_id", sort=True):
        y_true = group["correct"].to_numpy()
        prob = group["predicted_probability"].to_numpy()
        metrics = overall_metrics(y_true, prob)
        rows.append(
            {
                "student_id": student_id,
                "n_rows": int(len(group)),
                "mean_correct": float(group["correct"].mean()),
                "mean_predicted_probability": float(group["predicted_probability"].mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


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


def calibration_plot(table: pd.DataFrame, output_path: Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.plot(
        table["mean_predicted_probability"],
        table["observed_rate"],
        marker="o",
        linewidth=1.5,
        color="#1f77b4",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed correctness rate")
    ax.set_title("Model 2 Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def predict_probabilities(
    model,
    idata,
    eval_df: pd.DataFrame,
    *,
    include_group_specific: bool,
    sample_new_groups: bool,
    batch_size: int,
) -> np.ndarray:
    n_rows = len(eval_df)
    probabilities = np.empty(n_rows, dtype=float)
    predictor_columns = ["correct", "practice_feature", "student_id", "item_id"]

    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        chunk = eval_df.iloc[start:stop][predictor_columns]
        predictions = model.predict(
            idata,
            data=chunk,
            inplace=False,
            kind="response_params",
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
        )
        probabilities[start:stop] = predictions.posterior["p"].mean(dim=("chain", "draw")).to_numpy()
        print(f"Predicted rows {start + 1}-{stop} of {n_rows}")
        del predictions
        gc.collect()

    return np.clip(probabilities, 1e-6, 1 - 1e-6)


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    prepend_compiler_to_path(config.get("compiler_bin_dir"))

    trials = load_trials(Path(config["processed_trials_path"]))
    train_df = trials.loc[trials["split"] == "train", ["correct", "practice_feature", "student_id", "item_id"]].copy()
    eval_df = trials.loc[trials["split"] == config["evaluate_split"]].copy()
    if config.get("primary_eval_only", True):
        eval_df = eval_df.loc[eval_df["primary_eval_eligible"] == 1].copy()
    eval_df = eval_df.reset_index(drop=True)

    model = build_model(train_df)
    idata = az.from_netcdf(Path(config["idata_path"]))
    batch_size = int(config.get("prediction_batch_size", 1000))
    if batch_size < 1:
        raise ValueError("prediction_batch_size must be at least 1.")
    prob = predict_probabilities(
        model,
        idata,
        eval_df,
        include_group_specific=bool(config.get("include_group_specific", True)),
        sample_new_groups=bool(config.get("sample_new_groups", False)),
        batch_size=batch_size,
    )
    eval_df["predicted_probability"] = prob

    y_true = eval_df["correct"].to_numpy()
    overall = overall_metrics(y_true, prob)
    overall_table = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in overall.items()]
    )

    learner_table = learner_metrics(eval_df[["student_id", "correct", "predicted_probability"]])
    calib_table = calibration_table(y_true, prob)

    overall_metrics_path = Path(config["overall_metrics_path"])
    learner_metrics_path = Path(config["learner_metrics_path"])
    calibration_table_path = Path(config["calibration_table_path"])
    calibration_figure_path = Path(config["calibration_figure_path"])
    ensure_parent(overall_metrics_path)
    ensure_parent(learner_metrics_path)
    ensure_parent(calibration_table_path)

    overall_table.to_csv(overall_metrics_path, index=False)
    learner_table.to_csv(learner_metrics_path, index=False)
    calib_table.to_csv(calibration_table_path, index=False)
    calibration_plot(calib_table, calibration_figure_path)

    summary = {
        "evaluation_rows": int(len(eval_df)),
        "evaluation_students": int(eval_df["student_id"].nunique()),
        "evaluate_split": config["evaluate_split"],
        "primary_eval_only": bool(config.get("primary_eval_only", True)),
        "metrics": overall,
        "overall_metrics_path": str(overall_metrics_path),
        "learner_metrics_path": str(learner_metrics_path),
        "calibration_table_path": str(calibration_table_path),
        "calibration_figure_path": str(calibration_figure_path),
    }
    write_json(Path(config["evaluation_summary_path"]), summary)

    print(f"Saved overall metrics to {overall_metrics_path}")
    print(f"Saved learner metrics to {learner_metrics_path}")
    print(f"Saved calibration outputs to {calibration_table_path} and {calibration_figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
