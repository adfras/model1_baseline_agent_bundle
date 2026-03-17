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

from model1_common import build_model, ensure_parent, load_json, load_trials, prepend_compiler_to_path


DEFAULT_CONFIG_PATH = Path("config/model1_evaluate.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the held-out performance of Model 1.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def default_row_predictions_path(config: dict) -> Path:
    configured = config.get("row_predictions_path")
    if configured is not None:
        return Path(configured)
    overall_path = Path(config["overall_metrics_path"])
    stem = overall_path.stem
    suffix = "_overall_metrics"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return overall_path.with_name(f"{stem}_row_predictions.csv")


def default_attempt_window_metrics_path(config: dict) -> Path | None:
    configured = config.get("attempt_window_metrics_path")
    if configured is not None:
        return Path(configured)
    attempt_windows = config.get("attempt_windows")
    if not attempt_windows:
        return None
    overall_path = Path(config["overall_metrics_path"])
    stem = overall_path.stem
    suffix = "_overall_metrics"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return overall_path.with_name(f"{stem}_attempt_window_metrics.csv")


def infer_track_name(config: dict) -> str:
    configured = config.get("track_name")
    if configured:
        return str(configured)
    processed_path = str(config.get("processed_trials_path", "")).lower()
    return "track_b" if "track_b" in processed_path else "track_a"


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
    ax.set_title("Model 1 Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def logistic_normal_mean(linear_mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(1.0 + (np.pi * variance / 8.0))
    return expit(linear_mean / scale)


def resolve_item_indices(
    eval_df: pd.DataFrame,
    item_levels: list[str],
    *,
    new_item_strategy: str,
    context_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    item_lookup = {item_id: index for index, item_id in enumerate(item_levels)}
    item_idx = eval_df["item_id"].map(item_lookup)
    unseen_mask = item_idx.isna().to_numpy(dtype=bool)
    if unseen_mask.any() and new_item_strategy != "zero_effect":
        unseen = eval_df.loc[unseen_mask, "item_id"].astype("string").unique().tolist()
        raise ValueError(f"Encountered unseen item ids during {context_label}: {unseen[:5]}")
    item_idx = item_idx.fillna(-1).to_numpy(dtype="int64")
    return item_idx, unseen_mask


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


def update_student_posterior(
    offset: np.ndarray,
    outcome: int,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    *,
    newton_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = prior_mean.copy()
    for _ in range(newton_steps):
        eta = np.clip(offset + alpha, -30.0, 30.0)
        prob = expit(eta)
        grad = -(alpha - prior_mean) / np.clip(prior_var, 1e-9, None) + outcome - prob
        hess = -1.0 / np.clip(prior_var, 1e-9, None) - prob * (1.0 - prob)
        alpha = alpha - grad / np.clip(hess, -np.inf, -1e-9)

    eta = np.clip(offset + alpha, -30.0, 30.0)
    prob = expit(eta)
    hess = -1.0 / np.clip(prior_var, 1e-9, None) - prob * (1.0 - prob)
    posterior_var = -1.0 / np.clip(hess, -np.inf, -1e-9)
    return alpha, posterior_var


def extract_online_posterior(idata, max_draws: int | None) -> dict[str, object]:
    posterior = idata.posterior
    intercept = posterior["Intercept"].stack(sample=("chain", "draw")).values.astype("float64")
    practice_beta = posterior["practice_feature"].stack(sample=("chain", "draw")).values.astype("float64")
    student_sigma = posterior["1|student_id_sigma"].stack(sample=("chain", "draw")).values.astype("float64")
    item_effect_da = posterior["1|item_id"].stack(sample=("chain", "draw")).transpose("sample", "item_id__factor_dim")
    item_effect = item_effect_da.values.astype("float64")
    item_levels = [str(value) for value in item_effect_da.coords["item_id__factor_dim"].values.tolist()]

    n_draws = intercept.shape[0]
    if max_draws is not None and 1 <= max_draws < n_draws:
        index = np.linspace(0, n_draws - 1, max_draws, dtype=int)
        intercept = intercept[index]
        practice_beta = practice_beta[index]
        student_sigma = student_sigma[index]
        item_effect = item_effect[index, :]

    return {
        "Intercept": intercept,
        "practice_feature": practice_beta,
        "student_sigma": student_sigma,
        "item_effect": item_effect,
        "item_levels": item_levels,
    }


def predict_probabilities_track_b_online(
    eval_df: pd.DataFrame,
    online_posterior: dict[str, object],
    *,
    newton_steps: int,
    new_item_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intercept = np.asarray(online_posterior["Intercept"], dtype="float64")
    practice_beta = np.asarray(online_posterior["practice_feature"], dtype="float64")
    student_sigma = np.asarray(online_posterior["student_sigma"], dtype="float64")
    item_effect = np.asarray(online_posterior["item_effect"], dtype="float64")
    item_idx, unseen_item_mask = resolve_item_indices(
        eval_df,
        online_posterior["item_levels"],
        new_item_strategy=new_item_strategy,
        context_label="online Track B evaluation",
    )

    ordered = (
        eval_df[["student_id", "trial_index_within_student", "attempt_id"]]
        .copy()
        .assign(row_index=np.arange(len(eval_df), dtype="int64"))
        .sort_values(["student_id", "trial_index_within_student", "attempt_id"], kind="mergesort")
    )
    ordered_indices = ordered["row_index"].to_numpy(dtype="int64")

    probabilities = np.empty(len(eval_df), dtype="float64")
    posterior_mean_before = np.empty(len(eval_df), dtype="float64")
    posterior_sd_before = np.empty(len(eval_df), dtype="float64")

    grouped = ordered.groupby("student_id", sort=False)["row_index"].apply(list)
    for student_id, row_indices in grouped.items():
        prior_mean = np.zeros_like(intercept)
        prior_var = student_sigma**2
        for step_index, row_index in enumerate(row_indices):
            item_column = item_idx[row_index]
            practice_value = float(eval_df.at[row_index, "practice_feature"])
            item_term = item_effect[:, item_column] if item_column >= 0 else 0.0
            offset = intercept + item_term + practice_beta * practice_value
            probabilities[row_index] = float(logistic_normal_mean(offset + prior_mean, prior_var).mean())
            posterior_mean_before[row_index] = float(prior_mean.mean())
            posterior_sd_before[row_index] = float(np.sqrt(prior_var).mean())

            outcome = int(eval_df.at[row_index, "correct"])
            prior_mean, prior_var = update_student_posterior(
                offset,
                outcome,
                prior_mean,
                prior_var,
                newton_steps=newton_steps,
            )

        print(f"Predicted student {student_id} with {len(row_indices)} online updates")

    return (
        np.clip(probabilities, 1e-6, 1 - 1e-6),
        posterior_mean_before,
        posterior_sd_before,
        unseen_item_mask.astype("int64"),
    )


def build_row_predictions(
    eval_df: pd.DataFrame,
    *,
    model_name: str,
    track_name: str,
    evaluate_split: str,
    evaluation_mode: str,
) -> pd.DataFrame:
    output = eval_df.copy()
    output["model_name"] = model_name
    output["track"] = track_name
    output["evaluate_split"] = evaluate_split
    output["evaluation_mode"] = evaluation_mode
    columns = [
        "attempt_id",
        "student_id",
        "item_id",
        "kc_id",
        "timestamp",
        "split",
        "correct",
        "predicted_probability",
        "trial_index_within_student",
        "overall_opportunity",
        "kc_opportunity",
        "practice_feature",
        "kc_practice_feature",
        "item_seen_in_train",
        "new_item_in_test",
        "primary_eval_eligible",
        "used_zero_item_effect",
        "online_student_effect_mean_before",
        "online_student_effect_sd_before",
        "model_name",
        "track",
        "evaluate_split",
        "evaluation_mode",
    ]
    keep = [column for column in columns if column in output.columns]
    return output[keep].copy()


def attempt_window_metrics(df: pd.DataFrame, windows: list[list[int]]) -> pd.DataFrame:
    rows = []
    for start, stop in windows:
        window_df = df.loc[df["trial_index_within_student"].between(int(start), int(stop))].copy()
        if window_df.empty:
            continue
        metrics = overall_metrics(
            window_df["correct"].to_numpy(dtype="int8"),
            window_df["predicted_probability"].to_numpy(dtype="float64"),
        )
        rows.append(
            {
                "attempt_start": int(start),
                "attempt_end": int(stop),
                "n_rows": int(len(window_df)),
                "n_students": int(window_df["student_id"].nunique()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


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
    evaluation_mode = str(config.get("evaluation_mode", "batch"))
    if evaluation_mode == "track_b_online":
        online_posterior = extract_online_posterior(idata, config.get("online_posterior_draws"))
        prob, student_effect_mean, student_effect_sd, used_zero_item_effect = predict_probabilities_track_b_online(
            eval_df,
            online_posterior,
            newton_steps=int(config.get("online_newton_steps", 5)),
            new_item_strategy=str(config.get("new_item_strategy", "error")),
        )
        eval_df["online_student_effect_mean_before"] = student_effect_mean
        eval_df["online_student_effect_sd_before"] = student_effect_sd
        eval_df["used_zero_item_effect"] = used_zero_item_effect
    else:
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
    row_predictions_path = default_row_predictions_path(config)
    attempt_window_metrics_path = default_attempt_window_metrics_path(config)
    ensure_parent(overall_metrics_path)
    ensure_parent(learner_metrics_path)
    ensure_parent(calibration_table_path)
    ensure_parent(row_predictions_path)

    overall_table.to_csv(overall_metrics_path, index=False)
    learner_table.to_csv(learner_metrics_path, index=False)
    calib_table.to_csv(calibration_table_path, index=False)
    calibration_plot(calib_table, calibration_figure_path)
    row_predictions = build_row_predictions(
        eval_df,
        model_name="model1",
        track_name=infer_track_name(config),
        evaluate_split=str(config["evaluate_split"]),
        evaluation_mode=evaluation_mode,
    )
    row_predictions.to_csv(row_predictions_path, index=False)

    window_table = None
    if attempt_window_metrics_path is not None:
        window_table = attempt_window_metrics(
            row_predictions,
            [list(window) for window in config.get("attempt_windows", [])],
        )
        ensure_parent(attempt_window_metrics_path)
        window_table.to_csv(attempt_window_metrics_path, index=False)

    summary = {
        "evaluation_rows": int(len(eval_df)),
        "evaluation_students": int(eval_df["student_id"].nunique()),
        "evaluate_split": config["evaluate_split"],
        "evaluation_mode": evaluation_mode,
        "primary_eval_only": bool(config.get("primary_eval_only", True)),
        "seen_item_rows": int((eval_df["new_item_in_test"] == 0).sum()) if "new_item_in_test" in eval_df.columns else None,
        "new_item_rows": int((eval_df["new_item_in_test"] == 1).sum()) if "new_item_in_test" in eval_df.columns else None,
        "metrics": overall,
        "overall_metrics_path": str(overall_metrics_path),
        "learner_metrics_path": str(learner_metrics_path),
        "calibration_table_path": str(calibration_table_path),
        "calibration_figure_path": str(calibration_figure_path),
        "row_predictions_path": str(row_predictions_path),
        "track": infer_track_name(config),
    }
    if "used_zero_item_effect" in eval_df.columns:
        summary["zero_effect_item_rows"] = int(eval_df["used_zero_item_effect"].sum())
    if attempt_window_metrics_path is not None:
        summary["attempt_window_metrics_path"] = str(attempt_window_metrics_path)
    write_json(Path(config["evaluation_summary_path"]), summary)

    print(f"Saved overall metrics to {overall_metrics_path}")
    print(f"Saved learner metrics to {learner_metrics_path}")
    print(f"Saved calibration outputs to {calibration_table_path} and {calibration_figure_path}")
    print(f"Saved row-level predictions to {row_predictions_path}")
    if attempt_window_metrics_path is not None:
        print(f"Saved attempt-window metrics to {attempt_window_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
