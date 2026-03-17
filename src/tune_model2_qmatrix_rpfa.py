from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from evaluate_model2_qmatrix_pfa import load_context, overall_metrics, predict_probabilities
from kc_history_common import resolve_history_value_columns
from qmatrix_pfa_common import (
    build_context,
    build_model2_qmatrix_pfa,
    ensure_parent,
    fit_model,
    load_attempt_kc_long_pfa,
    load_json,
    load_trials,
    prepare_attempt_kc_long_for_history,
    prepare_pfa_dataset,
    prepend_compiler_to_path,
    save_posterior_npz_pfa,
    summarize_vector_draws,
    summarize_vi_history,
)


DEFAULT_CONFIG_PATH = Path("config/phase1_multikc_qmatrix_rpfa_model2_tuning.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the recency alpha grid for explicit Q-matrix R-PFA Model 2.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def alpha_slug(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    prepend_compiler_to_path(config.get("compiler_bin_dir"))

    trials = load_trials(Path(config["processed_trials_path"]))
    base_attempt_kc_long = load_attempt_kc_long_pfa(Path(config["attempt_kc_long_path"]))
    train_df = trials.loc[trials["split"] == "train"].copy()
    eval_df = trials.loc[trials["split"] == str(config.get("evaluate_split", "test"))].copy()
    if bool(config.get("primary_eval_only", True)):
        eval_df = eval_df.loc[eval_df["primary_eval_eligible"] == 1].copy()
    eval_df = eval_df.reset_index(drop=True)

    history_mode = "rpfa"
    success_value_column, failure_value_column = resolve_history_value_columns(history_mode)
    due_review_hours = float(config.get("due_review_hours", 48.0))
    alpha_grid = [float(value) for value in config["alpha_grid"]]
    tie_margin = float(config.get("tie_margin", 0.0002))
    output_root = Path(config["output_root"])

    comparison_rows: list[dict] = []
    for alpha in alpha_grid:
        alpha_dir = output_root / f"alpha_{alpha_slug(alpha)}"
        posterior_draws_path = alpha_dir / "model2_posterior_draws.npz"
        posterior_summary_path = alpha_dir / "model2_posterior_summary.csv"
        kc_success_summary_path = alpha_dir / "model2_kc_success_summary.csv"
        kc_failure_summary_path = alpha_dir / "model2_kc_failure_summary.csv"
        student_slope_summary_path = alpha_dir / "model2_student_slope_summary.csv"
        fit_summary_path = alpha_dir / "model2_fit_summary.json"
        vi_history_path = alpha_dir / "model2_vi_history.csv"
        overall_metrics_path = alpha_dir / "model2_overall_metrics.csv"
        evaluation_summary_path = alpha_dir / "model2_evaluation_summary.json"

        attempt_kc_long = prepare_attempt_kc_long_for_history(
            base_attempt_kc_long,
            history_mode=history_mode,
            decay_alpha=alpha,
            due_review_hours=due_review_hours,
        )
        context = build_context(train_df, attempt_kc_long)
        train_data = prepare_pfa_dataset(
            trials,
            attempt_kc_long,
            context,
            split="train",
            success_value_column=success_value_column,
            failure_value_column=failure_value_column,
        )
        model = build_model2_qmatrix_pfa(train_data, context)

        started = time.perf_counter()
        idata, vi_losses = fit_model(
            model,
            inference_method=str(config["inference_method"]),
            vi_method=str(config.get("vi_method", "advi")),
            vi_iterations=int(config.get("vi_iterations", 20000)),
            posterior_draws=int(config["posterior_draws"]),
            random_seed=int(config["random_seed"]),
            draws=int(config.get("draws", 1000)),
            tune=int(config.get("tune", 1000)),
            chains=int(config.get("chains", 2)),
            cores=int(config.get("cores", 1)),
            target_accept=float(config.get("target_accept", 0.9)),
        )
        elapsed_seconds = time.perf_counter() - started

        posterior_summary = az.summary(idata, kind="stats")
        ensure_parent(posterior_summary_path)
        posterior_summary.to_csv(posterior_summary_path)

        save_posterior_npz_pfa(
            idata,
            posterior_draws_path,
            context,
            model_kind="model2",
            history_mode=history_mode,
            decay_alpha=alpha,
        )

        kc_success_summary = summarize_vector_draws(
            idata.posterior["kc_success"].stack(sample=("chain", "draw")).transpose("sample", "kc").values,
            context.kc_levels,
            value_name="kc_id",
        )
        ensure_parent(kc_success_summary_path)
        kc_success_summary.to_csv(kc_success_summary_path, index=False)

        kc_failure_summary = summarize_vector_draws(
            idata.posterior["kc_failure"].stack(sample=("chain", "draw")).transpose("sample", "kc").values,
            context.kc_levels,
            value_name="kc_id",
        )
        ensure_parent(kc_failure_summary_path)
        kc_failure_summary.to_csv(kc_failure_summary_path, index=False)

        student_slope_summary = summarize_vector_draws(
            idata.posterior["student_slope"].stack(sample=("chain", "draw")).transpose("sample", "student").values,
            context.student_levels,
            value_name="student_id",
        )
        ensure_parent(student_slope_summary_path)
        student_slope_summary.to_csv(student_slope_summary_path, index=False)

        vi_diagnostics = summarize_vi_history(vi_losses)
        if vi_losses is not None:
            vi_history = pd.DataFrame({"iteration": range(1, len(vi_losses) + 1), "loss": vi_losses})
            ensure_parent(vi_history_path)
            vi_history.to_csv(vi_history_path, index=False)

        fit_summary = {
            "model_kind": "explicit_qmatrix_rpfa_model2_tuning",
            "history_mode": history_mode,
            "decay_alpha": alpha,
            "inference_method": str(config["inference_method"]),
            "random_seed": int(config["random_seed"]),
            "train_rows": int(len(train_data.df)),
            "train_students": int(len(context.student_levels)),
            "train_items": int(len(context.item_levels)),
            "train_kcs": int(len(context.kc_levels)),
            "posterior_draws_path": str(posterior_draws_path),
            "posterior_summary_path": str(posterior_summary_path),
            "elapsed_seconds": elapsed_seconds,
            **vi_diagnostics,
        }
        write_json(fit_summary_path, fit_summary)

        posterior_npz = np.load(posterior_draws_path, allow_pickle=True)
        eval_context = load_context(posterior_npz)
        probabilities = predict_probabilities(
            posterior_npz,
            eval_context,
            eval_df,
            attempt_kc_long,
            batch_size=int(config.get("prediction_batch_size", 1000)),
            success_value_column=success_value_column,
            failure_value_column=failure_value_column,
        )
        posterior_npz.close()

        metrics = overall_metrics(eval_df["correct"].to_numpy(), probabilities)
        ensure_parent(overall_metrics_path)
        pd.DataFrame([metrics]).to_csv(overall_metrics_path, index=False)
        write_json(
            evaluation_summary_path,
            {
                "model_kind": "explicit_qmatrix_rpfa_model2_tuning",
                "history_mode": history_mode,
                "decay_alpha": alpha,
                "evaluate_split": str(config.get("evaluate_split", "test")),
                "primary_eval_only": bool(config.get("primary_eval_only", True)),
                "evaluation_rows": int(len(eval_df)),
                "evaluation_students": int(eval_df["student_id"].nunique()),
                "metrics": metrics,
                "overall_metrics_path": str(overall_metrics_path),
            },
        )

        comparison_rows.append(
            {
                "alpha": alpha,
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "auc": metrics["auc"],
                "calibration_intercept": metrics["calibration_intercept"],
                "calibration_slope": metrics["calibration_slope"],
                "accuracy": metrics["accuracy"],
                "elapsed_seconds": elapsed_seconds,
                "fit_summary_path": str(fit_summary_path),
                "evaluation_summary_path": str(evaluation_summary_path),
            }
        )

        del model
        del idata
        del attempt_kc_long
        del train_data
        gc.collect()

    comparison = pd.DataFrame(comparison_rows).sort_values(["log_loss", "alpha"], kind="mergesort").reset_index(drop=True)
    comparison_table_path = Path(config["comparison_table_path"])
    ensure_parent(comparison_table_path)
    comparison.to_csv(comparison_table_path, index=False)

    best_log_loss = float(comparison["log_loss"].min())
    tie_eligible = comparison.loc[comparison["log_loss"] <= best_log_loss + tie_margin].copy()
    selected = tie_eligible.sort_values("alpha", kind="mergesort").iloc[-1]

    selection_summary = {
        "history_mode": history_mode,
        "alpha_grid": alpha_grid,
        "tie_margin": tie_margin,
        "best_log_loss": best_log_loss,
        "selected_alpha": float(selected["alpha"]),
        "selected_rule": "min log loss, then choose the largest alpha within tie margin",
        "comparison_table_path": str(comparison_table_path),
        "selected_fit_summary_path": str(selected["fit_summary_path"]),
        "selected_evaluation_summary_path": str(selected["evaluation_summary_path"]),
    }
    write_json(Path(config["selection_summary_path"]), selection_summary)

    print(f"Saved alpha comparison to {comparison_table_path}")
    print(f"Selected alpha: {selected['alpha']}")
    print(f"Saved selection summary to {config['selection_summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
