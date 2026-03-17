from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from qmatrix_pfa_common import (
    build_context,
    build_model3_qmatrix_pfa,
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
from kc_history_common import resolve_history_value_columns


DEFAULT_CONFIG_PATH = Path("config/phase1_multikc_qmatrix_pfa_model3_fit.json")

SCALAR_VARS = [
    "Intercept",
    "student_intercept_sigma",
    "student_slope_sigma",
    "item_sigma",
    "kc_intercept_sigma",
    "kc_success_sigma",
    "kc_failure_sigma",
    "state_sigma_global",
    "rho",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit explicit Q-matrix PFA Model 3 with PyMC.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def summarize_scalar_draws(idata, variables: list[str]) -> pd.DataFrame:
    rows = []
    for variable in variables:
        draws = idata.posterior[variable].stack(sample=("chain", "draw")).values.reshape(-1)
        interval = az.hdi(draws, hdi_prob=0.94)
        rows.append(
            {
                "parameter": variable,
                "mean": float(np.mean(draws)),
                "sd": float(np.std(draws, ddof=1)),
                "hdi_3%": float(interval[0]),
                "hdi_97%": float(interval[1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    prepend_compiler_to_path(config.get("compiler_bin_dir"))

    trials = load_trials(Path(config["processed_trials_path"]))
    history_mode = str(config.get("history_mode", "pfa")).lower()
    decay_alpha = float(config.get("decay_alpha", 1.0))
    due_review_hours = float(config.get("due_review_hours", 48.0))
    attempt_kc_long = prepare_attempt_kc_long_for_history(
        load_attempt_kc_long_pfa(Path(config["attempt_kc_long_path"])),
        history_mode=history_mode,
        decay_alpha=decay_alpha,
        due_review_hours=due_review_hours,
    )
    success_value_column, failure_value_column = resolve_history_value_columns(history_mode)
    state_bin_width = int(config["state_bin_width"])

    train_df = trials.loc[trials["split"] == "train"].copy()
    context = build_context(train_df, attempt_kc_long)
    train_data = prepare_pfa_dataset(
        trials,
        attempt_kc_long,
        context,
        split="train",
        state_bin_width=state_bin_width,
        success_value_column=success_value_column,
        failure_value_column=failure_value_column,
    )

    model = build_model3_qmatrix_pfa(train_data, context)

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
    posterior_summary_path = Path(config["posterior_summary_path"])
    ensure_parent(posterior_summary_path)
    posterior_summary.to_csv(posterior_summary_path)

    posterior_draws_path = Path(config["posterior_draws_path"])
    save_posterior_npz_pfa(
        idata,
        posterior_draws_path,
        context,
        model_kind="model3",
        state_bin_width=state_bin_width,
        history_mode=history_mode,
        decay_alpha=decay_alpha,
    )

    structural_summary = summarize_scalar_draws(idata, SCALAR_VARS)
    structural_summary_path = Path(config["structural_summary_path"])
    ensure_parent(structural_summary_path)
    structural_summary.to_csv(structural_summary_path, index=False)

    volatility_summary = summarize_vector_draws(
        idata.posterior["state_sigma_student"].stack(sample=("chain", "draw")).transpose("sample", "student").values,
        context.student_levels,
        value_name="student_id",
    )
    volatility_summary_path = Path(config["volatility_summary_path"])
    ensure_parent(volatility_summary_path)
    volatility_summary.to_csv(volatility_summary_path, index=False)

    vi_diagnostics = summarize_vi_history(vi_losses)
    if vi_losses is not None:
        vi_history = pd.DataFrame({"iteration": range(1, len(vi_losses) + 1), "loss": vi_losses})
        vi_history_path = Path(config["vi_history_path"])
        ensure_parent(vi_history_path)
        vi_history.to_csv(vi_history_path, index=False)
        diagnostics_summary_path = Path(config["diagnostics_summary_path"])
        ensure_parent(diagnostics_summary_path)
        pd.DataFrame([{"metric": key, "value": value} for key, value in vi_diagnostics.items()]).to_csv(
            diagnostics_summary_path,
            index=False,
        )

    fit_summary = {
        "model_kind": f"explicit_qmatrix_{history_mode}_model3",
        "inference_method": str(config["inference_method"]),
        "history_mode": history_mode,
        "decay_alpha": decay_alpha,
        "due_review_hours": due_review_hours,
        "random_seed": int(config["random_seed"]),
        "state_bin_width": state_bin_width,
        "train_rows": int(len(train_data.df)),
        "train_students": int(len(context.student_levels)),
        "train_items": int(len(context.item_levels)),
        "train_kcs": int(len(context.kc_levels)),
        "train_state_steps": int(train_data.n_state_steps or 0),
        "posterior_draws_path": str(posterior_draws_path),
        "posterior_summary_path": str(posterior_summary_path),
        "structural_summary_path": str(structural_summary_path),
        "volatility_summary_path": str(volatility_summary_path),
        "elapsed_seconds": elapsed_seconds,
        **vi_diagnostics,
    }
    write_json(Path(config["fit_summary_path"]), fit_summary)

    print(f"Saved posterior draws to {posterior_draws_path}")
    print(f"Saved fit summary to {config['fit_summary_path']}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
