from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arviz as az
import pandas as pd

from qmatrix_common import (
    build_context,
    build_model1_qmatrix,
    ensure_parent,
    fit_model,
    load_attempt_kc_long,
    load_json,
    load_trials,
    prepare_dataset,
    prepend_compiler_to_path,
    save_posterior_npz,
    summarize_vector_draws,
    summarize_vi_history,
)


DEFAULT_CONFIG_PATH = Path("config/phase1_multikc_qmatrix_model1_fit.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit explicit Q-matrix Model 1 with PyMC.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    prepend_compiler_to_path(config.get("compiler_bin_dir"))

    trials = load_trials(Path(config["processed_trials_path"]))
    attempt_kc_long = load_attempt_kc_long(Path(config["attempt_kc_long_path"]))

    train_df = trials.loc[trials["split"] == "train"].copy()
    context = build_context(train_df, attempt_kc_long)
    train_data = prepare_dataset(trials, attempt_kc_long, context, split="train")

    model = build_model1_qmatrix(train_data, context)

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
    save_posterior_npz(idata, posterior_draws_path, context, model_kind="model1")

    kc_practice_summary = summarize_vector_draws(
        idata.posterior["kc_practice"].stack(sample=("chain", "draw")).transpose("sample", "kc").values,
        context.kc_levels,
        value_name="kc_id",
    )
    kc_practice_summary_path = Path(config["kc_practice_summary_path"])
    ensure_parent(kc_practice_summary_path)
    kc_practice_summary.to_csv(kc_practice_summary_path, index=False)

    vi_diagnostics = summarize_vi_history(vi_losses)
    if vi_losses is not None:
        vi_history = pd.DataFrame({"iteration": range(1, len(vi_losses) + 1), "loss": vi_losses})
        vi_history_path = Path(config["vi_history_path"])
        ensure_parent(vi_history_path)
        vi_history.to_csv(vi_history_path, index=False)

    fit_summary = {
        "model_kind": "explicit_qmatrix_model1",
        "inference_method": str(config["inference_method"]),
        "random_seed": int(config["random_seed"]),
        "train_rows": int(len(train_data.df)),
        "train_students": int(len(context.student_levels)),
        "train_items": int(len(context.item_levels)),
        "train_kcs": int(len(context.kc_levels)),
        "posterior_draws_path": str(posterior_draws_path),
        "posterior_summary_path": str(posterior_summary_path),
        "kc_practice_summary_path": str(kc_practice_summary_path),
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
