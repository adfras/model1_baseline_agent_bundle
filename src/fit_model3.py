from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from model1_common import ensure_parent, load_json, load_trials, prepend_compiler_to_path
from model3_common import MODEL3_DESCRIPTION, build_track_b_model, prepare_track_b_training_data


DEFAULT_CONFIG_PATH = Path("config/model3_track_b_fit.json")


SCALAR_VARS = [
    "Intercept",
    "practice_feature",
    "student_intercept_sigma",
    "student_slope_sigma",
    "item_sigma",
    "state_sigma_global",
    "rho",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the initial Model 3 Track B volatility model with PyMC.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def summarize_vi_history(losses: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(losses, dtype="float64")
    if arr.size == 0:
        return {
            "vi_loss_initial": None,
            "vi_loss_final": None,
            "vi_loss_best": None,
            "vi_best_iteration": None,
            "vi_relative_improvement": None,
            "vi_tail_mean": None,
            "vi_tail_sd": None,
            "vi_prev_tail_mean": None,
            "vi_tail_relative_change": None,
        }

    tail = min(500, arr.size)
    tail_mean = float(arr[-tail:].mean())
    tail_sd = float(arr[-tail:].std(ddof=1)) if tail > 1 else 0.0

    prev_tail_mean = None
    tail_relative_change = None
    if arr.size >= 2 * tail:
        prev = arr[-2 * tail : -tail]
        prev_tail_mean = float(prev.mean())
        if prev_tail_mean != 0.0:
            tail_relative_change = float((tail_mean - prev_tail_mean) / abs(prev_tail_mean))

    initial = float(arr[0])
    final = float(arr[-1])
    best_index = int(arr.argmin())
    best_loss = float(arr[best_index])
    relative_improvement = None
    if initial != 0.0:
        relative_improvement = float((initial - final) / abs(initial))

    return {
        "vi_loss_initial": initial,
        "vi_loss_final": final,
        "vi_loss_best": best_loss,
        "vi_best_iteration": best_index + 1,
        "vi_relative_improvement": relative_improvement,
        "vi_tail_mean": tail_mean,
        "vi_tail_sd": tail_sd,
        "vi_prev_tail_mean": prev_tail_mean,
        "vi_tail_relative_change": tail_relative_change,
    }


def summarize_vector_draws(values: np.ndarray, labels: list[str], value_name: str) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(labels):
        draws = values[:, index]
        interval = az.hdi(draws, hdi_prob=0.94)
        rows.append(
            {
                value_name: label,
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

    if config.get("inference_method", "vi") != "vi":
        raise ValueError("The initial Model 3 implementation currently supports VI only.")

    trials = load_trials(Path(config["processed_trials_path"]))
    state_bin_width = int(config["state_bin_width"])
    data = prepare_track_b_training_data(trials, state_bin_width=state_bin_width)

    model = build_track_b_model(data)

    with model:
        approx = pm.fit(
            n=int(config["vi_iterations"]),
            method=config.get("vi_method", "advi"),
            random_seed=int(config["random_seed"]),
            progressbar=True,
        )

        posterior_draws = int(config["posterior_draws"])
        trace = approx.sample(posterior_draws, random_seed=int(config["random_seed"]), return_inferencedata=False)
        sampled = {
            "Intercept": np.asarray(trace.get_values("Intercept", combine=True)),
            "practice_feature": np.asarray(trace.get_values("practice_feature", combine=True)),
            "student_intercept_sigma": np.asarray(trace.get_values("student_intercept_sigma", combine=True)),
            "student_slope_sigma": np.asarray(trace.get_values("student_slope_sigma", combine=True)),
            "item_sigma": np.asarray(trace.get_values("item_sigma", combine=True)),
            "state_sigma_global": np.asarray(trace.get_values("state_sigma_global", combine=True)),
            "rho": np.asarray(trace.get_values("rho", combine=True)),
            "student_intercept": np.asarray(trace.get_values("student_intercept", combine=True)),
            "student_slope": np.asarray(trace.get_values("student_slope", combine=True)),
            "item_effect": np.asarray(trace.get_values("item_effect", combine=True)),
            "state_sigma_student": np.asarray(trace.get_values("state_sigma_student", combine=True)),
            "latent_state": np.asarray(trace.get_values("latent_state", combine=True)),
        }

    posterior_draws_path = Path(config["posterior_draws_path"])
    ensure_parent(posterior_draws_path)
    np.savez_compressed(
        posterior_draws_path,
        item_levels=np.asarray(data.item_levels, dtype=object),
        student_levels=np.asarray(data.student_levels, dtype=object),
        state_bin_width=np.asarray([state_bin_width], dtype="int64"),
        **sampled,
    )

    structural_rows = []
    for variable in SCALAR_VARS:
        draws = sampled[variable].reshape(-1)
        interval = az.hdi(draws, hdi_prob=0.94)
        structural_rows.append(
            {
                "parameter": variable,
                "mean": float(np.mean(draws)),
                "sd": float(np.std(draws, ddof=1)),
                "hdi_3%": float(interval[0]),
                "hdi_97%": float(interval[1]),
            }
        )
    structural_summary = pd.DataFrame(structural_rows)
    structural_summary_path = Path(config["structural_summary_path"])
    ensure_parent(structural_summary_path)
    structural_summary.to_csv(structural_summary_path, index=False)

    volatility_summary = summarize_vector_draws(
        sampled["state_sigma_student"],
        data.student_levels,
        value_name="student_id",
    )
    volatility_summary_path = Path(config["volatility_summary_path"])
    ensure_parent(volatility_summary_path)
    volatility_summary.to_csv(volatility_summary_path, index=False)

    vi_history = pd.DataFrame(
        {
            "iteration": range(1, len(approx.hist) + 1),
            "loss": [float(value) for value in approx.hist],
        }
    )
    vi_history_path = Path(config["vi_history_path"])
    ensure_parent(vi_history_path)
    vi_history.to_csv(vi_history_path, index=False)
    vi_diagnostics = summarize_vi_history([float(value) for value in approx.hist])

    diagnostics_summary_path = Path(config["diagnostics_summary_path"])
    diagnostics_summary = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in vi_diagnostics.items()]
    )
    ensure_parent(diagnostics_summary_path)
    diagnostics_summary.to_csv(diagnostics_summary_path, index=False)

    fit_summary = {
        "model_description": MODEL3_DESCRIPTION,
        "processed_trials_path": config["processed_trials_path"],
        "state_bin_width": state_bin_width,
        "inference_method": "vi",
        "vi_method": config.get("vi_method", "advi"),
        "random_seed": int(config["random_seed"]),
        "train_rows": int(len(data.train_df)),
        "train_students": int(data.n_students),
        "train_items": int(data.n_items),
        "train_state_steps": int(data.n_state_steps),
        "posterior_draws": int(config["posterior_draws"]),
        "vi_iterations": int(config["vi_iterations"]),
        "posterior_draws_path": str(posterior_draws_path),
        "structural_summary_path": str(structural_summary_path),
        "volatility_summary_path": str(volatility_summary_path),
        "vi_history_path": str(vi_history_path),
        "diagnostics_summary_path": str(diagnostics_summary_path),
        **vi_diagnostics,
    }
    write_json(Path(config["fit_summary_path"]), fit_summary)

    print(f"Saved posterior draws to {posterior_draws_path}")
    print(f"Saved fit summary to {config['fit_summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
