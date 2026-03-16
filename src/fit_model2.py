from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arviz as az
import pandas as pd

from model1_common import ensure_parent, load_json, load_trials, prepend_compiler_to_path
from model2_common import build_model


DEFAULT_CONFIG_PATH = Path("config/model2_fit.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the Model 2 random-slope baseline with Bambi/PyMC.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def maybe_write_student_slope_summary(
    posterior_summary: pd.DataFrame,
    output_path_value: str | None,
) -> str | None:
    if output_path_value is None:
        return None
    mask = posterior_summary.index.to_series().astype("string").str.contains(
        "practice_feature\\|student_id",
        regex=True,
    )
    slope_summary = posterior_summary.loc[mask].copy()
    output_path = Path(output_path_value)
    ensure_parent(output_path)
    slope_summary.to_csv(output_path)
    return str(output_path)


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    prepend_compiler_to_path(config.get("compiler_bin_dir"))

    trials = load_trials(Path(config["processed_trials_path"]))
    train_df = trials.loc[trials["split"] == "train", ["correct", "practice_feature", "student_id", "item_id"]].copy()

    model = build_model(train_df)

    started = time.perf_counter()
    inference_method = config["inference_method"]
    random_seed = int(config["random_seed"])

    if inference_method == "vi":
        approx = model.fit(
            inference_method="vi",
            method=config.get("vi_method", "advi"),
            n=int(config["vi_iterations"]),
            random_seed=random_seed,
        )
        idata = approx.sample(
            int(config["posterior_draws"]),
            random_seed=random_seed,
            return_inferencedata=True,
        )
        vi_history = pd.DataFrame(
            {
                "iteration": range(1, len(approx.hist) + 1),
                "loss": [float(value) for value in approx.hist],
            }
        )
        vi_history_path = Path(config["vi_history_path"])
        ensure_parent(vi_history_path)
        vi_history.to_csv(vi_history_path, index=False)
        fit_extra = {
            "vi_iterations": int(config["vi_iterations"]),
            "posterior_draws": int(config["posterior_draws"]),
            "vi_loss_initial": float(approx.hist[0]) if len(approx.hist) else None,
            "vi_loss_final": float(approx.hist[-1]) if len(approx.hist) else None,
            "vi_history_path": str(vi_history_path),
        }
    else:
        idata = model.fit(
            inference_method=inference_method,
            draws=int(config["draws"]),
            tune=int(config["tune"]),
            chains=int(config["chains"]),
            cores=int(config["cores"]),
            random_seed=random_seed,
            target_accept=float(config["target_accept"]),
        )
        fit_extra = {
            "draws": int(config["draws"]),
            "tune": int(config["tune"]),
            "chains": int(config["chains"]),
            "cores": int(config["cores"]),
            "target_accept": float(config["target_accept"]),
        }

    elapsed_seconds = time.perf_counter() - started

    idata_path = Path(config["idata_path"])
    ensure_parent(idata_path)
    idata.to_netcdf(idata_path)

    posterior_summary = az.summary(idata, kind="stats")
    posterior_summary_path = Path(config["posterior_summary_path"])
    ensure_parent(posterior_summary_path)
    posterior_summary.to_csv(posterior_summary_path)

    diagnostics_path_value = config.get("diagnostics_summary_path")
    diagnostics_summary = None
    if diagnostics_path_value is not None:
        diagnostics_summary = az.summary(idata, kind="diagnostics")
        diagnostics_summary_path = Path(diagnostics_path_value)
        ensure_parent(diagnostics_summary_path)
        diagnostics_summary.to_csv(diagnostics_summary_path)

    student_slope_summary_path = maybe_write_student_slope_summary(
        posterior_summary,
        config.get("student_slope_summary_path"),
    )

    fit_summary = {
        "formula": str(model.formula),
        "inference_method": inference_method,
        "random_seed": random_seed,
        "train_rows": int(len(train_df)),
        "train_students": int(train_df["student_id"].nunique()),
        "train_items": int(train_df["item_id"].nunique()),
        "idata_path": str(idata_path),
        "posterior_summary_path": str(posterior_summary_path),
        "elapsed_seconds": elapsed_seconds,
        **fit_extra,
    }
    if student_slope_summary_path is not None:
        fit_summary["student_slope_summary_path"] = student_slope_summary_path
    if diagnostics_path_value is not None:
        fit_summary["diagnostics_summary_path"] = diagnostics_path_value
    if inference_method != "vi" and hasattr(idata, "sample_stats"):
        sample_stats = idata.sample_stats
        divergences = None
        if "diverging" in sample_stats:
            divergences = int(sample_stats["diverging"].sum().item())
        fit_summary["divergences"] = divergences
        if diagnostics_summary is not None:
            if "r_hat" in diagnostics_summary.columns:
                fit_summary["max_r_hat"] = float(diagnostics_summary["r_hat"].max())
            if "ess_bulk" in diagnostics_summary.columns:
                fit_summary["min_ess_bulk"] = float(diagnostics_summary["ess_bulk"].min())
            if "ess_tail" in diagnostics_summary.columns:
                fit_summary["min_ess_tail"] = float(diagnostics_summary["ess_tail"].min())
    write_json(Path(config["fit_summary_path"]), fit_summary)

    print(f"Saved posterior draws to {idata_path}")
    print(f"Saved fit summary to {config['fit_summary_path']}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
