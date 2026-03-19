from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qmatrix_common import ensure_parent, load_json, summarize_vector_draws


DEFAULT_CONFIG_PATH = Path("config/phase1_qmatrix_learner_profiles.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export learner-state profile tables from saved Phase 1 explicit Q-matrix posterior draws."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def add_rank_pct(frame: pd.DataFrame, source_column: str, output_column: str) -> pd.DataFrame:
    result = frame.copy()
    result[output_column] = result[source_column].rank(method="average", pct=True)
    return result


def load_posterior(path: Path) -> np.lib.npyio.NpzFile:
    return np.load(path, allow_pickle=True)


def summarize_student_draws(
    draws: np.ndarray,
    student_levels: list[str],
    *,
    value_name: str,
) -> pd.DataFrame:
    summary = summarize_vector_draws(draws, student_levels, value_name)
    return summary.sort_values(value_name, kind="mergesort").reset_index(drop=True)


def rename_profile_columns(frame: pd.DataFrame, *, id_column: str, prefix: str) -> pd.DataFrame:
    return frame.rename(
        columns={
            id_column: "student_id",
            "mean": f"{prefix}_mean",
            "hdi_3%": f"{prefix}_hdi_low",
            "hdi_97%": f"{prefix}_hdi_high",
        }
    ).loc[:, ["student_id", f"{prefix}_mean", f"{prefix}_hdi_low", f"{prefix}_hdi_high"]]


def build_model2_profiles(model2_posterior: np.lib.npyio.NpzFile) -> pd.DataFrame:
    student_levels = model2_posterior["student_levels"].astype(str).tolist()
    intercept_draws = model2_posterior["Intercept"].astype(np.float64)
    baseline_draws = intercept_draws[:, None] + model2_posterior["student_intercept"].astype(np.float64)
    growth_draws = model2_posterior["student_slope"].astype(np.float64)

    baseline_summary = rename_profile_columns(
        summarize_student_draws(baseline_draws, student_levels, value_name="student_id"),
        id_column="student_id",
        prefix="baseline",
    )
    growth_summary = rename_profile_columns(
        summarize_student_draws(growth_draws, student_levels, value_name="student_id"),
        id_column="student_id",
        prefix="growth",
    )

    merged = baseline_summary.merge(growth_summary, on="student_id", how="inner", validate="one_to_one")
    merged = add_rank_pct(merged, "baseline_mean", "baseline_rank_pct")
    merged = add_rank_pct(merged, "growth_mean", "growth_rank_pct")
    return merged.sort_values("student_id", kind="mergesort").reset_index(drop=True)


def build_model3_profiles(model3_posterior: np.lib.npyio.NpzFile) -> pd.DataFrame:
    student_levels = model3_posterior["student_levels"].astype(str).tolist()
    intercept_draws = model3_posterior["Intercept"].astype(np.float64)
    baseline_draws = intercept_draws[:, None] + model3_posterior["student_intercept"].astype(np.float64)
    growth_draws = model3_posterior["student_slope"].astype(np.float64)
    stability_draws = model3_posterior["state_sigma_student"].astype(np.float64)

    baseline_summary = rename_profile_columns(
        summarize_student_draws(baseline_draws, student_levels, value_name="student_id"),
        id_column="student_id",
        prefix="baseline",
    )
    growth_summary = rename_profile_columns(
        summarize_student_draws(growth_draws, student_levels, value_name="student_id"),
        id_column="student_id",
        prefix="growth",
    )
    stability_summary = rename_profile_columns(
        summarize_student_draws(stability_draws, student_levels, value_name="student_id"),
        id_column="student_id",
        prefix="stability",
    )

    merged = baseline_summary.merge(growth_summary, on="student_id", how="inner", validate="one_to_one")
    merged = merged.merge(stability_summary, on="student_id", how="inner", validate="one_to_one")
    merged = add_rank_pct(merged, "baseline_mean", "baseline_rank_pct")
    merged = add_rank_pct(merged, "growth_mean", "growth_rank_pct")
    merged = add_rank_pct(merged, "stability_mean", "stability_rank_pct")
    return merged.sort_values("student_id", kind="mergesort").reset_index(drop=True)


def build_model3_latent_state_profiles(model3_posterior: np.lib.npyio.NpzFile) -> pd.DataFrame:
    student_levels = model3_posterior["student_levels"].astype(str).tolist()
    latent_state = model3_posterior["latent_state"].astype(np.float64)
    state_steps = latent_state.shape[1]

    rows: list[dict[str, float | int | str]] = []
    for state_bin in range(state_steps):
        summary = summarize_student_draws(latent_state[:, state_bin, :], student_levels, value_name="student_id")
        summary = summary.rename(
            columns={
                "mean": "latent_state_mean",
                "hdi_3%": "latent_state_hdi_low",
                "hdi_97%": "latent_state_hdi_high",
            }
        )
        summary["state_bin"] = state_bin
        rows.extend(
            summary.loc[:, ["student_id", "state_bin", "latent_state_mean", "latent_state_hdi_low", "latent_state_hdi_high"]]
            .sort_values("student_id", kind="mergesort")
            .to_dict(orient="records")
        )

    return pd.DataFrame(rows).sort_values(["student_id", "state_bin"], kind="mergesort").reset_index(drop=True)


def distribution_summary(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = frame[column].to_numpy(dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def correlation_rows(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    columns = {
        "baseline_mean": "baseline",
        "growth_mean": "growth",
        "stability_mean": "stability",
    }
    rows: list[dict[str, float | str]] = []
    for left in columns:
        for right in columns:
            rows.append(
                {
                    "left_dimension": columns[left],
                    "right_dimension": columns[right],
                    "pearson_r": float(frame[left].corr(frame[right])),
                }
            )
    return rows


def max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    return float(np.max(np.abs(left.to_numpy(dtype=np.float64) - right.to_numpy(dtype=np.float64))))


def build_validation_summary(
    model2_profiles: pd.DataFrame,
    model3_profiles: pd.DataFrame,
    model3_latent_state: pd.DataFrame,
    *,
    model2_student_slope_summary_path: Path,
    model3_volatility_summary_path: Path,
    state_bin_width: int,
) -> dict:
    model2_growth_reference = pd.read_csv(model2_student_slope_summary_path)
    model2_growth_reference["student_id"] = model2_growth_reference["student_id"].astype(str)
    model2_growth_reference = model2_growth_reference.sort_values("student_id", kind="mergesort").reset_index(drop=True)

    model3_stability_reference = pd.read_csv(model3_volatility_summary_path)
    model3_stability_reference["student_id"] = model3_stability_reference["student_id"].astype(str)
    model3_stability_reference = model3_stability_reference.sort_values("student_id", kind="mergesort").reset_index(drop=True)

    merged_growth = model2_profiles.merge(
        model2_growth_reference,
        on="student_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    merged_stability = model3_profiles.merge(
        model3_stability_reference,
        on="student_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )

    return {
        "model2_student_count": int(len(model2_profiles)),
        "model3_student_count": int(len(model3_profiles)),
        "model2_duplicate_student_ids": int(model2_profiles["student_id"].duplicated().sum()),
        "model3_duplicate_student_ids": int(model3_profiles["student_id"].duplicated().sum()),
        "model3_latent_state_duplicate_rows": int(
            model3_latent_state.duplicated(subset=["student_id", "state_bin"]).sum()
        ),
        "model3_latent_state_row_count": int(len(model3_latent_state)),
        "model3_latent_state_expected_row_count": int(len(model3_profiles) * model3_latent_state["state_bin"].nunique()),
        "model3_state_bin_count": int(model3_latent_state["state_bin"].nunique()),
        "model3_state_bin_width": int(state_bin_width),
        "model2_growth_reference_max_abs_diff": {
            "mean": max_abs_diff(merged_growth["growth_mean"], merged_growth["mean"]),
            "hdi_low": max_abs_diff(merged_growth["growth_hdi_low"], merged_growth["hdi_3%"]),
            "hdi_high": max_abs_diff(merged_growth["growth_hdi_high"], merged_growth["hdi_97%"]),
        },
        "model3_stability_reference_max_abs_diff": {
            "mean": max_abs_diff(merged_stability["stability_mean"], merged_stability["mean"]),
            "hdi_low": max_abs_diff(merged_stability["stability_hdi_low"], merged_stability["hdi_3%"]),
            "hdi_high": max_abs_diff(merged_stability["stability_hdi_high"], merged_stability["hdi_97%"]),
        },
    }


def write_summary_markdown(
    path: Path,
    *,
    model2_profiles: pd.DataFrame,
    model3_profiles: pd.DataFrame,
    model3_latent_state: pd.DataFrame,
    summary_payload: dict,
) -> None:
    ensure_parent(path)

    def metric_lines(title: str, stats: dict[str, float]) -> list[str]:
        return [
            f"### {title}",
            "",
            f"- mean: `{stats['mean']:.4f}`",
            f"- sd: `{stats['sd']:.4f}`",
            f"- p10 / p50 / p90: `{stats['p10']:.4f}` / `{stats['p50']:.4f}` / `{stats['p90']:.4f}`",
            "",
        ]

    lines = [
        "# Phase 1 Explicit-Q Learner State Profiles",
        "",
        "This note promotes learner-state estimation to a first-class Phase 1 DBE deliverable.",
        "",
        "Source of truth:",
        "",
        "- scientific explicit Q-matrix posterior draws only",
        "- no refitting",
        "- 94% HDIs throughout",
        "",
        "## Exported tables",
        "",
        f"- Model 2 learner profiles: `{summary_payload['model2_profile_output_path']}`",
        f"- Model 3 learner profiles: `{summary_payload['model3_profile_output_path']}`",
        f"- Model 3 latent states: `{summary_payload['model3_latent_state_output_path']}`",
        "",
        f"- students exported: `{len(model3_profiles)}`",
        f"- Model 3 state bins exported: `{model3_latent_state['state_bin'].nunique()}`",
        f"- state-bin width: `{summary_payload['model3_state_bin_width']}`",
        "",
        "## Model 3 learner-dimension distributions",
        "",
    ]

    lines.extend(metric_lines("Baseline", summary_payload["model3_distribution_summary"]["baseline"]))
    lines.extend(metric_lines("Growth", summary_payload["model3_distribution_summary"]["growth"]))
    lines.extend(metric_lines("Stability", summary_payload["model3_distribution_summary"]["stability"]))

    lines.extend(
        [
            "## Correlations Among Learner Dimensions",
            "",
            "| left | right | pearson_r |",
            "| --- | --- | ---: |",
        ]
    )
    for row in summary_payload["model3_correlation_table"]:
        lines.append(f"| {row['left_dimension']} | {row['right_dimension']} | {row['pearson_r']:.4f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `baseline` is the learner-specific starting level on the logit scale: global intercept plus learner intercept.",
            "- `growth` is the learner-specific practice-response slope on the logit scale.",
            "- `stability` is the learner-specific latent state scale from Model 3; larger values indicate more time-varying deviation around the learner's baseline-plus-growth trajectory.",
            "- `latent_state_mean` by `state_bin` shows where a learner's transient state is above or below their longer-run trajectory at that practice stage.",
            "",
            "These exports are meant to support learner-state reporting and later Phase 2 replication, not to claim a new adaptive-policy win on DBE.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    model2_posterior_path = Path(config["model2_posterior_draws_path"])
    model3_posterior_path = Path(config["model3_posterior_draws_path"])
    model2_profile_output_path = Path(config["model2_profile_output_path"])
    model3_profile_output_path = Path(config["model3_profile_output_path"])
    model3_latent_state_output_path = Path(config["model3_latent_state_output_path"])
    summary_json_path = Path(config["summary_json_path"])
    validation_json_path = Path(config["validation_json_path"])
    summary_markdown_path = Path(config["summary_markdown_path"])

    with load_posterior(model2_posterior_path) as model2_posterior, load_posterior(model3_posterior_path) as model3_posterior:
        model2_profiles = build_model2_profiles(model2_posterior)
        model3_profiles = build_model3_profiles(model3_posterior)
        model3_latent_state = build_model3_latent_state_profiles(model3_posterior)
        state_bin_width = int(np.asarray(model3_posterior["state_bin_width"]).reshape(-1)[0])

    for path in [model2_profile_output_path, model3_profile_output_path, model3_latent_state_output_path]:
        ensure_parent(path)

    model2_profiles.to_csv(model2_profile_output_path, index=False)
    model3_profiles.to_csv(model3_profile_output_path, index=False)
    model3_latent_state.to_csv(model3_latent_state_output_path, index=False)

    summary_payload = {
        "model2_profile_output_path": str(model2_profile_output_path),
        "model3_profile_output_path": str(model3_profile_output_path),
        "model3_latent_state_output_path": str(model3_latent_state_output_path),
        "model3_state_bin_width": state_bin_width,
        "model2_distribution_summary": {
            "baseline": distribution_summary(model2_profiles, "baseline_mean"),
            "growth": distribution_summary(model2_profiles, "growth_mean"),
        },
        "model3_distribution_summary": {
            "baseline": distribution_summary(model3_profiles, "baseline_mean"),
            "growth": distribution_summary(model3_profiles, "growth_mean"),
            "stability": distribution_summary(model3_profiles, "stability_mean"),
        },
        "model3_correlation_table": correlation_rows(model3_profiles),
    }
    ensure_parent(summary_json_path)
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    validation_summary = build_validation_summary(
        model2_profiles,
        model3_profiles,
        model3_latent_state,
        model2_student_slope_summary_path=Path(config["model2_student_slope_summary_path"]),
        model3_volatility_summary_path=Path(config["model3_volatility_summary_path"]),
        state_bin_width=state_bin_width,
    )
    ensure_parent(validation_json_path)
    with validation_json_path.open("w", encoding="utf-8") as handle:
        json.dump(validation_summary, handle, indent=2)

    write_summary_markdown(
        summary_markdown_path,
        model2_profiles=model2_profiles,
        model3_profiles=model3_profiles,
        model3_latent_state=model3_latent_state,
        summary_payload=summary_payload,
    )

    print(f"Saved Model 2 learner profiles to {model2_profile_output_path}")
    print(f"Saved Model 3 learner profiles to {model3_profile_output_path}")
    print(f"Saved Model 3 latent-state profiles to {model3_latent_state_output_path}")
    print(f"Saved learner-profile summary to {summary_json_path}")
    print(f"Saved learner-profile validation to {validation_json_path}")
    print(f"Saved learner-profile note to {summary_markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
