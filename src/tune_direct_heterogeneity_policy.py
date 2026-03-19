from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from policy_suite_common import stable_student_bucket, summarize_policy_rows, write_json
from qmatrix_common import ensure_parent, load_json, load_trials


DEFAULT_CONFIG_PATH = Path("config/phase1_direct_heterogeneity_policy.json")
ACTION_ORDER = {
    "confidence_building": 0,
    "balanced_challenge": 1,
    "harder_challenge": 2,
    "spacing_aware_review": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate a direct heterogeneity-informed item policy over a small action slate."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def load_action_rows(config: dict) -> pd.DataFrame:
    new_learning = pd.read_csv(Path(config["new_learning_policy_rows_path"]))
    new_learning = new_learning.loc[
        new_learning["policy_name"].isin(["confidence_building", "balanced_challenge", "harder_challenge"])
    ].copy()

    spacing = pd.read_csv(Path(config["spacing_policy_rows_path"]))
    spacing = spacing.loc[spacing["policy_name"] == "spacing_aware_review"].copy()

    action_rows = pd.concat([new_learning, spacing], ignore_index=True)
    action_rows["student_id"] = action_rows["student_id"].astype(str)
    action_rows["attempt_id"] = action_rows["attempt_id"].astype("int64")
    action_rows["policy_name"] = action_rows["policy_name"].astype(str)
    return action_rows


def add_student_state_features(action_rows: pd.DataFrame, config: dict) -> pd.DataFrame:
    trials = load_trials(Path(config["processed_trials_path"]))
    trials = trials.loc[:, ["student_id", "attempt_id", "overall_opportunity"]].copy()
    trials["student_id"] = trials["student_id"].astype(str)
    trials["attempt_id"] = trials["attempt_id"].astype("int64")

    profiles = pd.read_csv(Path(config["model3_profile_path"]))
    profiles["student_id"] = profiles["student_id"].astype(str)

    latent_state = pd.read_csv(Path(config["model3_latent_state_path"]))
    latent_state["student_id"] = latent_state["student_id"].astype(str)
    latent_state["state_bin"] = latent_state["state_bin"].astype("int64")

    state_bin_width = int(config["state_bin_width"])

    merged = action_rows.merge(trials, on=["student_id", "attempt_id"], how="left", validate="many_to_one")
    merged["state_bin"] = (merged["overall_opportunity"].astype("int64") // state_bin_width).astype("int64")
    merged = merged.merge(profiles, on="student_id", how="left", validate="many_to_one")
    merged = merged.merge(
        latent_state.loc[:, ["student_id", "state_bin", "latent_state_mean"]],
        on=["student_id", "state_bin"],
        how="left",
        validate="many_to_one",
    )

    merged["latent_state_mean"] = merged["latent_state_mean"].fillna(0.0)
    merged["state_signal"] = (
        merged["latent_state_mean"] / merged["stability_mean"].clip(lower=float(config["stability_floor"]))
    ).clip(-1.0, 1.0)
    merged["recent_failure_log"] = np.log1p(merged["recent_failure_score"].astype(float))
    merged["is_review_action"] = (merged["policy_name"] == "spacing_aware_review").astype("int64")
    merged["action_order"] = merged["policy_name"].map(ACTION_ORDER).astype("int64")
    merged["student_bucket"] = merged["student_id"].map(lambda value: stable_student_bucket(value))
    merged["split_role"] = np.where(
        merged["student_bucket"] < int(100 * float(config["calibration_student_share"])),
        "calibration",
        "evaluation",
    )
    return merged


def compute_dynamic_target(frame: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    target = (
        float(params["base_target"])
        - float(params["baseline_weight"]) * (frame["baseline_rank_pct"] - 0.5)
        - float(params["growth_weight"]) * (frame["growth_rank_pct"] - 0.5)
        + float(params["stability_weight"]) * (frame["stability_rank_pct"] - 0.5)
        - float(params["state_weight"]) * frame["state_signal"]
    )
    return target.clip(float(params["target_min"]), float(params["target_max"]))


def select_direct_heterogeneity_rows(action_rows: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    frame = action_rows.copy()
    frame["target_probability"] = compute_dynamic_target(frame, params)
    frame["selection_gap"] = (frame["recommended_probability"] - frame["target_probability"]).abs()
    frame["utility"] = (
        -frame["selection_gap"]
        + float(params["remediation_weight"]) * frame["recent_failure_log"]
        + float(params["review_bonus"]) * frame["is_review_action"] * frame["due_review_flag"]
        - float(params["seen_item_penalty"]) * frame["recommended_seen_item"]
    )

    selected = (
        frame.sort_values(
            ["student_id", "attempt_id", "utility", "selection_gap", "recommended_seen_item", "action_order", "recommended_item_id"],
            ascending=[True, True, False, True, True, True, True],
            kind="mergesort",
        )
        .groupby(["student_id", "attempt_id"], sort=False)
        .head(1)
        .copy()
    )

    selected["selected_policy_name"] = selected["policy_name"]
    selected["policy_name"] = "direct_heterogeneity_policy"
    selected["route_reason"] = "heterogeneity_utility"
    selected["target_gap"] = (selected["recommended_probability"] - selected["target_probability"]).abs()
    selected["actual_target_gap"] = (selected["actual_next_probability"] - selected["target_probability"]).abs()
    return selected


def select_operational_freeze_rows(action_rows: pd.DataFrame) -> pd.DataFrame:
    frame = action_rows.copy()
    due_ready = frame.groupby(["student_id", "attempt_id"], sort=False)["due_review_flag"].transform("max")
    preferred = np.where(
        (due_ready == 1) & (frame["policy_name"] == "spacing_aware_review"),
        0,
        np.where(frame["policy_name"] == "confidence_building", 1, 2),
    )
    frame["selection_priority"] = preferred
    selected = (
        frame.sort_values(
            ["student_id", "attempt_id", "selection_priority", "action_order"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
        .groupby(["student_id", "attempt_id"], sort=False)
        .head(1)
        .copy()
    )
    selected["selected_policy_name"] = selected["policy_name"]
    selected["policy_name"] = "operational_freeze_proxy"
    selected["route_reason"] = np.where(selected["selected_policy_name"] == "spacing_aware_review", "due_review_ready", "default_confidence")
    return selected


def select_fixed_policy_rows(action_rows: pd.DataFrame, policy_name: str, output_name: str) -> pd.DataFrame:
    selected = action_rows.loc[action_rows["policy_name"] == policy_name].copy()
    selected["selected_policy_name"] = policy_name
    selected["policy_name"] = output_name
    selected["route_reason"] = policy_name
    return selected


def summarize_strategy(strategy_rows: pd.DataFrame, *, max_eval_step: int) -> dict:
    summary = summarize_policy_rows(strategy_rows.copy(), max_eval_step=max_eval_step)
    summary["selected_policy_shares"] = (
        strategy_rows["selected_policy_name"].value_counts(normalize=True, sort=True).sort_index().to_dict()
        if len(strategy_rows)
        else {}
    )
    summary["selected_policy_counts"] = (
        strategy_rows["selected_policy_name"].value_counts(normalize=False, sort=True).sort_index().to_dict()
        if len(strategy_rows)
        else {}
    )
    return summary


def calibration_objective(summary: dict, baseline_summary: dict) -> tuple[float, float, float]:
    return (
        float(summary["student_avg_target_gap_1_10"] - baseline_summary["student_avg_target_gap_1_10"]),
        -float(summary["policy_advantage_over_actual_1_10"] - baseline_summary["policy_advantage_over_actual_1_10"]),
        float(summary["recommendation_stability_mean_abs_diff"] - baseline_summary["recommendation_stability_mean_abs_diff"]),
    )


def best_grid_row(grid_df: pd.DataFrame) -> pd.Series:
    ordered = grid_df.sort_values(
        [
            "calibration_delta_target_gap_1_10",
            "calibration_delta_policy_advantage_1_10",
            "calibration_delta_stability",
        ],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return ordered.iloc[0]


def make_comparison_payload(candidate_summary: dict, baseline_summary: dict) -> dict[str, float]:
    return {
        "delta_target_gap_1_10": float(candidate_summary["student_avg_target_gap_1_10"] - baseline_summary["student_avg_target_gap_1_10"]),
        "delta_policy_advantage_1_10": float(
            candidate_summary["policy_advantage_over_actual_1_10"] - baseline_summary["policy_advantage_over_actual_1_10"]
        ),
        "delta_stability": float(
            candidate_summary["recommendation_stability_mean_abs_diff"] - baseline_summary["recommendation_stability_mean_abs_diff"]
        ),
        "delta_band_hit_rate_1_10": float(
            candidate_summary["recommended_target_band_hit_rate_1_10"] - baseline_summary["recommended_target_band_hit_rate_1_10"]
        ),
    }


def write_decision_note(path: Path, summary: dict) -> None:
    ensure_parent(path)
    selected = summary["selected_parameters"]
    eval_comparison = summary["evaluation_comparison_to_operational_freeze"]
    eval_direct = summary["evaluation_strategy_summaries"]["direct_heterogeneity_policy"]
    eval_freeze = summary["evaluation_strategy_summaries"]["operational_freeze_proxy"]

    lines = [
        "# Direct Heterogeneity Policy Decision",
        "",
        "This note records the first direct heterogeneity-informed next-item branch on DBE.",
        "",
        "## Frozen baseline",
        "",
        "- mean scorer: raw explicit Q-matrix `R-PFA Model 2`",
        "- replay freeze: `alpha = 0.9`",
        "- review threshold: `24` hours",
        "- comparison baseline here: `operational_freeze_proxy` = `spacing_aware_review` when due, else fixed `confidence_building`",
        "",
        "## Direct policy idea",
        "",
        "This branch does not use a calibration side-channel.",
        "",
        "Instead, it chooses directly among a small action slate:",
        "",
        "- `confidence_building`",
        "- `balanced_challenge`",
        "- `harder_challenge`",
        "- `spacing_aware_review`",
        "",
        "using scientific Model 3 learner heterogeneity signals inside the decision itself:",
        "",
        "- learner baseline rank",
        "- learner growth rank",
        "- learner stability rank",
        "- current latent-state signal",
        "",
        "## Selected parameters",
        "",
        f"- base target: `{selected['base_target']}`",
        f"- baseline weight: `{selected['baseline_weight']}`",
        f"- growth weight: `{selected['growth_weight']}`",
        f"- stability weight: `{selected['stability_weight']}`",
        f"- state weight: `{selected['state_weight']}`",
        f"- remediation weight: `{selected['remediation_weight']}`",
        f"- review bonus: `{selected['review_bonus']}`",
        f"- seen-item penalty: `{selected['seen_item_penalty']}`",
        "",
        "## Evaluation result",
        "",
        f"- direct policy target gap `1-10`: `{eval_direct['student_avg_target_gap_1_10']:.6f}`",
        f"- freeze proxy target gap `1-10`: `{eval_freeze['student_avg_target_gap_1_10']:.6f}`",
        f"- delta: `{eval_comparison['delta_target_gap_1_10']:+.6f}`",
        "",
        f"- direct policy advantage `1-10`: `{eval_direct['policy_advantage_over_actual_1_10']:.6f}`",
        f"- freeze proxy policy advantage `1-10`: `{eval_freeze['policy_advantage_over_actual_1_10']:.6f}`",
        f"- delta: `{eval_comparison['delta_policy_advantage_1_10']:+.6f}`",
        "",
        f"- direct stability: `{eval_direct['recommendation_stability_mean_abs_diff']:.6f}`",
        f"- freeze proxy stability: `{eval_freeze['recommendation_stability_mean_abs_diff']:.6f}`",
        f"- delta: `{eval_comparison['delta_stability']:+.6f}`",
        "",
        "## Interpretation",
        "",
        "- This is the first branch in the repo where heterogeneity changes the next-item choice directly rather than entering as a calibration add-on.",
        "- It should be read as an offline small-slate decision experiment, not a causal learning-gain result.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    action_rows = load_action_rows(config)
    action_rows = add_student_state_features(action_rows, config)

    calibration_rows = action_rows.loc[action_rows["split_role"] == "calibration"].copy()
    evaluation_rows = action_rows.loc[action_rows["split_role"] == "evaluation"].copy()

    baseline_calibration_rows = select_operational_freeze_rows(calibration_rows)
    baseline_evaluation_rows = select_operational_freeze_rows(evaluation_rows)
    baseline_calibration_summary = summarize_strategy(
        baseline_calibration_rows,
        max_eval_step=int(config["max_eval_step"]),
    )
    baseline_evaluation_summary = summarize_strategy(
        baseline_evaluation_rows,
        max_eval_step=int(config["max_eval_step"]),
    )

    grid_records: list[dict[str, float | int]] = []
    for base_target, baseline_weight, growth_weight, stability_weight, state_weight, remediation_weight, review_bonus, seen_item_penalty in product(
        config["base_target_grid"],
        config["baseline_weight_grid"],
        config["growth_weight_grid"],
        config["stability_weight_grid"],
        config["state_weight_grid"],
        config["remediation_weight_grid"],
        config["review_bonus_grid"],
        config["seen_item_penalty_grid"],
    ):
        params = {
            "base_target": float(base_target),
            "baseline_weight": float(baseline_weight),
            "growth_weight": float(growth_weight),
            "stability_weight": float(stability_weight),
            "state_weight": float(state_weight),
            "remediation_weight": float(remediation_weight),
            "review_bonus": float(review_bonus),
            "seen_item_penalty": float(seen_item_penalty),
            "target_min": float(config["target_min"]),
            "target_max": float(config["target_max"]),
        }
        direct_rows = select_direct_heterogeneity_rows(calibration_rows, params)
        direct_summary = summarize_strategy(direct_rows, max_eval_step=int(config["max_eval_step"]))
        objective = calibration_objective(direct_summary, baseline_calibration_summary)
        grid_records.append(
            {
                **params,
                "calibration_target_gap_1_10": float(direct_summary["student_avg_target_gap_1_10"]),
                "calibration_policy_advantage_1_10": float(direct_summary["policy_advantage_over_actual_1_10"]),
                "calibration_stability": float(direct_summary["recommendation_stability_mean_abs_diff"]),
                "calibration_delta_target_gap_1_10": float(objective[0]),
                "calibration_delta_policy_advantage_1_10": float(
                    direct_summary["policy_advantage_over_actual_1_10"] - baseline_calibration_summary["policy_advantage_over_actual_1_10"]
                ),
                "calibration_delta_stability": float(
                    direct_summary["recommendation_stability_mean_abs_diff"]
                    - baseline_calibration_summary["recommendation_stability_mean_abs_diff"]
                ),
                "calibration_route_share_confidence": float(direct_summary["selected_policy_shares"].get("confidence_building", 0.0)),
                "calibration_route_share_balanced": float(direct_summary["selected_policy_shares"].get("balanced_challenge", 0.0)),
                "calibration_route_share_harder": float(direct_summary["selected_policy_shares"].get("harder_challenge", 0.0)),
                "calibration_route_share_review": float(direct_summary["selected_policy_shares"].get("spacing_aware_review", 0.0)),
            }
        )

    grid_df = pd.DataFrame(grid_records)
    selected = best_grid_row(grid_df)
    selected_params = {
        "base_target": float(selected["base_target"]),
        "baseline_weight": float(selected["baseline_weight"]),
        "growth_weight": float(selected["growth_weight"]),
        "stability_weight": float(selected["stability_weight"]),
        "state_weight": float(selected["state_weight"]),
        "remediation_weight": float(selected["remediation_weight"]),
        "review_bonus": float(selected["review_bonus"]),
        "seen_item_penalty": float(selected["seen_item_penalty"]),
        "target_min": float(config["target_min"]),
        "target_max": float(config["target_max"]),
    }

    strategy_rows = {
        "operational_freeze_proxy": baseline_evaluation_rows,
        "direct_heterogeneity_policy": select_direct_heterogeneity_rows(evaluation_rows, selected_params),
        "fixed_confidence_building": select_fixed_policy_rows(evaluation_rows, "confidence_building", "fixed_confidence_building"),
        "fixed_balanced_challenge": select_fixed_policy_rows(evaluation_rows, "balanced_challenge", "fixed_balanced_challenge"),
        "fixed_harder_challenge": select_fixed_policy_rows(evaluation_rows, "harder_challenge", "fixed_harder_challenge"),
        "fixed_spacing_review": select_fixed_policy_rows(evaluation_rows, "spacing_aware_review", "fixed_spacing_review"),
    }

    strategy_summaries = {
        name: summarize_strategy(rows, max_eval_step=int(config["max_eval_step"])) for name, rows in strategy_rows.items()
    }
    comparison = make_comparison_payload(
        strategy_summaries["direct_heterogeneity_policy"],
        strategy_summaries["operational_freeze_proxy"],
    )
    operational_pass = bool(
        comparison["delta_target_gap_1_10"] < 0.0
        and comparison["delta_stability"] <= float(config["stability_tolerance"])
    )

    output_root = Path(config["output_root"])
    ensure_parent(output_root / "placeholder.txt")
    grid_output_path = output_root / "direct_policy_grid_search.csv"
    selected_row_output_path = output_root / "direct_policy_selected_rows.csv"
    summary_output_path = output_root / "direct_policy_summary.json"

    grid_df.to_csv(grid_output_path, index=False)
    strategy_rows["direct_heterogeneity_policy"].to_csv(selected_row_output_path, index=False)

    summary = {
        "track_name": str(config["track_name"]),
        "state_sources": {
            "model3_profile_path": str(config["model3_profile_path"]),
            "model3_latent_state_path": str(config["model3_latent_state_path"]),
            "new_learning_policy_rows_path": str(config["new_learning_policy_rows_path"]),
            "spacing_policy_rows_path": str(config["spacing_policy_rows_path"]),
        },
        "calibration_student_share": float(config["calibration_student_share"]),
        "selected_parameters": selected_params,
        "baseline_definition": "spacing_aware_review when due_review_flag=1, else confidence_building",
        "strategy_summaries": strategy_summaries,
        "evaluation_strategy_summaries": strategy_summaries,
        "evaluation_comparison_to_operational_freeze": comparison,
        "operational_pass": operational_pass,
        "stability_tolerance": float(config["stability_tolerance"]),
        "grid_output_path": str(grid_output_path),
        "selected_row_output_path": str(selected_row_output_path),
    }
    write_json(summary_output_path, summary)

    decision_note_path = Path(config["decision_note_path"])
    write_decision_note(decision_note_path, summary)

    print(f"Saved direct heterogeneity grid search to {grid_output_path}")
    print(f"Saved direct heterogeneity selected rows to {selected_row_output_path}")
    print(f"Saved direct heterogeneity summary to {summary_output_path}")
    print(f"Saved direct heterogeneity memo to {decision_note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
