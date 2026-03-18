from __future__ import annotations

import argparse
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from policy_suite_common import (
    build_attempt_event_lookup,
    build_candidate_frame,
    build_item_kc_lookup,
    choose_policy_item,
    score_candidates_model2,
    summarize_policy_rows,
    write_json,
)
from qmatrix_common import ensure_parent, load_json, load_trials
from qmatrix_pfa_common import load_attempt_kc_long_pfa, prepare_attempt_kc_long_for_history


DEFAULT_CONFIG_PATH = Path("config/phase1_adaptive_policy_simple_router_qmatrix_rpfa.json")
FIXED_NEW_LEARNING_POLICIES = ("balanced_challenge", "confidence_building", "harder_challenge")
ROUTER_POLICIES = ("balanced_challenge", "confidence_building", "spacing_aware_review")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate a simple two-mode RPFA router using Model 2 scores."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def safe_mean(values: deque[float] | list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_median(values: deque[float] | list[float], *, default: float) -> float:
    if not values:
        return float(default)
    return float(np.median(np.asarray(values, dtype=np.float64)))


def compute_behavior_state(
    *,
    recent_correct_window: deque[int],
    recent_hint_window: deque[int],
    recent_selection_change_window: deque[int],
    recent_duration_window: deque[float],
    baseline_duration_values: list[float],
    default_duration_seconds: float,
) -> dict[str, float]:
    recent_attempt_count = len(recent_correct_window)
    recent_success_rate_attempt = safe_mean(recent_correct_window) if recent_attempt_count else 0.5
    recent_hint_rate = safe_mean(recent_hint_window)
    recent_selection_change_rate = safe_mean(recent_selection_change_window)
    baseline_duration_median = safe_median(baseline_duration_values, default=default_duration_seconds)
    recent_duration_median = safe_median(recent_duration_window, default=baseline_duration_median)
    response_time_inflation = recent_duration_median / max(baseline_duration_median, 1e-6)
    return {
        "recent_attempt_count": float(recent_attempt_count),
        "recent_success_rate_attempt": float(recent_success_rate_attempt),
        "recent_hint_rate": float(recent_hint_rate),
        "recent_selection_change_rate": float(recent_selection_change_rate),
        "baseline_duration_median": float(baseline_duration_median),
        "recent_duration_median": float(recent_duration_median),
        "response_time_inflation": float(response_time_inflation),
    }


def policy_result_columns(policy_name: str) -> dict[str, str]:
    prefix = f"{policy_name}__"
    return {
        "policy_name": prefix + "policy_name",
        "candidate_pool_mode": prefix + "candidate_pool_mode",
        "recommended_item_id": prefix + "recommended_item_id",
        "recommended_probability": prefix + "recommended_probability",
        "target_probability": prefix + "target_probability",
        "target_gap": prefix + "target_gap",
        "in_target_band": prefix + "in_target_band",
        "fallback_used": prefix + "fallback_used",
        "recent_failure_score": prefix + "recent_failure_score",
        "due_review_flag": prefix + "due_review_flag",
        "due_review_hours": prefix + "due_review_hours",
        "recommended_seen_item": prefix + "recommended_seen_item",
        "student_item_exposure_count": prefix + "student_item_exposure_count",
        "linked_kc_exposure_total": prefix + "linked_kc_exposure_total",
        "actual_target_gap": prefix + "actual_target_gap",
        "recommended_kc_count": prefix + "recommended_kc_count",
        "recommended_single_kc": prefix + "recommended_single_kc",
    }


def extract_policy_result(
    *,
    policy_name: str,
    selected: pd.Series,
    policy_meta: dict,
    actual_next_probability: float,
    recommended_kc_count: int,
) -> dict[str, object]:
    columns = policy_result_columns(policy_name)
    target_probability = float(policy_meta["target_probability"])
    return {
        columns["policy_name"]: policy_name,
        columns["candidate_pool_mode"]: str(policy_meta["candidate_pool_mode"]),
        columns["recommended_item_id"]: str(selected["item_id"]),
        columns["recommended_probability"]: float(selected["predicted_probability"]),
        columns["target_probability"]: target_probability,
        columns["target_gap"]: abs(float(selected["predicted_probability"]) - target_probability),
        columns["in_target_band"]: int(policy_meta["in_target_band"]),
        columns["fallback_used"]: str(policy_meta["fallback_used"]),
        columns["recent_failure_score"]: float(selected["recent_failure_score"]),
        columns["due_review_flag"]: int(selected["due_review_flag"]),
        columns["due_review_hours"]: float(selected["due_review_hours"]) if pd.notna(selected["due_review_hours"]) else np.nan,
        columns["recommended_seen_item"]: int(selected["recommended_seen_item"]),
        columns["student_item_exposure_count"]: int(selected["student_item_exposure_count"]),
        columns["linked_kc_exposure_total"]: float(selected["linked_kc_exposure_total"]),
        columns["actual_target_gap"]: abs(actual_next_probability - target_probability),
        columns["recommended_kc_count"]: int(recommended_kc_count),
        columns["recommended_single_kc"]: int(recommended_kc_count == 1),
    }


def build_standardized_rows(rows: pd.DataFrame, *, policy_name: str) -> pd.DataFrame:
    columns = policy_result_columns(policy_name)
    standardized = pd.DataFrame(
        {
            "student_id": rows["student_id"].to_numpy(),
            "attempt_id": rows["attempt_id"].to_numpy(),
            "actual_item_id": rows["actual_item_id"].to_numpy(),
            "actual_correct": rows["actual_correct"].to_numpy(),
            "eval_step": rows["eval_step"].to_numpy(),
            "policy_name": np.repeat(policy_name, len(rows)),
            "candidate_pool_mode": rows[columns["candidate_pool_mode"]].to_numpy(),
            "recommended_item_id": rows[columns["recommended_item_id"]].to_numpy(),
            "recommended_probability": rows[columns["recommended_probability"]].to_numpy(),
            "target_probability": rows[columns["target_probability"]].to_numpy(),
            "target_gap": rows[columns["target_gap"]].to_numpy(),
            "in_target_band": rows[columns["in_target_band"]].to_numpy(),
            "fallback_used": rows[columns["fallback_used"]].to_numpy(),
            "recent_failure_score": rows[columns["recent_failure_score"]].to_numpy(),
            "due_review_flag": rows[columns["due_review_flag"]].to_numpy(),
            "due_review_hours": rows[columns["due_review_hours"]].to_numpy(),
            "recommended_seen_item": rows[columns["recommended_seen_item"]].to_numpy(),
            "student_item_exposure_count": rows[columns["student_item_exposure_count"]].to_numpy(),
            "linked_kc_exposure_total": rows[columns["linked_kc_exposure_total"]].to_numpy(),
            "candidate_count": rows["candidate_count"].to_numpy(),
            "actual_next_probability": rows["actual_next_probability"].to_numpy(),
            "actual_target_gap": rows[columns["actual_target_gap"]].to_numpy(),
            "track": rows["track"].to_numpy(),
            "model_name": rows["model_name"].to_numpy(),
            "history_mode": rows["history_mode"].to_numpy(),
            "decay_alpha": rows["decay_alpha"].to_numpy(),
        }
    )
    if "route_reason" in rows.columns:
        standardized["route_reason"] = rows["route_reason"].to_numpy()
    return standardized


def friction_flag(row: pd.Series, *, rule_name: str, config: dict) -> int:
    if rule_name == "current":
        hint_threshold = float(config["current_hint_rate_threshold"])
        selection_threshold = float(config["current_selection_change_rate_threshold"])
        duration_threshold = float(config["current_duration_inflation_threshold"])
    elif rule_name == "stricter":
        hint_threshold = float(config["stricter_hint_rate_threshold"])
        selection_threshold = float(config["stricter_selection_change_rate_threshold"])
        duration_threshold = float(config["stricter_duration_inflation_threshold"])
    else:
        raise ValueError(f"Unsupported friction rule: {rule_name}")

    return int(
        float(row["recent_hint_rate"]) >= hint_threshold
        or float(row["recent_selection_change_rate"]) >= selection_threshold
        or float(row["response_time_inflation"]) >= duration_threshold
    )


def route_simple_policy(
    row: pd.Series,
    *,
    early_step_cutoff: int,
    low_proficiency_threshold: float,
    recent_failure_threshold: float,
    friction_rule_name: str,
    config: dict,
) -> tuple[str, str, dict[str, int]]:
    due_review_ready = int(row["due_review_available"]) == 1
    early_step = int(row["eval_step"]) <= early_step_cutoff
    low_predicted_proficiency = float(row["balanced_reference_probability"]) <= low_proficiency_threshold
    high_recent_failure = float(row["recent_failure_total"]) >= recent_failure_threshold
    high_friction = friction_flag(row, rule_name=friction_rule_name, config=config) == 1

    if due_review_ready:
        return "spacing_aware_review", "due_review_ready", {
            "due_review_ready": 1,
            "early_step": int(early_step),
            "low_predicted_proficiency": int(low_predicted_proficiency),
            "high_recent_failure": int(high_recent_failure),
            "high_friction": int(high_friction),
        }

    if early_step or low_predicted_proficiency or high_recent_failure or high_friction:
        reasons: list[str] = []
        if early_step:
            reasons.append("early_step")
        if low_predicted_proficiency:
            reasons.append("low_predicted_proficiency")
        if high_recent_failure:
            reasons.append("high_recent_failure")
        if high_friction:
            reasons.append("high_friction")
        return "confidence_building", "+".join(reasons), {
            "due_review_ready": 0,
            "early_step": int(early_step),
            "low_predicted_proficiency": int(low_predicted_proficiency),
            "high_recent_failure": int(high_recent_failure),
            "high_friction": int(high_friction),
        }

    return "balanced_challenge", "default_balanced", {
        "due_review_ready": 0,
        "early_step": int(early_step),
        "low_predicted_proficiency": int(low_predicted_proficiency),
        "high_recent_failure": int(high_recent_failure),
        "high_friction": int(high_friction),
    }


def assemble_routed_rows(
    base_rows: pd.DataFrame,
    *,
    early_step_cutoff: int,
    low_proficiency_threshold: float,
    recent_failure_threshold: float,
    friction_rule_name: str,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    routed_records: list[dict[str, object]] = []
    for row in base_rows.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        policy_name, route_reason, route_flags = route_simple_policy(
            row_series,
            early_step_cutoff=early_step_cutoff,
            low_proficiency_threshold=low_proficiency_threshold,
            recent_failure_threshold=recent_failure_threshold,
            friction_rule_name=friction_rule_name,
            config=config,
        )
        columns = policy_result_columns(policy_name)
        recommended_kc_count = getattr(row, columns["recommended_kc_count"], np.nan)
        recommended_single_kc = getattr(row, columns["recommended_single_kc"], np.nan)
        routed_records.append(
            {
                "student_id": str(row.student_id),
                "attempt_id": int(row.attempt_id),
                "actual_item_id": str(row.actual_item_id),
                "actual_correct": int(row.actual_correct),
                "eval_step": int(row.eval_step),
                "policy_name": "simple_two_mode_router",
                "selected_policy_name": policy_name,
                "route_reason": route_reason,
                "candidate_pool_mode": getattr(row, columns["candidate_pool_mode"]),
                "recommended_item_id": getattr(row, columns["recommended_item_id"]),
                "recommended_probability": getattr(row, columns["recommended_probability"]),
                "target_probability": getattr(row, columns["target_probability"]),
                "target_gap": getattr(row, columns["target_gap"]),
                "in_target_band": getattr(row, columns["in_target_band"]),
                "fallback_used": getattr(row, columns["fallback_used"]),
                "recent_failure_score": getattr(row, columns["recent_failure_score"]),
                "due_review_flag": getattr(row, columns["due_review_flag"]),
                "due_review_hours": getattr(row, columns["due_review_hours"]),
                "recommended_seen_item": getattr(row, columns["recommended_seen_item"]),
                "student_item_exposure_count": getattr(row, columns["student_item_exposure_count"]),
                "linked_kc_exposure_total": getattr(row, columns["linked_kc_exposure_total"]),
                "candidate_count": int(row.candidate_count),
                "actual_next_probability": float(row.actual_next_probability),
                "actual_target_gap": getattr(row, columns["actual_target_gap"]),
                "recent_failure_total": float(row.recent_failure_total),
                "recent_success_total": float(row.recent_success_total),
                "recent_success_rate_kc": float(row.recent_success_rate_kc),
                "recent_success_rate_attempt": float(row.recent_success_rate_attempt),
                "recent_hint_rate": float(row.recent_hint_rate),
                "recent_selection_change_rate": float(row.recent_selection_change_rate),
                "response_time_inflation": float(row.response_time_inflation),
                "baseline_duration_median": float(row.baseline_duration_median),
                "recent_duration_median": float(row.recent_duration_median),
                "recent_attempt_count": int(row.recent_attempt_count),
                "balanced_reference_probability": float(row.balanced_reference_probability),
                "mean_unseen_probability": float(row.mean_unseen_probability),
                "due_review_available": int(route_flags["due_review_ready"]),
                "high_recent_failure_context": int(route_flags["high_recent_failure"]),
                "high_friction_context": int(route_flags["high_friction"]),
                "lower_predicted_proficiency": int(route_flags["low_predicted_proficiency"]),
                "early_step_context": int(route_flags["early_step"]),
                "actual_kc_count": int(row.actual_kc_count),
                "actual_single_kc": int(row.actual_single_kc),
                "recommended_kc_count": recommended_kc_count,
                "recommended_single_kc": recommended_single_kc,
                "track": str(row.track),
                "model_name": str(row.model_name),
                "history_mode": str(row.history_mode),
                "decay_alpha": float(row.decay_alpha),
                "router_early_step_cutoff": int(early_step_cutoff),
                "router_low_proficiency_threshold": float(low_proficiency_threshold),
                "router_recent_failure_threshold": float(recent_failure_threshold),
                "router_friction_rule": friction_rule_name,
            }
        )

    routed_rows = pd.DataFrame(routed_records)
    overall_summary = summarize_policy_rows(routed_rows.copy(), max_eval_step=int(config.get("max_eval_step", 10)))
    new_learning_rows = routed_rows.loc[routed_rows["selected_policy_name"] != "spacing_aware_review"].copy()
    review_rows = routed_rows.loc[routed_rows["selected_policy_name"] == "spacing_aware_review"].copy()
    new_learning_summary = summarize_policy_rows(new_learning_rows.copy(), max_eval_step=int(config.get("max_eval_step", 10)))
    review_summary = summarize_policy_rows(review_rows.copy(), max_eval_step=int(config.get("max_eval_step", 10)))

    keys = ["student_id", "attempt_id", "actual_item_id", "actual_correct", "eval_step"]
    baseline_source_rows = base_rows.merge(
        new_learning_rows.loc[:, keys].drop_duplicates(),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    baseline_summaries: dict[str, dict[str, float | int]] = {}
    for policy_name in FIXED_NEW_LEARNING_POLICIES:
        baseline_rows = build_standardized_rows(baseline_source_rows, policy_name=policy_name)
        baseline_summaries[policy_name] = summarize_policy_rows(
            baseline_rows,
            max_eval_step=int(config.get("max_eval_step", 10)),
        )

    best_baseline_name = min(
        ("balanced_challenge", "confidence_building"),
        key=lambda policy: float(baseline_summaries[policy]["student_avg_target_gap_1_10"]),
    )
    best_baseline_summary = baseline_summaries[best_baseline_name]
    comparison = {
        "best_fixed_new_learning_baseline_name": best_baseline_name,
        "router_new_learning_target_gap_1_10": float(new_learning_summary["student_avg_target_gap_1_10"]),
        "best_fixed_new_learning_target_gap_1_10": float(best_baseline_summary["student_avg_target_gap_1_10"]),
        "delta_router_minus_best_fixed_target_gap_1_10": float(
            new_learning_summary["student_avg_target_gap_1_10"] - best_baseline_summary["student_avg_target_gap_1_10"]
        ),
        "router_new_learning_policy_advantage_1_10": float(new_learning_summary["policy_advantage_over_actual_1_10"]),
        "best_fixed_new_learning_policy_advantage_1_10": float(
            best_baseline_summary["policy_advantage_over_actual_1_10"]
        ),
        "delta_router_minus_best_fixed_policy_advantage_1_10": float(
            new_learning_summary["policy_advantage_over_actual_1_10"]
            - best_baseline_summary["policy_advantage_over_actual_1_10"]
        ),
        "router_new_learning_stability": float(new_learning_summary["recommendation_stability_mean_abs_diff"]),
        "best_fixed_new_learning_stability": float(best_baseline_summary["recommendation_stability_mean_abs_diff"]),
        "delta_router_minus_best_fixed_stability": float(
            new_learning_summary["recommendation_stability_mean_abs_diff"]
            - best_baseline_summary["recommendation_stability_mean_abs_diff"]
        ),
        "promotion_passes_primary_rule": bool(
            float(new_learning_summary["student_avg_target_gap_1_10"])
            < float(best_baseline_summary["student_avg_target_gap_1_10"])
        ),
    }

    route_counts = routed_rows["selected_policy_name"].value_counts(normalize=False).sort_index().to_dict()
    route_shares = routed_rows["selected_policy_name"].value_counts(normalize=True).sort_index().to_dict()
    route_reason_counts = routed_rows["route_reason"].value_counts(normalize=False).sort_index().to_dict()

    summary = {
        "overall_summary": overall_summary,
        "new_learning_summary": new_learning_summary,
        "review_summary": review_summary,
        "baseline_summaries": baseline_summaries,
        "comparison_to_best_fixed_new_learning_baseline": comparison,
        "route_counts": route_counts,
        "route_shares": route_shares,
        "route_reason_counts": route_reason_counts,
    }
    return routed_rows, summary


def select_best_grid_row(grid_df: pd.DataFrame) -> pd.Series:
    sort_df = grid_df.sort_values(
        [
            "router_new_learning_target_gap_1_10",
            "router_new_learning_policy_advantage_1_10",
            "router_new_learning_stability",
        ],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return sort_df.iloc[0]


def load_numeric_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in frame.columns:
        if column in {"student_id", "actual_item_id", "policy_name", "candidate_pool_mode", "recommended_item_id", "fallback_used", "track", "model_name", "history_mode", "selected_policy_name", "route_reason"}:
            continue
        try:
            frame[column] = pd.to_numeric(frame[column])
        except (TypeError, ValueError):
            pass
    if "student_id" in frame.columns:
        frame["student_id"] = frame["student_id"].astype(str)
    if "actual_item_id" in frame.columns:
        frame["actual_item_id"] = frame["actual_item_id"].astype(str)
    if "recommended_item_id" in frame.columns:
        frame["recommended_item_id"] = frame["recommended_item_id"].astype(str)
    return frame


def rename_policy_columns(frame: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    keys = ["student_id", "attempt_id", "actual_item_id", "actual_correct", "eval_step"]
    renamed = frame.loc[:, [column for column in frame.columns if column not in {"policy_name"}]].copy()
    prefix = f"{policy_name}__"
    rename_map = {column: prefix + column for column in renamed.columns if column not in keys}
    return renamed.rename(columns=rename_map)


def build_base_rows_from_precomputed_outputs(config: dict) -> pd.DataFrame:
    context_rows = load_numeric_csv(Path(config["context_rows_path"]))
    context_rows = context_rows.drop_duplicates(subset=["student_id", "attempt_id", "actual_item_id", "eval_step"]).copy()

    fixed_rows = load_numeric_csv(Path(config["new_learning_policy_rows_path"]))
    fixed_rows = fixed_rows.loc[fixed_rows["policy_name"].isin(FIXED_NEW_LEARNING_POLICIES)].copy()

    spacing_rows = load_numeric_csv(Path(config["spacing_policy_rows_path"]))
    spacing_rows = spacing_rows.loc[spacing_rows["policy_name"] == "spacing_aware_review"].copy()

    keys = ["student_id", "attempt_id", "actual_item_id", "actual_correct", "eval_step"]
    base_rows = context_rows.loc[
        :,
        [
            "student_id",
            "attempt_id",
            "actual_item_id",
            "actual_correct",
            "eval_step",
            "candidate_count",
            "actual_next_probability",
            "recent_success_total",
            "recent_failure_total",
            "recent_success_rate_kc",
            "recent_success_rate_attempt",
            "recent_hint_rate",
            "recent_selection_change_rate",
            "response_time_inflation",
            "baseline_duration_median",
            "recent_duration_median",
            "recent_attempt_count",
            "balanced_reference_probability",
            "mean_unseen_probability",
            "due_review_available",
            "actual_kc_count",
            "actual_single_kc",
            "track",
            "model_name",
            "history_mode",
            "decay_alpha",
        ],
    ].copy()

    for policy_name in FIXED_NEW_LEARNING_POLICIES:
        policy_frame = rename_policy_columns(
            fixed_rows.loc[fixed_rows["policy_name"] == policy_name].copy(),
            policy_name,
        )
        base_rows = base_rows.merge(policy_frame, on=keys, how="left", validate="one_to_one")

    spacing_frame = rename_policy_columns(spacing_rows.copy(), "spacing_aware_review")
    base_rows = base_rows.merge(spacing_frame, on=keys, how="left", validate="one_to_one")
    return base_rows


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    if {"context_rows_path", "new_learning_policy_rows_path", "spacing_policy_rows_path"}.issubset(config.keys()):
        base_rows = build_base_rows_from_precomputed_outputs(config)
        base_row_output_path = Path(config["base_row_output_path"])
        ensure_parent(base_row_output_path)
        base_rows.to_csv(base_row_output_path, index=False)

        recent_failure_quantiles = [float(value) for value in config["recent_failure_quantiles"]]
        low_proficiency_quantiles = [float(value) for value in config["low_proficiency_quantiles"]]
        early_step_cutoffs = [int(value) for value in config["early_step_cutoffs"]]
        friction_rule_names = [str(value) for value in config["friction_rule_names"]]

        recent_failure_thresholds = {
            quantile: float(base_rows["recent_failure_total"].quantile(quantile))
            for quantile in recent_failure_quantiles
        }
        low_proficiency_thresholds = {
            quantile: float(base_rows["balanced_reference_probability"].quantile(quantile))
            for quantile in low_proficiency_quantiles
        }

        grid_records: list[dict[str, object]] = []
        for early_step_cutoff, low_proficiency_quantile, recent_failure_quantile, friction_rule_name in product(
            early_step_cutoffs,
            low_proficiency_quantiles,
            recent_failure_quantiles,
            friction_rule_names,
        ):
            low_proficiency_threshold = low_proficiency_thresholds[float(low_proficiency_quantile)]
            recent_failure_threshold = recent_failure_thresholds[float(recent_failure_quantile)]
            routed_rows, routed_summary = assemble_routed_rows(
                base_rows,
                early_step_cutoff=early_step_cutoff,
                low_proficiency_threshold=low_proficiency_threshold,
                recent_failure_threshold=recent_failure_threshold,
                friction_rule_name=friction_rule_name,
                config=config,
            )
            comparison = routed_summary["comparison_to_best_fixed_new_learning_baseline"]
            grid_records.append(
                {
                    "early_step_cutoff": int(early_step_cutoff),
                    "low_proficiency_quantile": float(low_proficiency_quantile),
                    "low_proficiency_threshold": float(low_proficiency_threshold),
                    "recent_failure_quantile": float(recent_failure_quantile),
                    "recent_failure_threshold": float(recent_failure_threshold),
                    "friction_rule_name": friction_rule_name,
                    "overall_target_gap_1_10": float(routed_summary["overall_summary"]["student_avg_target_gap_1_10"]),
                    "new_learning_target_gap_1_10": float(routed_summary["new_learning_summary"]["student_avg_target_gap_1_10"]),
                    "new_learning_policy_advantage_1_10": float(routed_summary["new_learning_summary"]["policy_advantage_over_actual_1_10"]),
                    "new_learning_stability": float(routed_summary["new_learning_summary"]["recommendation_stability_mean_abs_diff"]),
                    "review_target_gap_1_10": float(routed_summary["review_summary"]["student_avg_target_gap_1_10"]),
                    "review_seen_item_rate": float(routed_summary["review_summary"]["seen_item_recommendation_rate"]),
                    "review_fallback_rate": float(routed_summary["review_summary"]["fallback_rate"]),
                    "review_due_review_coverage_rate": float(routed_summary["review_summary"]["due_review_coverage_rate"]),
                    "route_share_balanced": float(routed_summary["route_shares"].get("balanced_challenge", 0.0)),
                    "route_share_confidence": float(routed_summary["route_shares"].get("confidence_building", 0.0)),
                    "route_share_spacing": float(routed_summary["route_shares"].get("spacing_aware_review", 0.0)),
                    "best_fixed_new_learning_baseline_name": str(comparison["best_fixed_new_learning_baseline_name"]),
                    "best_fixed_new_learning_target_gap_1_10": float(comparison["best_fixed_new_learning_target_gap_1_10"]),
                    "delta_router_minus_best_fixed_target_gap_1_10": float(comparison["delta_router_minus_best_fixed_target_gap_1_10"]),
                    "delta_router_minus_best_fixed_policy_advantage_1_10": float(comparison["delta_router_minus_best_fixed_policy_advantage_1_10"]),
                    "delta_router_minus_best_fixed_stability": float(comparison["delta_router_minus_best_fixed_stability"]),
                    "promotion_passes_primary_rule": int(comparison["promotion_passes_primary_rule"]),
                }
            )

        grid_df = pd.DataFrame(grid_records)
        best_router_selector = select_best_grid_row(
            grid_df.rename(
                columns={
                    "new_learning_target_gap_1_10": "router_new_learning_target_gap_1_10",
                    "new_learning_policy_advantage_1_10": "router_new_learning_policy_advantage_1_10",
                    "new_learning_stability": "router_new_learning_stability",
                }
            )
        )
        best_router_rows, best_router_summary = assemble_routed_rows(
            base_rows,
            early_step_cutoff=int(best_router_selector["early_step_cutoff"]),
            low_proficiency_threshold=float(best_router_selector["low_proficiency_threshold"]),
            recent_failure_threshold=float(best_router_selector["recent_failure_threshold"]),
            friction_rule_name=str(best_router_selector["friction_rule_name"]),
            config=config,
        )

        grid_output_path = Path(config["grid_output_path"])
        ensure_parent(grid_output_path)
        grid_df.to_csv(grid_output_path, index=False)

        selected_row_output_path = Path(config["selected_row_output_path"])
        ensure_parent(selected_row_output_path)
        best_router_rows.to_csv(selected_row_output_path, index=False)

        selected_summary = {
            "policy_name": "simple_two_mode_router",
            "scorer_model_name": "model2_qmatrix_rpfa",
            "history_mode": "rpfa",
            "decay_alpha": float(config["decay_alpha"]),
            "due_review_hours": float(config["due_review_hours"]),
            "operational_freeze": {
                "scorer": "explicit_qmatrix_rpfa_model2",
                "decay_alpha": float(config["decay_alpha"]),
                "spacing_due_review_hours": float(config["due_review_hours"]),
                "model3_role": "exploratory_uncertainty_signal_only",
            },
            "search_space": {
                "early_step_cutoffs": early_step_cutoffs,
                "low_proficiency_quantiles": low_proficiency_quantiles,
                "recent_failure_quantiles": recent_failure_quantiles,
                "friction_rule_names": friction_rule_names,
            },
            "quantile_thresholds": {
                "low_proficiency_thresholds": low_proficiency_thresholds,
                "recent_failure_thresholds": recent_failure_thresholds,
            },
            "selected_thresholds": {
                "early_step_cutoff": int(best_router_selector["early_step_cutoff"]),
                "low_proficiency_quantile": float(best_router_selector["low_proficiency_quantile"]),
                "low_proficiency_threshold": float(best_router_selector["low_proficiency_threshold"]),
                "recent_failure_quantile": float(best_router_selector["recent_failure_quantile"]),
                "recent_failure_threshold": float(best_router_selector["recent_failure_threshold"]),
                "friction_rule_name": str(best_router_selector["friction_rule_name"]),
            },
            "base_row_output_path": str(base_row_output_path),
            "grid_output_path": str(grid_output_path),
            "selected_row_output_path": str(selected_row_output_path),
            "overall_summary": best_router_summary["overall_summary"],
            "new_learning_summary": best_router_summary["new_learning_summary"],
            "review_summary": best_router_summary["review_summary"],
            "baseline_summaries": best_router_summary["baseline_summaries"],
            "comparison_to_best_fixed_new_learning_baseline": best_router_summary["comparison_to_best_fixed_new_learning_baseline"],
            "route_counts": best_router_summary["route_counts"],
            "route_shares": best_router_summary["route_shares"],
            "route_reason_counts": best_router_summary["route_reason_counts"],
        }

        selected_summary_output_path = Path(config["selected_summary_output_path"])
        write_json(selected_summary_output_path, selected_summary)

        print(f"Saved simple-router base rows to {base_row_output_path}")
        print(f"Saved simple-router grid search to {grid_output_path}")
        print(f"Saved selected simple-router rows to {selected_row_output_path}")
        print(f"Saved selected simple-router summary to {selected_summary_output_path}")
        return 0

    trials = load_trials(Path(config["processed_trials_path"]))
    attempt_kc_long = prepare_attempt_kc_long_for_history(
        load_attempt_kc_long_pfa(Path(config["attempt_kc_long_path"])),
        history_mode="rpfa",
        decay_alpha=float(config["decay_alpha"]),
        due_review_hours=float(config["due_review_hours"]),
    )
    posterior = np.load(Path(config["posterior_draws_path"]), allow_pickle=True)

    student_levels = [str(value) for value in posterior["student_levels"].tolist()]
    item_levels = [str(value) for value in posterior["item_levels"].tolist()]
    kc_levels = [str(value) for value in posterior["kc_levels"].tolist()]
    student_lookup = {value: index for index, value in enumerate(student_levels)}
    item_lookup = {value: index for index, value in enumerate(item_levels)}
    kc_lookup = {value: index for index, value in enumerate(kc_levels)}

    item_kc_matrix, item_to_kc_indices = build_item_kc_lookup(attempt_kc_long, item_levels, kc_levels)
    attempt_event_lookup = build_attempt_event_lookup(attempt_kc_long, kc_lookup=kc_lookup)

    posterior_means = {
        "Intercept_mean": float(posterior["Intercept"].mean()),
        "student_intercept_mean": posterior["student_intercept"].mean(axis=0),
        "student_slope_mean": posterior["student_slope"].mean(axis=0),
        "kc_success_mean": posterior["kc_success"].mean(axis=0),
        "kc_failure_mean": posterior["kc_failure"].mean(axis=0),
    }
    item_effect_mean = posterior["item_effect"].mean(axis=0)
    kc_intercept_mean = posterior["kc_intercept"].mean(axis=0)
    posterior.close()

    static_item_term = item_effect_mean + item_kc_matrix @ kc_intercept_mean

    train_df = trials.loc[trials["split"] == "train"].copy()
    test_df = trials.loc[trials["split"] == "test"].copy()

    recent_attempt_window_size = int(config.get("recent_attempt_window_size", 5))
    duration_cap_seconds = float(config.get("duration_cap_seconds", 600.0))
    default_duration_seconds = float(config.get("default_duration_seconds", 30.0))
    max_eval_step = int(config.get("max_eval_step", 10))
    primary_eval_only = bool(config.get("primary_eval_only", True))

    base_records: list[dict[str, object]] = []
    for student_id, student_test in test_df.groupby("student_id", sort=False):
        student_id = str(student_id)
        if student_id not in student_lookup:
            continue
        student_index = student_lookup[student_id]
        train_history = train_df.loc[train_df["student_id"] == student_id].sort_values(
            ["timestamp", "attempt_id"],
            kind="mergesort",
        )
        student_test = student_test.sort_values(["timestamp", "attempt_id"], kind="mergesort")

        opportunity_counts = np.zeros(len(kc_levels), dtype=np.float64)
        success_decay = np.zeros(len(kc_levels), dtype=np.float64)
        failure_decay = np.zeros(len(kc_levels), dtype=np.float64)
        last_seen_timestamp_ns = np.full(len(kc_levels), -1, dtype=np.int64)
        item_exposure_counts: dict[str, int] = {}

        recent_correct_window: deque[int] = deque(maxlen=recent_attempt_window_size)
        recent_hint_window: deque[int] = deque(maxlen=recent_attempt_window_size)
        recent_selection_change_window: deque[int] = deque(maxlen=recent_attempt_window_size)
        recent_duration_window: deque[float] = deque(maxlen=recent_attempt_window_size)
        baseline_duration_values: list[float] = []

        def update_behavior_history(row_like) -> None:
            recent_correct_window.append(int(row_like.correct))
            hint_used = int(getattr(row_like, "hint_used", 0))
            recent_hint_window.append(hint_used)
            selection_change = int(getattr(row_like, "selection_change", 0))
            recent_selection_change_window.append(int(selection_change > 1))
            duration_seconds = float(getattr(row_like, "duration_seconds", np.nan))
            if np.isfinite(duration_seconds) and duration_seconds > 0:
                clipped = min(duration_seconds, duration_cap_seconds)
                recent_duration_window.append(clipped)
                baseline_duration_values.append(clipped)

        def update_from_attempt(attempt_id: int) -> None:
            event = attempt_event_lookup[int(attempt_id)]
            timestamp_ns = int(event["timestamp"].to_datetime64().astype("datetime64[ns]").astype("int64"))
            item_id_local = str(event["item_id"])
            item_exposure_counts[item_id_local] = item_exposure_counts.get(item_id_local, 0) + 1
            for row_event in event["rows"]:
                kc_index = int(row_event["kc_index"])
                opportunity_counts[kc_index] += float(row_event["kc_exposure_increment"])
                success_decay[kc_index] = float(config["decay_alpha"]) * (
                    success_decay[kc_index] + float(row_event["kc_success_increment"])
                )
                failure_decay[kc_index] = float(config["decay_alpha"]) * (
                    failure_decay[kc_index] + float(row_event["kc_failure_increment"])
                )
                last_seen_timestamp_ns[kc_index] = timestamp_ns

        for row in train_history.itertuples(index=False):
            update_from_attempt(int(row.attempt_id))
            update_behavior_history(row)

        eval_step = 0
        for row in student_test.itertuples(index=False):
            step_is_eval = True
            if primary_eval_only and int(row.primary_eval_eligible) != 1:
                step_is_eval = False
            if step_is_eval:
                eval_step += 1
                if eval_step > max_eval_step:
                    step_is_eval = False

            if step_is_eval:
                current_timestamp_ns = int(pd.Timestamp(row.timestamp).to_datetime64().astype("datetime64[ns]").astype("int64"))
                candidate_indices = np.arange(len(item_levels), dtype=np.int64)
                practice_vector = np.log1p(opportunity_counts)
                candidate_probabilities = score_candidates_model2(
                    posterior_means,
                    student_index=student_index,
                    candidate_indices=candidate_indices,
                    item_kc_matrix=item_kc_matrix,
                    success_feature_vector=success_decay,
                    failure_feature_vector=failure_decay,
                    practice_vector=practice_vector,
                    static_item_term=static_item_term,
                )
                candidates = build_candidate_frame(
                    item_levels=item_levels,
                    candidate_probabilities=candidate_probabilities,
                    item_to_kc_indices=item_to_kc_indices,
                    item_exposure_counts=item_exposure_counts,
                    opportunity_counts=opportunity_counts,
                    failure_feature_vector=failure_decay,
                    current_timestamp_ns=current_timestamp_ns,
                    last_seen_timestamp_ns=last_seen_timestamp_ns,
                    due_review_hours_threshold=float(config["due_review_hours"]),
                )
                actual_next_probability = float(candidate_probabilities[item_lookup[str(row.item_id)]])
                due_review_available = int((candidates["due_review_flag"] == 1).any())
                behavior_state = compute_behavior_state(
                    recent_correct_window=recent_correct_window,
                    recent_hint_window=recent_hint_window,
                    recent_selection_change_window=recent_selection_change_window,
                    recent_duration_window=recent_duration_window,
                    baseline_duration_values=baseline_duration_values,
                    default_duration_seconds=default_duration_seconds,
                )

                policy_payload: dict[str, object] = {}
                for policy_name in ROUTER_POLICIES + ("harder_challenge",):
                    selected, policy_meta = choose_policy_item(policy_name, candidates)
                    recommended_item_id = str(selected["item_id"])
                    recommended_kc_count = len(item_to_kc_indices.get(recommended_item_id, []))
                    policy_payload.update(
                        extract_policy_result(
                            policy_name=policy_name,
                            selected=selected,
                            policy_meta=policy_meta,
                            actual_next_probability=actual_next_probability,
                            recommended_kc_count=recommended_kc_count,
                        )
                    )

                recent_success_total = float(success_decay.sum())
                recent_failure_total = float(failure_decay.sum())
                balanced_probability = float(
                    policy_payload[policy_result_columns("balanced_challenge")["recommended_probability"]]
                )
                unseen_candidates = candidates.loc[candidates["is_unseen_candidate"] == 1]
                mean_unseen_probability = (
                    float(unseen_candidates["predicted_probability"].mean())
                    if len(unseen_candidates)
                    else float(candidates["predicted_probability"].mean())
                )

                base_records.append(
                    {
                        "student_id": student_id,
                        "attempt_id": int(row.attempt_id),
                        "actual_item_id": str(row.item_id),
                        "actual_correct": int(row.correct),
                        "eval_step": eval_step,
                        "candidate_count": int(len(candidates)),
                        "actual_next_probability": actual_next_probability,
                        "due_review_available": due_review_available,
                        "recent_success_total": recent_success_total,
                        "recent_failure_total": recent_failure_total,
                        "recent_success_rate_kc": (
                            recent_success_total / (recent_success_total + recent_failure_total)
                            if (recent_success_total + recent_failure_total) > 0
                            else 0.5
                        ),
                        "recent_success_rate_attempt": float(behavior_state["recent_success_rate_attempt"]),
                        "recent_hint_rate": float(behavior_state["recent_hint_rate"]),
                        "recent_selection_change_rate": float(behavior_state["recent_selection_change_rate"]),
                        "response_time_inflation": float(behavior_state["response_time_inflation"]),
                        "baseline_duration_median": float(behavior_state["baseline_duration_median"]),
                        "recent_duration_median": float(behavior_state["recent_duration_median"]),
                        "recent_attempt_count": int(behavior_state["recent_attempt_count"]),
                        "balanced_reference_probability": balanced_probability,
                        "mean_unseen_probability": mean_unseen_probability,
                        "actual_kc_count": int(row.kc_count),
                        "actual_single_kc": int(int(row.kc_count) == 1),
                        "track": str(config.get("track_name", "phase1_adaptive_policy_simple_router_qmatrix_rpfa")),
                        "model_name": "model2_qmatrix_rpfa",
                        "history_mode": "rpfa",
                        "decay_alpha": float(config["decay_alpha"]),
                        **policy_payload,
                    }
                )

            update_from_attempt(int(row.attempt_id))
            update_behavior_history(row)

    base_rows = pd.DataFrame(base_records)
    base_row_output_path = Path(config["base_row_output_path"])
    ensure_parent(base_row_output_path)
    base_rows.to_csv(base_row_output_path, index=False)

    recent_failure_quantiles = [float(value) for value in config["recent_failure_quantiles"]]
    low_proficiency_quantiles = [float(value) for value in config["low_proficiency_quantiles"]]
    early_step_cutoffs = [int(value) for value in config["early_step_cutoffs"]]
    friction_rule_names = [str(value) for value in config["friction_rule_names"]]

    recent_failure_thresholds = {
        quantile: float(base_rows["recent_failure_total"].quantile(quantile)) for quantile in recent_failure_quantiles
    }
    low_proficiency_thresholds = {
        quantile: float(base_rows["balanced_reference_probability"].quantile(quantile))
        for quantile in low_proficiency_quantiles
    }

    grid_records: list[dict[str, object]] = []
    for early_step_cutoff, low_proficiency_quantile, recent_failure_quantile, friction_rule_name in product(
        early_step_cutoffs,
        low_proficiency_quantiles,
        recent_failure_quantiles,
        friction_rule_names,
    ):
        low_proficiency_threshold = low_proficiency_thresholds[float(low_proficiency_quantile)]
        recent_failure_threshold = recent_failure_thresholds[float(recent_failure_quantile)]

        routed_rows, routed_summary = assemble_routed_rows(
            base_rows,
            early_step_cutoff=early_step_cutoff,
            low_proficiency_threshold=low_proficiency_threshold,
            recent_failure_threshold=recent_failure_threshold,
            friction_rule_name=friction_rule_name,
            config=config,
        )
        comparison = routed_summary["comparison_to_best_fixed_new_learning_baseline"]
        grid_records.append(
            {
                "early_step_cutoff": int(early_step_cutoff),
                "low_proficiency_quantile": float(low_proficiency_quantile),
                "low_proficiency_threshold": float(low_proficiency_threshold),
                "recent_failure_quantile": float(recent_failure_quantile),
                "recent_failure_threshold": float(recent_failure_threshold),
                "friction_rule_name": friction_rule_name,
                "overall_target_gap_1_10": float(routed_summary["overall_summary"]["student_avg_target_gap_1_10"]),
                "new_learning_target_gap_1_10": float(routed_summary["new_learning_summary"]["student_avg_target_gap_1_10"]),
                "new_learning_policy_advantage_1_10": float(
                    routed_summary["new_learning_summary"]["policy_advantage_over_actual_1_10"]
                ),
                "new_learning_stability": float(
                    routed_summary["new_learning_summary"]["recommendation_stability_mean_abs_diff"]
                ),
                "review_target_gap_1_10": float(routed_summary["review_summary"]["student_avg_target_gap_1_10"]),
                "review_seen_item_rate": float(routed_summary["review_summary"]["seen_item_recommendation_rate"]),
                "review_fallback_rate": float(routed_summary["review_summary"]["fallback_rate"]),
                "review_due_review_coverage_rate": float(routed_summary["review_summary"]["due_review_coverage_rate"]),
                "route_share_balanced": float(routed_summary["route_shares"].get("balanced_challenge", 0.0)),
                "route_share_confidence": float(routed_summary["route_shares"].get("confidence_building", 0.0)),
                "route_share_spacing": float(routed_summary["route_shares"].get("spacing_aware_review", 0.0)),
                "best_fixed_new_learning_baseline_name": str(
                    comparison["best_fixed_new_learning_baseline_name"]
                ),
                "best_fixed_new_learning_target_gap_1_10": float(
                    comparison["best_fixed_new_learning_target_gap_1_10"]
                ),
                "delta_router_minus_best_fixed_target_gap_1_10": float(
                    comparison["delta_router_minus_best_fixed_target_gap_1_10"]
                ),
                "delta_router_minus_best_fixed_policy_advantage_1_10": float(
                    comparison["delta_router_minus_best_fixed_policy_advantage_1_10"]
                ),
                "delta_router_minus_best_fixed_stability": float(
                    comparison["delta_router_minus_best_fixed_stability"]
                ),
                "promotion_passes_primary_rule": int(comparison["promotion_passes_primary_rule"]),
            }
        )

    grid_df = pd.DataFrame(grid_records)
    best_router_selector = select_best_grid_row(
        grid_df.rename(
            columns={
                "new_learning_target_gap_1_10": "router_new_learning_target_gap_1_10",
                "new_learning_policy_advantage_1_10": "router_new_learning_policy_advantage_1_10",
                "new_learning_stability": "router_new_learning_stability",
            }
        )
    )
    best_router_rows, best_router_summary = assemble_routed_rows(
        base_rows,
        early_step_cutoff=int(best_router_selector["early_step_cutoff"]),
        low_proficiency_threshold=float(best_router_selector["low_proficiency_threshold"]),
        recent_failure_threshold=float(best_router_selector["recent_failure_threshold"]),
        friction_rule_name=str(best_router_selector["friction_rule_name"]),
        config=config,
    )

    grid_output_path = Path(config["grid_output_path"])
    ensure_parent(grid_output_path)
    grid_df.to_csv(grid_output_path, index=False)

    selected_row_output_path = Path(config["selected_row_output_path"])
    ensure_parent(selected_row_output_path)
    best_router_rows.to_csv(selected_row_output_path, index=False)

    selected_summary = {
        "policy_name": "simple_two_mode_router",
        "scorer_model_name": "model2_qmatrix_rpfa",
        "history_mode": "rpfa",
        "decay_alpha": float(config["decay_alpha"]),
        "due_review_hours": float(config["due_review_hours"]),
        "operational_freeze": {
            "scorer": "explicit_qmatrix_rpfa_model2",
            "decay_alpha": float(config["decay_alpha"]),
            "spacing_due_review_hours": float(config["due_review_hours"]),
            "model3_role": "exploratory_uncertainty_signal_only",
        },
        "search_space": {
            "early_step_cutoffs": early_step_cutoffs,
            "low_proficiency_quantiles": low_proficiency_quantiles,
            "recent_failure_quantiles": recent_failure_quantiles,
            "friction_rule_names": friction_rule_names,
        },
        "quantile_thresholds": {
            "low_proficiency_thresholds": low_proficiency_thresholds,
            "recent_failure_thresholds": recent_failure_thresholds,
        },
        "selected_thresholds": {
            "early_step_cutoff": int(best_router_selector["early_step_cutoff"]),
            "low_proficiency_quantile": float(best_router_selector["low_proficiency_quantile"]),
            "low_proficiency_threshold": float(best_router_selector["low_proficiency_threshold"]),
            "recent_failure_quantile": float(best_router_selector["recent_failure_quantile"]),
            "recent_failure_threshold": float(best_router_selector["recent_failure_threshold"]),
            "friction_rule_name": str(best_router_selector["friction_rule_name"]),
        },
        "base_row_output_path": str(base_row_output_path),
        "grid_output_path": str(grid_output_path),
        "selected_row_output_path": str(selected_row_output_path),
        "overall_summary": best_router_summary["overall_summary"],
        "new_learning_summary": best_router_summary["new_learning_summary"],
        "review_summary": best_router_summary["review_summary"],
        "baseline_summaries": best_router_summary["baseline_summaries"],
        "comparison_to_best_fixed_new_learning_baseline": best_router_summary[
            "comparison_to_best_fixed_new_learning_baseline"
        ],
        "route_counts": best_router_summary["route_counts"],
        "route_shares": best_router_summary["route_shares"],
        "route_reason_counts": best_router_summary["route_reason_counts"],
    }

    selected_summary_output_path = Path(config["selected_summary_output_path"])
    write_json(selected_summary_output_path, selected_summary)

    print(f"Saved simple-router base rows to {base_row_output_path}")
    print(f"Saved simple-router grid search to {grid_output_path}")
    print(f"Saved selected simple-router rows to {selected_row_output_path}")
    print(f"Saved selected simple-router summary to {selected_summary_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
