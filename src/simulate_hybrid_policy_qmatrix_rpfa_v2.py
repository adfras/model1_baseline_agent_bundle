from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from policy_suite_common import (
    build_attempt_event_lookup,
    build_candidate_frame,
    build_item_kc_lookup,
    choose_policy_item,
    model3_future_state_summary,
    score_candidates_model2,
    summarize_policy_rows,
    write_json,
)
from qmatrix_common import load_json, load_trials
from qmatrix_pfa_common import load_attempt_kc_long_pfa, prepare_attempt_kc_long_for_history


DEFAULT_CONFIG_PATH = Path("config/phase1_adaptive_policy_hybrid_qmatrix_rpfa_v2.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hybrid RPFA policy router v2 using Model 2 means and Model 3 uncertainty."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def safe_mean(values: deque[float] | list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_median(values: deque[float] | list[float], default: float) -> float:
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
    failure_streak: int,
    default_duration: float,
) -> dict[str, float]:
    attempt_count = len(recent_correct_window)
    recent_success_rate = safe_mean(recent_correct_window) if attempt_count else 0.5
    recent_failure_rate = 1.0 - recent_success_rate if attempt_count else 0.5
    recent_hint_rate = safe_mean(recent_hint_window)
    recent_selection_change_rate = safe_mean(recent_selection_change_window)
    baseline_duration_median = safe_median(baseline_duration_values, default=default_duration)
    recent_duration_median = safe_median(recent_duration_window, default=baseline_duration_median)
    response_time_inflation = recent_duration_median / max(baseline_duration_median, 1e-6)
    return {
        "recent_attempt_count": float(attempt_count),
        "recent_success_rate_attempt": float(recent_success_rate),
        "recent_failure_rate_attempt": float(recent_failure_rate),
        "recent_hint_rate": float(recent_hint_rate),
        "recent_selection_change_rate": float(recent_selection_change_rate),
        "baseline_duration_median": float(baseline_duration_median),
        "recent_duration_median": float(recent_duration_median),
        "response_time_inflation": float(response_time_inflation),
        "failure_streak": float(failure_streak),
    }


def route_policy_v2(
    *,
    candidates: pd.DataFrame,
    recent_attempt_count: int,
    uncertainty_sd: float,
    recent_success_rate_attempt: float,
    recent_failure_rate_attempt: float,
    recent_hint_rate: float,
    recent_selection_change_rate: float,
    response_time_inflation: float,
    failure_streak: int,
    balanced_reference_probability: float,
    config: dict,
) -> tuple[str, str, dict[str, int | float]]:
    due_review_available = bool((candidates["due_review_flag"] == 1).any())
    attempt_window_min = int(config.get("recent_attempt_window_min", 2))
    failure_streak_threshold = int(config.get("failure_streak_threshold", 2))
    recent_failure_rate_threshold = float(config.get("recent_failure_rate_threshold", 0.50))
    recent_success_high_threshold = float(config.get("recent_success_high_threshold", 0.82))
    recent_success_low_threshold = float(config.get("recent_success_low_threshold", 0.58))
    uncertainty_high_threshold = float(config.get("uncertainty_high_sd_threshold", 0.34))
    uncertainty_low_threshold = float(config.get("uncertainty_low_sd_threshold", 0.12))
    duration_inflation_threshold = float(config.get("duration_inflation_threshold", 1.75))
    hint_rate_threshold = float(config.get("hint_rate_threshold", 0.15))
    selection_change_rate_threshold = float(config.get("selection_change_rate_threshold", 0.35))
    balanced_low_probability_threshold = float(config.get("balanced_low_probability_threshold", 0.68))
    balanced_high_probability_threshold = float(config.get("balanced_high_probability_threshold", 0.78))

    enough_history = recent_attempt_count >= attempt_window_min
    acute_failure = int(
        failure_streak >= failure_streak_threshold or recent_failure_rate_attempt >= recent_failure_rate_threshold
    )
    friction_high = int(
        recent_hint_rate >= hint_rate_threshold
        or recent_selection_change_rate >= selection_change_rate_threshold
        or response_time_inflation >= duration_inflation_threshold
    )
    low_proficiency = int(balanced_reference_probability < balanced_low_probability_threshold)
    high_proficiency = int(balanced_reference_probability >= balanced_high_probability_threshold)
    high_uncertainty = int(uncertainty_sd >= uncertainty_high_threshold)
    low_uncertainty = int(uncertainty_sd < uncertainty_low_threshold)

    if due_review_available and not acute_failure and not friction_high and low_uncertainty:
        return "spacing_aware_review", "due_review_ready", {
            "acute_failure": acute_failure,
            "friction_high": friction_high,
            "low_proficiency": low_proficiency,
            "high_proficiency": high_proficiency,
            "high_uncertainty": high_uncertainty,
            "low_uncertainty": low_uncertainty,
            "due_review_available": int(due_review_available),
        }
    if acute_failure and (friction_high or low_proficiency):
        return "failure_aware_remediation", "acute_failure_or_friction", {
            "acute_failure": acute_failure,
            "friction_high": friction_high,
            "low_proficiency": low_proficiency,
            "high_proficiency": high_proficiency,
            "high_uncertainty": high_uncertainty,
            "low_uncertainty": low_uncertainty,
            "due_review_available": int(due_review_available),
        }
    if high_uncertainty and not acute_failure:
        return "diagnostic_challenge", "high_uncertainty", {
            "acute_failure": acute_failure,
            "friction_high": friction_high,
            "low_proficiency": low_proficiency,
            "high_proficiency": high_proficiency,
            "high_uncertainty": high_uncertainty,
            "low_uncertainty": low_uncertainty,
            "due_review_available": int(due_review_available),
        }
    if (
        recent_success_rate_attempt <= recent_success_low_threshold
        or friction_high
        or low_proficiency
        or not enough_history
    ):
        return "confidence_building", "mild_struggle_or_low_proficiency", {
            "acute_failure": acute_failure,
            "friction_high": friction_high,
            "low_proficiency": low_proficiency,
            "high_proficiency": high_proficiency,
            "high_uncertainty": high_uncertainty,
            "low_uncertainty": low_uncertainty,
            "due_review_available": int(due_review_available),
        }
    if recent_success_rate_attempt >= recent_success_high_threshold and high_proficiency and low_uncertainty and not friction_high:
        return "harder_challenge", "confident_progression", {
            "acute_failure": acute_failure,
            "friction_high": friction_high,
            "low_proficiency": low_proficiency,
            "high_proficiency": high_proficiency,
            "high_uncertainty": high_uncertainty,
            "low_uncertainty": low_uncertainty,
            "due_review_available": int(due_review_available),
        }
    return "balanced_challenge", "default_balanced", {
        "acute_failure": acute_failure,
        "friction_high": friction_high,
        "low_proficiency": low_proficiency,
        "high_proficiency": high_proficiency,
        "high_uncertainty": high_uncertainty,
        "low_uncertainty": low_uncertainty,
        "due_review_available": int(due_review_available),
    }


def summarize_subgroups(rows: pd.DataFrame, *, max_eval_step: int) -> dict[str, dict]:
    subgroup_specs = {
        "review_eligible_context": rows["due_review_available"] == 1,
        "high_recent_failure_context": rows["high_recent_failure_context"] == 1,
        "high_friction_context": rows["high_friction_context"] == 1,
        "lower_predicted_proficiency": rows["lower_predicted_proficiency"] == 1,
        "actual_single_kc": rows["actual_single_kc"] == 1,
        "actual_multi_kc": rows["actual_single_kc"] == 0,
        "early_steps_1_5": rows["eval_step"] <= 5,
        "later_steps_6_10": rows["eval_step"] >= 6,
    }
    summaries: dict[str, dict] = {}
    for name, mask in subgroup_specs.items():
        subset = rows.loc[mask].copy()
        summaries[name] = summarize_policy_rows(subset, max_eval_step=max_eval_step)
    return summaries


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    trials = load_trials(Path(config["processed_trials_path"]))
    attempt_kc_long = prepare_attempt_kc_long_for_history(
        load_attempt_kc_long_pfa(Path(config["attempt_kc_long_path"])),
        history_mode="rpfa",
        decay_alpha=float(config["decay_alpha"]),
        due_review_hours=float(config.get("due_review_hours", 24.0)),
    )

    model2_posterior = np.load(Path(config["model2_posterior_draws_path"]), allow_pickle=True)
    model3_posterior = np.load(Path(config["model3_posterior_draws_path"]), allow_pickle=True)

    student_levels = [str(value) for value in model2_posterior["student_levels"].tolist()]
    item_levels = [str(value) for value in model2_posterior["item_levels"].tolist()]
    kc_levels = [str(value) for value in model2_posterior["kc_levels"].tolist()]
    student_lookup = {value: index for index, value in enumerate(student_levels)}
    item_lookup = {value: index for index, value in enumerate(item_levels)}
    kc_lookup = {value: index for index, value in enumerate(kc_levels)}

    item_kc_matrix, item_to_kc_indices = build_item_kc_lookup(attempt_kc_long, item_levels, kc_levels)
    attempt_event_lookup = build_attempt_event_lookup(attempt_kc_long, kc_lookup=kc_lookup)

    model2_means = {
        "Intercept_mean": float(model2_posterior["Intercept"].mean()),
        "student_intercept_mean": model2_posterior["student_intercept"].mean(axis=0),
        "student_slope_mean": model2_posterior["student_slope"].mean(axis=0),
        "kc_success_mean": model2_posterior["kc_success"].mean(axis=0),
        "kc_failure_mean": model2_posterior["kc_failure"].mean(axis=0),
    }
    item_effect_mean = model2_posterior["item_effect"].mean(axis=0)
    kc_intercept_mean = model2_posterior["kc_intercept"].mean(axis=0)
    static_item_term = item_effect_mean + item_kc_matrix @ kc_intercept_mean

    model3_means = {
        "rho_mean": float(model3_posterior["rho"].mean()),
        "state_sigma_student_mean": model3_posterior["state_sigma_student"].mean(axis=0),
        "latent_state_mean": model3_posterior["latent_state"].mean(axis=0),
        "state_bin_width": int(np.asarray(model3_posterior["state_bin_width"]).reshape(-1)[0]),
    }

    model2_posterior.close()
    model3_posterior.close()

    train_df = trials.loc[trials["split"] == "train"].copy()
    test_df = trials.loc[trials["split"] == "test"].copy()

    state_bin_width = int(model3_means["state_bin_width"])
    last_train_bins = (
        train_df.assign(state_bin=(train_df["overall_opportunity"] // state_bin_width).astype("int64"))
        .groupby("student_id", sort=False)["state_bin"]
        .max()
    )

    max_eval_step = int(config.get("max_eval_step", 10))
    primary_eval_only = bool(config.get("primary_eval_only", True))
    due_review_hours_threshold = float(config.get("due_review_hours", 24.0))
    recent_attempt_window_size = int(config.get("recent_attempt_window_size", 5))
    duration_cap_seconds = float(config.get("duration_cap_seconds", 600.0))
    default_duration_seconds = float(config.get("default_duration_seconds", 30.0))

    records: list[dict] = []
    for student_id, student_test in test_df.groupby("student_id", sort=False):
        student_id = str(student_id)
        if student_id not in student_lookup:
            continue
        student_index = student_lookup[student_id]
        train_history = train_df.loc[train_df["student_id"] == student_id].sort_values(
            ["timestamp", "attempt_id"], kind="mergesort"
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
        failure_streak = 0

        def update_behavior_history(row_like) -> None:
            nonlocal failure_streak
            correct = int(row_like.correct)
            recent_correct_window.append(correct)
            failure_streak = 0 if correct == 1 else failure_streak + 1

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
                    model2_means,
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
                    due_review_hours_threshold=due_review_hours_threshold,
                )

                balanced_selected, _ = choose_policy_item("balanced_challenge", candidates)
                balanced_reference_probability = float(balanced_selected["predicted_probability"])
                unseen_candidates = candidates.loc[candidates["is_unseen_candidate"] == 1]
                mean_unseen_probability = (
                    float(unseen_candidates["predicted_probability"].mean())
                    if len(unseen_candidates)
                    else float(candidates["predicted_probability"].mean())
                )
                review_eligible_count = int((candidates["due_review_flag"] == 1).sum())

                last_train_bin = int(last_train_bins.loc[student_id])
                _, future_state_variance = model3_future_state_summary(
                    model3_means,
                    student_index=student_index,
                    step_overall_opportunity=int(row.overall_opportunity),
                    last_train_bin=last_train_bin,
                )
                uncertainty_sd = float(np.sqrt(max(future_state_variance, 0.0)))
                recent_success_total = float(success_decay.sum())
                recent_failure_total = float(failure_decay.sum())
                behavior_state = compute_behavior_state(
                    recent_correct_window=recent_correct_window,
                    recent_hint_window=recent_hint_window,
                    recent_selection_change_window=recent_selection_change_window,
                    recent_duration_window=recent_duration_window,
                    baseline_duration_values=baseline_duration_values,
                    failure_streak=failure_streak,
                    default_duration=default_duration_seconds,
                )
                chosen_policy_name, route_reason, route_flags = route_policy_v2(
                    candidates=candidates,
                    recent_attempt_count=int(behavior_state["recent_attempt_count"]),
                    uncertainty_sd=uncertainty_sd,
                    recent_success_rate_attempt=float(behavior_state["recent_success_rate_attempt"]),
                    recent_failure_rate_attempt=float(behavior_state["recent_failure_rate_attempt"]),
                    recent_hint_rate=float(behavior_state["recent_hint_rate"]),
                    recent_selection_change_rate=float(behavior_state["recent_selection_change_rate"]),
                    response_time_inflation=float(behavior_state["response_time_inflation"]),
                    failure_streak=int(behavior_state["failure_streak"]),
                    balanced_reference_probability=balanced_reference_probability,
                    config=config,
                )
                selected, policy_meta = choose_policy_item(chosen_policy_name, candidates)
                target_probability = float(policy_meta["target_probability"])
                actual_next_probability = float(candidate_probabilities[item_lookup[str(row.item_id)]])
                recommended_item_id = str(selected["item_id"])
                recommended_kc_count = len(item_to_kc_indices.get(recommended_item_id, []))

                records.append(
                    {
                        "student_id": student_id,
                        "attempt_id": int(row.attempt_id),
                        "actual_item_id": str(row.item_id),
                        "actual_correct": int(row.correct),
                        "eval_step": eval_step,
                        "policy_name": "hybrid_uncertainty_router_v2",
                        "selected_policy_name": chosen_policy_name,
                        "route_reason": route_reason,
                        "candidate_pool_mode": str(policy_meta["candidate_pool_mode"]),
                        "recommended_item_id": recommended_item_id,
                        "recommended_probability": float(selected["predicted_probability"]),
                        "target_probability": target_probability,
                        "target_gap": abs(float(selected["predicted_probability"]) - target_probability),
                        "in_target_band": int(policy_meta["in_target_band"]),
                        "fallback_used": str(policy_meta["fallback_used"]),
                        "recent_failure_score": float(selected["recent_failure_score"]),
                        "due_review_flag": int(selected["due_review_flag"]),
                        "due_review_hours": float(selected["due_review_hours"]) if pd.notna(selected["due_review_hours"]) else np.nan,
                        "recommended_seen_item": int(selected["recommended_seen_item"]),
                        "student_item_exposure_count": int(selected["student_item_exposure_count"]),
                        "linked_kc_exposure_total": float(selected["linked_kc_exposure_total"]),
                        "candidate_count": int(len(candidates)),
                        "actual_next_probability": actual_next_probability,
                        "actual_target_gap": abs(actual_next_probability - target_probability),
                        "uncertainty_sd": uncertainty_sd,
                        "recent_success_total": recent_success_total,
                        "recent_failure_total": recent_failure_total,
                        "recent_success_rate_kc": (
                            recent_success_total / (recent_success_total + recent_failure_total)
                            if (recent_success_total + recent_failure_total) > 0
                            else 0.5
                        ),
                        "recent_success_rate_attempt": float(behavior_state["recent_success_rate_attempt"]),
                        "recent_failure_rate_attempt": float(behavior_state["recent_failure_rate_attempt"]),
                        "recent_hint_rate": float(behavior_state["recent_hint_rate"]),
                        "recent_selection_change_rate": float(behavior_state["recent_selection_change_rate"]),
                        "response_time_inflation": float(behavior_state["response_time_inflation"]),
                        "baseline_duration_median": float(behavior_state["baseline_duration_median"]),
                        "recent_duration_median": float(behavior_state["recent_duration_median"]),
                        "failure_streak": int(behavior_state["failure_streak"]),
                        "recent_attempt_count": int(behavior_state["recent_attempt_count"]),
                        "balanced_reference_probability": balanced_reference_probability,
                        "mean_unseen_probability": mean_unseen_probability,
                        "due_review_available": int(route_flags["due_review_available"]),
                        "high_recent_failure_context": int(route_flags["acute_failure"]),
                        "high_friction_context": int(route_flags["friction_high"]),
                        "lower_predicted_proficiency": int(route_flags["low_proficiency"]),
                        "high_predicted_proficiency": int(route_flags["high_proficiency"]),
                        "high_uncertainty_context": int(route_flags["high_uncertainty"]),
                        "low_uncertainty_context": int(route_flags["low_uncertainty"]),
                        "actual_kc_count": int(row.kc_count),
                        "actual_single_kc": int(int(row.kc_count) == 1),
                        "recommended_kc_count": int(recommended_kc_count),
                        "recommended_single_kc": int(recommended_kc_count == 1),
                        "actual_hint_used_previous_proxy": float(behavior_state["recent_hint_rate"]),
                        "track": str(config.get("track_name", "phase1_adaptive_policy_hybrid_qmatrix_rpfa_v2")),
                        "model_name": "hybrid_v2_model2_mean_model3_uncertainty",
                        "history_mode": "rpfa",
                        "decay_alpha": float(config["decay_alpha"]),
                    }
                )

            update_from_attempt(int(row.attempt_id))
            update_behavior_history(row)

    rows = pd.DataFrame(records)
    row_output_path = Path(config["row_output_path"])
    row_output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(row_output_path, index=False)

    overall_summary = summarize_policy_rows(rows.copy(), max_eval_step=max_eval_step)
    route_summaries = {}
    for policy_name, group in rows.groupby("selected_policy_name", sort=True):
        route_summaries[policy_name] = summarize_policy_rows(group.copy(), max_eval_step=max_eval_step)
    subgroup_summaries = summarize_subgroups(rows.copy(), max_eval_step=max_eval_step)

    route_counts = rows["selected_policy_name"].value_counts(normalize=False, sort=True).sort_index().to_dict() if len(rows) else {}
    route_shares = rows["selected_policy_name"].value_counts(normalize=True, sort=True).sort_index().to_dict() if len(rows) else {}
    route_reason_counts = rows["route_reason"].value_counts(normalize=False, sort=True).sort_index().to_dict() if len(rows) else {}

    summary = {
        "policy_name": "hybrid_uncertainty_router_v2",
        "history_mode": "rpfa",
        "decay_alpha": float(config["decay_alpha"]),
        "model2_posterior_draws_path": str(config["model2_posterior_draws_path"]),
        "model3_posterior_draws_path": str(config["model3_posterior_draws_path"]),
        "due_review_hours": due_review_hours_threshold,
        "max_eval_step": max_eval_step,
        "primary_eval_only": primary_eval_only,
        "evaluation_rows": int(len(rows)),
        "evaluation_students": int(rows["student_id"].nunique()) if len(rows) else 0,
        "thresholds": {
            "recent_attempt_window_size": recent_attempt_window_size,
            "recent_attempt_window_min": int(config.get("recent_attempt_window_min", 2)),
            "failure_streak_threshold": int(config.get("failure_streak_threshold", 2)),
            "recent_failure_rate_threshold": float(config.get("recent_failure_rate_threshold", 0.50)),
            "recent_success_high_threshold": float(config.get("recent_success_high_threshold", 0.82)),
            "recent_success_low_threshold": float(config.get("recent_success_low_threshold", 0.58)),
            "uncertainty_high_sd_threshold": float(config.get("uncertainty_high_sd_threshold", 0.34)),
            "uncertainty_low_sd_threshold": float(config.get("uncertainty_low_sd_threshold", 0.12)),
            "duration_inflation_threshold": float(config.get("duration_inflation_threshold", 1.75)),
            "hint_rate_threshold": float(config.get("hint_rate_threshold", 0.15)),
            "selection_change_rate_threshold": float(config.get("selection_change_rate_threshold", 0.35)),
            "balanced_low_probability_threshold": float(config.get("balanced_low_probability_threshold", 0.68)),
            "balanced_high_probability_threshold": float(config.get("balanced_high_probability_threshold", 0.78)),
            "duration_cap_seconds": duration_cap_seconds,
        },
        "overall_summary": overall_summary,
        "route_summaries": route_summaries,
        "subgroup_summaries": subgroup_summaries,
        "route_counts": route_counts,
        "route_shares": route_shares,
        "route_reason_counts": route_reason_counts,
        "row_output_path": str(row_output_path),
    }
    summary_output_path = Path(config["summary_output_path"])
    write_json(summary_output_path, summary)

    print(f"Saved hybrid v2 policy rows to {row_output_path}")
    print(f"Saved hybrid v2 policy summary to {summary_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
