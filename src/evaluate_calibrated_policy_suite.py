from __future__ import annotations

import argparse
import hashlib
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from policy_suite_common import (
    build_attempt_event_lookup,
    build_item_kc_lookup,
    model3_future_state_summary,
    score_candidates_model2,
    summarize_policy_rows,
    write_json,
)
from qmatrix_common import load_json, load_trials
from qmatrix_pfa_common import load_attempt_kc_long_pfa, prepare_attempt_kc_long_for_history
from simulate_hybrid_policy_qmatrix_rpfa_v2 import compute_behavior_state


DEFAULT_CONFIG_PATH = Path("config/phase1_calibrated_policy_suite.json")
NEW_LEARNING_POLICIES = ["balanced_challenge", "harder_challenge", "confidence_building"]
POLICY_SPECS = {
    "balanced_challenge": {"target_probability": 0.72, "target_band_low": 0.65, "target_band_high": 0.80},
    "harder_challenge": {"target_probability": 0.60, "target_band_low": 0.55, "target_band_high": 0.65},
    "confidence_building": {"target_probability": 0.85, "target_band_low": 0.80, "target_band_high": 0.90},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun the fixed Model 2 new-learning policies with a frozen uncertainty calibration side-channel."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def stable_student_bucket(student_id: str, *, modulo: int = 100) -> int:
    digest = hashlib.sha1(student_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def clip_probability(probability: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def apply_calibrator(
    raw_probabilities: np.ndarray,
    *,
    method_name: str,
    context_flags: dict[str, int | float],
    uncertainty_sd: float,
    calibration_summary: dict,
) -> np.ndarray:
    raw_probabilities = clip_probability(raw_probabilities)
    if method_name == "model2_raw":
        return raw_probabilities

    coefficients = calibration_summary["methods"][method_name]["coefficients"]
    z = logit(raw_probabilities)
    linear = np.full_like(z, float(coefficients["intercept"]), dtype=np.float64)

    if method_name == "model2_platt":
        linear = linear + float(coefficients["model2_logit"]) * z
        return clip_probability(expit(linear))

    linear = linear + float(coefficients["model2_logit"]) * z
    for context_name in [
        "early_steps_1_5",
        "due_review_available",
        "high_recent_failure_context",
        "high_friction_context",
        "lower_predicted_proficiency",
    ]:
        linear = linear + float(coefficients[context_name]) * float(context_flags[context_name])

    if method_name == "model2_context_calibrated":
        return clip_probability(expit(linear))

    if method_name != "model2_plus_model3_uncertainty":
        raise ValueError(f"Unsupported method_name: {method_name}")

    standardization = calibration_summary["uncertainty_standardization"]
    band_thresholds = calibration_summary["uncertainty_band_thresholds"]
    uncertainty_z = (float(uncertainty_sd) - float(standardization["mean"])) / max(float(standardization["std"]), 1e-6)
    low_band = float(uncertainty_z <= float(band_thresholds["low_cut"]))
    mid_band = float(
        (uncertainty_z > float(band_thresholds["low_cut"])) and (uncertainty_z <= float(band_thresholds["high_cut"]))
    )
    high_band = float(uncertainty_z > float(band_thresholds["high_cut"]))

    linear = linear + float(coefficients["uncertainty_band_mid"]) * mid_band
    linear = linear + float(coefficients["uncertainty_band_high"]) * high_band
    linear = linear + float(coefficients["model2_logit_x_uncertainty_band_low"]) * z * low_band
    linear = linear + float(coefficients["model2_logit_x_uncertainty_band_mid"]) * z * mid_band
    linear = linear + float(coefficients["model2_logit_x_uncertainty_band_high"]) * z * high_band
    return clip_probability(expit(linear))


def choose_unseen_policy_item(
    predicted_probabilities: np.ndarray,
    *,
    unseen_mask: np.ndarray,
    target_probability: float,
    target_band_low: float,
    target_band_high: float,
    student_item_exposure: np.ndarray,
    linked_kc_exposure_total: np.ndarray,
    item_order_rank: np.ndarray,
) -> tuple[int, int]:
    available = np.where(unseen_mask)[0]
    if len(available) == 0:
        available = np.arange(len(predicted_probabilities), dtype=np.int64)

    gaps = np.abs(predicted_probabilities[available] - float(target_probability))
    order = np.lexsort(
        (
            item_order_rank[available],
            linked_kc_exposure_total[available],
            student_item_exposure[available],
            gaps,
        )
    )
    selected_index = int(available[int(order[0])])
    selected_probability = float(predicted_probabilities[selected_index])
    in_target_band = int(float(target_band_low) <= selected_probability <= float(target_band_high))
    return selected_index, in_target_band


def compare_policy_summary(challenger: dict, baseline: dict) -> dict[str, float]:
    return {
        "target_gap_1_10_delta": float(
            challenger["student_avg_target_gap_1_10"] - baseline["student_avg_target_gap_1_10"]
        ),
        "policy_advantage_1_10_delta": float(
            challenger["policy_advantage_over_actual_1_10"] - baseline["policy_advantage_over_actual_1_10"]
        ),
        "stability_delta": float(
            challenger["recommendation_stability_mean_abs_diff"] - baseline["recommendation_stability_mean_abs_diff"]
        ),
        "band_hit_rate_1_10_delta": float(
            challenger["recommended_target_band_hit_rate_1_10"] - baseline["recommended_target_band_hit_rate_1_10"]
        ),
    }


def average_metric(policy_summaries: dict[str, dict], policy_names: list[str], metric: str) -> float:
    values = [float(policy_summaries[policy_name][metric]) for policy_name in policy_names]
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    hybrid_threshold_config = load_json(Path(config["hybrid_threshold_config_path"]))
    calibration_summary = load_json(Path(config["uncertainty_calibration_summary_path"]))

    trials = load_trials(Path(config["processed_trials_path"]))
    attempt_kc_long = prepare_attempt_kc_long_for_history(
        load_attempt_kc_long_pfa(Path(config["attempt_kc_long_path"])),
        history_mode="rpfa",
        decay_alpha=float(config["decay_alpha"]),
        due_review_hours=float(config["due_review_hours"]),
    )

    model2_posterior = np.load(Path(config["model2_posterior_draws_path"]), allow_pickle=True)
    model3_posterior = np.load(Path(config["model3_posterior_draws_path"]), allow_pickle=True)

    student_levels = [str(value) for value in model2_posterior["student_levels"].tolist()]
    item_levels = [str(value) for value in model2_posterior["item_levels"].tolist()]
    kc_levels = [str(value) for value in model2_posterior["kc_levels"].tolist()]
    student_lookup = {value: index for index, value in enumerate(student_levels)}
    item_lookup = {value: index for index, value in enumerate(item_levels)}
    kc_lookup = {value: index for index, value in enumerate(kc_levels)}
    item_order_rank = np.asarray(np.argsort(np.argsort(np.asarray(item_levels, dtype=object))), dtype=np.int64)

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

    student_split_cutoff = int(round(float(config["calibration_student_share"]) * 100.0))
    evaluation_students = {
        student_id for student_id in student_levels if stable_student_bucket(student_id, modulo=100) >= student_split_cutoff
    }

    train_df = trials.loc[trials["split"] == "train"].copy()
    test_df = trials.loc[(trials["split"] == "test") & (trials["student_id"].astype(str).isin(evaluation_students))].copy()

    state_bin_width = int(model3_means["state_bin_width"])
    last_train_bins = (
        train_df.assign(state_bin=(train_df["overall_opportunity"] // state_bin_width).astype("int64"))
        .groupby("student_id", sort=False)["state_bin"]
        .max()
    )

    due_review_hours_threshold = float(config["due_review_hours"])
    max_eval_step = int(config["max_eval_step"])
    primary_eval_only = bool(config["primary_eval_only"])
    recent_attempt_window_size = int(hybrid_threshold_config["recent_attempt_window_size"])
    duration_cap_seconds = float(hybrid_threshold_config["duration_cap_seconds"])
    default_duration_seconds = float(hybrid_threshold_config["default_duration_seconds"])

    policy_names = [str(name) for name in config["policy_names"]]
    method_names = ["model2_raw", "model2_context_calibrated", "model2_plus_model3_uncertainty"]

    records: list[dict] = []
    for student_id, student_test in test_df.groupby("student_id", sort=False):
        student_id = str(student_id)
        if student_id not in student_lookup:
            continue
        student_index = student_lookup[student_id]
        train_history = train_df.loc[train_df["student_id"].astype(str) == student_id].sort_values(
            ["timestamp", "attempt_id"], kind="mergesort"
        )
        student_test = student_test.sort_values(["timestamp", "attempt_id"], kind="mergesort")

        opportunity_counts = np.zeros(len(kc_levels), dtype=np.float64)
        success_decay = np.zeros(len(kc_levels), dtype=np.float64)
        failure_decay = np.zeros(len(kc_levels), dtype=np.float64)
        last_seen_timestamp_ns = np.full(len(kc_levels), -1, dtype=np.int64)
        item_exposure_counts = np.zeros(len(item_levels), dtype=np.int64)

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
            item_exposure_counts[item_lookup[item_id_local]] += 1
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
                practice_vector = np.log1p(opportunity_counts)
                raw_candidate_probabilities = score_candidates_model2(
                    model2_means,
                    student_index=student_index,
                    candidate_indices=np.arange(len(item_levels), dtype=np.int64),
                    item_kc_matrix=item_kc_matrix,
                    success_feature_vector=success_decay,
                    failure_feature_vector=failure_decay,
                    practice_vector=practice_vector,
                    static_item_term=static_item_term,
                )

                unseen_mask = item_exposure_counts == 0
                linked_kc_exposure_total = item_kc_matrix @ opportunity_counts
                recent_failure_score = item_kc_matrix @ failure_decay

                current_timestamp_ns = int(pd.Timestamp(row.timestamp).to_datetime64().astype("datetime64[ns]").astype("int64"))
                due_review_flag = np.zeros(len(item_levels), dtype=np.int64)
                due_review_hours = np.full(len(item_levels), np.nan, dtype=np.float64)
                for item_index, item_id in enumerate(item_levels):
                    kc_indices = item_to_kc_indices.get(item_id, [])
                    if not kc_indices:
                        continue
                    seen_timestamps = last_seen_timestamp_ns[kc_indices]
                    seen_timestamps = seen_timestamps[seen_timestamps >= 0]
                    if len(seen_timestamps) == 0:
                        continue
                    hours = (current_timestamp_ns - seen_timestamps) / 3_600_000_000_000.0
                    due_review_hours[item_index] = float(np.max(hours))
                    due_review_flag[item_index] = int(np.any(hours >= due_review_hours_threshold))

                balanced_index, _ = choose_unseen_policy_item(
                    raw_candidate_probabilities,
                    unseen_mask=unseen_mask,
                    target_probability=POLICY_SPECS["balanced_challenge"]["target_probability"],
                    target_band_low=POLICY_SPECS["balanced_challenge"]["target_band_low"],
                    target_band_high=POLICY_SPECS["balanced_challenge"]["target_band_high"],
                    student_item_exposure=item_exposure_counts,
                    linked_kc_exposure_total=linked_kc_exposure_total,
                    item_order_rank=item_order_rank,
                )
                balanced_reference_probability = float(raw_candidate_probabilities[balanced_index])

                last_train_bin = int(last_train_bins.loc[student_id])
                _, future_state_variance = model3_future_state_summary(
                    model3_means,
                    student_index=student_index,
                    step_overall_opportunity=int(row.overall_opportunity),
                    last_train_bin=last_train_bin,
                )
                uncertainty_sd = float(np.sqrt(max(future_state_variance, 0.0)))

                behavior_state = compute_behavior_state(
                    recent_correct_window=recent_correct_window,
                    recent_hint_window=recent_hint_window,
                    recent_selection_change_window=recent_selection_change_window,
                    recent_duration_window=recent_duration_window,
                    baseline_duration_values=baseline_duration_values,
                    failure_streak=failure_streak,
                    default_duration=default_duration_seconds,
                )

                acute_failure = int(
                    int(behavior_state["failure_streak"]) >= int(hybrid_threshold_config["failure_streak_threshold"])
                    or float(behavior_state["recent_failure_rate_attempt"]) >= float(hybrid_threshold_config["recent_failure_rate_threshold"])
                )
                friction_high = int(
                    float(behavior_state["recent_hint_rate"]) >= float(hybrid_threshold_config["hint_rate_threshold"])
                    or float(behavior_state["recent_selection_change_rate"]) >= float(hybrid_threshold_config["selection_change_rate_threshold"])
                    or float(behavior_state["response_time_inflation"]) >= float(hybrid_threshold_config["duration_inflation_threshold"])
                )
                low_proficiency = int(
                    balanced_reference_probability < float(hybrid_threshold_config["balanced_low_probability_threshold"])
                )
                context_flags = {
                    "early_steps_1_5": int(eval_step <= 5),
                    "due_review_available": int(np.any(due_review_flag == 1)),
                    "high_recent_failure_context": acute_failure,
                    "high_friction_context": friction_high,
                    "lower_predicted_proficiency": low_proficiency,
                }

                for method_name in method_names:
                    calibrated_candidate_probabilities = apply_calibrator(
                        raw_candidate_probabilities,
                        method_name=method_name,
                        context_flags=context_flags,
                        uncertainty_sd=uncertainty_sd,
                        calibration_summary=calibration_summary,
                    )
                    actual_next_probability = float(
                        apply_calibrator(
                            np.asarray([raw_candidate_probabilities[item_lookup[str(row.item_id)]]], dtype=np.float64),
                            method_name=method_name,
                            context_flags=context_flags,
                            uncertainty_sd=uncertainty_sd,
                            calibration_summary=calibration_summary,
                        )[0]
                    )

                    for policy_name in policy_names:
                        spec = POLICY_SPECS[policy_name]
                        selected_index, in_target_band = choose_unseen_policy_item(
                            calibrated_candidate_probabilities,
                            unseen_mask=unseen_mask,
                            target_probability=float(spec["target_probability"]),
                            target_band_low=float(spec["target_band_low"]),
                            target_band_high=float(spec["target_band_high"]),
                            student_item_exposure=item_exposure_counts,
                            linked_kc_exposure_total=linked_kc_exposure_total,
                            item_order_rank=item_order_rank,
                        )
                        selected_probability = float(calibrated_candidate_probabilities[selected_index])
                        target_probability = float(spec["target_probability"])
                        records.append(
                            {
                                "student_id": student_id,
                                "attempt_id": int(row.attempt_id),
                                "actual_item_id": str(row.item_id),
                                "actual_correct": int(row.correct),
                                "eval_step": eval_step,
                                "policy_name": policy_name,
                                "calibration_method": method_name,
                                "recommended_item_id": str(item_levels[selected_index]),
                                "recommended_probability": selected_probability,
                                "target_probability": target_probability,
                                "target_gap": abs(selected_probability - target_probability),
                                "in_target_band": int(in_target_band),
                                "fallback_used": "none",
                                "recent_failure_score": float(recent_failure_score[selected_index]),
                                "due_review_flag": int(due_review_flag[selected_index]),
                                "due_review_hours": float(due_review_hours[selected_index]) if np.isfinite(due_review_hours[selected_index]) else np.nan,
                                "recommended_seen_item": int(item_exposure_counts[selected_index] > 0),
                                "student_item_exposure_count": int(item_exposure_counts[selected_index]),
                                "linked_kc_exposure_total": float(linked_kc_exposure_total[selected_index]),
                                "candidate_count": int(unseen_mask.sum()),
                                "actual_next_probability": actual_next_probability,
                                "actual_target_gap": abs(actual_next_probability - target_probability),
                                "uncertainty_sd": uncertainty_sd,
                                "due_review_available": int(context_flags["due_review_available"]),
                                "high_recent_failure_context": int(context_flags["high_recent_failure_context"]),
                                "high_friction_context": int(context_flags["high_friction_context"]),
                                "lower_predicted_proficiency": int(context_flags["lower_predicted_proficiency"]),
                                "track": str(config["track_name"]),
                                "model_name": "model2_qmatrix_rpfa",
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

    method_policy_summaries: dict[str, dict[str, dict]] = {}
    for method_name, method_rows in rows.groupby("calibration_method", sort=True):
        method_policy_summaries[method_name] = {}
        for policy_name, group in method_rows.groupby("policy_name", sort=True):
            method_policy_summaries[method_name][policy_name] = summarize_policy_rows(group.copy(), max_eval_step=max_eval_step)

    method_new_learning_summary: dict[str, dict[str, float]] = {}
    for method_name, policy_map in method_policy_summaries.items():
        method_new_learning_summary[method_name] = {
            "mean_target_gap_1_10": average_metric(policy_map, NEW_LEARNING_POLICIES, "student_avg_target_gap_1_10"),
            "mean_policy_advantage_1_10": average_metric(policy_map, NEW_LEARNING_POLICIES, "policy_advantage_over_actual_1_10"),
            "mean_stability": average_metric(policy_map, NEW_LEARNING_POLICIES, "recommendation_stability_mean_abs_diff"),
        }

    baseline_method = str(config["baseline_method"])
    challenger_method = str(config["challenger_method"])
    primary_policy = str(config["primary_policy"])
    stability_tolerance = float(config["stability_tolerance"])

    primary_policy_delta = compare_policy_summary(
        method_policy_summaries[challenger_method][primary_policy],
        method_policy_summaries[baseline_method][primary_policy],
    )
    new_learning_delta = {
        "mean_target_gap_1_10_delta": float(
            method_new_learning_summary[challenger_method]["mean_target_gap_1_10"]
            - method_new_learning_summary[baseline_method]["mean_target_gap_1_10"]
        ),
        "mean_policy_advantage_1_10_delta": float(
            method_new_learning_summary[challenger_method]["mean_policy_advantage_1_10"]
            - method_new_learning_summary[baseline_method]["mean_policy_advantage_1_10"]
        ),
        "mean_stability_delta": float(
            method_new_learning_summary[challenger_method]["mean_stability"]
            - method_new_learning_summary[baseline_method]["mean_stability"]
        ),
    }

    pass_conditions = {
        "primary_policy_target_gap_improves": bool(primary_policy_delta["target_gap_1_10_delta"] < 0.0),
        "mean_new_learning_target_gap_improves": bool(new_learning_delta["mean_target_gap_1_10_delta"] < 0.0),
        "primary_policy_stability_within_tolerance": bool(primary_policy_delta["stability_delta"] <= stability_tolerance),
    }
    operational_pass = all(pass_conditions.values())

    comparison_rows: list[dict[str, object]] = []
    for policy_name in policy_names:
        baseline_summary = method_policy_summaries[baseline_method][policy_name]
        challenger_summary = method_policy_summaries[challenger_method][policy_name]
        raw_summary = method_policy_summaries["model2_raw"][policy_name]
        uncertainty_vs_context = compare_policy_summary(challenger_summary, baseline_summary)
        context_vs_raw = compare_policy_summary(baseline_summary, raw_summary)
        comparison_rows.append(
            {
                "policy_name": policy_name,
                "raw_target_gap_1_10": float(raw_summary["student_avg_target_gap_1_10"]),
                "context_target_gap_1_10": float(baseline_summary["student_avg_target_gap_1_10"]),
                "uncertainty_target_gap_1_10": float(challenger_summary["student_avg_target_gap_1_10"]),
                "delta_uncertainty_minus_context_target_gap_1_10": uncertainty_vs_context["target_gap_1_10_delta"],
                "delta_context_minus_raw_target_gap_1_10": context_vs_raw["target_gap_1_10_delta"],
                "raw_policy_advantage_1_10": float(raw_summary["policy_advantage_over_actual_1_10"]),
                "context_policy_advantage_1_10": float(baseline_summary["policy_advantage_over_actual_1_10"]),
                "uncertainty_policy_advantage_1_10": float(challenger_summary["policy_advantage_over_actual_1_10"]),
                "delta_uncertainty_minus_context_policy_advantage_1_10": uncertainty_vs_context["policy_advantage_1_10_delta"],
                "raw_stability": float(raw_summary["recommendation_stability_mean_abs_diff"]),
                "context_stability": float(baseline_summary["recommendation_stability_mean_abs_diff"]),
                "uncertainty_stability": float(challenger_summary["recommendation_stability_mean_abs_diff"]),
                "delta_uncertainty_minus_context_stability": uncertainty_vs_context["stability_delta"],
            }
        )

    summary = {
        "calibration_student_share": float(config["calibration_student_share"]),
        "evaluation_students": int(len(evaluation_students)),
        "evaluation_rows": int(len(rows)),
        "policy_names": policy_names,
        "method_policy_summaries": method_policy_summaries,
        "method_new_learning_summary": method_new_learning_summary,
        "baseline_method": baseline_method,
        "challenger_method": challenger_method,
        "primary_policy": primary_policy,
        "stability_tolerance": stability_tolerance,
        "primary_policy_delta": primary_policy_delta,
        "new_learning_delta": new_learning_delta,
        "pass_conditions": pass_conditions,
        "operational_pass": operational_pass,
        "row_output_path": str(row_output_path),
    }

    summary_output_path = Path(config["summary_output_path"])
    write_json(summary_output_path, summary)

    comparison_output_path = Path(config["comparison_output_path"])
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(comparison_output_path, index=False)

    print(f"Saved calibrated policy rows to {row_output_path}")
    print(f"Saved calibrated policy summary to {summary_output_path}")
    print(f"Saved calibrated policy comparison to {comparison_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
