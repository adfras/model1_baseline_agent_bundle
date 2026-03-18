from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from policy_suite_common import (
    build_attempt_event_lookup,
    build_item_kc_lookup,
    build_kc_constrained_slate_mask,
    fit_logistic_calibrator,
    mean_over_linked_kcs,
    model3_future_state_summary,
    score_candidates_model2,
    stable_student_bucket,
    summarize_policy_rows,
    summarize_prediction_frame,
    write_json,
)
from qmatrix_common import load_json, load_trials
from qmatrix_pfa_common import load_attempt_kc_long_pfa, prepare_attempt_kc_long_for_history


DEFAULT_CONFIG_PATH = Path("config/phase1_local_uncertainty_policy_suite.json")
POLICY_SPECS = {
    "balanced_challenge": {"target_probability": 0.72, "target_band_low": 0.65, "target_band_high": 0.80},
    "harder_challenge": {"target_probability": 0.60, "target_band_low": 0.55, "target_band_high": 0.65},
    "confidence_building": {"target_probability": 0.85, "target_band_low": 0.80, "target_band_high": 0.90},
}
METHOD_SPECS = {
    "model2_raw": {"feature_names": []},
    "policy_band_calibrated": {"feature_names": []},
    "policy_band_plus_local_residuals": {
        "feature_names": [
            "recent_abs_residual_mean_5",
            "recent_abs_residual_mean_10",
            "recent_residual_var_10",
            "recent_hint_rate_5",
            "recent_selection_change_rate_5",
            "response_time_inflation_5",
            "recent_trust_feedback_mean_5",
            "recent_difficulty_feedback_mean_5",
            "early_steps_1_5",
            "candidate_kc_abs_residual_mean",
            "candidate_kc_residual_var",
            "candidate_kc_failure_decay_mean",
            "candidate_kc_success_decay_mean",
        ]
    },
    "policy_band_plus_local_residuals_plus_model3": {
        "feature_names": [
            "recent_abs_residual_mean_5",
            "recent_abs_residual_mean_10",
            "recent_residual_var_10",
            "recent_hint_rate_5",
            "recent_selection_change_rate_5",
            "response_time_inflation_5",
            "recent_trust_feedback_mean_5",
            "recent_difficulty_feedback_mean_5",
            "early_steps_1_5",
            "candidate_kc_abs_residual_mean",
            "candidate_kc_residual_var",
            "candidate_kc_failure_decay_mean",
            "candidate_kc_success_decay_mean",
            "uncertainty_sd",
        ]
    },
}
CONTINUOUS_FEATURES = {
    "recent_abs_residual_mean_5",
    "recent_abs_residual_mean_10",
    "recent_residual_var_10",
    "recent_hint_rate_5",
    "recent_selection_change_rate_5",
    "response_time_inflation_5",
    "recent_trust_feedback_mean_5",
    "recent_difficulty_feedback_mean_5",
    "candidate_kc_abs_residual_mean",
    "candidate_kc_residual_var",
    "candidate_kc_failure_decay_mean",
    "candidate_kc_success_decay_mean",
    "uncertainty_sd",
}
NEW_LEARNING_POLICIES = ["balanced_challenge", "harder_challenge", "confidence_building"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local residual-heterogeneity and Model 3 uncertainty on a KC-constrained fixed policy suite."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def clip_probability(probability: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1.0e-6)


def safe_recent_mean(values: deque[float], *, window: int, default: float) -> float:
    if not values:
        return float(default)
    data = np.asarray(list(values)[-window:], dtype=np.float64)
    return float(np.mean(data)) if len(data) else float(default)


def safe_recent_var(values: deque[float], *, window: int, default: float) -> float:
    if not values:
        return float(default)
    data = np.asarray(list(values)[-window:], dtype=np.float64)
    return float(np.var(data, ddof=0)) if len(data) else float(default)


def safe_recent_median(values: deque[float], *, window: int, default: float) -> float:
    if not values:
        return float(default)
    data = np.asarray(list(values)[-window:], dtype=np.float64)
    return float(np.median(data)) if len(data) else float(default)


def choose_policy_index(
    predicted_probabilities: np.ndarray,
    *,
    candidate_mask: np.ndarray,
    target_probability: float,
    target_band_low: float,
    target_band_high: float,
    student_item_exposure: np.ndarray,
    linked_kc_exposure_total: np.ndarray,
    item_order_rank: np.ndarray,
) -> tuple[int, int]:
    available = np.where(candidate_mask)[0]
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


def average_metric(policy_summaries: dict[str, dict], metric: str) -> float:
    values = [float(policy_summaries[policy_name][metric]) for policy_name in NEW_LEARNING_POLICIES]
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def standardize_feature(values: np.ndarray, *, mean_value: float, std_value: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - float(mean_value)) / max(float(std_value), 1e-6)


def build_design_matrix(
    raw_probabilities: np.ndarray,
    feature_payload: dict[str, np.ndarray | float],
    *,
    method_name: str,
    calibration_spec: dict | None,
    fit_mode: bool,
) -> tuple[np.ndarray, list[str], dict[str, dict[str, float]]]:
    if method_name == "model2_raw":
        raise ValueError("Raw probabilities do not use a design matrix.")

    raw_probabilities = clip_probability(raw_probabilities)
    base_logit = logit(raw_probabilities)
    columns = [np.ones(len(base_logit), dtype=np.float64), base_logit]
    feature_names = ["intercept", "model2_logit"]
    standardization: dict[str, dict[str, float]] = {}

    for feature_name in METHOD_SPECS[method_name]["feature_names"]:
        values = np.asarray(feature_payload[feature_name], dtype=np.float64)
        if feature_name in CONTINUOUS_FEATURES:
            if fit_mode:
                mean_value = float(np.mean(values))
                std_value = float(np.std(values, ddof=0))
                standardization[feature_name] = {"mean": mean_value, "std": max(std_value, 1e-6)}
            else:
                assert calibration_spec is not None
                standardization[feature_name] = calibration_spec["standardization"][feature_name]
            stats = standardization[feature_name]
            values = standardize_feature(values, mean_value=stats["mean"], std_value=stats["std"])
        columns.append(values)
        feature_names.append(feature_name)
    X = np.column_stack(columns).astype(np.float64)
    return X, feature_names, standardization


def apply_policy_calibrator(
    raw_probabilities: np.ndarray,
    feature_payload: dict[str, np.ndarray | float],
    *,
    method_name: str,
    calibration_spec: dict | None,
) -> np.ndarray:
    raw_probabilities = clip_probability(raw_probabilities)
    if method_name == "model2_raw":
        return raw_probabilities
    assert calibration_spec is not None
    X, _, _ = build_design_matrix(
        raw_probabilities,
        feature_payload,
        method_name=method_name,
        calibration_spec=calibration_spec,
        fit_mode=False,
    )
    beta = np.asarray(calibration_spec["coefficients"], dtype=np.float64)
    return clip_probability(expit(X @ beta))


def load_model_means(config: dict) -> tuple[dict, dict, list[str], list[str], list[str]]:
    model2_posterior = np.load(Path(config["model2_posterior_draws_path"]), allow_pickle=True)
    model3_posterior = np.load(Path(config["model3_posterior_draws_path"]), allow_pickle=True)

    student_levels = [str(value) for value in model2_posterior["student_levels"].tolist()]
    item_levels = [str(value) for value in model2_posterior["item_levels"].tolist()]
    kc_levels = [str(value) for value in model2_posterior["kc_levels"].tolist()]

    model2_means = {
        "Intercept_mean": float(model2_posterior["Intercept"].mean()),
        "student_intercept_mean": model2_posterior["student_intercept"].mean(axis=0),
        "student_slope_mean": model2_posterior["student_slope"].mean(axis=0),
        "kc_success_mean": model2_posterior["kc_success"].mean(axis=0),
        "kc_failure_mean": model2_posterior["kc_failure"].mean(axis=0),
        "item_effect_mean": model2_posterior["item_effect"].mean(axis=0),
        "kc_intercept_mean": model2_posterior["kc_intercept"].mean(axis=0),
    }
    model3_means = {
        "rho_mean": float(model3_posterior["rho"].mean()),
        "state_sigma_student_mean": model3_posterior["state_sigma_student"].mean(axis=0),
        "latent_state_mean": model3_posterior["latent_state"].mean(axis=0),
        "state_bin_width": int(np.asarray(model3_posterior["state_bin_width"]).reshape(-1)[0]),
    }
    model2_posterior.close()
    model3_posterior.close()
    return model2_means, model3_means, student_levels, item_levels, kc_levels


def initialize_student_state(
    *,
    recent_attempt_window_5: int,
    recent_attempt_window_10: int,
) -> dict[str, object]:
    return {
        "opportunity_counts": None,
        "success_decay": None,
        "failure_decay": None,
        "last_seen_timestamp_ns": None,
        "item_exposure_counts": None,
        "kc_abs_residual_decay": None,
        "kc_sq_residual_decay": None,
        "recent_attempt_events": deque(maxlen=recent_attempt_window_10),
        "recent_residuals": deque(maxlen=recent_attempt_window_10),
        "recent_abs_residuals": deque(maxlen=recent_attempt_window_10),
        "recent_hint_window": deque(maxlen=recent_attempt_window_5),
        "recent_selection_change_window": deque(maxlen=recent_attempt_window_5),
        "recent_duration_window": deque(maxlen=recent_attempt_window_5),
        "recent_trust_feedback_window": deque(maxlen=recent_attempt_window_5),
        "recent_difficulty_feedback_window": deque(maxlen=recent_attempt_window_5),
        "baseline_duration_values": deque(maxlen=recent_attempt_window_10),
    }


def prepare_state_arrays(state: dict[str, object], *, num_items: int, num_kcs: int) -> None:
    state["opportunity_counts"] = np.zeros(num_kcs, dtype=np.float64)
    state["success_decay"] = np.zeros(num_kcs, dtype=np.float64)
    state["failure_decay"] = np.zeros(num_kcs, dtype=np.float64)
    state["last_seen_timestamp_ns"] = np.full(num_kcs, -1, dtype=np.int64)
    state["item_exposure_counts"] = np.zeros(num_items, dtype=np.int64)
    state["kc_abs_residual_decay"] = np.zeros(num_kcs, dtype=np.float64)
    state["kc_sq_residual_decay"] = np.zeros(num_kcs, dtype=np.float64)


def build_step_features(
    *,
    success_decay: np.ndarray,
    failure_decay: np.ndarray,
    kc_abs_residual_decay: np.ndarray,
    kc_sq_residual_decay: np.ndarray,
    item_kc_matrix: np.ndarray,
    recent_abs_residuals: deque[float],
    recent_residuals: deque[float],
    recent_hint_window: deque[float],
    recent_selection_change_window: deque[float],
    recent_duration_window: deque[float],
    recent_trust_feedback_window: deque[float],
    recent_difficulty_feedback_window: deque[float],
    baseline_duration_values: deque[float],
    uncertainty_sd: float,
    eval_step: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    candidate_kc_abs_residual_mean = mean_over_linked_kcs(item_kc_matrix, kc_abs_residual_decay)
    candidate_kc_sq_residual_mean = mean_over_linked_kcs(item_kc_matrix, kc_sq_residual_decay)
    candidate_kc_residual_var = np.maximum(candidate_kc_sq_residual_mean - np.square(candidate_kc_abs_residual_mean), 0.0)
    candidate_kc_failure_decay_mean = mean_over_linked_kcs(item_kc_matrix, failure_decay)
    candidate_kc_success_decay_mean = mean_over_linked_kcs(item_kc_matrix, success_decay)

    baseline_duration = float(np.median(np.asarray(baseline_duration_values, dtype=np.float64))) if baseline_duration_values else 1.0
    recent_duration = safe_recent_median(recent_duration_window, window=5, default=baseline_duration)
    response_time_inflation = recent_duration / max(baseline_duration, 1e-6)

    global_features = {
        "recent_abs_residual_mean_5": safe_recent_mean(recent_abs_residuals, window=5, default=0.0),
        "recent_abs_residual_mean_10": safe_recent_mean(recent_abs_residuals, window=10, default=0.0),
        "recent_residual_var_10": safe_recent_var(recent_residuals, window=10, default=0.0),
        "recent_hint_rate_5": safe_recent_mean(recent_hint_window, window=5, default=0.0),
        "recent_selection_change_rate_5": safe_recent_mean(recent_selection_change_window, window=5, default=0.0),
        "response_time_inflation_5": float(response_time_inflation),
        "recent_trust_feedback_mean_5": safe_recent_mean(recent_trust_feedback_window, window=5, default=0.0),
        "recent_difficulty_feedback_mean_5": safe_recent_mean(recent_difficulty_feedback_window, window=5, default=0.0),
        "early_steps_1_5": float(eval_step <= 5),
        "uncertainty_sd": float(uncertainty_sd),
    }
    candidate_features = {
        "candidate_kc_abs_residual_mean": candidate_kc_abs_residual_mean,
        "candidate_kc_residual_var": candidate_kc_residual_var,
        "candidate_kc_failure_decay_mean": candidate_kc_failure_decay_mean,
        "candidate_kc_success_decay_mean": candidate_kc_success_decay_mean,
    }
    return global_features, candidate_features


def combine_feature_payload(
    *,
    global_features: dict[str, float],
    candidate_features: dict[str, np.ndarray],
    item_index: int | None = None,
) -> dict[str, np.ndarray | float]:
    payload: dict[str, np.ndarray | float] = {name: value for name, value in global_features.items()}
    for feature_name, values in candidate_features.items():
        if item_index is None:
            payload[feature_name] = np.asarray(values, dtype=np.float64)
        else:
            payload[feature_name] = np.asarray([float(values[item_index])], dtype=np.float64)
    if item_index is None:
        n_candidates = len(next(iter(candidate_features.values())))
        for feature_name, value in global_features.items():
            payload[feature_name] = np.full(n_candidates, float(value), dtype=np.float64)
    return payload


def update_state_from_row(
    state: dict[str, object],
    *,
    row,
    attempt_event_lookup: dict[int, dict],
    item_lookup: dict[str, int],
    raw_actual_probability: float,
    decay_alpha: float,
) -> None:
    residual = float(int(row.correct) - float(raw_actual_probability))
    abs_residual = abs(residual)
    state["recent_residuals"].append(residual)
    state["recent_abs_residuals"].append(abs_residual)
    state["recent_hint_window"].append(float(getattr(row, "hint_used", 0)))
    state["recent_selection_change_window"].append(float(int(getattr(row, "selection_change", 0)) > 1))
    state["recent_trust_feedback_window"].append(float(getattr(row, "trust_feedback", 0)))
    state["recent_difficulty_feedback_window"].append(float(getattr(row, "difficulty_feedback", 0)))

    duration_seconds = float(getattr(row, "duration_seconds", np.nan))
    if np.isfinite(duration_seconds) and duration_seconds > 0:
        state["recent_duration_window"].append(duration_seconds)
        state["baseline_duration_values"].append(duration_seconds)

    event = attempt_event_lookup[int(row.attempt_id)]
    timestamp_ns = int(event["timestamp"].to_datetime64().astype("datetime64[ns]").astype("int64"))
    item_id_local = str(event["item_id"])
    state["item_exposure_counts"][item_lookup[item_id_local]] += 1
    state["recent_attempt_events"].append(event["rows"])

    for row_event in event["rows"]:
        kc_index = int(row_event["kc_index"])
        state["opportunity_counts"][kc_index] += float(row_event["kc_exposure_increment"])
        state["success_decay"][kc_index] = decay_alpha * (
            state["success_decay"][kc_index] + float(row_event["kc_success_increment"])
        )
        state["failure_decay"][kc_index] = decay_alpha * (
            state["failure_decay"][kc_index] + float(row_event["kc_failure_increment"])
        )
        state["kc_abs_residual_decay"][kc_index] = decay_alpha * (
            state["kc_abs_residual_decay"][kc_index] + abs_residual
        )
        state["kc_sq_residual_decay"][kc_index] = decay_alpha * (
            state["kc_sq_residual_decay"][kc_index] + residual * residual
        )
        state["last_seen_timestamp_ns"][kc_index] = timestamp_ns


def collect_policy_training_rows(
    config: dict,
    *,
    trials: pd.DataFrame,
    attempt_event_lookup: dict[int, dict],
    item_kc_matrix: np.ndarray,
    model2_means: dict,
    model3_means: dict,
    student_levels: list[str],
    item_levels: list[str],
) -> pd.DataFrame:
    student_lookup = {value: index for index, value in enumerate(student_levels)}
    item_lookup = {value: index for index, value in enumerate(item_levels)}
    static_item_term = model2_means["item_effect_mean"] + item_kc_matrix @ model2_means["kc_intercept_mean"]

    calibration_cutoff = int(round(float(config["calibration_student_share"]) * 100.0))
    state_bin_width = int(model3_means["state_bin_width"])
    train_df = trials.loc[trials["split"] == "train"].copy()
    test_df = trials.loc[trials["split"] == "test"].copy()
    last_train_bins = (
        train_df.assign(state_bin=(train_df["overall_opportunity"] // state_bin_width).astype("int64"))
        .groupby("student_id", sort=False)["state_bin"]
        .max()
    )

    records: list[dict] = []
    for student_id, student_test in test_df.groupby("student_id", sort=False):
        student_id = str(student_id)
        if student_id not in student_lookup:
            continue
        student_index = student_lookup[student_id]
        student_split = "calibration" if stable_student_bucket(student_id, modulo=100) < calibration_cutoff else "evaluation"
        train_history = train_df.loc[train_df["student_id"].astype(str) == student_id].sort_values(
            ["timestamp", "attempt_id"], kind="mergesort"
        )
        student_test = student_test.sort_values(["timestamp", "attempt_id"], kind="mergesort")

        state = initialize_student_state(
            recent_attempt_window_5=int(config["recent_attempt_window_5"]),
            recent_attempt_window_10=int(config["recent_attempt_window_10"]),
        )
        prepare_state_arrays(state, num_items=len(item_levels), num_kcs=item_kc_matrix.shape[1])

        for row in train_history.itertuples(index=False):
            practice_vector = np.log1p(state["opportunity_counts"])
            raw_candidate_probabilities = score_candidates_model2(
                model2_means,
                student_index=student_index,
                candidate_indices=np.arange(len(item_levels), dtype=np.int64),
                item_kc_matrix=item_kc_matrix,
                success_feature_vector=state["success_decay"],
                failure_feature_vector=state["failure_decay"],
                practice_vector=practice_vector,
                static_item_term=static_item_term,
            )
            raw_actual_probability = float(raw_candidate_probabilities[item_lookup[str(row.item_id)]])
            update_state_from_row(
                state,
                row=row,
                attempt_event_lookup=attempt_event_lookup,
                item_lookup=item_lookup,
                raw_actual_probability=raw_actual_probability,
                decay_alpha=float(config["decay_alpha"]),
            )

        eval_step = 0
        for row in student_test.itertuples(index=False):
            step_is_eval = True
            if bool(config["primary_eval_only"]) and int(row.primary_eval_eligible) != 1:
                step_is_eval = False
            if step_is_eval:
                eval_step += 1
                if eval_step > int(config["max_eval_step"]):
                    step_is_eval = False

            practice_vector = np.log1p(state["opportunity_counts"])
            raw_candidate_probabilities = score_candidates_model2(
                model2_means,
                student_index=student_index,
                candidate_indices=np.arange(len(item_levels), dtype=np.int64),
                item_kc_matrix=item_kc_matrix,
                success_feature_vector=state["success_decay"],
                failure_feature_vector=state["failure_decay"],
                practice_vector=practice_vector,
                static_item_term=static_item_term,
            )
            actual_item_index = item_lookup[str(row.item_id)]
            raw_actual_probability = float(raw_candidate_probabilities[actual_item_index])

            if step_is_eval:
                slate_info = build_kc_constrained_slate_mask(
                    item_kc_matrix=item_kc_matrix,
                    item_levels=item_levels,
                    item_exposure_counts=state["item_exposure_counts"],
                    recent_attempt_events=state["recent_attempt_events"],
                    decay_alpha=float(config["decay_alpha"]),
                    frontier_top_kcs=int(config["frontier_top_kcs"]),
                    slate_min_candidate_count=int(config["slate_min_candidate_count"]),
                )
                last_train_bin = int(last_train_bins.loc[student_id])
                _, future_state_variance = model3_future_state_summary(
                    model3_means,
                    student_index=student_index,
                    step_overall_opportunity=int(row.overall_opportunity),
                    last_train_bin=last_train_bin,
                )
                uncertainty_sd = float(np.sqrt(max(future_state_variance, 0.0)))
                global_features, candidate_features = build_step_features(
                    success_decay=state["success_decay"],
                    failure_decay=state["failure_decay"],
                    kc_abs_residual_decay=state["kc_abs_residual_decay"],
                    kc_sq_residual_decay=state["kc_sq_residual_decay"],
                    item_kc_matrix=item_kc_matrix,
                    recent_abs_residuals=state["recent_abs_residuals"],
                    recent_residuals=state["recent_residuals"],
                    recent_hint_window=state["recent_hint_window"],
                    recent_selection_change_window=state["recent_selection_change_window"],
                    recent_duration_window=state["recent_duration_window"],
                    recent_trust_feedback_window=state["recent_trust_feedback_window"],
                    recent_difficulty_feedback_window=state["recent_difficulty_feedback_window"],
                    baseline_duration_values=state["baseline_duration_values"],
                    uncertainty_sd=uncertainty_sd,
                    eval_step=eval_step,
                )
                actual_feature_payload = combine_feature_payload(
                    global_features=global_features,
                    candidate_features=candidate_features,
                    item_index=actual_item_index,
                )
                actual_item_unseen = int(state["item_exposure_counts"][actual_item_index] == 0)
                actual_item_in_slate = int(bool(slate_info["candidate_mask"][actual_item_index]))

                for policy_name in config["policy_names"]:
                    records.append(
                        {
                            "student_id": student_id,
                            "student_split": student_split,
                            "attempt_id": int(row.attempt_id),
                            "actual_item_id": str(row.item_id),
                            "actual_correct": int(row.correct),
                            "eval_step": eval_step,
                            "policy_name": policy_name,
                            "model2_raw_probability": raw_actual_probability,
                            "actual_item_unseen": actual_item_unseen,
                            "actual_item_in_slate": actual_item_in_slate,
                            "training_row_mode": "unseen_and_in_slate",
                            "candidate_pool_mode": str(slate_info["candidate_pool_mode"]),
                            "candidate_count": int(slate_info["candidate_count"]),
                            "kc_frontier_top5": str(slate_info["kc_frontier_top5"]),
                            "kc_frontier_fallback_used": str(slate_info["kc_frontier_fallback_used"]),
                            "track": str(config["track_name"]),
                            **{name: float(np.asarray(value).reshape(-1)[0]) for name, value in actual_feature_payload.items()},
                        }
                    )

            update_state_from_row(
                state,
                row=row,
                attempt_event_lookup=attempt_event_lookup,
                item_lookup=item_lookup,
                raw_actual_probability=raw_actual_probability,
                decay_alpha=float(config["decay_alpha"]),
            )

    return pd.DataFrame(records)


def fit_policy_calibrators(config: dict, training_rows: pd.DataFrame) -> tuple[dict, dict]:
    min_rows = int(config["min_policy_calibration_rows"])
    l2_penalty = float(config["l2_penalty"])
    calibrators: dict[str, dict[str, dict]] = {}
    actual_next_summary: dict[str, dict[str, dict]] = {}

    for policy_name in config["policy_names"]:
        policy_rows = training_rows.loc[training_rows["policy_name"] == policy_name].copy()
        calibration_rows = policy_rows.loc[policy_rows["student_split"] == "calibration"].copy()
        evaluation_rows = policy_rows.loc[policy_rows["student_split"] == "evaluation"].copy()

        eligible = calibration_rows.loc[
            (calibration_rows["actual_item_unseen"] == 1) & (calibration_rows["actual_item_in_slate"] == 1)
        ].copy()
        training_row_mode = "unseen_and_in_slate"
        if len(eligible) < min_rows:
            eligible = calibration_rows.loc[calibration_rows["actual_item_unseen"] == 1].copy()
            training_row_mode = "unseen_only"
        if len(eligible) < min_rows:
            eligible = calibration_rows.copy()
            training_row_mode = "all_actual_next_rows"

        evaluation_eligible = evaluation_rows.copy()
        if training_row_mode == "unseen_and_in_slate":
            evaluation_eligible = evaluation_rows.loc[
                (evaluation_rows["actual_item_unseen"] == 1) & (evaluation_rows["actual_item_in_slate"] == 1)
            ].copy()
        elif training_row_mode == "unseen_only":
            evaluation_eligible = evaluation_rows.loc[evaluation_rows["actual_item_unseen"] == 1].copy()

        calibrators[policy_name] = {}
        actual_next_summary[policy_name] = {
            "training_row_mode": training_row_mode,
            "calibration_rows_used": int(len(eligible)),
            "evaluation_rows_used": int(len(evaluation_eligible)),
            "methods": {},
        }
        if len(evaluation_eligible):
            evaluation_eligible["model2_raw"] = clip_probability(evaluation_eligible["model2_raw_probability"].to_numpy(dtype=np.float64))
            actual_next_summary[policy_name]["methods"]["model2_raw"] = summarize_prediction_frame(
                evaluation_eligible,
                "model2_raw",
            )

        for method_name in ["policy_band_calibrated", "policy_band_plus_local_residuals", "policy_band_plus_local_residuals_plus_model3"]:
            feature_payload = {
                feature_name: eligible[feature_name].to_numpy(dtype=np.float64)
                for feature_name in METHOD_SPECS[method_name]["feature_names"]
            }
            X_cal, feature_names, standardization = build_design_matrix(
                eligible["model2_raw_probability"].to_numpy(dtype=np.float64),
                feature_payload,
                method_name=method_name,
                calibration_spec=None,
                fit_mode=True,
            )
            y_cal = eligible["actual_correct"].to_numpy(dtype=np.float64)
            beta, objective_value = fit_logistic_calibrator(X_cal, y_cal, l2_penalty=l2_penalty)
            calibrators[policy_name][method_name] = {
                "feature_names": feature_names,
                "coefficients": beta.tolist(),
                "standardization": standardization,
                "objective_value": objective_value,
                "training_row_mode": training_row_mode,
            }
            if len(evaluation_eligible):
                eval_feature_payload = {
                    feature_name: evaluation_eligible[feature_name].to_numpy(dtype=np.float64)
                    for feature_name in METHOD_SPECS[method_name]["feature_names"]
                }
                calibrated = apply_policy_calibrator(
                    evaluation_eligible["model2_raw_probability"].to_numpy(dtype=np.float64),
                    eval_feature_payload,
                    method_name=method_name,
                    calibration_spec=calibrators[policy_name][method_name],
                )
                evaluation_frame = evaluation_eligible.copy()
                evaluation_frame[method_name] = calibrated
                actual_next_summary[policy_name]["methods"][method_name] = summarize_prediction_frame(
                    evaluation_frame,
                    method_name,
                )

    return calibrators, actual_next_summary


def rerun_policy_suite(
    config: dict,
    *,
    trials: pd.DataFrame,
    attempt_event_lookup: dict[int, dict],
    item_kc_matrix: np.ndarray,
    model2_means: dict,
    model3_means: dict,
    student_levels: list[str],
    item_levels: list[str],
    calibrators: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    student_lookup = {value: index for index, value in enumerate(student_levels)}
    item_lookup = {value: index for index, value in enumerate(item_levels)}
    item_order_rank = np.asarray(np.argsort(np.argsort(np.asarray(item_levels, dtype=object))), dtype=np.int64)
    static_item_term = model2_means["item_effect_mean"] + item_kc_matrix @ model2_means["kc_intercept_mean"]

    calibration_cutoff = int(round(float(config["calibration_student_share"]) * 100.0))
    state_bin_width = int(model3_means["state_bin_width"])
    train_df = trials.loc[trials["split"] == "train"].copy()
    test_df = trials.loc[trials["split"] == "test"].copy()
    last_train_bins = (
        train_df.assign(state_bin=(train_df["overall_opportunity"] // state_bin_width).astype("int64"))
        .groupby("student_id", sort=False)["state_bin"]
        .max()
    )

    records: list[dict] = []
    for student_id, student_test in test_df.groupby("student_id", sort=False):
        student_id = str(student_id)
        if student_id not in student_lookup:
            continue
        if stable_student_bucket(student_id, modulo=100) < calibration_cutoff:
            continue
        student_index = student_lookup[student_id]
        train_history = train_df.loc[train_df["student_id"].astype(str) == student_id].sort_values(
            ["timestamp", "attempt_id"], kind="mergesort"
        )
        student_test = student_test.sort_values(["timestamp", "attempt_id"], kind="mergesort")

        state = initialize_student_state(
            recent_attempt_window_5=int(config["recent_attempt_window_5"]),
            recent_attempt_window_10=int(config["recent_attempt_window_10"]),
        )
        prepare_state_arrays(state, num_items=len(item_levels), num_kcs=item_kc_matrix.shape[1])

        for row in train_history.itertuples(index=False):
            practice_vector = np.log1p(state["opportunity_counts"])
            raw_candidate_probabilities = score_candidates_model2(
                model2_means,
                student_index=student_index,
                candidate_indices=np.arange(len(item_levels), dtype=np.int64),
                item_kc_matrix=item_kc_matrix,
                success_feature_vector=state["success_decay"],
                failure_feature_vector=state["failure_decay"],
                practice_vector=practice_vector,
                static_item_term=static_item_term,
            )
            raw_actual_probability = float(raw_candidate_probabilities[item_lookup[str(row.item_id)]])
            update_state_from_row(
                state,
                row=row,
                attempt_event_lookup=attempt_event_lookup,
                item_lookup=item_lookup,
                raw_actual_probability=raw_actual_probability,
                decay_alpha=float(config["decay_alpha"]),
            )

        eval_step = 0
        for row in student_test.itertuples(index=False):
            step_is_eval = True
            if bool(config["primary_eval_only"]) and int(row.primary_eval_eligible) != 1:
                step_is_eval = False
            if step_is_eval:
                eval_step += 1
                if eval_step > int(config["max_eval_step"]):
                    step_is_eval = False

            practice_vector = np.log1p(state["opportunity_counts"])
            raw_candidate_probabilities = score_candidates_model2(
                model2_means,
                student_index=student_index,
                candidate_indices=np.arange(len(item_levels), dtype=np.int64),
                item_kc_matrix=item_kc_matrix,
                success_feature_vector=state["success_decay"],
                failure_feature_vector=state["failure_decay"],
                practice_vector=practice_vector,
                static_item_term=static_item_term,
            )
            actual_item_index = item_lookup[str(row.item_id)]
            raw_actual_probability = float(raw_candidate_probabilities[actual_item_index])

            if step_is_eval:
                linked_kc_exposure_total = item_kc_matrix @ state["opportunity_counts"]
                recent_failure_score = mean_over_linked_kcs(item_kc_matrix, state["failure_decay"])
                slate_info = build_kc_constrained_slate_mask(
                    item_kc_matrix=item_kc_matrix,
                    item_levels=item_levels,
                    item_exposure_counts=state["item_exposure_counts"],
                    recent_attempt_events=state["recent_attempt_events"],
                    decay_alpha=float(config["decay_alpha"]),
                    frontier_top_kcs=int(config["frontier_top_kcs"]),
                    slate_min_candidate_count=int(config["slate_min_candidate_count"]),
                )
                last_train_bin = int(last_train_bins.loc[student_id])
                _, future_state_variance = model3_future_state_summary(
                    model3_means,
                    student_index=student_index,
                    step_overall_opportunity=int(row.overall_opportunity),
                    last_train_bin=last_train_bin,
                )
                uncertainty_sd = float(np.sqrt(max(future_state_variance, 0.0)))
                global_features, candidate_features = build_step_features(
                    success_decay=state["success_decay"],
                    failure_decay=state["failure_decay"],
                    kc_abs_residual_decay=state["kc_abs_residual_decay"],
                    kc_sq_residual_decay=state["kc_sq_residual_decay"],
                    item_kc_matrix=item_kc_matrix,
                    recent_abs_residuals=state["recent_abs_residuals"],
                    recent_residuals=state["recent_residuals"],
                    recent_hint_window=state["recent_hint_window"],
                    recent_selection_change_window=state["recent_selection_change_window"],
                    recent_duration_window=state["recent_duration_window"],
                    recent_trust_feedback_window=state["recent_trust_feedback_window"],
                    recent_difficulty_feedback_window=state["recent_difficulty_feedback_window"],
                    baseline_duration_values=state["baseline_duration_values"],
                    uncertainty_sd=uncertainty_sd,
                    eval_step=eval_step,
                )
                candidate_feature_payload = combine_feature_payload(
                    global_features=global_features,
                    candidate_features=candidate_features,
                    item_index=None,
                )
                actual_feature_payload = combine_feature_payload(
                    global_features=global_features,
                    candidate_features=candidate_features,
                    item_index=actual_item_index,
                )
                candidate_mask = np.asarray(slate_info["candidate_mask"], dtype=bool)
                if not np.any(candidate_mask):
                    candidate_mask = state["item_exposure_counts"] == 0

                for policy_name in config["policy_names"]:
                    spec = POLICY_SPECS[policy_name]
                    for method_name in METHOD_SPECS:
                        calibration_spec = None
                        if method_name != "model2_raw":
                            calibration_spec = calibrators[policy_name][method_name]
                        candidate_probabilities = apply_policy_calibrator(
                            raw_candidate_probabilities,
                            candidate_feature_payload,
                            method_name=method_name,
                            calibration_spec=calibration_spec,
                        )
                        actual_next_probability = float(
                            apply_policy_calibrator(
                                np.asarray([raw_actual_probability], dtype=np.float64),
                                actual_feature_payload,
                                method_name=method_name,
                                calibration_spec=calibration_spec,
                            )[0]
                        )
                        selected_index, in_target_band = choose_policy_index(
                            candidate_probabilities,
                            candidate_mask=candidate_mask,
                            target_probability=float(spec["target_probability"]),
                            target_band_low=float(spec["target_band_low"]),
                            target_band_high=float(spec["target_band_high"]),
                            student_item_exposure=state["item_exposure_counts"],
                            linked_kc_exposure_total=linked_kc_exposure_total,
                            item_order_rank=item_order_rank,
                        )
                        selected_probability = float(candidate_probabilities[selected_index])
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
                                "target_probability": float(spec["target_probability"]),
                                "target_gap": abs(selected_probability - float(spec["target_probability"])),
                                "in_target_band": int(in_target_band),
                                "fallback_used": str(slate_info["kc_frontier_fallback_used"]),
                                "recent_failure_score": float(recent_failure_score[selected_index]),
                                "due_review_flag": 0,
                                "due_review_hours": np.nan,
                                "recommended_seen_item": int(state["item_exposure_counts"][selected_index] > 0),
                                "student_item_exposure_count": int(state["item_exposure_counts"][selected_index]),
                                "linked_kc_exposure_total": float(linked_kc_exposure_total[selected_index]),
                                "candidate_count": int(slate_info["candidate_count"]),
                                "actual_next_probability": actual_next_probability,
                                "actual_target_gap": abs(actual_next_probability - float(spec["target_probability"])),
                                "candidate_pool_mode": str(slate_info["candidate_pool_mode"]),
                                "kc_frontier_top5": str(slate_info["kc_frontier_top5"]),
                                "kc_frontier_fallback_used": str(slate_info["kc_frontier_fallback_used"]),
                                "track": str(config["track_name"]),
                                "model_name": "model2_qmatrix_rpfa",
                                "history_mode": "rpfa",
                                "decay_alpha": float(config["decay_alpha"]),
                            }
                        )

            update_state_from_row(
                state,
                row=row,
                attempt_event_lookup=attempt_event_lookup,
                item_lookup=item_lookup,
                raw_actual_probability=raw_actual_probability,
                decay_alpha=float(config["decay_alpha"]),
            )

    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    trials = load_trials(Path(config["processed_trials_path"]))
    attempt_kc_long = prepare_attempt_kc_long_for_history(
        load_attempt_kc_long_pfa(Path(config["attempt_kc_long_path"])),
        history_mode="rpfa",
        decay_alpha=float(config["decay_alpha"]),
        due_review_hours=float(config["due_review_hours"]),
    )
    model2_means, model3_means, student_levels, item_levels, kc_levels = load_model_means(config)
    kc_lookup = {value: index for index, value in enumerate(kc_levels)}
    item_kc_matrix, _ = build_item_kc_lookup(attempt_kc_long, item_levels, kc_levels)
    attempt_event_lookup = build_attempt_event_lookup(attempt_kc_long, kc_lookup=kc_lookup)

    training_rows = collect_policy_training_rows(
        config,
        trials=trials,
        attempt_event_lookup=attempt_event_lookup,
        item_kc_matrix=item_kc_matrix,
        model2_means=model2_means,
        model3_means=model3_means,
        student_levels=student_levels,
        item_levels=item_levels,
    )

    calibration_rows_output_path = Path(config["calibration_rows_output_path"])
    calibration_rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    training_rows.to_csv(calibration_rows_output_path, index=False)

    calibrators, actual_next_calibration_summary = fit_policy_calibrators(config, training_rows)

    policy_rows = rerun_policy_suite(
        config,
        trials=trials,
        attempt_event_lookup=attempt_event_lookup,
        item_kc_matrix=item_kc_matrix,
        model2_means=model2_means,
        model3_means=model3_means,
        student_levels=student_levels,
        item_levels=item_levels,
        calibrators=calibrators,
    )
    row_output_path = Path(config["row_output_path"])
    row_output_path.parent.mkdir(parents=True, exist_ok=True)
    policy_rows.to_csv(row_output_path, index=False)

    method_policy_summaries: dict[str, dict[str, dict]] = {}
    for method_name, method_rows in policy_rows.groupby("calibration_method", sort=True):
        method_policy_summaries[method_name] = {}
        for policy_name, group in method_rows.groupby("policy_name", sort=True):
            method_policy_summaries[method_name][policy_name] = summarize_policy_rows(
                group.copy(),
                max_eval_step=int(config["max_eval_step"]),
            )

    method_new_learning_summary = {}
    for method_name, policy_map in method_policy_summaries.items():
        method_new_learning_summary[method_name] = {
            "mean_target_gap_1_10": average_metric(policy_map, "student_avg_target_gap_1_10"),
            "mean_policy_advantage_1_10": average_metric(policy_map, "policy_advantage_over_actual_1_10"),
            "mean_stability": average_metric(policy_map, "recommendation_stability_mean_abs_diff"),
            "mean_band_hit_rate_1_10": average_metric(policy_map, "recommended_target_band_hit_rate_1_10"),
            "mean_candidate_count": average_metric(policy_map, "mean_candidate_count"),
            "mean_fallback_rate": average_metric(policy_map, "fallback_rate"),
        }

    baseline_method = "model2_raw"
    challenger_method = "policy_band_plus_local_residuals_plus_model3"
    residual_only_method = "policy_band_plus_local_residuals"
    primary_policy = str(config["primary_policy"])
    stability_tolerance = float(config["stability_tolerance"])

    primary_challenger = method_policy_summaries[challenger_method][primary_policy]
    primary_baseline = method_policy_summaries[baseline_method][primary_policy]
    primary_delta = {
        "target_gap_1_10_delta": float(primary_challenger["student_avg_target_gap_1_10"] - primary_baseline["student_avg_target_gap_1_10"]),
        "policy_advantage_1_10_delta": float(primary_challenger["policy_advantage_over_actual_1_10"] - primary_baseline["policy_advantage_over_actual_1_10"]),
        "stability_delta": float(primary_challenger["recommendation_stability_mean_abs_diff"] - primary_baseline["recommendation_stability_mean_abs_diff"]),
        "band_hit_rate_1_10_delta": float(
            primary_challenger["recommended_target_band_hit_rate_1_10"] - primary_baseline["recommended_target_band_hit_rate_1_10"]
        ),
    }
    pooled_delta = {
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
        "mean_band_hit_rate_1_10_delta": float(
            method_new_learning_summary[challenger_method]["mean_band_hit_rate_1_10"]
            - method_new_learning_summary[baseline_method]["mean_band_hit_rate_1_10"]
        ),
    }
    residual_vs_challenger_delta = {
        "mean_target_gap_1_10_delta": float(
            method_new_learning_summary[challenger_method]["mean_target_gap_1_10"]
            - method_new_learning_summary[residual_only_method]["mean_target_gap_1_10"]
        ),
        "mean_policy_advantage_1_10_delta": float(
            method_new_learning_summary[challenger_method]["mean_policy_advantage_1_10"]
            - method_new_learning_summary[residual_only_method]["mean_policy_advantage_1_10"]
        ),
        "mean_stability_delta": float(
            method_new_learning_summary[challenger_method]["mean_stability"]
            - method_new_learning_summary[residual_only_method]["mean_stability"]
        ),
    }

    pass_conditions = {
        "confidence_building_target_gap_improves": bool(primary_delta["target_gap_1_10_delta"] < 0.0),
        "pooled_target_gap_improves": bool(pooled_delta["mean_target_gap_1_10_delta"] < 0.0),
        "confidence_building_stability_within_tolerance": bool(primary_delta["stability_delta"] <= stability_tolerance),
        "pooled_stability_within_tolerance": bool(pooled_delta["mean_stability_delta"] <= stability_tolerance),
    }
    operational_pass = all(pass_conditions.values())

    comparison_rows = []
    for policy_name in config["policy_names"]:
        raw_summary = method_policy_summaries["model2_raw"][policy_name]
        policy_calibrated_summary = method_policy_summaries["policy_band_calibrated"][policy_name]
        residual_summary = method_policy_summaries["policy_band_plus_local_residuals"][policy_name]
        challenger_summary = method_policy_summaries["policy_band_plus_local_residuals_plus_model3"][policy_name]
        comparison_rows.append(
            {
                "policy_name": policy_name,
                "raw_target_gap_1_10": float(raw_summary["student_avg_target_gap_1_10"]),
                "policy_band_target_gap_1_10": float(policy_calibrated_summary["student_avg_target_gap_1_10"]),
                "residual_only_target_gap_1_10": float(residual_summary["student_avg_target_gap_1_10"]),
                "residual_plus_model3_target_gap_1_10": float(challenger_summary["student_avg_target_gap_1_10"]),
                "delta_residual_only_minus_raw": float(
                    residual_summary["student_avg_target_gap_1_10"] - raw_summary["student_avg_target_gap_1_10"]
                ),
                "delta_residual_plus_model3_minus_raw": float(
                    challenger_summary["student_avg_target_gap_1_10"] - raw_summary["student_avg_target_gap_1_10"]
                ),
                "delta_residual_plus_model3_minus_residual_only": float(
                    challenger_summary["student_avg_target_gap_1_10"] - residual_summary["student_avg_target_gap_1_10"]
                ),
                "raw_policy_advantage_1_10": float(raw_summary["policy_advantage_over_actual_1_10"]),
                "residual_only_policy_advantage_1_10": float(residual_summary["policy_advantage_over_actual_1_10"]),
                "residual_plus_model3_policy_advantage_1_10": float(challenger_summary["policy_advantage_over_actual_1_10"]),
                "raw_stability": float(raw_summary["recommendation_stability_mean_abs_diff"]),
                "residual_only_stability": float(residual_summary["recommendation_stability_mean_abs_diff"]),
                "residual_plus_model3_stability": float(challenger_summary["recommendation_stability_mean_abs_diff"]),
                "raw_mean_candidate_count": float(raw_summary["mean_candidate_count"]),
                "residual_plus_model3_mean_candidate_count": float(challenger_summary["mean_candidate_count"]),
                "raw_fallback_rate": float(raw_summary["fallback_rate"]),
                "residual_plus_model3_fallback_rate": float(challenger_summary["fallback_rate"]),
            }
        )

    calibration_summary_output_path = Path(config["actual_next_calibration_summary_output_path"])
    write_json(calibration_summary_output_path, actual_next_calibration_summary)

    summary = {
        "calibration_student_share": float(config["calibration_student_share"]),
        "policy_names": list(config["policy_names"]),
        "calibration_rows_output_path": str(calibration_rows_output_path),
        "policy_row_output_path": str(row_output_path),
        "baseline_method": baseline_method,
        "challenger_method": challenger_method,
        "residual_only_method": residual_only_method,
        "primary_policy": primary_policy,
        "stability_tolerance": stability_tolerance,
        "method_policy_summaries": method_policy_summaries,
        "method_new_learning_summary": method_new_learning_summary,
        "actual_next_calibration_summary": actual_next_calibration_summary,
        "primary_delta": primary_delta,
        "pooled_delta": pooled_delta,
        "residual_vs_challenger_delta": residual_vs_challenger_delta,
        "pass_conditions": pass_conditions,
        "operational_pass": operational_pass,
    }
    summary_output_path = Path(config["summary_output_path"])
    write_json(summary_output_path, summary)

    comparison_output_path = Path(config["comparison_output_path"])
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(comparison_output_path, index=False)

    print(f"Saved calibration-feature rows to {calibration_rows_output_path}")
    print(f"Saved local uncertainty policy rows to {row_output_path}")
    print(f"Saved summary to {summary_output_path}")
    print(f"Saved comparison to {comparison_output_path}")
    print(f"Saved actual-next calibration summary to {calibration_summary_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
