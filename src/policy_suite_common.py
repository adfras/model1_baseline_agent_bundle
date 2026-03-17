from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from kc_history_common import resolve_history_value_columns
from qmatrix_common import ensure_parent, load_trials
from qmatrix_pfa_common import load_attempt_kc_long_pfa, prepare_attempt_kc_long_for_history


POLICY_LIBRARY = {
    "balanced_challenge": {
        "target_probability": 0.72,
        "target_band_low": 0.65,
        "target_band_high": 0.80,
        "candidate_pool_mode": "unseen_only",
    },
    "harder_challenge": {
        "target_probability": 0.60,
        "target_band_low": 0.55,
        "target_band_high": 0.65,
        "candidate_pool_mode": "unseen_only",
    },
    "confidence_building": {
        "target_probability": 0.85,
        "target_band_low": 0.80,
        "target_band_high": 0.90,
        "candidate_pool_mode": "unseen_only",
    },
    "failure_aware_remediation": {
        "target_probability": 0.75,
        "target_band_low": 0.65,
        "target_band_high": 0.85,
        "candidate_pool_mode": "unseen_only",
    },
    "spacing_aware_review": {
        "target_probability": 0.80,
        "target_band_low": 0.70,
        "target_band_high": 0.90,
        "candidate_pool_mode": "unseen_plus_due_review",
    },
}


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def logistic_normal_mean(linear_mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(1.0 + (np.pi * variance / 8.0))
    return expit(linear_mean / scale)


def build_item_kc_lookup(
    attempt_kc_long: pd.DataFrame,
    item_levels: list[str],
    kc_levels: list[str],
) -> tuple[np.ndarray, dict[str, list[int]]]:
    item_lookup = {item_id: index for index, item_id in enumerate(item_levels)}
    kc_lookup = {kc_id: index for index, kc_id in enumerate(kc_levels)}

    item_kc = (
        attempt_kc_long.loc[:, ["item_id", "kc_id"]]
        .drop_duplicates()
        .sort_values(["item_id", "kc_id"], kind="mergesort")
    )

    matrix = np.zeros((len(item_levels), len(kc_levels)), dtype=np.float64)
    item_to_kc_indices: dict[str, list[int]] = {item_id: [] for item_id in item_levels}
    for row in item_kc.itertuples(index=False):
        item_id = str(row.item_id)
        kc_id = str(row.kc_id)
        if item_id not in item_lookup or kc_id not in kc_lookup:
            continue
        item_index = item_lookup[item_id]
        kc_index = kc_lookup[kc_id]
        matrix[item_index, kc_index] = 1.0
        item_to_kc_indices[item_id].append(kc_index)
    return matrix, item_to_kc_indices


def build_attempt_event_lookup(
    attempt_kc_long: pd.DataFrame,
    *,
    kc_lookup: dict[str, int],
) -> dict[int, dict]:
    events: dict[int, dict] = {}
    for attempt_id, group in attempt_kc_long.groupby("attempt_id", sort=False):
        group = group.sort_values(["kc_relationship_id", "kc_id"], kind="mergesort")
        rows = []
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "kc_index": kc_lookup[str(row.kc_id)],
                    "kc_exposure_increment": float(row.kc_exposure_increment),
                    "kc_success_increment": float(row.kc_success_increment),
                    "kc_failure_increment": float(row.kc_failure_increment),
                }
            )
        timestamp = pd.to_datetime(group["timestamp"].iloc[0], utc=True, errors="raise")
        events[int(attempt_id)] = {
            "item_id": str(group["item_id"].iloc[0]),
            "timestamp": timestamp,
            "rows": rows,
        }
    return events


def student_average(values: pd.Series, student_ids: pd.Series) -> float:
    grouped = pd.DataFrame({"student_id": student_ids, "value": values}).groupby("student_id", sort=False)["value"].mean()
    return float(grouped.mean()) if len(grouped) else float("nan")


def summarize_policy_rows(rows: pd.DataFrame, *, max_eval_step: int) -> dict:
    if rows.empty:
        return {"recommendation_rows": 0, "recommendation_students": 0}

    early_5 = rows.loc[rows["eval_step"] <= 5]
    early_max = rows.loc[rows["eval_step"] <= max_eval_step]
    stability = (
        rows.sort_values(["student_id", "eval_step"], kind="mergesort")
        .groupby("student_id", sort=False)["recommended_probability"]
        .apply(lambda s: s.diff().abs().mean())
        .dropna()
    )
    due_rows = rows.loc[rows["due_review_flag"] == 1]
    return {
        "recommendation_rows": int(len(rows)),
        "recommendation_students": int(rows["student_id"].nunique()),
        "student_avg_target_gap_overall": student_average(rows["target_gap"], rows["student_id"]),
        "student_avg_target_gap_1_5": student_average(early_5["target_gap"], early_5["student_id"]),
        "student_avg_target_gap_1_10": student_average(early_max["target_gap"], early_max["student_id"]),
        "student_avg_actual_target_gap_overall": student_average(rows["actual_target_gap"], rows["student_id"]),
        "student_avg_actual_target_gap_1_5": student_average(early_5["actual_target_gap"], early_5["student_id"]),
        "student_avg_actual_target_gap_1_10": student_average(early_max["actual_target_gap"], early_max["student_id"]),
        "recommended_target_band_hit_rate_overall": float(rows["in_target_band"].mean()),
        "recommended_target_band_hit_rate_1_5": float(early_5["in_target_band"].mean()) if len(early_5) else float("nan"),
        "recommended_target_band_hit_rate_1_10": float(early_max["in_target_band"].mean()) if len(early_max) else float("nan"),
        "policy_advantage_over_actual_overall": float((rows["actual_target_gap"] - rows["target_gap"]).mean()),
        "policy_advantage_over_actual_1_5": float((early_5["actual_target_gap"] - early_5["target_gap"]).mean()) if len(early_5) else float("nan"),
        "policy_advantage_over_actual_1_10": float((early_max["actual_target_gap"] - early_max["target_gap"]).mean()) if len(early_max) else float("nan"),
        "recommendation_stability_mean_abs_diff": float(stability.mean()) if len(stability) else float("nan"),
        "recent_failure_coverage_rate": float((rows["recent_failure_score"] > 0).mean()),
        "mean_recent_failure_score": float(rows["recent_failure_score"].mean()),
        "due_review_coverage_rate": float(rows["due_review_flag"].mean()),
        "mean_due_review_hours": float(due_rows["due_review_hours"].mean()) if len(due_rows) else float("nan"),
        "mean_candidate_count": float(rows["candidate_count"].mean()),
        "fallback_rate": float((rows["fallback_used"] != "none").mean()),
        "seen_item_recommendation_rate": float(rows["recommended_seen_item"].mean()),
    }


def score_candidates_model2(
    posterior_means: dict[str, np.ndarray | float | None],
    *,
    student_index: int,
    candidate_indices: np.ndarray,
    item_kc_matrix: np.ndarray,
    success_feature_vector: np.ndarray,
    failure_feature_vector: np.ndarray,
    practice_vector: np.ndarray,
    static_item_term: np.ndarray,
) -> np.ndarray:
    candidate_matrix = item_kc_matrix[candidate_indices]
    practice_total = candidate_matrix @ practice_vector
    linear = (
        float(posterior_means["Intercept_mean"])
        + float(posterior_means["student_intercept_mean"][student_index])
        + static_item_term[candidate_indices]
        + candidate_matrix
        @ (
            posterior_means["kc_success_mean"] * success_feature_vector
            + posterior_means["kc_failure_mean"] * failure_feature_vector
        )
        + practice_total * float(posterior_means["student_slope_mean"][student_index])
    )
    return expit(linear)


def score_candidates_model3(
    posterior_means: dict[str, np.ndarray | float | None],
    *,
    student_index: int,
    candidate_indices: np.ndarray,
    item_kc_matrix: np.ndarray,
    success_feature_vector: np.ndarray,
    failure_feature_vector: np.ndarray,
    practice_vector: np.ndarray,
    static_item_term: np.ndarray,
    step_overall_opportunity: int,
    last_train_bin: int,
) -> np.ndarray:
    state_bin_width = int(posterior_means["state_bin_width"])
    step_bin = int(step_overall_opportunity // state_bin_width)
    delta = max(step_bin - last_train_bin, 0)

    candidate_matrix = item_kc_matrix[candidate_indices]
    practice_total = candidate_matrix @ practice_vector
    rho = float(posterior_means["rho_mean"])
    latent_state = float(posterior_means["latent_state_mean"][last_train_bin, student_index])
    state_sigma_student = float(posterior_means["state_sigma_student_mean"][student_index])

    rho_power = float(np.power(rho, delta))
    future_state_mean = latent_state * rho_power
    if delta == 0:
        future_state_variance = 0.0
    else:
        future_state_variance = (
            (state_sigma_student**2) * (1.0 - float(np.power(rho, 2 * delta))) / max(1.0 - rho**2, 1e-6)
        )

    linear_mean = (
        float(posterior_means["Intercept_mean"])
        + float(posterior_means["student_intercept_mean"][student_index])
        + static_item_term[candidate_indices]
        + candidate_matrix
        @ (
            posterior_means["kc_success_mean"] * success_feature_vector
            + posterior_means["kc_failure_mean"] * failure_feature_vector
        )
        + practice_total * float(posterior_means["student_slope_mean"][student_index])
        + future_state_mean
    )
    return logistic_normal_mean(linear_mean, np.full_like(linear_mean, future_state_variance))


def build_candidate_frame(
    *,
    item_levels: list[str],
    candidate_probabilities: np.ndarray,
    item_to_kc_indices: dict[str, list[int]],
    item_exposure_counts: dict[str, int],
    opportunity_counts: np.ndarray,
    failure_feature_vector: np.ndarray,
    current_timestamp_ns: int,
    last_seen_timestamp_ns: np.ndarray,
    due_review_hours_threshold: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for item_index, item_id in enumerate(item_levels):
        kc_indices = item_to_kc_indices.get(item_id, [])
        student_item_exposure_count = int(item_exposure_counts.get(item_id, 0))
        linked_kc_exposure_total = float(np.sum(opportunity_counts[kc_indices])) if kc_indices else 0.0
        recent_failure_score = float(np.sum(failure_feature_vector[kc_indices])) if kc_indices else 0.0

        due_hours = np.nan
        due_flag = 0
        if kc_indices:
            seen_timestamps = last_seen_timestamp_ns[kc_indices]
            seen_timestamps = seen_timestamps[seen_timestamps >= 0]
            if len(seen_timestamps):
                hours = (current_timestamp_ns - seen_timestamps) / 3_600_000_000_000.0
                due_hours = float(np.max(hours))
                due_flag = int(np.any(hours >= due_review_hours_threshold))

        rows.append(
            {
                "item_id": item_id,
                "item_index": item_index,
                "predicted_probability": float(candidate_probabilities[item_index]),
                "student_item_exposure_count": student_item_exposure_count,
                "linked_kc_exposure_total": linked_kc_exposure_total,
                "recent_failure_score": recent_failure_score,
                "due_review_flag": due_flag,
                "due_review_hours": due_hours,
                "is_unseen_candidate": int(student_item_exposure_count == 0),
                "recommended_seen_item": int(student_item_exposure_count > 0),
            }
        )
    return pd.DataFrame(rows)


def choose_balanced_like(
    candidates: pd.DataFrame,
    *,
    target_probability: float,
    target_band_low: float,
    target_band_high: float,
    candidate_pool_mode: str,
    fallback_used: str = "none",
    prioritize_due_hours: bool = False,
) -> tuple[pd.Series, str, int]:
    working = candidates.copy()
    if candidate_pool_mode == "unseen_only":
        working = working.loc[working["is_unseen_candidate"] == 1].copy()
    elif candidate_pool_mode != "unseen_plus_due_review":
        raise ValueError(f"Unsupported candidate_pool_mode: {candidate_pool_mode}")

    if working.empty:
        working = candidates.copy()
        if fallback_used == "none":
            fallback_used = "all_items_fallback"

    working["selection_gap"] = (working["predicted_probability"] - target_probability).abs()
    sort_columns = ["selection_gap"]
    ascending = [True]
    if prioritize_due_hours:
        working["due_review_hours_rank"] = working["due_review_hours"].fillna(-1.0)
        sort_columns.append("due_review_hours_rank")
        ascending.append(False)
    sort_columns.extend(["student_item_exposure_count", "linked_kc_exposure_total", "item_id"])
    ascending.extend([True, True, True])
    selected = working.sort_values(sort_columns, ascending=ascending, kind="mergesort").iloc[0]
    in_band = int(target_band_low <= float(selected["predicted_probability"]) <= target_band_high)
    return selected, fallback_used, in_band


def choose_policy_item(policy_name: str, candidates: pd.DataFrame) -> tuple[pd.Series, dict]:
    spec = POLICY_LIBRARY[policy_name]
    target_probability = float(spec["target_probability"])
    target_band_low = float(spec["target_band_low"])
    target_band_high = float(spec["target_band_high"])

    if policy_name in {"balanced_challenge", "harder_challenge", "confidence_building"}:
        selected, fallback_used, in_band = choose_balanced_like(
            candidates,
            target_probability=target_probability,
            target_band_low=target_band_low,
            target_band_high=target_band_high,
            candidate_pool_mode=str(spec["candidate_pool_mode"]),
        )
        return selected, {
            "target_probability": target_probability,
            "target_band_low": target_band_low,
            "target_band_high": target_band_high,
            "candidate_pool_mode": str(spec["candidate_pool_mode"]),
            "fallback_used": fallback_used,
            "in_target_band": in_band,
        }

    if policy_name == "failure_aware_remediation":
        unseen = candidates.loc[candidates["is_unseen_candidate"] == 1].copy()
        if unseen.empty:
            selected, fallback_used, in_band = choose_balanced_like(
                candidates,
                target_probability=POLICY_LIBRARY["balanced_challenge"]["target_probability"],
                target_band_low=POLICY_LIBRARY["balanced_challenge"]["target_band_low"],
                target_band_high=POLICY_LIBRARY["balanced_challenge"]["target_band_high"],
                candidate_pool_mode="unseen_only",
                fallback_used="balanced_challenge_empty_unseen",
            )
            return selected, {
                "target_probability": target_probability,
                "target_band_low": target_band_low,
                "target_band_high": target_band_high,
                "candidate_pool_mode": "unseen_only",
                "fallback_used": fallback_used,
                "in_target_band": in_band,
            }

        positive = unseen.loc[unseen["recent_failure_score"] > 0].copy()
        if positive.empty:
            selected, fallback_used, in_band = choose_balanced_like(
                unseen,
                target_probability=POLICY_LIBRARY["balanced_challenge"]["target_probability"],
                target_band_low=POLICY_LIBRARY["balanced_challenge"]["target_band_low"],
                target_band_high=POLICY_LIBRARY["balanced_challenge"]["target_band_high"],
                candidate_pool_mode="unseen_only",
                fallback_used="balanced_challenge_no_recent_failure",
            )
            return selected, {
                "target_probability": target_probability,
                "target_band_low": target_band_low,
                "target_band_high": target_band_high,
                "candidate_pool_mode": "unseen_only",
                "fallback_used": fallback_used,
                "in_target_band": in_band,
            }

        threshold = float(positive["recent_failure_score"].quantile(0.75))
        restricted = positive.loc[positive["recent_failure_score"] >= threshold].copy()
        if restricted.empty:
            restricted = positive.copy()

        selected, fallback_used, in_band = choose_balanced_like(
            restricted,
            target_probability=target_probability,
            target_band_low=target_band_low,
            target_band_high=target_band_high,
            candidate_pool_mode="unseen_only",
        )
        return selected, {
            "target_probability": target_probability,
            "target_band_low": target_band_low,
            "target_band_high": target_band_high,
            "candidate_pool_mode": "unseen_only",
            "fallback_used": fallback_used,
            "in_target_band": in_band,
        }

    if policy_name == "spacing_aware_review":
        due_candidates = candidates.loc[candidates["due_review_flag"] == 1].copy()
        if due_candidates.empty:
            selected, fallback_used, in_band = choose_balanced_like(
                candidates,
                target_probability=POLICY_LIBRARY["balanced_challenge"]["target_probability"],
                target_band_low=POLICY_LIBRARY["balanced_challenge"]["target_band_low"],
                target_band_high=POLICY_LIBRARY["balanced_challenge"]["target_band_high"],
                candidate_pool_mode="unseen_only",
                fallback_used="balanced_challenge_no_due_review",
            )
            return selected, {
                "target_probability": target_probability,
                "target_band_low": target_band_low,
                "target_band_high": target_band_high,
                "candidate_pool_mode": "unseen_plus_due_review",
                "fallback_used": fallback_used,
                "in_target_band": in_band,
            }

        selected, fallback_used, in_band = choose_balanced_like(
            due_candidates,
            target_probability=target_probability,
            target_band_low=target_band_low,
            target_band_high=target_band_high,
            candidate_pool_mode="unseen_plus_due_review",
            prioritize_due_hours=True,
        )
        return selected, {
            "target_probability": target_probability,
            "target_band_low": target_band_low,
            "target_band_high": target_band_high,
            "candidate_pool_mode": "unseen_plus_due_review",
            "fallback_used": fallback_used,
            "in_target_band": in_band,
        }

    raise ValueError(f"Unsupported policy_name: {policy_name}")


def run_policy_suite(config: dict) -> int:
    trials = load_trials(Path(config["processed_trials_path"]))
    posterior = np.load(Path(config["posterior_draws_path"]), allow_pickle=True)

    model_kind = str(config["model_kind"]).lower()
    if model_kind not in {"model2", "model3"}:
        raise ValueError("model_kind must be one of: model2, model3")

    history_mode = str(
        config.get(
            "history_mode",
            posterior["history_mode"].reshape(-1)[0] if "history_mode" in posterior.files else "pfa",
        )
    ).lower()
    decay_alpha = float(
        config.get(
            "decay_alpha",
            np.asarray(posterior["decay_alpha"]).reshape(-1)[0] if "decay_alpha" in posterior.files else 1.0,
        )
    )
    due_review_hours_threshold = float(config.get("due_review_hours", 48.0))
    attempt_kc_long = prepare_attempt_kc_long_for_history(
        load_attempt_kc_long_pfa(Path(config["attempt_kc_long_path"])),
        history_mode=history_mode,
        decay_alpha=decay_alpha,
        due_review_hours=due_review_hours_threshold,
    )
    resolve_history_value_columns(history_mode)

    policy_names = [str(name) for name in config.get("policy_names", list(POLICY_LIBRARY.keys()))]
    unknown_policies = sorted(set(policy_names).difference(POLICY_LIBRARY.keys()))
    if unknown_policies:
        raise ValueError(f"Unsupported policy names: {unknown_policies}")

    row_output_path = Path(config["row_output_path"])
    summary_output_path = Path(config["summary_output_path"])
    max_eval_step = int(config.get("max_eval_step", 10))
    primary_eval_only = bool(config.get("primary_eval_only", True))

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
        "student_slope_mean": posterior["student_slope"].mean(axis=0) if "student_slope" in posterior.files else None,
        "kc_success_mean": posterior["kc_success"].mean(axis=0) if "kc_success" in posterior.files else None,
        "kc_failure_mean": posterior["kc_failure"].mean(axis=0) if "kc_failure" in posterior.files else None,
        "rho_mean": float(posterior["rho"].mean()) if "rho" in posterior.files else None,
        "state_sigma_student_mean": posterior["state_sigma_student"].mean(axis=0) if "state_sigma_student" in posterior.files else None,
        "latent_state_mean": posterior["latent_state"].mean(axis=0) if "latent_state" in posterior.files else None,
        "state_bin_width": int(np.asarray(posterior["state_bin_width"]).reshape(-1)[0]) if "state_bin_width" in posterior.files else None,
    }
    item_effect_mean = posterior["item_effect"].mean(axis=0)
    kc_intercept_mean = posterior["kc_intercept"].mean(axis=0)
    posterior.close()

    static_item_term = item_effect_mean + item_kc_matrix @ kc_intercept_mean

    train_df = trials.loc[trials["split"] == "train"].copy()
    test_df = trials.loc[trials["split"] == "test"].copy()

    if model_kind == "model3":
        state_bin_width = int(posterior_means["state_bin_width"])
        last_train_bins = (
            train_df.assign(state_bin=(train_df["overall_opportunity"] // state_bin_width).astype("int64"))
            .groupby("student_id", sort=False)["state_bin"]
            .max()
        )
    else:
        last_train_bins = None

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
        success_counts = np.zeros(len(kc_levels), dtype=np.float64)
        failure_counts = np.zeros(len(kc_levels), dtype=np.float64)
        success_decay = np.zeros(len(kc_levels), dtype=np.float64)
        failure_decay = np.zeros(len(kc_levels), dtype=np.float64)
        last_seen_timestamp_ns = np.full(len(kc_levels), -1, dtype=np.int64)
        item_exposure_counts: dict[str, int] = {}

        def update_from_attempt(attempt_id: int) -> None:
            event = attempt_event_lookup[int(attempt_id)]
            timestamp_ns = int(event["timestamp"].to_datetime64().astype("datetime64[ns]").astype("int64"))
            item_id_local = str(event["item_id"])
            item_exposure_counts[item_id_local] = item_exposure_counts.get(item_id_local, 0) + 1
            for row in event["rows"]:
                kc_index = int(row["kc_index"])
                opportunity_counts[kc_index] += float(row["kc_exposure_increment"])
                success_counts[kc_index] += float(row["kc_success_increment"])
                failure_counts[kc_index] += float(row["kc_failure_increment"])
                success_decay[kc_index] = decay_alpha * (success_decay[kc_index] + float(row["kc_success_increment"]))
                failure_decay[kc_index] = decay_alpha * (failure_decay[kc_index] + float(row["kc_failure_increment"]))
                last_seen_timestamp_ns[kc_index] = timestamp_ns

        for row in train_history.itertuples(index=False):
            update_from_attempt(int(row.attempt_id))

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
                success_feature_vector = success_decay if history_mode == "rpfa" else success_counts
                failure_feature_vector = failure_decay if history_mode == "rpfa" else failure_counts

                if model_kind == "model2":
                    candidate_probabilities = score_candidates_model2(
                        posterior_means,
                        student_index=student_index,
                        candidate_indices=candidate_indices,
                        item_kc_matrix=item_kc_matrix,
                        success_feature_vector=success_feature_vector,
                        failure_feature_vector=failure_feature_vector,
                        practice_vector=practice_vector,
                        static_item_term=static_item_term,
                    )
                else:
                    last_train_bin = int(last_train_bins.loc[student_id])
                    candidate_probabilities = score_candidates_model3(
                        posterior_means,
                        student_index=student_index,
                        candidate_indices=candidate_indices,
                        item_kc_matrix=item_kc_matrix,
                        success_feature_vector=success_feature_vector,
                        failure_feature_vector=failure_feature_vector,
                        practice_vector=practice_vector,
                        static_item_term=static_item_term,
                        step_overall_opportunity=int(row.overall_opportunity),
                        last_train_bin=last_train_bin,
                    )

                candidates = build_candidate_frame(
                    item_levels=item_levels,
                    candidate_probabilities=candidate_probabilities,
                    item_to_kc_indices=item_to_kc_indices,
                    item_exposure_counts=item_exposure_counts,
                    opportunity_counts=opportunity_counts,
                    failure_feature_vector=failure_feature_vector,
                    current_timestamp_ns=current_timestamp_ns,
                    last_seen_timestamp_ns=last_seen_timestamp_ns,
                    due_review_hours_threshold=due_review_hours_threshold,
                )
                actual_next_probability = float(candidate_probabilities[item_lookup[str(row.item_id)]])

                for policy_name in policy_names:
                    selected, policy_meta = choose_policy_item(policy_name, candidates)
                    target_probability = float(policy_meta["target_probability"])
                    records.append(
                        {
                            "student_id": student_id,
                            "attempt_id": int(row.attempt_id),
                            "actual_item_id": str(row.item_id),
                            "actual_correct": int(row.correct),
                            "eval_step": eval_step,
                            "policy_name": policy_name,
                            "candidate_pool_mode": str(policy_meta["candidate_pool_mode"]),
                            "recommended_item_id": str(selected["item_id"]),
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
                            "track": str(config.get("track_name", f"adaptive_policy_{model_kind}_{history_mode}")),
                            "model_name": f"{model_kind}_qmatrix_{history_mode}",
                            "history_mode": history_mode,
                            "decay_alpha": decay_alpha,
                        }
                    )

            update_from_attempt(int(row.attempt_id))

    rows = pd.DataFrame(records)
    ensure_parent(row_output_path)
    rows.to_csv(row_output_path, index=False)

    policy_summaries = {}
    for policy_name, group in rows.groupby("policy_name", sort=True):
        policy_summaries[policy_name] = summarize_policy_rows(group.copy(), max_eval_step=max_eval_step)

    summary = {
        "model_kind": model_kind,
        "history_mode": history_mode,
        "decay_alpha": decay_alpha,
        "due_review_hours": due_review_hours_threshold,
        "policy_names": policy_names,
        "evaluation_rows": int(len(rows)),
        "evaluation_students": int(rows["student_id"].nunique()) if len(rows) else 0,
        "max_eval_step": max_eval_step,
        "primary_eval_only": primary_eval_only,
        "row_output_path": str(row_output_path),
        "policy_summaries": policy_summaries,
    }
    write_json(summary_output_path, summary)

    print(f"Saved adaptive policy rows to {row_output_path}")
    print(f"Saved adaptive policy summary to {summary_output_path}")
    return 0
