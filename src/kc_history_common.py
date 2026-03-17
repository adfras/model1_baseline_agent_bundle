from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_HISTORY_MODES = {"pfa", "rpfa"}


def resolve_history_value_columns(history_mode: str) -> tuple[str, str]:
    normalized = history_mode.strip().lower()
    if normalized not in SUPPORTED_HISTORY_MODES:
        raise ValueError(f"Unsupported history_mode '{history_mode}'. Expected one of {sorted(SUPPORTED_HISTORY_MODES)}.")
    if normalized == "pfa":
        return "kc_prior_success_count", "kc_prior_failure_count"
    return "kc_prior_success_decay", "kc_prior_failure_decay"


def add_decay_features(
    attempt_kc_long: pd.DataFrame,
    *,
    decay_alpha: float,
    due_review_hours: float = 48.0,
) -> pd.DataFrame:
    if not 0.0 < decay_alpha <= 1.0:
        raise ValueError("decay_alpha must lie in (0, 1].")
    if due_review_hours <= 0.0:
        raise ValueError("due_review_hours must be positive.")

    required = {
        "student_id",
        "kc_id",
        "timestamp",
        "attempt_id",
        "kc_success_increment",
        "kc_failure_increment",
    }
    missing = sorted(required.difference(attempt_kc_long.columns))
    if missing:
        raise ValueError(f"Attempt-KC table is missing required columns for decay features: {missing}")

    df = attempt_kc_long.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")

    df = df.sort_values(["student_id", "timestamp", "attempt_id", "kc_id"], kind="mergesort").reset_index(drop=True)

    timestamp_ns = df["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
    success_increment = df["kc_success_increment"].to_numpy(dtype="float64")
    failure_increment = df["kc_failure_increment"].to_numpy(dtype="float64")

    prior_success_decay = np.zeros(len(df), dtype="float64")
    prior_failure_decay = np.zeros(len(df), dtype="float64")
    last_seen_hours = np.full(len(df), np.nan, dtype="float64")

    grouped_indices = df.groupby(["student_id", "kc_id"], sort=False).indices
    nanoseconds_per_hour = 3_600_000_000_000.0

    for indexer in grouped_indices.values():
        positions = np.asarray(indexer, dtype="int64")
        prev_success_decay = 0.0
        prev_failure_decay = 0.0
        prev_success_increment = 0.0
        prev_failure_increment = 0.0
        prev_timestamp_ns: int | None = None

        for pos in positions:
            current_success_decay = decay_alpha * (prev_success_decay + prev_success_increment)
            current_failure_decay = decay_alpha * (prev_failure_decay + prev_failure_increment)

            prior_success_decay[pos] = current_success_decay
            prior_failure_decay[pos] = current_failure_decay

            if prev_timestamp_ns is not None:
                last_seen_hours[pos] = (timestamp_ns[pos] - prev_timestamp_ns) / nanoseconds_per_hour

            prev_success_decay = current_success_decay
            prev_failure_decay = current_failure_decay
            prev_success_increment = success_increment[pos]
            prev_failure_increment = failure_increment[pos]
            prev_timestamp_ns = int(timestamp_ns[pos])

    df["kc_prior_success_decay"] = prior_success_decay
    df["kc_prior_failure_decay"] = prior_failure_decay
    df["kc_last_seen_hours"] = last_seen_hours
    df["kc_due_review_default"] = np.where(
        np.isfinite(last_seen_hours) & (last_seen_hours >= due_review_hours),
        1,
        0,
    ).astype("int64")
    df["decay_alpha_used"] = float(decay_alpha)
    df["due_review_hours_threshold"] = float(due_review_hours)
    return df


def materialize_decay_features(
    attempt_kc_long_path: Path,
    *,
    decay_alpha: float,
    due_review_hours: float = 48.0,
) -> pd.DataFrame:
    df = pd.read_csv(attempt_kc_long_path)
    return add_decay_features(df, decay_alpha=decay_alpha, due_review_hours=due_review_hours)
