from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan.basic import scan

from kc_history_common import add_decay_features, resolve_history_value_columns
from qmatrix_common import (
    QMatrixContext,
    build_context,
    ensure_parent,
    fit_model,
    load_json,
    load_trials,
    prepend_compiler_to_path,
    summarize_vector_draws,
    summarize_vi_history,
    to_numpy_draws,
)


def load_attempt_kc_long_pfa(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")

    string_columns = ["student_id", "item_id", "kc_id", "kc_name"]
    for column in string_columns:
        if column in df.columns:
            df[column] = df[column].astype("string")

    int_columns = [
        "attempt_id",
        "correct",
        "kc_relationship_id",
        "kc_count",
        "kc_due_review_default",
    ]
    for column in int_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="raise").astype("int64")

    float_columns = [
        "kc_opportunity",
        "kc_exposure_increment",
        "kc_success_increment",
        "kc_failure_increment",
        "kc_prior_success_count",
        "kc_prior_failure_count",
        "kc_prior_success_decay",
        "kc_prior_failure_decay",
        "kc_last_seen_hours",
        "decay_alpha_used",
        "due_review_hours_threshold",
        "kc_practice_component",
        "kc_success_component",
        "kc_failure_component",
        "kc_weight_equal",
    ]
    for column in float_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")
    if "kc_base_indicator" not in df.columns:
        df["kc_base_indicator"] = np.float32(1.0)
    return df


def prepare_attempt_kc_long_for_history(
    attempt_kc_long: pd.DataFrame,
    *,
    history_mode: str,
    decay_alpha: float = 1.0,
    due_review_hours: float = 48.0,
) -> pd.DataFrame:
    normalized = history_mode.strip().lower()
    if normalized == "rpfa":
        return add_decay_features(
            attempt_kc_long,
            decay_alpha=decay_alpha,
            due_review_hours=due_review_hours,
        )
    if normalized != "pfa":
        raise ValueError("history_mode must be one of: pfa, rpfa")
    return attempt_kc_long.copy()


def build_feature_matrix(
    df: pd.DataFrame,
    attempt_kc_long: pd.DataFrame,
    context: QMatrixContext,
    *,
    value_column: str,
) -> np.ndarray:
    attempt_order = pd.Series(np.arange(len(df), dtype="int64"), index=df["attempt_id"].to_numpy(dtype="int64"))
    filtered = attempt_kc_long.loc[
        attempt_kc_long["attempt_id"].isin(attempt_order.index)
        & attempt_kc_long["kc_id"].isin(context.kc_lookup.keys())
    ].copy()

    row_idx = filtered["attempt_id"].map(attempt_order).to_numpy(dtype="int64")
    col_idx = filtered["kc_id"].map(context.kc_lookup).to_numpy(dtype="int64")
    matrix = np.zeros((len(df), len(context.kc_levels)), dtype=np.float32)
    np.add.at(matrix, (row_idx, col_idx), filtered[value_column].to_numpy(dtype=np.float32))
    return matrix


@dataclass
class PFADataset:
    df: pd.DataFrame
    student_idx: np.ndarray
    item_idx: np.ndarray
    x_kc_base: np.ndarray
    x_kc_practice: np.ndarray
    x_kc_success: np.ndarray
    x_kc_failure: np.ndarray
    practice_total: np.ndarray
    correct: np.ndarray
    state_bin_idx: np.ndarray | None = None
    n_state_steps: int | None = None


def prepare_pfa_dataset(
    trials: pd.DataFrame,
    attempt_kc_long: pd.DataFrame,
    context: QMatrixContext,
    *,
    split: str,
    primary_eval_only: bool = False,
    state_bin_width: int | None = None,
    success_value_column: str = "kc_prior_success_count",
    failure_value_column: str = "kc_prior_failure_count",
) -> PFADataset:
    df = trials.loc[trials["split"] == split].copy()
    if primary_eval_only:
        df = df.loc[df["primary_eval_eligible"] == 1].copy()
    df = df.reset_index(drop=True)

    student_idx = df["student_id"].map(context.student_lookup)
    if student_idx.isna().any():
        unseen = df.loc[student_idx.isna(), "student_id"].unique().tolist()
        raise ValueError(f"Encountered unseen student ids: {unseen[:5]}")

    item_idx = df["item_id"].map(context.item_lookup)
    if item_idx.isna().any():
        unseen = df.loc[item_idx.isna(), "item_id"].unique().tolist()
        raise ValueError(f"Encountered unseen item ids: {unseen[:5]}")

    x_kc_base = build_feature_matrix(df, attempt_kc_long, context, value_column="kc_base_indicator")
    x_kc_practice = build_feature_matrix(df, attempt_kc_long, context, value_column="kc_practice_component")
    x_kc_success = build_feature_matrix(df, attempt_kc_long, context, value_column=success_value_column)
    x_kc_failure = build_feature_matrix(df, attempt_kc_long, context, value_column=failure_value_column)

    state_bin_idx = None
    n_state_steps = None
    if state_bin_width is not None:
        if state_bin_width < 1:
            raise ValueError("state_bin_width must be at least 1.")
        state_bin_idx = (df["overall_opportunity"] // state_bin_width).to_numpy(dtype="int64")
        n_state_steps = int(state_bin_idx.max()) + 1 if len(state_bin_idx) else 0

    return PFADataset(
        df=df,
        student_idx=student_idx.to_numpy(dtype="int64"),
        item_idx=item_idx.to_numpy(dtype="int64"),
        x_kc_base=x_kc_base,
        x_kc_practice=x_kc_practice,
        x_kc_success=x_kc_success,
        x_kc_failure=x_kc_failure,
        practice_total=x_kc_practice.sum(axis=1).astype("float32"),
        correct=df["correct"].to_numpy(dtype="int8"),
        state_bin_idx=state_bin_idx,
        n_state_steps=n_state_steps,
    )


def build_model2_qmatrix_pfa(dataset: PFADataset, context: QMatrixContext) -> pm.Model:
    coords = {
        "student": context.student_levels,
        "item": context.item_levels,
        "kc": context.kc_levels,
    }

    with pm.Model(coords=coords) as model:
        student_idx = pm.Data("student_idx", dataset.student_idx)
        item_idx = pm.Data("item_idx", dataset.item_idx)
        x_kc_base = pm.Data("x_kc_base", dataset.x_kc_base)
        x_kc_success = pm.Data("x_kc_success", dataset.x_kc_success)
        x_kc_failure = pm.Data("x_kc_failure", dataset.x_kc_failure)
        practice_total = pm.Data("practice_total", dataset.practice_total)

        intercept = pm.Normal("Intercept", mu=0.0, sigma=1.5)

        student_intercept_sigma = pm.HalfNormal("student_intercept_sigma", sigma=1.0)
        student_intercept_offset = pm.Normal("student_intercept_offset", mu=0.0, sigma=1.0, dims="student")
        student_intercept = pm.Deterministic(
            "student_intercept",
            student_intercept_sigma * student_intercept_offset,
            dims="student",
        )

        student_slope_sigma = pm.HalfNormal("student_slope_sigma", sigma=0.5)
        student_slope_offset = pm.Normal("student_slope_offset", mu=0.0, sigma=1.0, dims="student")
        student_slope = pm.Deterministic(
            "student_slope",
            student_slope_sigma * student_slope_offset,
            dims="student",
        )

        item_sigma = pm.HalfNormal("item_sigma", sigma=1.0)
        item_offset = pm.Normal("item_offset", mu=0.0, sigma=1.0, dims="item")
        item_effect = pm.Deterministic("item_effect", item_sigma * item_offset, dims="item")

        kc_intercept_sigma = pm.HalfNormal("kc_intercept_sigma", sigma=1.0)
        kc_intercept_offset = pm.Normal("kc_intercept_offset", mu=0.0, sigma=1.0, dims="kc")
        kc_intercept = pm.Deterministic("kc_intercept", kc_intercept_sigma * kc_intercept_offset, dims="kc")

        kc_success_sigma = pm.HalfNormal("kc_success_sigma", sigma=0.5)
        kc_success_offset = pm.Normal("kc_success_offset", mu=0.0, sigma=1.0, dims="kc")
        kc_success = pm.Deterministic("kc_success", kc_success_sigma * kc_success_offset, dims="kc")

        kc_failure_sigma = pm.HalfNormal("kc_failure_sigma", sigma=0.5)
        kc_failure_offset = pm.Normal("kc_failure_offset", mu=0.0, sigma=1.0, dims="kc")
        kc_failure = pm.Deterministic("kc_failure", kc_failure_sigma * kc_failure_offset, dims="kc")

        linear = (
            intercept
            + student_intercept[student_idx]
            + item_effect[item_idx]
            + pt.dot(x_kc_base, kc_intercept)
            + pt.dot(x_kc_success, kc_success)
            + pt.dot(x_kc_failure, kc_failure)
            + student_slope[student_idx] * practice_total
        )

        pm.Bernoulli("correct_obs", logit_p=linear, observed=dataset.correct)

    return model


def build_model3_qmatrix_pfa(dataset: PFADataset, context: QMatrixContext) -> pm.Model:
    if dataset.state_bin_idx is None or dataset.n_state_steps is None:
        raise ValueError("Model 3 requires state_bin_idx and n_state_steps in the dataset.")

    coords = {
        "student": context.student_levels,
        "item": context.item_levels,
        "kc": context.kc_levels,
        "state_step": list(range(dataset.n_state_steps)),
    }
    if dataset.n_state_steps > 1:
        coords["state_step_rest"] = list(range(1, dataset.n_state_steps))

    with pm.Model(coords=coords) as model:
        student_idx = pm.Data("student_idx", dataset.student_idx)
        item_idx = pm.Data("item_idx", dataset.item_idx)
        state_step_idx = pm.Data("state_step_idx", dataset.state_bin_idx)
        x_kc_base = pm.Data("x_kc_base", dataset.x_kc_base)
        x_kc_success = pm.Data("x_kc_success", dataset.x_kc_success)
        x_kc_failure = pm.Data("x_kc_failure", dataset.x_kc_failure)
        practice_total = pm.Data("practice_total", dataset.practice_total)

        intercept = pm.Normal("Intercept", mu=0.0, sigma=1.5)

        student_intercept_sigma = pm.HalfNormal("student_intercept_sigma", sigma=1.0)
        student_intercept_offset = pm.Normal("student_intercept_offset", mu=0.0, sigma=1.0, dims="student")
        student_intercept = pm.Deterministic(
            "student_intercept",
            student_intercept_sigma * student_intercept_offset,
            dims="student",
        )

        student_slope_sigma = pm.HalfNormal("student_slope_sigma", sigma=0.5)
        student_slope_offset = pm.Normal("student_slope_offset", mu=0.0, sigma=1.0, dims="student")
        student_slope = pm.Deterministic(
            "student_slope",
            student_slope_sigma * student_slope_offset,
            dims="student",
        )

        item_sigma = pm.HalfNormal("item_sigma", sigma=1.0)
        item_offset = pm.Normal("item_offset", mu=0.0, sigma=1.0, dims="item")
        item_effect = pm.Deterministic("item_effect", item_sigma * item_offset, dims="item")

        kc_intercept_sigma = pm.HalfNormal("kc_intercept_sigma", sigma=1.0)
        kc_intercept_offset = pm.Normal("kc_intercept_offset", mu=0.0, sigma=1.0, dims="kc")
        kc_intercept = pm.Deterministic("kc_intercept", kc_intercept_sigma * kc_intercept_offset, dims="kc")

        kc_success_sigma = pm.HalfNormal("kc_success_sigma", sigma=0.5)
        kc_success_offset = pm.Normal("kc_success_offset", mu=0.0, sigma=1.0, dims="kc")
        kc_success = pm.Deterministic("kc_success", kc_success_sigma * kc_success_offset, dims="kc")

        kc_failure_sigma = pm.HalfNormal("kc_failure_sigma", sigma=0.5)
        kc_failure_offset = pm.Normal("kc_failure_offset", mu=0.0, sigma=1.0, dims="kc")
        kc_failure = pm.Deterministic("kc_failure", kc_failure_sigma * kc_failure_offset, dims="kc")

        state_sigma_global = pm.HalfNormal("state_sigma_global", sigma=0.5)
        state_sigma_student = pm.HalfNormal("state_sigma_student", sigma=state_sigma_global, dims="student")
        rho = pm.Beta("rho", alpha=2.0, beta=2.0)

        initial_state_raw = pm.Normal("state_initial_raw", mu=0.0, sigma=1.0, dims="student")
        initial_state = initial_state_raw * state_sigma_student / pt.sqrt(1.0 - rho**2 + 1e-6)

        if dataset.n_state_steps > 1:
            innovation_raw = pm.Normal(
                "state_innovation_raw",
                mu=0.0,
                sigma=1.0,
                dims=("state_step_rest", "student"),
            )

            def ar_step(innovation_t, prev_state, rho_value, sigma_student):
                return rho_value * prev_state + innovation_t * sigma_student

            state_rest, _ = scan(
                fn=ar_step,
                sequences=[innovation_raw],
                outputs_info=[initial_state],
                non_sequences=[rho, state_sigma_student],
                strict=True,
            )
            latent_state = pt.concatenate([initial_state[None, :], state_rest], axis=0)
        else:
            latent_state = initial_state[None, :]

        latent_state = pm.Deterministic("latent_state", latent_state, dims=("state_step", "student"))

        linear = (
            intercept
            + student_intercept[student_idx]
            + item_effect[item_idx]
            + pt.dot(x_kc_base, kc_intercept)
            + pt.dot(x_kc_success, kc_success)
            + pt.dot(x_kc_failure, kc_failure)
            + student_slope[student_idx] * practice_total
            + latent_state[state_step_idx, student_idx]
        )

        pm.Bernoulli("correct_obs", logit_p=linear, observed=dataset.correct)

    return model


def save_posterior_npz_pfa(
    idata: az.InferenceData,
    output_path: Path,
    context: QMatrixContext,
    *,
    model_kind: str,
    state_bin_width: int | None = None,
    history_mode: str = "pfa",
    decay_alpha: float = 1.0,
) -> None:
    ensure_parent(output_path)
    posterior = idata.posterior

    payload: dict[str, np.ndarray] = {
        "student_levels": np.asarray(context.student_levels, dtype=object),
        "item_levels": np.asarray(context.item_levels, dtype=object),
        "kc_levels": np.asarray(context.kc_levels, dtype=object),
        "history_mode": np.asarray([history_mode], dtype=object),
        "decay_alpha": np.asarray([decay_alpha], dtype="float64"),
        "Intercept": to_numpy_draws(posterior["Intercept"].values, 2),
        "student_intercept_sigma": to_numpy_draws(posterior["student_intercept_sigma"].values, 2),
        "item_sigma": to_numpy_draws(posterior["item_sigma"].values, 2),
        "kc_intercept_sigma": to_numpy_draws(posterior["kc_intercept_sigma"].values, 2),
        "kc_success_sigma": to_numpy_draws(posterior["kc_success_sigma"].values, 2),
        "kc_failure_sigma": to_numpy_draws(posterior["kc_failure_sigma"].values, 2),
        "student_intercept": to_numpy_draws(posterior["student_intercept"].values, 3),
        "item_effect": to_numpy_draws(posterior["item_effect"].values, 3),
        "kc_intercept": to_numpy_draws(posterior["kc_intercept"].values, 3),
        "kc_success": to_numpy_draws(posterior["kc_success"].values, 3),
        "kc_failure": to_numpy_draws(posterior["kc_failure"].values, 3),
    }

    if model_kind in {"model2", "model3"}:
        payload["student_slope_sigma"] = to_numpy_draws(posterior["student_slope_sigma"].values, 2)
        payload["student_slope"] = to_numpy_draws(posterior["student_slope"].values, 3)

    if model_kind == "model3":
        payload["state_sigma_global"] = to_numpy_draws(posterior["state_sigma_global"].values, 2)
        payload["state_sigma_student"] = to_numpy_draws(posterior["state_sigma_student"].values, 3)
        payload["rho"] = to_numpy_draws(posterior["rho"].values, 2)
        payload["latent_state"] = to_numpy_draws(posterior["latent_state"].values, 4)
        if state_bin_width is None:
            raise ValueError("state_bin_width is required when saving model3 posterior draws.")
        payload["state_bin_width"] = np.asarray([state_bin_width], dtype="int64")

    np.savez_compressed(output_path, **payload)
