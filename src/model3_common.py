from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan.basic import scan


MODEL3_DESCRIPTION = (
    "correct ~ practice_feature + new-student random effects + item effect + "
    "latent AR(1) state over attempt bins"
)


@dataclass
class TrackBTrainingData:
    train_df: pd.DataFrame
    student_levels: list[str]
    item_levels: list[str]
    student_idx: np.ndarray
    item_idx: np.ndarray
    state_bin_idx: np.ndarray
    correct: np.ndarray
    practice_feature: np.ndarray
    n_students: int
    n_items: int
    n_state_steps: int
    state_bin_width: int


def prepare_track_b_training_data(trials: pd.DataFrame, state_bin_width: int) -> TrackBTrainingData:
    if state_bin_width < 1:
        raise ValueError("state_bin_width must be at least 1.")

    train_df = trials.loc[trials["split"] == "train"].copy()
    train_df = train_df.sort_values(
        ["student_id", "trial_index_within_student", "attempt_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    train_df["state_bin"] = (train_df["overall_opportunity"] // state_bin_width).astype("int64")

    student_levels = sorted(train_df["student_id"].astype("string").unique().tolist())
    item_levels = sorted(train_df["item_id"].astype("string").unique().tolist())

    student_lookup = {student_id: index for index, student_id in enumerate(student_levels)}
    item_lookup = {item_id: index for index, item_id in enumerate(item_levels)}

    student_idx = train_df["student_id"].map(student_lookup).to_numpy(dtype="int64")
    item_idx = train_df["item_id"].map(item_lookup).to_numpy(dtype="int64")
    state_bin_idx = train_df["state_bin"].to_numpy(dtype="int64")

    return TrackBTrainingData(
        train_df=train_df,
        student_levels=student_levels,
        item_levels=item_levels,
        student_idx=student_idx,
        item_idx=item_idx,
        state_bin_idx=state_bin_idx,
        correct=train_df["correct"].to_numpy(dtype="int8"),
        practice_feature=train_df["practice_feature"].to_numpy(dtype="float64"),
        n_students=len(student_levels),
        n_items=len(item_levels),
        n_state_steps=int(state_bin_idx.max()) + 1,
        state_bin_width=state_bin_width,
    )


def build_track_b_model(data: TrackBTrainingData) -> pm.Model:
    coords: dict[str, list[object]] = {
        "student": data.student_levels,
        "item": data.item_levels,
        "state_step": list(range(data.n_state_steps)),
    }
    if data.n_state_steps > 1:
        coords["state_step_rest"] = list(range(1, data.n_state_steps))

    with pm.Model(coords=coords) as model:
        student_idx = pm.Data("student_idx", data.student_idx)
        item_idx = pm.Data("item_idx", data.item_idx)
        state_step_idx = pm.Data("state_step_idx", data.state_bin_idx)
        practice_feature = pm.Data("practice_feature_data", data.practice_feature)

        intercept = pm.Normal("Intercept", mu=0.0, sigma=1.5)
        practice_beta = pm.Normal("practice_feature", mu=0.0, sigma=1.0)

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

        state_sigma_global = pm.HalfNormal("state_sigma_global", sigma=0.5)
        state_sigma_student = pm.HalfNormal(
            "state_sigma_student",
            sigma=state_sigma_global,
            dims="student",
        )
        rho = pm.Beta("rho", alpha=2.0, beta=2.0)

        initial_state_raw = pm.Normal("state_initial_raw", mu=0.0, sigma=1.0, dims="student")
        initial_state = initial_state_raw * state_sigma_student / pt.sqrt(1.0 - rho**2 + 1e-6)

        if data.n_state_steps > 1:
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
            + (practice_beta + student_slope[student_idx]) * practice_feature
            + latent_state[state_step_idx, student_idx]
        )

        pm.Bernoulli("correct_obs", logit_p=linear, observed=data.correct)

    return model

