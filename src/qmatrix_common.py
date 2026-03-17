from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


def prepend_compiler_to_path(compiler_bin_dir: str | None) -> None:
    if not compiler_bin_dir:
        return
    compiler_path = Path(compiler_bin_dir)
    if not compiler_path.exists():
        return
    current = os.environ.get("PATH", "")
    compiler = str(compiler_path)
    if current.startswith(compiler + os.pathsep):
        return
    os.environ["PATH"] = compiler + os.pathsep + current


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_trials(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["student_id", "timestamp", "attempt_id"], kind="mergesort").reset_index(drop=True)
    df["student_id"] = df["student_id"].astype("string")
    df["item_id"] = df["item_id"].astype("string")
    df["split"] = df["split"].astype("string")
    df["correct"] = df["correct"].astype("int8")
    df["practice_feature"] = df["practice_feature"].astype("float64")
    df["trial_index_within_student"] = df["trial_index_within_student"].astype("int64")
    df["attempt_id"] = df["attempt_id"].astype("int64")
    return df


def load_attempt_kc_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["attempt_id"] = df["attempt_id"].astype("int64")
    df["student_id"] = df["student_id"].astype("string")
    df["item_id"] = df["item_id"].astype("string")
    df["kc_id"] = df["kc_id"].astype("string")
    df["kc_practice_component"] = df["kc_practice_component"].astype("float32")
    return df


@dataclass
class QMatrixContext:
    student_levels: list[str]
    item_levels: list[str]
    kc_levels: list[str]
    student_lookup: dict[str, int]
    item_lookup: dict[str, int]
    kc_lookup: dict[str, int]


@dataclass
class QMatrixDataset:
    df: pd.DataFrame
    student_idx: np.ndarray
    item_idx: np.ndarray
    x_kc_base: np.ndarray
    x_kc_practice: np.ndarray
    practice_total: np.ndarray
    correct: np.ndarray


def build_context(train_df: pd.DataFrame, attempt_kc_long: pd.DataFrame) -> QMatrixContext:
    train_attempt_ids = set(train_df["attempt_id"].astype(int).tolist())
    train_long = attempt_kc_long.loc[attempt_kc_long["attempt_id"].isin(train_attempt_ids)].copy()

    student_levels = sorted(train_df["student_id"].astype("string").unique().tolist())
    item_levels = sorted(train_df["item_id"].astype("string").unique().tolist())
    kc_levels = sorted(train_long["kc_id"].astype("string").unique().tolist())

    return QMatrixContext(
        student_levels=student_levels,
        item_levels=item_levels,
        kc_levels=kc_levels,
        student_lookup={value: index for index, value in enumerate(student_levels)},
        item_lookup={value: index for index, value in enumerate(item_levels)},
        kc_lookup={value: index for index, value in enumerate(kc_levels)},
    )


def build_design_matrices(
    df: pd.DataFrame,
    attempt_kc_long: pd.DataFrame,
    context: QMatrixContext,
) -> tuple[np.ndarray, np.ndarray]:
    attempt_order = pd.Series(np.arange(len(df), dtype="int64"), index=df["attempt_id"].to_numpy(dtype="int64"))
    filtered = attempt_kc_long.loc[
        attempt_kc_long["attempt_id"].isin(attempt_order.index)
        & attempt_kc_long["kc_id"].isin(context.kc_lookup.keys())
    ].copy()

    row_idx = filtered["attempt_id"].map(attempt_order).to_numpy(dtype="int64")
    col_idx = filtered["kc_id"].map(context.kc_lookup).to_numpy(dtype="int64")

    x_kc_base = np.zeros((len(df), len(context.kc_levels)), dtype=np.float32)
    x_kc_practice = np.zeros((len(df), len(context.kc_levels)), dtype=np.float32)

    np.add.at(x_kc_base, (row_idx, col_idx), 1.0)
    np.add.at(
        x_kc_practice,
        (row_idx, col_idx),
        filtered["kc_practice_component"].to_numpy(dtype=np.float32),
    )
    return x_kc_base, x_kc_practice


def prepare_dataset(
    trials: pd.DataFrame,
    attempt_kc_long: pd.DataFrame,
    context: QMatrixContext,
    *,
    split: str,
    primary_eval_only: bool = False,
) -> QMatrixDataset:
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

    x_kc_base, x_kc_practice = build_design_matrices(df, attempt_kc_long, context)
    return QMatrixDataset(
        df=df,
        student_idx=student_idx.to_numpy(dtype="int64"),
        item_idx=item_idx.to_numpy(dtype="int64"),
        x_kc_base=x_kc_base,
        x_kc_practice=x_kc_practice,
        practice_total=x_kc_practice.sum(axis=1).astype("float32"),
        correct=df["correct"].to_numpy(dtype="int8"),
    )


def build_model1_qmatrix(dataset: QMatrixDataset, context: QMatrixContext) -> pm.Model:
    coords = {
        "student": context.student_levels,
        "item": context.item_levels,
        "kc": context.kc_levels,
    }

    with pm.Model(coords=coords) as model:
        student_idx = pm.Data("student_idx", dataset.student_idx)
        item_idx = pm.Data("item_idx", dataset.item_idx)
        x_kc_base = pm.Data("x_kc_base", dataset.x_kc_base)
        x_kc_practice = pm.Data("x_kc_practice", dataset.x_kc_practice)

        intercept = pm.Normal("Intercept", mu=0.0, sigma=1.5)

        student_intercept_sigma = pm.HalfNormal("student_intercept_sigma", sigma=1.0)
        student_intercept_offset = pm.Normal("student_intercept_offset", mu=0.0, sigma=1.0, dims="student")
        student_intercept = pm.Deterministic(
            "student_intercept",
            student_intercept_sigma * student_intercept_offset,
            dims="student",
        )

        item_sigma = pm.HalfNormal("item_sigma", sigma=1.0)
        item_offset = pm.Normal("item_offset", mu=0.0, sigma=1.0, dims="item")
        item_effect = pm.Deterministic("item_effect", item_sigma * item_offset, dims="item")

        kc_intercept_sigma = pm.HalfNormal("kc_intercept_sigma", sigma=1.0)
        kc_intercept_offset = pm.Normal("kc_intercept_offset", mu=0.0, sigma=1.0, dims="kc")
        kc_intercept = pm.Deterministic("kc_intercept", kc_intercept_sigma * kc_intercept_offset, dims="kc")

        kc_practice_sigma = pm.HalfNormal("kc_practice_sigma", sigma=0.5)
        kc_practice_offset = pm.Normal("kc_practice_offset", mu=0.0, sigma=1.0, dims="kc")
        kc_practice = pm.Deterministic("kc_practice", kc_practice_sigma * kc_practice_offset, dims="kc")

        linear = (
            intercept
            + student_intercept[student_idx]
            + item_effect[item_idx]
            + pt.dot(x_kc_base, kc_intercept)
            + pt.dot(x_kc_practice, kc_practice)
        )

        pm.Bernoulli("correct_obs", logit_p=linear, observed=dataset.correct)

    return model


def build_model2_qmatrix(dataset: QMatrixDataset, context: QMatrixContext) -> pm.Model:
    coords = {
        "student": context.student_levels,
        "item": context.item_levels,
        "kc": context.kc_levels,
    }

    with pm.Model(coords=coords) as model:
        student_idx = pm.Data("student_idx", dataset.student_idx)
        item_idx = pm.Data("item_idx", dataset.item_idx)
        x_kc_base = pm.Data("x_kc_base", dataset.x_kc_base)
        x_kc_practice = pm.Data("x_kc_practice", dataset.x_kc_practice)
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

        kc_practice_sigma = pm.HalfNormal("kc_practice_sigma", sigma=0.5)
        kc_practice_offset = pm.Normal("kc_practice_offset", mu=0.0, sigma=1.0, dims="kc")
        kc_practice = pm.Deterministic("kc_practice", kc_practice_sigma * kc_practice_offset, dims="kc")

        linear = (
            intercept
            + student_intercept[student_idx]
            + item_effect[item_idx]
            + pt.dot(x_kc_base, kc_intercept)
            + pt.dot(x_kc_practice, kc_practice)
            + student_slope[student_idx] * practice_total
        )

        pm.Bernoulli("correct_obs", logit_p=linear, observed=dataset.correct)

    return model


def fit_model(
    model: pm.Model,
    *,
    inference_method: str,
    vi_method: str,
    vi_iterations: int,
    posterior_draws: int,
    random_seed: int,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
) -> tuple[az.InferenceData, list[float] | None]:
    with model:
        if inference_method == "vi":
            approx = pm.fit(
                n=vi_iterations,
                method=vi_method,
                random_seed=random_seed,
                progressbar=True,
            )
            idata = approx.sample(
                posterior_draws,
                random_seed=random_seed,
                return_inferencedata=True,
            )
            return idata, [float(value) for value in approx.hist]

        if inference_method not in {"pymc", "nuts", "mcmc"}:
            raise ValueError(f"Unsupported inference method: {inference_method}")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=True,
        )
        return idata, None


def summarize_vi_history(losses: list[float] | None) -> dict[str, float | int | None]:
    if losses is None:
        return {}

    arr = np.asarray(losses, dtype="float64")
    if arr.size == 0:
        return {
            "vi_loss_initial": None,
            "vi_loss_final": None,
            "vi_loss_best": None,
            "vi_best_iteration": None,
            "vi_relative_improvement": None,
            "vi_tail_mean": None,
            "vi_tail_sd": None,
            "vi_prev_tail_mean": None,
            "vi_tail_relative_change": None,
        }

    tail = min(500, arr.size)
    tail_mean = float(arr[-tail:].mean())
    tail_sd = float(arr[-tail:].std(ddof=1)) if tail > 1 else 0.0

    prev_tail_mean = None
    tail_relative_change = None
    if arr.size >= 2 * tail:
        prev = arr[-2 * tail : -tail]
        prev_tail_mean = float(prev.mean())
        if prev_tail_mean != 0.0:
            tail_relative_change = float((tail_mean - prev_tail_mean) / abs(prev_tail_mean))

    initial = float(arr[0])
    final = float(arr[-1])
    best_index = int(arr.argmin())
    best_loss = float(arr[best_index])
    relative_improvement = None
    if initial != 0.0:
        relative_improvement = float((initial - final) / abs(initial))

    return {
        "vi_loss_initial": initial,
        "vi_loss_final": final,
        "vi_loss_best": best_loss,
        "vi_best_iteration": best_index + 1,
        "vi_relative_improvement": relative_improvement,
        "vi_tail_mean": tail_mean,
        "vi_tail_sd": tail_sd,
        "vi_prev_tail_mean": prev_tail_mean,
        "vi_tail_relative_change": tail_relative_change,
    }


def to_numpy_draws(values, expected_dims: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != expected_dims:
        raise ValueError(f"Expected {expected_dims} dims, got {arr.ndim}")
    return arr.reshape((-1,) + arr.shape[2:])


def save_posterior_npz(
    idata: az.InferenceData,
    output_path: Path,
    context: QMatrixContext,
    *,
    model_kind: str,
) -> None:
    ensure_parent(output_path)
    posterior = idata.posterior

    payload: dict[str, np.ndarray] = {
        "student_levels": np.asarray(context.student_levels, dtype=object),
        "item_levels": np.asarray(context.item_levels, dtype=object),
        "kc_levels": np.asarray(context.kc_levels, dtype=object),
        "Intercept": to_numpy_draws(posterior["Intercept"].values, 2),
        "student_intercept_sigma": to_numpy_draws(posterior["student_intercept_sigma"].values, 2),
        "item_sigma": to_numpy_draws(posterior["item_sigma"].values, 2),
        "kc_intercept_sigma": to_numpy_draws(posterior["kc_intercept_sigma"].values, 2),
        "kc_practice_sigma": to_numpy_draws(posterior["kc_practice_sigma"].values, 2),
        "student_intercept": to_numpy_draws(posterior["student_intercept"].values, 3),
        "item_effect": to_numpy_draws(posterior["item_effect"].values, 3),
        "kc_intercept": to_numpy_draws(posterior["kc_intercept"].values, 3),
        "kc_practice": to_numpy_draws(posterior["kc_practice"].values, 3),
    }

    if model_kind == "model2":
        payload["student_slope_sigma"] = to_numpy_draws(posterior["student_slope_sigma"].values, 2)
        payload["student_slope"] = to_numpy_draws(posterior["student_slope"].values, 3)

    np.savez_compressed(output_path, **payload)


def summarize_vector_draws(values: np.ndarray, labels: list[str], value_name: str) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(labels):
        draws = values[:, index]
        interval = az.hdi(draws, hdi_prob=0.94)
        rows.append(
            {
                value_name: label,
                "mean": float(np.mean(draws)),
                "sd": float(np.std(draws, ddof=1)),
                "hdi_3%": float(interval[0]),
                "hdi_97%": float(interval[1]),
            }
        )
    return pd.DataFrame(rows)
