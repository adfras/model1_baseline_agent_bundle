from __future__ import annotations

import json
import os
from pathlib import Path

import bambi as bmb
import pandas as pd


MODEL1_FORMULA = "correct ~ practice_feature + (1|student_id) + (1|item_id)"


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
    df["student_id"] = df["student_id"].astype("string")
    df["item_id"] = df["item_id"].astype("string")
    df["split"] = df["split"].astype("string")
    df["correct"] = df["correct"].astype("int8")
    df["practice_feature"] = df["practice_feature"].astype("float64")
    for column in [
        "item_seen_in_train",
        "new_item_in_test",
        "primary_eval_eligible",
        "train_rows_for_student",
        "test_rows_for_student",
        "student_total_attempts",
        "trial_index_within_student",
        "overall_opportunity",
    ]:
        if column in df.columns:
            df[column] = df[column].astype("int64")
    return df


def build_model(df: pd.DataFrame) -> bmb.Model:
    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1.5),
        "practice_feature": bmb.Prior("Normal", mu=0, sigma=1.0),
        "1|student_id": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1.0)),
        "1|item_id": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1.0)),
    }
    return bmb.Model(
        MODEL1_FORMULA,
        df,
        family="bernoulli",
        priors=priors,
        auto_scale=False,
    )
