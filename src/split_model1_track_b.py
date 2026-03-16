from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from model1_common import ensure_parent, load_json, load_trials


DEFAULT_CONFIG_PATH = Path("config/model1_track_b_split.json")


@dataclass
class Config:
    base_trials_path: Path
    processed_trials_path: Path
    student_assignments_path: Path
    split_assignments_path: Path
    summary_path: Path
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    hash_salt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic student-wise Track B split for Model 1."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def load_config(path: Path) -> Config:
    raw = load_json(path)
    cfg = Config(
        base_trials_path=Path(raw["base_trials_path"]),
        processed_trials_path=Path(raw["processed_trials_path"]),
        student_assignments_path=Path(raw["student_assignments_path"]),
        split_assignments_path=Path(raw["split_assignments_path"]),
        summary_path=Path(raw["summary_path"]),
        train_fraction=float(raw["train_fraction"]),
        validation_fraction=float(raw["validation_fraction"]),
        test_fraction=float(raw["test_fraction"]),
        hash_salt=str(raw.get("hash_salt", "model1-track-b")),
    )
    total = cfg.train_fraction + cfg.validation_fraction + cfg.test_fraction
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Track B fractions must sum to 1.")
    for value in [cfg.train_fraction, cfg.validation_fraction, cfg.test_fraction]:
        if value <= 0.0 or value >= 1.0:
            raise ValueError("Each Track B fraction must be between 0 and 1.")
    return cfg


def split_key(student_id: str, salt: str) -> str:
    payload = f"{salt}:{student_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_student_splits(students: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    students = students.copy()
    students["split_key"] = students["student_id"].map(lambda value: split_key(value, cfg.hash_salt))
    students = students.sort_values(["split_key", "student_id"], kind="mergesort").reset_index(drop=True)
    students["split_rank"] = students.index + 1

    n_students = len(students)
    train_count = int(math.floor(n_students * cfg.train_fraction))
    validation_count = int(math.floor(n_students * cfg.validation_fraction))
    train_count = max(1, min(train_count, n_students - 2))
    validation_count = max(1, min(validation_count, n_students - train_count - 1))
    test_count = n_students - train_count - validation_count
    if test_count < 1:
        raise ValueError("Track B split left no students for test.")

    split_labels = (
        ["train"] * train_count
        + ["validation"] * validation_count
        + ["test"] * test_count
    )
    students["split"] = pd.Series(split_labels, dtype="string")
    return students


def summarize_split(track_b: pd.DataFrame, students: pd.DataFrame, cfg: Config) -> dict:
    summary: dict[str, object] = {
        "config": {
            "base_trials_path": str(cfg.base_trials_path),
            "processed_trials_path": str(cfg.processed_trials_path),
            "student_assignments_path": str(cfg.student_assignments_path),
            "split_assignments_path": str(cfg.split_assignments_path),
            "summary_path": str(cfg.summary_path),
            "train_fraction": cfg.train_fraction,
            "validation_fraction": cfg.validation_fraction,
            "test_fraction": cfg.test_fraction,
            "hash_salt": cfg.hash_salt,
        },
        "students": {
            "total_students": int(len(students)),
        },
        "rows": {
            "total_rows": int(len(track_b)),
            "training_items": int(track_b.loc[track_b["split"] == "train", "item_id"].nunique()),
        },
        "eval_item_overlap": {},
    }

    for split_name in ["train", "validation", "test"]:
        student_mask = students["split"] == split_name
        row_mask = track_b["split"] == split_name
        summary["students"][f"{split_name}_students"] = int(student_mask.sum())
        summary["rows"][f"{split_name}_rows"] = int(row_mask.sum())

    for split_name in ["validation", "test"]:
        row_mask = track_b["split"] == split_name
        eligible = int(track_b.loc[row_mask, "primary_eval_eligible"].sum())
        unseen = int(track_b.loc[row_mask, "new_item_in_test"].sum())
        total_rows = int(row_mask.sum())
        summary["eval_item_overlap"][split_name] = {
            "primary_eval_rows": eligible,
            "new_item_rows": unseen,
            "primary_eval_share": (eligible / total_rows) if total_rows else 0.0,
            "new_item_share": (unseen / total_rows) if total_rows else 0.0,
        }

    return summary


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def format_timestamp(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    base_trials = load_trials(cfg.base_trials_path).copy()
    students = (
        base_trials.groupby("student_id", sort=True)
        .agg(
            student_total_attempts=("attempt_id", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )
    students = assign_student_splits(students, cfg)

    split_map = dict(zip(students["student_id"], students["split"]))
    track_b = base_trials.copy()
    track_b["split"] = track_b["student_id"].map(split_map).astype("string")

    training_items = set(track_b.loc[track_b["split"] == "train", "item_id"].tolist())
    item_seen = track_b["item_id"].isin(training_items)
    eval_rows = track_b["split"].isin(["validation", "test"])
    track_b["item_seen_in_train"] = item_seen.astype("int64")
    track_b["new_item_in_test"] = (eval_rows & ~item_seen).astype("int64")
    track_b["primary_eval_eligible"] = (eval_rows & item_seen).astype("int64")

    student_assignments = students[
        [
            "student_id",
            "split",
            "split_rank",
            "student_total_attempts",
            "first_timestamp",
            "last_timestamp",
        ]
    ].copy()
    student_assignments["first_timestamp"] = student_assignments["first_timestamp"].map(format_timestamp)
    student_assignments["last_timestamp"] = student_assignments["last_timestamp"].map(format_timestamp)

    split_assignments = track_b[
        [
            "attempt_id",
            "student_id",
            "item_id",
            "split",
            "trial_index_within_student",
            "overall_opportunity",
            "practice_feature",
            "item_seen_in_train",
            "new_item_in_test",
            "primary_eval_eligible",
        ]
    ].copy()

    ensure_parent(cfg.processed_trials_path)
    track_b.to_csv(cfg.processed_trials_path, index=False)
    ensure_parent(cfg.student_assignments_path)
    student_assignments.to_csv(cfg.student_assignments_path, index=False)
    ensure_parent(cfg.split_assignments_path)
    split_assignments.to_csv(cfg.split_assignments_path, index=False)

    summary = summarize_split(track_b, students, cfg)
    write_json(cfg.summary_path, summary)

    print(f"Saved Track B trials to {cfg.processed_trials_path}")
    print(f"Saved Track B student assignments to {cfg.student_assignments_path}")
    print(f"Saved Track B row assignments to {cfg.split_assignments_path}")
    print(f"Saved Track B summary to {cfg.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
