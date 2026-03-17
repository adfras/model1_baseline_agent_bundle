from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG_PATH = Path("config/phase2_local_split_template.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic student-wise Phase 2 local splits.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    processed_trials_path = Path(config["processed_trials_path"])
    student_assignments_path = Path(config["student_assignments_path"])
    split_trials_path = Path(config["split_trials_path"])
    summary_path = Path(config["summary_path"])

    train_fraction = float(config["train_fraction"])
    calibration_fraction = float(config["calibration_fraction"])
    test_fraction = float(config["test_fraction"])
    min_history = int(config["min_history"])

    total_fraction = train_fraction + calibration_fraction + test_fraction
    if not math.isclose(total_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train_fraction + calibration_fraction + test_fraction must equal 1.")

    trials = pd.read_csv(processed_trials_path)
    student_counts = trials.groupby("student_id")["attempt_id"].size().rename("row_count").reset_index()
    eligible_students = student_counts.loc[student_counts["row_count"] >= min_history].copy()
    eligible_students["student_id_string"] = eligible_students["student_id"].astype("string")
    eligible_students = eligible_students.sort_values("student_id_string", kind="mergesort").reset_index(drop=True)

    n_students = len(eligible_students)
    if n_students < 3:
        raise ValueError("Need at least three eligible students for a train/calibration/test split.")

    n_train = max(1, int(math.floor(n_students * train_fraction)))
    n_calibration = max(1, int(math.floor(n_students * calibration_fraction)))
    if n_train + n_calibration >= n_students:
        n_calibration = max(1, n_students - n_train - 1)
    n_test = n_students - n_train - n_calibration
    if n_test < 1:
        raise ValueError("Not enough eligible students to allocate a non-empty test split.")

    split_labels = ["train"] * n_train + ["calibration"] * n_calibration + ["test"] * n_test
    assignments = eligible_students[["student_id", "row_count"]].copy()
    assignments["split"] = split_labels

    split_trials = trials.merge(assignments[["student_id", "split"]], on="student_id", how="inner")
    seen_fit_items = set(
        split_trials.loc[split_trials["split"].isin(["train", "calibration"]), "item_id"].astype("string").tolist()
    )
    split_trials["item_seen_in_fit"] = split_trials["item_id"].astype("string").isin(seen_fit_items).astype("int64")
    split_trials["new_item_in_test"] = (
        (split_trials["split"] == "test") & (split_trials["item_seen_in_fit"] == 0)
    ).astype("int64")

    ensure_parent(student_assignments_path)
    assignments.to_csv(student_assignments_path, index=False)
    ensure_parent(split_trials_path)
    split_trials.to_csv(split_trials_path, index=False)

    summary = {
        "config": {
            "processed_trials_path": str(processed_trials_path),
            "student_assignments_path": str(student_assignments_path),
            "split_trials_path": str(split_trials_path),
            "summary_path": str(summary_path),
            "train_fraction": train_fraction,
            "calibration_fraction": calibration_fraction,
            "test_fraction": test_fraction,
            "min_history": min_history,
        },
        "counts": {
            "eligible_students": int(n_students),
            "train_students": int((assignments["split"] == "train").sum()),
            "calibration_students": int((assignments["split"] == "calibration").sum()),
            "test_students": int((assignments["split"] == "test").sum()),
            "train_rows": int((split_trials["split"] == "train").sum()),
            "calibration_rows": int((split_trials["split"] == "calibration").sum()),
            "test_rows": int((split_trials["split"] == "test").sum()),
            "new_item_test_rows": int(split_trials["new_item_in_test"].sum()),
        },
    }

    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote Phase 2 student assignments to {student_assignments_path}")
    print(f"Eligible students: {n_students}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
