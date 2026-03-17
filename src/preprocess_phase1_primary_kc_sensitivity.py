from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFIG_PATH = Path("config/phase1_primary_kc_preprocess.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full-item primary-KC sensitivity table for Phase 1."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_bool_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": True, "false": False})
    if mapped.isna().any():
        unexpected = sorted(normalized[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"Unsupported boolean literals: {unexpected[:5]}")
    return mapped.astype(bool)


def parse_optional_bool_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": 1, "false": 0})
    return mapped.astype("Int64")


def build_schema_note(path: Path, summary: dict) -> None:
    ensure_parent(path)
    counts = summary["counts"]
    payload = f"""# Phase 1 Primary-KC Sensitivity Schema Note

This note records the full-item sensitivity table for Phase 1.

## Assignment rule

- All visible items are retained.
- Each item is assigned one deterministic `primary_kc_id`.
- The assignment rule is: use the KC from the earliest relationship row in `Question_KC_Relationships.csv` for that item.

This is a sensitivity analysis, not the primary discovery dataset.

## Main analysis columns

- `student_id`
- `item_id`
- `kc_id`
- `correct`
- `timestamp`
- `attempt_id`
- `trial_index_within_student`
- `overall_opportunity`
- `kc_opportunity`
- `kc_practice_feature`

Implementation note:

- `practice_feature` is retained as a compatibility alias and is set equal to `kc_practice_feature`.

## Sensitivity sample summary

- Visible rows before KC assignment: `{counts["rows_after_hidden_exclusion"]}`
- Assigned rows after primary-KC mapping: `{counts["assigned_rows"]}`
- Eligible learners: `{counts["eligible_students"]}`
- Assigned items: `{counts["assigned_items"]}`
- Assigned KCs: `{counts["assigned_kcs"]}`
- Questions with no KC link: `{counts["zero_kc_questions"]}`

## Validation checks

- KC opportunity monotone within student-KC: `{summary["validation"]["kc_opportunity_monotone"]}`
- Chronology violations after sorting: `{summary["validation"]["chronology_violations"]}`
"""
    path.write_text(payload, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    transactions_path = Path(config["transactions_path"])
    questions_path = Path(config["questions_path"])
    question_kc_path = Path(config["question_kc_relationships_path"])
    kcs_path = Path(config["kcs_path"])
    processed_trials_path = Path(config["processed_trials_path"])
    split_assignments_path = Path(config["split_assignments_path"])
    summary_path = Path(config["summary_path"])
    schema_note_path = Path(config["schema_note_path"])

    train_fraction = float(config["train_fraction"])
    min_history = int(config["min_history"])
    exclude_hidden = bool(config.get("exclude_hidden", True))

    transactions = pd.read_csv(transactions_path, encoding="utf-8-sig")
    transactions["start_time"] = pd.to_datetime(transactions["start_time"], utc=True, errors="raise")
    transactions["end_time"] = pd.to_datetime(transactions["end_time"], utc=True, errors="raise")
    questions = pd.read_csv(questions_path, encoding="utf-8-sig")
    question_kc = pd.read_csv(question_kc_path, encoding="utf-8-sig")
    kcs = pd.read_csv(kcs_path, encoding="utf-8-sig")

    raw_rows = int(len(transactions))
    raw_students = int(transactions["student_id"].nunique())
    raw_questions = int(transactions["question_id"].nunique())

    transactions["is_hidden_bool"] = parse_bool_series(transactions["is_hidden"])
    transactions["correct"] = parse_bool_series(transactions["answer_state"]).astype("int8")
    transactions["hint_used_clean"] = parse_optional_bool_series(transactions["hint_used"])
    visible = transactions.loc[~transactions["is_hidden_bool"]].copy() if exclude_hidden else transactions.copy()
    hidden_rows_excluded = raw_rows - len(visible)

    visible["duration_seconds"] = (visible["end_time"] - visible["start_time"]).dt.total_seconds()
    visible["timestamp"] = visible["start_time"]
    visible["attempt_id"] = visible["id"].astype("int64")
    visible["student_id"] = visible["student_id"].astype("int64")
    visible["item_id"] = visible["question_id"].astype("int64")

    question_kc = question_kc.sort_values(["question_id", "id"], kind="mergesort").copy()
    primary_kc_lookup = (
        question_kc.drop_duplicates(subset=["question_id"], keep="first")
        .rename(
            columns={
                "knowledgecomponent_id": "kc_id",
                "id": "primary_kc_relationship_id",
            }
        )[["question_id", "kc_id", "primary_kc_relationship_id"]]
        .astype({"question_id": "int64", "kc_id": "int64", "primary_kc_relationship_id": "int64"})
    )

    zero_kc_questions = int(raw_questions - primary_kc_lookup["question_id"].nunique())
    kc_lookup = kcs.rename(columns={"id": "kc_id", "name": "kc_name"})
    primary_kc_lookup = primary_kc_lookup.merge(kc_lookup[["kc_id", "kc_name"]], on="kc_id", how="left")

    sensitivity = visible.merge(
        primary_kc_lookup[["question_id", "kc_id", "kc_name", "primary_kc_relationship_id"]],
        left_on="item_id",
        right_on="question_id",
        how="inner",
    )

    rows_after_hidden_exclusion = int(len(visible))
    assigned_rows = int(len(sensitivity))

    question_meta = questions.rename(columns={"id": "item_id", "difficulty": "question_difficulty"})[
        ["item_id", "question_difficulty"]
    ].copy()
    sensitivity = sensitivity.merge(question_meta, on="item_id", how="left")
    sensitivity["question_difficulty"] = pd.to_numeric(
        sensitivity["question_difficulty"], errors="coerce"
    ).astype("Int64")
    sensitivity["trust_feedback"] = pd.to_numeric(sensitivity["trust_feedback"], errors="coerce")
    sensitivity["difficulty_feedback"] = pd.to_numeric(
        sensitivity["difficulty_feedback"], errors="coerce"
    )
    sensitivity["selection_change"] = pd.to_numeric(
        sensitivity["selection_change"], errors="coerce"
    ).astype("Int64")

    sensitivity = sensitivity.sort_values(
        ["student_id", "timestamp", "attempt_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    student_counts = sensitivity.groupby("student_id")["attempt_id"].transform("size")
    eligible = sensitivity.loc[student_counts >= min_history].copy()
    excluded_students_min_history = int(sensitivity["student_id"].nunique() - eligible["student_id"].nunique())

    eligible["trial_index_within_student"] = eligible.groupby("student_id").cumcount() + 1
    eligible["overall_opportunity"] = eligible["trial_index_within_student"] - 1
    eligible["kc_opportunity"] = eligible.groupby(["student_id", "kc_id"]).cumcount()
    eligible["kc_practice_feature"] = np.log1p(eligible["kc_opportunity"].to_numpy(dtype="float64"))
    eligible["practice_feature"] = eligible["kc_practice_feature"]
    eligible["student_total_attempts"] = eligible.groupby("student_id")["attempt_id"].transform("size")

    train_rows_for_student = np.floor(eligible["student_total_attempts"] * train_fraction).astype("int64")
    train_rows_for_student = train_rows_for_student.clip(lower=1)
    train_rows_for_student = np.minimum(
        train_rows_for_student.to_numpy(dtype="int64"),
        eligible["student_total_attempts"].to_numpy(dtype="int64") - 1,
    )
    eligible["train_rows_for_student"] = train_rows_for_student
    eligible["test_rows_for_student"] = (
        eligible["student_total_attempts"] - eligible["train_rows_for_student"]
    ).astype("int64")
    eligible["split"] = np.where(
        eligible["trial_index_within_student"] <= eligible["train_rows_for_student"],
        "train",
        "test",
    )

    training_items = set(eligible.loc[eligible["split"] == "train", "item_id"].astype(int).tolist())
    eligible["item_seen_in_train"] = eligible["item_id"].isin(training_items).astype("int64")
    eligible["new_item_in_test"] = (
        (eligible["split"] == "test") & (~eligible["item_id"].isin(training_items))
    ).astype("int64")
    eligible["primary_eval_eligible"] = (
        (eligible["split"] == "test") & eligible["item_id"].isin(training_items)
    ).astype("int64")

    chronology_violations = int(
        eligible.groupby("student_id")["timestamp"]
        .apply(lambda s: int((s.diff().dropna() < pd.Timedelta(0)).sum()))
        .sum()
    )
    kc_monotone = bool(
        eligible.groupby(["student_id", "kc_id"])["kc_opportunity"]
        .apply(lambda s: bool((s.diff().fillna(1) >= 0).all()))
        .all()
    )

    processed_columns = [
        "attempt_id",
        "student_id",
        "item_id",
        "kc_id",
        "kc_name",
        "primary_kc_relationship_id",
        "correct",
        "timestamp",
        "question_difficulty",
        "trial_index_within_student",
        "overall_opportunity",
        "kc_opportunity",
        "practice_feature",
        "kc_practice_feature",
        "student_total_attempts",
        "split",
        "train_rows_for_student",
        "test_rows_for_student",
        "item_seen_in_train",
        "new_item_in_test",
        "primary_eval_eligible",
        "difficulty_feedback",
        "trust_feedback",
        "hint_used_clean",
        "duration_seconds",
        "selection_change",
    ]
    processed = eligible.loc[:, processed_columns].copy()
    processed = processed.rename(columns={"hint_used_clean": "hint_used"})
    processed["timestamp"] = processed["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    split_columns = [
        "attempt_id",
        "student_id",
        "item_id",
        "kc_id",
        "split",
        "trial_index_within_student",
        "student_total_attempts",
        "train_rows_for_student",
        "test_rows_for_student",
        "item_seen_in_train",
        "new_item_in_test",
        "primary_eval_eligible",
    ]
    split_assignments = processed.loc[:, split_columns].copy()

    ensure_parent(processed_trials_path)
    processed.to_csv(processed_trials_path, index=False)
    ensure_parent(split_assignments_path)
    split_assignments.to_csv(split_assignments_path, index=False)

    summary = {
        "config": {
            "transactions_path": str(transactions_path),
            "questions_path": str(questions_path),
            "question_kc_relationships_path": str(question_kc_path),
            "kcs_path": str(kcs_path),
            "processed_trials_path": str(processed_trials_path),
            "split_assignments_path": str(split_assignments_path),
            "summary_path": str(summary_path),
            "schema_note_path": str(schema_note_path),
            "train_fraction": train_fraction,
            "min_history": min_history,
            "exclude_hidden": exclude_hidden,
            "assignment_strategy": "earliest_question_kc_relationship_id",
        },
        "counts": {
            "raw_rows": raw_rows,
            "raw_students": raw_students,
            "raw_questions": raw_questions,
            "hidden_rows_excluded": int(hidden_rows_excluded),
            "rows_after_hidden_exclusion": rows_after_hidden_exclusion,
            "assigned_rows": assigned_rows,
            "zero_kc_questions": int(zero_kc_questions),
            "eligible_students": int(processed["student_id"].nunique()),
            "excluded_students_min_history": excluded_students_min_history,
            "assigned_items": int(processed["item_id"].nunique()),
            "assigned_kcs": int(processed["kc_id"].nunique()),
            "processed_rows": int(len(processed)),
            "train_rows": int((processed["split"] == "train").sum()),
            "test_rows": int((processed["split"] == "test").sum()),
            "primary_eval_rows": int(processed["primary_eval_eligible"].sum()),
            "new_item_test_rows": int(processed["new_item_in_test"].sum()),
        },
        "validation": {
            "kc_opportunity_monotone": kc_monotone,
            "chronology_violations": chronology_violations,
            "practice_feature_alias_matches_kc_feature": bool(
                np.allclose(
                    processed["practice_feature"].to_numpy(dtype="float64"),
                    processed["kc_practice_feature"].to_numpy(dtype="float64"),
                )
            ),
        },
    }

    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    build_schema_note(schema_note_path, summary)

    print(f"Wrote primary-KC sensitivity table to {processed_trials_path}")
    print(f"Assigned rows after KC mapping: {assigned_rows}")
    print(f"Eligible students after min-history filter: {summary['counts']['eligible_students']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
