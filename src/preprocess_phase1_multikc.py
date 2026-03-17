from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFIG_PATH = Path("config/phase1_multikc_preprocess.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full-data multi-KC DBE-KT22 discovery table for Phase 1."
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


def join_unique_strings(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna().tolist()})
    return "|".join(values)


def build_schema_note(path: Path, summary: dict) -> None:
    ensure_parent(path)
    counts = summary["counts"]
    test_profile = summary["test_profile"]
    practice = summary["practice"]
    payload = f"""# Phase 1 Multi-KC Discovery Schema Note

This note records the full-data multi-KC Phase 1 discovery table built from DBE-KT22.

## Main design

- All visible attempts are retained if their item has at least one linked KC.
- Multi-KC items are not dropped.
- A long attempt-KC table is built internally so each attempt contributes one prior-opportunity update to every linked KC.
- The model-facing attempt table stays one row per attempt.

## Practice construction

- KC update mode: `{practice["kc_update_mode"]}`
- For each student-KC pair, `kc_opportunity` tracks prior exposure before the current attempt.
- For multi-KC items, the attempt contributes to every linked KC after the response using the configured update rule.
- The main model-facing practice term is:
  - `practice_feature = kc_practice_feature_sum`
  - where `kc_practice_feature_sum = sum(log1p(kc_opportunity_k))` across the attempt's linked KCs

Current update rule summary:

- KC increment per linked KC: `{practice["increment_description"]}`
- Practice aggregation: `{practice["aggregation_description"]}`

Additional summaries retained on each attempt:

- `kc_count`
- `kc_ids`
- `kc_names`
- `kc_opportunity_sum`
- `kc_opportunity_mean`
- `kc_opportunity_max`
- `kc_practice_feature_mean`
- `kc_practice_feature_weighted`
- `any_first_kc`
- `all_first_kc`

## Discovery sample summary

- Raw rows before hidden exclusion: `{counts["raw_rows"]}`
- Visible rows after hidden exclusion: `{counts["rows_after_hidden_exclusion"]}`
- Attempt rows with at least one KC: `{counts["attempt_rows_with_kc"]}`
- Long attempt-KC rows: `{counts["attempt_kc_rows"]}`
- Eligible learners: `{counts["eligible_students"]}`
- Eligible attempt rows: `{counts["eligible_rows"]}`
- Items with at least one KC: `{counts["items_with_kc"]}`
- KCs represented: `{counts["represented_kcs"]}`
- Single-KC attempt rows: `{counts["single_kc_attempt_rows"]}`
- Multi-KC attempt rows: `{counts["multi_kc_attempt_rows"]}`
- Questions with no KC link: `{counts["zero_kc_questions"]}`
- Mean KC count per attempt: `{counts["mean_kc_count_per_attempt"]}`
- Median KC count per attempt: `{counts["median_kc_count_per_attempt"]}`

## Validation checks

- KC opportunity monotone within student-KC: `{summary["validation"]["kc_opportunity_monotone"]}`
- Chronology violations after sorting: `{summary["validation"]["chronology_violations"]}`

## Held-out profile

- Mean KC count in test rows: `{test_profile["mean_kc_count_test"]}`
- Share of test rows with any first-seen KC: `{test_profile["any_first_kc_share_test"]}`
- Share of test rows with all linked KCs first-seen: `{test_profile["all_first_kc_share_test"]}`
- Mean `kc_opportunity_mean` in test rows: `{test_profile["kc_opportunity_mean_test"]}`
- Mean `practice_feature` in test rows: `{test_profile["practice_feature_mean_test"]}`
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
    attempt_kc_long_path = Path(config["attempt_kc_long_path"])

    train_fraction = float(config["train_fraction"])
    min_history = int(config["min_history"])
    exclude_hidden = bool(config.get("exclude_hidden", True))
    kc_update_mode = str(config.get("kc_update_mode", "full_credit"))
    if kc_update_mode not in {"full_credit", "fractional_equal"}:
        raise ValueError("kc_update_mode must be one of: full_credit, fractional_equal")

    transactions = pd.read_csv(transactions_path, encoding="utf-8-sig")
    transactions["start_time"] = pd.to_datetime(transactions["start_time"], utc=True, errors="raise")
    transactions["end_time"] = pd.to_datetime(transactions["end_time"], utc=True, errors="raise")
    questions = pd.read_csv(questions_path, encoding="utf-8-sig")
    question_kc = pd.read_csv(question_kc_path, encoding="utf-8-sig")
    kcs = pd.read_csv(kcs_path, encoding="utf-8-sig")

    raw_rows = int(len(transactions))
    raw_questions = int(transactions["question_id"].nunique())

    transactions["is_hidden_bool"] = parse_bool_series(transactions["is_hidden"])
    transactions["correct"] = parse_bool_series(transactions["answer_state"]).astype("int8")
    transactions["hint_used_clean"] = parse_optional_bool_series(transactions["hint_used"])
    visible = transactions.loc[~transactions["is_hidden_bool"]].copy() if exclude_hidden else transactions.copy()

    visible["duration_seconds"] = (visible["end_time"] - visible["start_time"]).dt.total_seconds()
    visible["timestamp"] = visible["start_time"]
    visible["attempt_id"] = visible["id"].astype("int64")
    visible["student_id"] = visible["student_id"].astype("int64")
    visible["item_id"] = visible["question_id"].astype("int64")

    question_meta = questions.rename(columns={"id": "item_id", "difficulty": "question_difficulty"})[
        ["item_id", "question_difficulty"]
    ].copy()

    kc_lookup = kcs.rename(columns={"id": "kc_id", "name": "kc_name"})[["kc_id", "kc_name"]].copy()

    question_kc = question_kc.rename(
        columns={
            "id": "kc_relationship_id",
            "knowledgecomponent_id": "kc_id",
        }
    ).copy()
    question_kc["question_id"] = question_kc["question_id"].astype("int64")
    question_kc["kc_id"] = question_kc["kc_id"].astype("int64")
    question_kc["kc_relationship_id"] = question_kc["kc_relationship_id"].astype("int64")
    question_kc = question_kc.merge(kc_lookup, on="kc_id", how="left")

    kc_counts = (
        question_kc.groupby("question_id", as_index=False)
        .agg(kc_count=("kc_id", "nunique"))
        .astype({"question_id": "int64", "kc_count": "int64"})
    )
    question_kc = question_kc.merge(kc_counts, on="question_id", how="left")

    zero_kc_questions = int(raw_questions - question_kc["question_id"].nunique())

    attempt_kc = visible.merge(
        question_kc[["question_id", "kc_id", "kc_name", "kc_relationship_id", "kc_count"]],
        left_on="item_id",
        right_on="question_id",
        how="inner",
    )
    attempt_kc = attempt_kc.merge(question_meta, on="item_id", how="left")
    attempt_kc["question_difficulty"] = pd.to_numeric(
        attempt_kc["question_difficulty"], errors="coerce"
    ).astype("Int64")
    attempt_kc["trust_feedback"] = pd.to_numeric(attempt_kc["trust_feedback"], errors="coerce")
    attempt_kc["difficulty_feedback"] = pd.to_numeric(attempt_kc["difficulty_feedback"], errors="coerce")
    attempt_kc["selection_change"] = pd.to_numeric(
        attempt_kc["selection_change"], errors="coerce"
    ).astype("Int64")

    attempt_kc = attempt_kc.sort_values(
        ["student_id", "timestamp", "attempt_id", "kc_relationship_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if kc_update_mode == "full_credit":
        attempt_kc["kc_exposure_increment"] = 1.0
        attempt_kc["kc_opportunity"] = attempt_kc.groupby(["student_id", "kc_id"]).cumcount().astype("float64")
        increment_description = "1.0 per linked KC"
    else:
        attempt_kc["kc_exposure_increment"] = 1.0 / attempt_kc["kc_count"].to_numpy(dtype="float64")
        grouped_exposure = attempt_kc.groupby(["student_id", "kc_id"])["kc_exposure_increment"]
        attempt_kc["kc_opportunity"] = (
            grouped_exposure.cumsum() - attempt_kc["kc_exposure_increment"]
        ).astype("float64")
        increment_description = "1 / kc_count per linked KC"
    attempt_kc["kc_practice_component"] = np.log1p(attempt_kc["kc_opportunity"].to_numpy(dtype="float64"))
    attempt_kc["kc_weight_equal"] = 1.0 / attempt_kc["kc_count"].to_numpy(dtype="float64")
    attempt_kc["kc_practice_component_weighted"] = (
        attempt_kc["kc_practice_component"] * attempt_kc["kc_weight_equal"]
    )

    aggregation = {
        "student_id": "first",
        "item_id": "first",
        "correct": "first",
        "timestamp": "first",
        "question_difficulty": "first",
        "difficulty_feedback": "first",
        "trust_feedback": "first",
        "hint_used_clean": "first",
        "duration_seconds": "first",
        "selection_change": "first",
        "kc_count": "first",
        "kc_id": join_unique_strings,
        "kc_name": join_unique_strings,
        "kc_opportunity": ["sum", "mean", "max", "min"],
        "kc_practice_component": ["sum", "mean", "max"],
        "kc_practice_component_weighted": "sum",
        "kc_exposure_increment": "sum",
    }

    attempt_table = attempt_kc.groupby("attempt_id", as_index=False).agg(aggregation)
    attempt_table.columns = [
        "attempt_id",
        "student_id",
        "item_id",
        "correct",
        "timestamp",
        "question_difficulty",
        "difficulty_feedback",
        "trust_feedback",
        "hint_used",
        "duration_seconds",
        "selection_change",
        "kc_count",
        "kc_ids",
        "kc_names",
        "kc_opportunity_sum",
        "kc_opportunity_mean",
        "kc_opportunity_max",
        "kc_opportunity_min",
        "kc_practice_feature_sum",
        "kc_practice_feature_mean",
        "kc_practice_feature_max",
        "kc_practice_feature_weighted",
        "kc_exposure_increment_sum",
    ]
    attempt_table["any_first_kc"] = (attempt_table["kc_opportunity_min"] == 0).astype("int64")
    attempt_table["all_first_kc"] = (attempt_table["kc_opportunity_max"] == 0).astype("int64")
    attempt_table["practice_feature"] = attempt_table["kc_practice_feature_sum"].astype("float64")
    attempt_table["kc_practice_feature"] = attempt_table["kc_practice_feature_sum"].astype("float64")

    attempt_table = attempt_table.sort_values(
        ["student_id", "timestamp", "attempt_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    student_counts = attempt_table.groupby("student_id")["attempt_id"].transform("size")
    eligible = attempt_table.loc[student_counts >= min_history].copy()

    eligible["trial_index_within_student"] = eligible.groupby("student_id").cumcount() + 1
    eligible["overall_opportunity"] = eligible["trial_index_within_student"] - 1
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

    eligible_attempt_ids = set(eligible["attempt_id"].astype(int).tolist())
    eligible_attempt_kc = attempt_kc.loc[attempt_kc["attempt_id"].isin(eligible_attempt_ids)].copy()
    kc_monotone = bool(
        eligible_attempt_kc.groupby(["student_id", "kc_id"])["kc_opportunity"]
        .apply(lambda s: bool((s.diff().fillna(1) >= 0).all()))
        .all()
    )

    processed_columns = [
        "attempt_id",
        "student_id",
        "item_id",
        "correct",
        "timestamp",
        "question_difficulty",
        "trial_index_within_student",
        "overall_opportunity",
        "student_total_attempts",
        "split",
        "train_rows_for_student",
        "test_rows_for_student",
        "item_seen_in_train",
        "new_item_in_test",
        "primary_eval_eligible",
        "practice_feature",
        "kc_practice_feature",
        "kc_count",
        "kc_ids",
        "kc_names",
        "kc_opportunity_sum",
        "kc_opportunity_mean",
        "kc_opportunity_max",
        "kc_opportunity_min",
        "kc_practice_feature_sum",
        "kc_practice_feature_mean",
        "kc_practice_feature_max",
        "kc_practice_feature_weighted",
        "kc_exposure_increment_sum",
        "any_first_kc",
        "all_first_kc",
        "difficulty_feedback",
        "trust_feedback",
        "hint_used",
        "duration_seconds",
        "selection_change",
    ]
    processed = eligible.loc[:, processed_columns].copy()
    processed["timestamp"] = processed["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    split_columns = [
        "attempt_id",
        "student_id",
        "item_id",
        "split",
        "trial_index_within_student",
        "train_rows_for_student",
        "test_rows_for_student",
        "item_seen_in_train",
        "new_item_in_test",
        "primary_eval_eligible",
        "practice_feature",
        "kc_practice_feature",
        "kc_count",
        "kc_ids",
        "kc_opportunity_mean",
        "any_first_kc",
        "all_first_kc",
    ]
    split_assignments = processed.loc[:, split_columns].copy()

    attempt_kc_long = eligible_attempt_kc[
        [
            "attempt_id",
            "student_id",
            "item_id",
            "kc_id",
            "kc_name",
            "kc_relationship_id",
            "kc_count",
            "kc_opportunity",
            "kc_practice_component",
            "kc_weight_equal",
        ]
    ].copy()

    ensure_parent(processed_trials_path)
    processed.to_csv(processed_trials_path, index=False)

    ensure_parent(split_assignments_path)
    split_assignments.to_csv(split_assignments_path, index=False)

    ensure_parent(attempt_kc_long_path)
    attempt_kc_long.to_csv(attempt_kc_long_path, index=False)

    test_rows = processed.loc[processed["split"] == "test"].copy()

    summary = {
        "counts": {
            "raw_rows": raw_rows,
            "rows_after_hidden_exclusion": int(len(visible)),
            "attempt_rows_with_kc": int(attempt_table["attempt_id"].nunique()),
            "attempt_kc_rows": int(len(attempt_kc)),
            "eligible_rows": int(len(processed)),
            "eligible_students": int(processed["student_id"].nunique()),
            "items_with_kc": int(processed["item_id"].nunique()),
            "represented_kcs": int(attempt_kc["kc_id"].nunique()),
            "single_kc_attempt_rows": int((attempt_table["kc_count"] == 1).sum()),
            "multi_kc_attempt_rows": int((attempt_table["kc_count"] > 1).sum()),
            "zero_kc_questions": zero_kc_questions,
            "mean_kc_count_per_attempt": float(attempt_table["kc_count"].mean()),
            "median_kc_count_per_attempt": float(attempt_table["kc_count"].median()),
            "train_rows": int((processed["split"] == "train").sum()),
            "test_rows": int((processed["split"] == "test").sum()),
        },
        "validation": {
            "kc_opportunity_monotone": kc_monotone,
            "chronology_violations": chronology_violations,
        },
        "practice": {
            "kc_update_mode": kc_update_mode,
            "increment_description": increment_description,
            "aggregation_description": "sum(log1p(kc_opportunity_k)) across linked KCs",
        },
        "test_profile": {
            "mean_kc_count_test": float(test_rows["kc_count"].mean()) if len(test_rows) else None,
            "any_first_kc_share_test": float(test_rows["any_first_kc"].mean()) if len(test_rows) else None,
            "all_first_kc_share_test": float(test_rows["all_first_kc"].mean()) if len(test_rows) else None,
            "kc_opportunity_mean_test": float(test_rows["kc_opportunity_mean"].mean()) if len(test_rows) else None,
            "practice_feature_mean_test": float(test_rows["practice_feature"].mean()) if len(test_rows) else None,
        },
        "paths": {
            "processed_trials_path": str(processed_trials_path),
            "split_assignments_path": str(split_assignments_path),
            "attempt_kc_long_path": str(attempt_kc_long_path),
        },
    }

    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    build_schema_note(schema_note_path, summary)

    print(f"Saved multi-KC discovery table to {processed_trials_path}")
    print(f"Saved long attempt-KC table to {attempt_kc_long_path}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
