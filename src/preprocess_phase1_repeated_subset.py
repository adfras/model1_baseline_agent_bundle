from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFIG_PATH = Path("config/phase1_repeated_subset_preprocess.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a repeated-practice Phase 1 discovery subset from the single-KC discovery table."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_schema_note(path: Path, summary: dict) -> None:
    ensure_parent(path)
    counts = summary["counts"]
    thresholds = summary["repeat_audit"]["thresholds"]
    threshold_lines = []
    for threshold in thresholds:
        threshold_lines.append(
            f"- `>= {threshold['min_kc_sequence_length']}` opportunities: "
            f"`{threshold['student_kc_sequences']}` student-KC sequences, "
            f"`{threshold['students']}` students, "
            f"`{threshold['kcs']}` KCs, "
            f"`{threshold['rows_in_kept_sequences']}` rows"
        )
    threshold_block = "\n".join(threshold_lines)

    payload = f"""# Phase 1 Repeated-Practice Subset Note

This note records the stronger repeated-practice discovery subset built from the single-KC public discovery table.

## Why this subset exists

- The original single-KC table still mixes many student-KC sequences with only one or two opportunities.
- That is weak support for identifying learner-specific growth.
- This subset keeps only student-KC trajectories with at least `{summary["config"]["min_kc_sequence_length"]}` opportunities.

## Audit of repeated student-KC practice

{threshold_block}

## Selected subset rule

- keep only student-KC sequences with `>= {summary["config"]["min_kc_sequence_length"]}` opportunities
- then require each student to retain at least `{summary["config"]["min_history"]}` rows in the filtered table

## Filtered subset summary

- rows after sequence filter: `{counts["rows_after_sequence_filter"]}`
- rows after student-history filter: `{counts["processed_rows"]}`
- retained learners: `{counts["eligible_students"]}`
- retained items: `{counts["items"]}`
- retained KCs: `{counts["kcs"]}`
- retained student-KC sequences: `{counts["student_kc_sequences"]}`
- train rows: `{counts["train_rows"]}`
- test rows: `{counts["test_rows"]}`
- primary-eval rows: `{counts["primary_eval_rows"]}`

## Validation checks

- KC opportunity monotone within retained student-KC: `{summary["validation"]["kc_opportunity_monotone"]}`
- chronology violations after filtering and resorting: `{summary["validation"]["chronology_violations"]}`
- unseen-item test rows: `{counts["new_item_test_rows"]}`

## Design choice

- The selected threshold is a compromise:
  - `4+` and `5+` opportunities collapse the analysis to too few KCs
  - `3+` retains a stronger repeated-practice signal while preserving a usable learner and KC sample
"""
    path.write_text(payload, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    input_trials_path = Path(config["input_trials_path"])
    processed_trials_path = Path(config["processed_trials_path"])
    split_assignments_path = Path(config["split_assignments_path"])
    summary_path = Path(config["summary_path"])
    schema_note_path = Path(config["schema_note_path"])

    min_kc_sequence_length = int(config["min_kc_sequence_length"])
    train_fraction = float(config["train_fraction"])
    min_history = int(config["min_history"])
    audit_thresholds = [int(value) for value in config.get("audit_thresholds", [1, 2, 3, 4, 5, 6, 8, 10])]

    trials = pd.read_csv(input_trials_path, parse_dates=["timestamp"])
    trials = trials.sort_values(["student_id", "timestamp", "attempt_id"], kind="mergesort").reset_index(drop=True)

    student_kc_sequences = (
        trials.groupby(["student_id", "kc_id"], as_index=False)
        .agg(
            kc_sequence_length=("attempt_id", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
        .sort_values(["student_id", "kc_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    audit_rows = []
    for threshold in audit_thresholds:
        kept_sequences = student_kc_sequences.loc[
            student_kc_sequences["kc_sequence_length"] >= threshold
        ].copy()
        audit_rows.append(
            {
                "min_kc_sequence_length": threshold,
                "student_kc_sequences": int(len(kept_sequences)),
                "students": int(kept_sequences["student_id"].nunique()),
                "kcs": int(kept_sequences["kc_id"].nunique()),
                "rows_in_kept_sequences": int(kept_sequences["kc_sequence_length"].sum()),
            }
        )

    repeated_sequences = student_kc_sequences.loc[
        student_kc_sequences["kc_sequence_length"] >= min_kc_sequence_length,
        ["student_id", "kc_id", "kc_sequence_length"],
    ].copy()
    filtered = trials.merge(repeated_sequences, on=["student_id", "kc_id"], how="inner")
    rows_after_sequence_filter = int(len(filtered))

    student_filtered_counts = filtered.groupby("student_id")["attempt_id"].transform("size")
    eligible = filtered.loc[student_filtered_counts >= min_history].copy()
    excluded_students_min_history = int(filtered["student_id"].nunique() - eligible["student_id"].nunique())

    eligible = eligible.sort_values(["student_id", "timestamp", "attempt_id"], kind="mergesort").reset_index(drop=True)
    eligible["trial_index_within_student"] = eligible.groupby("student_id").cumcount() + 1
    eligible["overall_opportunity"] = eligible["trial_index_within_student"] - 1
    eligible["student_total_attempts"] = eligible.groupby("student_id")["attempt_id"].transform("size")
    eligible["kc_opportunity"] = eligible.groupby(["student_id", "kc_id"]).cumcount()
    eligible["kc_practice_feature"] = np.log1p(eligible["kc_opportunity"].to_numpy(dtype="float64"))
    eligible["practice_feature"] = eligible["kc_practice_feature"]

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
        "correct",
        "timestamp",
        "question_difficulty",
        "trial_index_within_student",
        "overall_opportunity",
        "kc_opportunity",
        "practice_feature",
        "kc_practice_feature",
        "kc_sequence_length",
        "student_total_attempts",
        "split",
        "train_rows_for_student",
        "test_rows_for_student",
        "item_seen_in_train",
        "new_item_in_test",
        "primary_eval_eligible",
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
        "kc_id",
        "kc_sequence_length",
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
            "input_trials_path": str(input_trials_path),
            "processed_trials_path": str(processed_trials_path),
            "split_assignments_path": str(split_assignments_path),
            "summary_path": str(summary_path),
            "schema_note_path": str(schema_note_path),
            "min_kc_sequence_length": min_kc_sequence_length,
            "train_fraction": train_fraction,
            "min_history": min_history,
            "audit_thresholds": audit_thresholds,
        },
        "repeat_audit": {
            "thresholds": audit_rows,
            "kc_sequence_length_quantiles": {
                key: float(value)
                for key, value in student_kc_sequences["kc_sequence_length"]
                .quantile([0.5, 0.75, 0.9, 0.95, 0.99])
                .to_dict()
                .items()
            },
        },
        "counts": {
            "input_rows": int(len(trials)),
            "input_students": int(trials["student_id"].nunique()),
            "input_items": int(trials["item_id"].nunique()),
            "input_kcs": int(trials["kc_id"].nunique()),
            "rows_after_sequence_filter": rows_after_sequence_filter,
            "eligible_students": int(processed["student_id"].nunique()),
            "excluded_students_min_history": excluded_students_min_history,
            "processed_rows": int(len(processed)),
            "items": int(processed["item_id"].nunique()),
            "kcs": int(processed["kc_id"].nunique()),
            "student_kc_sequences": int(
                processed[["student_id", "kc_id"]].drop_duplicates().shape[0]
            ),
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

    print(f"Wrote repeated-practice subset to {processed_trials_path}")
    print(f"Rows after sequence filter: {rows_after_sequence_filter}")
    print(f"Eligible students after min-history filter: {summary['counts']['eligible_students']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
