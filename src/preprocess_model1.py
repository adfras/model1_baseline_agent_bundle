from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("config/model1_preprocess.json")


@dataclass
class Config:
    transactions_path: Path
    questions_path: Path
    processed_trials_path: Path
    split_assignments_path: Path
    summary_path: Path
    train_fraction: float
    min_history: int
    exclude_hidden: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess DBE-KT22 for the Model 1 baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON config file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    cfg = Config(
        transactions_path=Path(raw["transactions_path"]),
        questions_path=Path(raw["questions_path"]),
        processed_trials_path=Path(raw["processed_trials_path"]),
        split_assignments_path=Path(raw["split_assignments_path"]),
        summary_path=Path(raw["summary_path"]),
        train_fraction=float(raw["train_fraction"]),
        min_history=int(raw["min_history"]),
        exclude_hidden=bool(raw["exclude_hidden"]),
    )

    if not 0.0 < cfg.train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if cfg.min_history < 2:
        raise ValueError("min_history must be at least 2.")
    return cfg


def parse_timestamp(value: str) -> datetime:
    formats = (
        "%Y-%m-%d %H:%M:%S.%f %z",
        "%Y-%m-%d %H:%M:%S %z",
    )
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value!r}")


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"Unsupported boolean literal: {value!r}")


def load_question_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata[row["id"]] = {
                "question_difficulty": int(row["difficulty"]) if row["difficulty"] else None,
                "question_title": row.get("question_title") or "",
            }
    return metadata


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    questions = load_question_metadata(cfg.questions_path)

    by_student: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_students: set[str] = set()
    visible_students: set[str] = set()
    raw_questions: set[str] = set()
    visible_questions: set[str] = set()
    raw_rows = 0
    hidden_rows_excluded = 0
    negative_duration_count = 0
    zero_duration_count = 0
    start_min: datetime | None = None
    start_max: datetime | None = None

    with cfg.transactions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_rows += 1
            raw_students.add(row["student_id"])
            raw_questions.add(row["question_id"])

            start_ts = parse_timestamp(row["start_time"])
            end_ts = parse_timestamp(row["end_time"])
            duration_seconds = (end_ts - start_ts).total_seconds()
            if duration_seconds < 0:
                negative_duration_count += 1
            elif duration_seconds == 0:
                zero_duration_count += 1

            if start_min is None or start_ts < start_min:
                start_min = start_ts
            if start_max is None or start_ts > start_max:
                start_max = start_ts

            is_hidden = parse_bool(row["is_hidden"])
            if cfg.exclude_hidden and is_hidden:
                hidden_rows_excluded += 1
                continue

            visible_students.add(row["student_id"])
            visible_questions.add(row["question_id"])

            question_meta = questions.get(row["question_id"], {})
            record = {
                "attempt_id": int(row["id"]),
                "student_id": int(row["student_id"]),
                "item_id": int(row["question_id"]),
                "correct": 1 if parse_bool(row["answer_state"]) else 0,
                "timestamp": start_ts,
                "question_difficulty": question_meta.get("question_difficulty"),
            }
            by_student[row["student_id"]].append(record)

    eligible_students = {
        student_id: rows for student_id, rows in by_student.items() if len(rows) >= cfg.min_history
    }
    excluded_students = {
        student_id: len(rows) for student_id, rows in by_student.items() if len(rows) < cfg.min_history
    }

    processed_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    training_items: set[int] = set()
    duplicate_student_start_pairs = 0

    for student_id in sorted(eligible_students, key=lambda value: int(value)):
        rows = eligible_students[student_id]
        rows.sort(key=lambda row: (row["timestamp"], row["attempt_id"]))

        seen_timestamps: dict[str, int] = defaultdict(int)
        for row in rows:
            seen_timestamps[row["timestamp"].isoformat()] += 1
        duplicate_student_start_pairs += sum(
            1 for count in seen_timestamps.values() if count > 1
        )

        total_attempts = len(rows)
        train_rows = int(math.floor(total_attempts * cfg.train_fraction))
        train_rows = max(1, min(train_rows, total_attempts - 1))
        test_rows = total_attempts - train_rows

        for idx, row in enumerate(rows, start=1):
            overall_opportunity = idx - 1
            split = "train" if idx <= train_rows else "test"
            if split == "train":
                training_items.add(row["item_id"])

            processed_rows.append(
                {
                    "attempt_id": row["attempt_id"],
                    "student_id": row["student_id"],
                    "item_id": row["item_id"],
                    "correct": row["correct"],
                    "timestamp": row["timestamp"].isoformat(),
                    "question_difficulty": row["question_difficulty"]
                    if row["question_difficulty"] is not None
                    else "",
                    "trial_index_within_student": idx,
                    "overall_opportunity": overall_opportunity,
                    "practice_feature": f"{math.log1p(overall_opportunity):.12f}",
                    "student_total_attempts": total_attempts,
                    "split": split,
                    "train_rows_for_student": train_rows,
                    "test_rows_for_student": test_rows,
                }
            )

    for row in processed_rows:
        item_seen_in_train = row["item_id"] in training_items
        new_item_in_test = row["split"] == "test" and not item_seen_in_train
        primary_eval_eligible = row["split"] == "test" and item_seen_in_train

        row["item_seen_in_train"] = int(item_seen_in_train)
        row["new_item_in_test"] = int(new_item_in_test)
        row["primary_eval_eligible"] = int(primary_eval_eligible)

        split_rows.append(
            {
                "attempt_id": row["attempt_id"],
                "student_id": row["student_id"],
                "item_id": row["item_id"],
                "split": row["split"],
                "trial_index_within_student": row["trial_index_within_student"],
                "student_total_attempts": row["student_total_attempts"],
                "train_rows_for_student": row["train_rows_for_student"],
                "test_rows_for_student": row["test_rows_for_student"],
                "item_seen_in_train": row["item_seen_in_train"],
                "new_item_in_test": row["new_item_in_test"],
                "primary_eval_eligible": row["primary_eval_eligible"],
            }
        )

    processed_fieldnames = [
        "attempt_id",
        "student_id",
        "item_id",
        "correct",
        "timestamp",
        "question_difficulty",
        "trial_index_within_student",
        "overall_opportunity",
        "practice_feature",
        "student_total_attempts",
        "split",
        "train_rows_for_student",
        "test_rows_for_student",
        "item_seen_in_train",
        "new_item_in_test",
        "primary_eval_eligible",
    ]
    split_fieldnames = [
        "attempt_id",
        "student_id",
        "item_id",
        "split",
        "trial_index_within_student",
        "student_total_attempts",
        "train_rows_for_student",
        "test_rows_for_student",
        "item_seen_in_train",
        "new_item_in_test",
        "primary_eval_eligible",
    ]

    write_csv(cfg.processed_trials_path, processed_fieldnames, processed_rows)
    write_csv(cfg.split_assignments_path, split_fieldnames, split_rows)

    train_count = sum(1 for row in processed_rows if row["split"] == "train")
    test_count = sum(1 for row in processed_rows if row["split"] == "test")
    primary_eval_count = sum(row["primary_eval_eligible"] for row in processed_rows)
    new_item_test_count = sum(row["new_item_in_test"] for row in processed_rows)

    summary = {
        "config": {
            "transactions_path": str(cfg.transactions_path),
            "questions_path": str(cfg.questions_path),
            "processed_trials_path": str(cfg.processed_trials_path),
            "split_assignments_path": str(cfg.split_assignments_path),
            "summary_path": str(cfg.summary_path),
            "train_fraction": cfg.train_fraction,
            "min_history": cfg.min_history,
            "exclude_hidden": cfg.exclude_hidden,
        },
        "counts": {
            "raw_rows": raw_rows,
            "raw_students": len(raw_students),
            "raw_questions": len(raw_questions),
            "hidden_rows_excluded": hidden_rows_excluded,
            "rows_after_hidden_exclusion": sum(len(rows) for rows in by_student.values()),
            "visible_students": len(visible_students),
            "visible_questions": len(visible_questions),
            "eligible_students": len(eligible_students),
            "excluded_students_min_history": len(excluded_students),
            "processed_rows": len(processed_rows),
            "train_rows": train_count,
            "test_rows": test_count,
            "primary_eval_rows": primary_eval_count,
            "new_item_test_rows": new_item_test_count,
            "training_items": len(training_items),
        },
        "timing": {
            "start_time_min": start_min.isoformat() if start_min is not None else None,
            "start_time_max": start_max.isoformat() if start_max is not None else None,
            "negative_duration_count": negative_duration_count,
            "zero_duration_count": zero_duration_count,
            "duplicate_student_start_pairs_among_eligible_students": duplicate_student_start_pairs,
        },
        "student_attempt_distribution": {
            "eligible_min_attempts": min((len(rows) for rows in eligible_students.values()), default=0),
            "eligible_max_attempts": max((len(rows) for rows in eligible_students.values()), default=0),
        },
        "new_item_holdout": {
            "new_item_test_share": (new_item_test_count / test_count) if test_count else 0.0,
            "primary_eval_share_of_test": (primary_eval_count / test_count) if test_count else 0.0,
        },
    }
    write_json(cfg.summary_path, summary)

    print(
        "Saved "
        f"{len(processed_rows)} processed rows for {len(eligible_students)} students "
        f"to {cfg.processed_trials_path}"
    )
    print(f"Saved split assignments to {cfg.split_assignments_path}")
    print(f"Saved preprocessing summary to {cfg.summary_path}")
    print(
        "Test rows: "
        f"{test_count} | primary eval eligible: {primary_eval_count} | new-item test rows: {new_item_test_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
