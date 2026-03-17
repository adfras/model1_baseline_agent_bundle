from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFIG_PATH = Path("config/phase2_local_preprocess_template.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a local learner-response table for Phase 2.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_correct(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unique = set(numeric.dropna().astype(int).unique().tolist())
        if not unique.issubset({0, 1}):
            raise ValueError("correct_column must contain only 0/1 values.")
        return numeric.astype("int8")

    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": 1, "false": 0, "correct": 1, "incorrect": 0})
    if mapped.isna().any():
        unexpected = sorted(normalized[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"Unsupported correct_column values: {unexpected[:5]}")
    return mapped.astype("int8")


def build_schema_note(path: Path, summary: dict) -> None:
    ensure_parent(path)
    payload = f"""# Phase 2 Local Schema Note

This note records the normalized local-data schema for Phase 2.

## Required analysis columns

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

## Source mapping

- Student column: `{summary["source_mapping"]["student_id_column"]}`
- Item column: `{summary["source_mapping"]["item_id_column"]}`
- KC column: `{summary["source_mapping"]["kc_id_column"]}`
- Correct column: `{summary["source_mapping"]["correct_column"]}`
- Timestamp source: `{summary["source_mapping"]["timestamp_source"]}`

## Counts

- Raw rows: `{summary["counts"]["raw_rows"]}`
- Processed rows: `{summary["counts"]["processed_rows"]}`
- Students: `{summary["counts"]["students"]}`
- Items: `{summary["counts"]["items"]}`
- KCs: `{summary["counts"]["kcs"]}`
"""
    path.write_text(payload, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    input_path = Path(config["input_path"])
    processed_trials_path = Path(config["processed_trials_path"])
    summary_path = Path(config["summary_path"])
    schema_note_path = Path(config["schema_note_path"])

    if str(config.get("input_format", "csv")).lower() != "csv":
        raise ValueError("Only CSV input_format is currently supported.")

    raw = pd.read_csv(input_path)
    raw_rows = int(len(raw))

    student_col = config["student_id_column"]
    item_col = config["item_id_column"]
    kc_col = config["kc_id_column"]
    correct_col = config["correct_column"]
    timestamp_col = config.get("timestamp_column") or None
    order_col = config.get("attempt_order_column") or None
    attempt_id_col = config.get("attempt_id_column") or None

    if not timestamp_col and not order_col:
        raise ValueError("Provide either timestamp_column or attempt_order_column.")

    normalized = pd.DataFrame(
        {
            "student_id": raw[student_col].astype("string"),
            "item_id": raw[item_col].astype("string"),
            "kc_id": raw[kc_col].astype("string"),
            "correct": parse_correct(raw[correct_col]),
        }
    )

    if attempt_id_col:
        normalized["attempt_id"] = raw[attempt_id_col].astype("string")
    else:
        normalized["attempt_id"] = pd.Series(np.arange(1, len(raw) + 1), dtype="int64").astype("string")

    if timestamp_col:
        timestamp = pd.to_datetime(raw[timestamp_col], utc=True, errors="raise")
        timestamp_source = timestamp_col
    else:
        order_values = pd.to_numeric(raw[order_col], errors="raise")
        timestamp = pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(order_values, unit="s")
        timestamp_source = order_col

    normalized["timestamp"] = timestamp
    if order_col:
        normalized["source_attempt_order"] = pd.to_numeric(raw[order_col], errors="coerce").astype("Int64")

    optional_columns = {
        "question_difficulty": config.get("question_difficulty_column"),
        "hint_used": config.get("hint_used_column"),
        "trust_feedback": config.get("trust_feedback_column"),
        "difficulty_feedback": config.get("difficulty_feedback_column"),
        "duration_seconds": config.get("duration_seconds_column"),
        "selection_change": config.get("selection_change_column"),
    }
    for target, source in optional_columns.items():
        if source:
            normalized[target] = raw[source]

    normalized = normalized.dropna(subset=["student_id", "item_id", "kc_id", "correct", "timestamp"]).copy()
    normalized = normalized.sort_values(
        ["student_id", "timestamp", "attempt_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    normalized["trial_index_within_student"] = normalized.groupby("student_id").cumcount() + 1
    normalized["overall_opportunity"] = normalized["trial_index_within_student"] - 1
    normalized["kc_opportunity"] = normalized.groupby(["student_id", "kc_id"]).cumcount()
    normalized["kc_practice_feature"] = np.log1p(normalized["kc_opportunity"].to_numpy(dtype="float64"))
    normalized["practice_feature"] = normalized["kc_practice_feature"]
    normalized["student_total_attempts"] = normalized.groupby("student_id")["attempt_id"].transform("size")
    normalized["timestamp"] = normalized["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    ensure_parent(processed_trials_path)
    normalized.to_csv(processed_trials_path, index=False)

    summary = {
        "source_mapping": {
            "input_path": str(input_path),
            "student_id_column": student_col,
            "item_id_column": item_col,
            "kc_id_column": kc_col,
            "correct_column": correct_col,
            "timestamp_source": timestamp_source,
            "timestamp_column": timestamp_col,
            "attempt_order_column": order_col,
            "attempt_id_column": attempt_id_col,
        },
        "counts": {
            "raw_rows": raw_rows,
            "processed_rows": int(len(normalized)),
            "students": int(normalized["student_id"].nunique()),
            "items": int(normalized["item_id"].nunique()),
            "kcs": int(normalized["kc_id"].nunique()),
        },
        "validation": {
            "kc_opportunity_monotone": bool(
                normalized.groupby(["student_id", "kc_id"])["kc_opportunity"]
                .apply(lambda s: bool((s.diff().fillna(1) >= 0).all()))
                .all()
            ),
            "practice_feature_alias_matches_kc_feature": bool(
                np.allclose(
                    normalized["practice_feature"].to_numpy(dtype="float64"),
                    normalized["kc_practice_feature"].to_numpy(dtype="float64"),
                )
            ),
        },
    }

    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    build_schema_note(schema_note_path, summary)

    print(f"Wrote normalized local Phase 2 table to {processed_trials_path}")
    print(f"Processed rows: {len(normalized)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
