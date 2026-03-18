from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize spacing-policy due-review threshold runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as label|summary_json_path|rows_csv_path",
    )
    parser.add_argument("--target-gap-tie-margin", type=float, default=0.001)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def student_average(values: pd.Series, student_ids: pd.Series) -> float:
    grouped = pd.DataFrame({"student_id": student_ids, "value": values}).groupby("student_id", sort=False)["value"].mean()
    return float(grouped.mean()) if len(grouped) else float("nan")


def parse_run_spec(run_spec: str) -> tuple[str, Path, Path]:
    parts = run_spec.split("|")
    if len(parts) != 3:
        raise ValueError(f"Run spec must be label|summary|rows, got: {run_spec}")
    label, summary_path, rows_path = parts
    return label, Path(summary_path), Path(rows_path)


def summarize_run(label: str, summary_path: Path, rows_path: Path) -> dict:
    summary = load_json(summary_path)
    rows = pd.read_csv(rows_path)
    spacing_rows = rows.loc[rows["policy_name"] == "spacing_aware_review"].copy()
    eligible = spacing_rows.loc[spacing_rows["fallback_used"] == "none"].copy()

    result = {
        "label": label,
        "decay_alpha": float(summary["decay_alpha"]),
        "due_review_hours": float(summary["due_review_hours"]),
        "spacing_rows": int(len(spacing_rows)),
        "review_eligible_rows": int(len(eligible)),
        "review_eligible_rate": float(len(eligible) / len(spacing_rows)) if len(spacing_rows) else float("nan"),
        "overall_fallback_rate": float((spacing_rows["fallback_used"] != "none").mean()) if len(spacing_rows) else float("nan"),
        "overall_seen_item_rate": float(spacing_rows["recommended_seen_item"].mean()) if len(spacing_rows) else float("nan"),
        "overall_target_gap_1_10": float(summary["policy_summaries"]["spacing_aware_review"]["student_avg_target_gap_1_10"]),
        "overall_policy_advantage_1_10": float(summary["policy_summaries"]["spacing_aware_review"]["policy_advantage_over_actual_1_10"]),
    }
    if len(eligible):
        result.update(
            {
                "eligible_student_avg_target_gap": student_average(eligible["target_gap"], eligible["student_id"]),
                "eligible_student_avg_policy_advantage": student_average(
                    eligible["actual_target_gap"] - eligible["target_gap"],
                    eligible["student_id"],
                ),
                "eligible_band_hit_rate": float(eligible["in_target_band"].mean()),
                "eligible_seen_item_rate": float(eligible["recommended_seen_item"].mean()),
                "eligible_mean_due_review_hours": float(eligible["due_review_hours"].mean()),
            }
        )
    else:
        result.update(
            {
                "eligible_student_avg_target_gap": float("nan"),
                "eligible_student_avg_policy_advantage": float("nan"),
                "eligible_band_hit_rate": float("nan"),
                "eligible_seen_item_rate": float("nan"),
                "eligible_mean_due_review_hours": float("nan"),
            }
        )
    return result


def choose_threshold(results: list[dict], tie_margin: float) -> tuple[dict, str]:
    ordered = sorted(results, key=lambda item: item["due_review_hours"])
    best = ordered[0]
    reason = ""
    for candidate in ordered[1:]:
        if candidate["eligible_student_avg_target_gap"] + tie_margin < best["eligible_student_avg_target_gap"]:
            best = candidate
            reason = "Lower review-eligible target gap."
            continue
        if abs(candidate["eligible_student_avg_target_gap"] - best["eligible_student_avg_target_gap"]) <= tie_margin:
            if candidate["review_eligible_rate"] > best["review_eligible_rate"]:
                best = candidate
                reason = "Target-gap tie within margin; higher review-eligible rate."
                continue
            if candidate["review_eligible_rate"] == best["review_eligible_rate"] and (
                candidate["eligible_student_avg_policy_advantage"] > best["eligible_student_avg_policy_advantage"]
            ):
                best = candidate
                reason = "Target-gap and eligible-rate tie; higher review-eligible policy advantage."
    if not reason:
        reason = "Best review-eligible target gap, with the broadest review eligibility among the compared thresholds."
    return best, reason


def main() -> int:
    args = parse_args()
    results = [summarize_run(*parse_run_spec(spec)) for spec in args.run]
    table = pd.DataFrame(results).sort_values("due_review_hours", kind="mergesort")
    ensure_parent(args.output_csv)
    table.to_csv(args.output_csv, index=False)

    selected, selection_reason = choose_threshold(results, args.target_gap_tie_margin)
    blocks = []
    for row in table.itertuples(index=False):
        blocks.append(
            f"""### {row.label}

- Due-review threshold: `{row.due_review_hours:.0f}` hours
- Review-eligible rate: `{row.review_eligible_rate:.4f}`
- Review-eligible target gap: `{row.eligible_student_avg_target_gap:.5f}`
- Review-eligible policy advantage: `{row.eligible_student_avg_policy_advantage:.5f}`
- Review-eligible band-hit rate: `{row.eligible_band_hit_rate:.4f}`
- Review-eligible seen-item rate: `{row.eligible_seen_item_rate:.4f}`
- Overall fallback rate: `{row.overall_fallback_rate:.4f}`
- Overall seen-item rate: `{row.overall_seen_item_rate:.4f}`
"""
        )

    markdown = f"""# Spacing Policy Due-Review Threshold Grid

This note compares `spacing_aware_review` across a due-review-hours grid on the **operational Model 2 R-PFA branch**.

Selection rule used here:

- primary selector: lower student-averaged target gap on the **review-eligible subset**
- tie margin on that target gap: `{args.target_gap_tie_margin}`
- first tie break: higher review-eligible rate
- second tie break: higher review-eligible policy advantage over the actual next item

Selected threshold:

- due-review hours: `{selected['due_review_hours']:.0f}`
- label: `{selected['label']}`
- reason: {selection_reason}

Important interpretation rule:

- this is a **review-mode** comparison
- the eligible-subset metrics are the main evidence
- the overall fallback rate is reported separately so review-mode coverage is visible

{chr(10).join(blocks)}
"""

    ensure_parent(args.output_md)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Saved spacing threshold table to {args.output_csv}")
    print(f"Saved spacing threshold note to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
