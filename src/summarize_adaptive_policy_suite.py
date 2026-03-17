from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the offline adaptive policy suite comparison.")
    parser.add_argument("--model2-summary", type=Path, required=True)
    parser.add_argument("--model3-summary", type=Path, required=True)
    parser.add_argument("--model2-rows", type=Path, required=True)
    parser.add_argument("--model3-rows", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compare_same_item_rate(model2_rows: pd.DataFrame, model3_rows: pd.DataFrame) -> pd.DataFrame:
    merged = model2_rows.merge(
        model3_rows,
        on=["student_id", "attempt_id", "policy_name"],
        suffixes=("_model2", "_model3"),
        how="inner",
    )
    comparison = (
        merged.assign(
            same_recommended_item=(
                merged["recommended_item_id_model2"].astype("string")
                == merged["recommended_item_id_model3"].astype("string")
            ).astype(int)
        )
        .groupby("policy_name", sort=True)["same_recommended_item"]
        .mean()
        .reset_index()
        .rename(columns={"same_recommended_item": "same_recommended_item_rate"})
    )
    return comparison


def format_policy_block(policy_name: str, model2_metrics: dict, model3_metrics: dict, same_rate: float | None) -> str:
    same_rate_text = "n/a" if same_rate is None else f"{same_rate:.4f}"
    return f"""### {policy_name}

- Model 2 target gap `1-5`: `{model2_metrics['student_avg_target_gap_1_5']:.5f}`
- Model 3 target gap `1-5`: `{model3_metrics['student_avg_target_gap_1_5']:.5f}`
- Model 2 target gap `1-10`: `{model2_metrics['student_avg_target_gap_1_10']:.5f}`
- Model 3 target gap `1-10`: `{model3_metrics['student_avg_target_gap_1_10']:.5f}`
- Model 2 band-hit rate `1-10`: `{model2_metrics['recommended_target_band_hit_rate_1_10']:.4f}`
- Model 3 band-hit rate `1-10`: `{model3_metrics['recommended_target_band_hit_rate_1_10']:.4f}`
- Model 2 policy advantage over actual `1-10`: `{model2_metrics['policy_advantage_over_actual_1_10']:.5f}`
- Model 3 policy advantage over actual `1-10`: `{model3_metrics['policy_advantage_over_actual_1_10']:.5f}`
- Model 2 stability mean abs diff: `{model2_metrics['recommendation_stability_mean_abs_diff']:.5f}`
- Model 3 stability mean abs diff: `{model3_metrics['recommendation_stability_mean_abs_diff']:.5f}`
- Model 2 recent-failure coverage: `{model2_metrics['recent_failure_coverage_rate']:.4f}`
- Model 3 recent-failure coverage: `{model3_metrics['recent_failure_coverage_rate']:.4f}`
- Model 2 due-review coverage: `{model2_metrics['due_review_coverage_rate']:.4f}`
- Model 3 due-review coverage: `{model3_metrics['due_review_coverage_rate']:.4f}`
- Model 2 fallback rate: `{model2_metrics['fallback_rate']:.4f}`
- Model 3 fallback rate: `{model3_metrics['fallback_rate']:.4f}`
- Model 2 seen-item recommendation rate: `{model2_metrics['seen_item_recommendation_rate']:.4f}`
- Model 3 seen-item recommendation rate: `{model3_metrics['seen_item_recommendation_rate']:.4f}`
- Model 2 mean candidate count: `{model2_metrics['mean_candidate_count']:.2f}`
- Model 3 mean candidate count: `{model3_metrics['mean_candidate_count']:.2f}`
- Same recommended item rate: `{same_rate_text}`
"""


def main() -> int:
    args = parse_args()
    model2_summary = load_json(args.model2_summary)
    model3_summary = load_json(args.model3_summary)
    model2_rows = pd.read_csv(args.model2_rows)
    model3_rows = pd.read_csv(args.model3_rows)
    same_rate = compare_same_item_rate(model2_rows, model3_rows)
    same_rate_lookup = dict(zip(same_rate["policy_name"], same_rate["same_recommended_item_rate"]))

    policies = sorted(set(model2_summary["policy_summaries"]).intersection(model3_summary["policy_summaries"]))
    blocks = [
        format_policy_block(
            policy_name,
            model2_summary["policy_summaries"][policy_name],
            model3_summary["policy_summaries"][policy_name],
            same_rate_lookup.get(policy_name),
        )
        for policy_name in policies
    ]

    markdown = f"""# Adaptive Policy Suite Comparison

This note compares the offline policy suite for explicit Q-matrix R-PFA Model 2 and Model 3.

Model metadata:

- Model 2 history mode: `{model2_summary['history_mode']}`, decay alpha `{model2_summary['decay_alpha']}`
- Model 3 history mode: `{model3_summary['history_mode']}`, decay alpha `{model3_summary['decay_alpha']}`
- Evaluation window: first `{model2_summary['max_eval_step']}` held-out primary-evaluation steps per student
- Due-review threshold: `{model2_summary['due_review_hours']}` hours

Current reading:

- Model 2 remains the default policy model unless Model 3 clearly wins on policy-facing metrics.
- This report is an offline target-control / policy-behavior comparison, not a causal learning-gain estimate.

{chr(10).join(blocks)}
"""
    ensure_parent(args.output_md)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Saved policy comparison note to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
