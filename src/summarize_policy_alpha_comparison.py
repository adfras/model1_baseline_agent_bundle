from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NEW_LEARNING_POLICIES = [
    "balanced_challenge",
    "harder_challenge",
    "confidence_building",
]

SUPPORT_POLICIES = [
    "failure_aware_remediation",
    "spacing_aware_review",
]

KEY_METRICS = [
    "student_avg_target_gap_1_10",
    "policy_advantage_over_actual_1_10",
    "recommendation_stability_mean_abs_diff",
    "recommended_target_band_hit_rate_1_10",
    "fallback_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare offline policy-suite metrics across two RPFA alpha settings.")
    parser.add_argument("--summary-a", type=Path, required=True)
    parser.add_argument("--summary-b", type=Path, required=True)
    parser.add_argument("--rows-a", type=Path, required=True)
    parser.add_argument("--rows-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, required=True)
    parser.add_argument("--label-b", type=str, required=True)
    parser.add_argument("--display-a", type=str)
    parser.add_argument("--display-b", type=str)
    parser.add_argument("--target-gap-tie-margin", type=float, default=0.0002)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def same_item_rates(rows_a: pd.DataFrame, rows_b: pd.DataFrame) -> dict[str, float]:
    merged = rows_a.merge(
        rows_b,
        on=["student_id", "attempt_id", "policy_name"],
        suffixes=("_a", "_b"),
        how="inner",
    )
    if merged.empty:
        return {}
    summary = (
        merged.assign(
            same_recommended_item=(
                merged["recommended_item_id_a"].astype("string")
                == merged["recommended_item_id_b"].astype("string")
            ).astype(float)
        )
        .groupby("policy_name", sort=True)["same_recommended_item"]
        .mean()
    )
    return {str(index): float(value) for index, value in summary.items()}


def aggregate_metrics(summary: dict, policy_names: list[str]) -> dict[str, float]:
    metrics = [summary["policy_summaries"][name] for name in policy_names]
    return {
        "policy_count": float(len(policy_names)),
        "mean_target_gap_1_10": float(sum(m["student_avg_target_gap_1_10"] for m in metrics) / len(metrics)),
        "mean_policy_advantage_1_10": float(sum(m["policy_advantage_over_actual_1_10"] for m in metrics) / len(metrics)),
        "mean_stability": float(sum(m["recommendation_stability_mean_abs_diff"] for m in metrics) / len(metrics)),
        "mean_band_hit_rate_1_10": float(sum(m["recommended_target_band_hit_rate_1_10"] for m in metrics) / len(metrics)),
        "mean_fallback_rate": float(sum(m["fallback_rate"] for m in metrics) / len(metrics)),
    }


def choose_alpha(label_a: str, agg_a: dict, label_b: str, agg_b: dict, tie_margin: float) -> tuple[str, str]:
    gap_a = agg_a["mean_target_gap_1_10"]
    gap_b = agg_b["mean_target_gap_1_10"]
    if gap_a + tie_margin < gap_b:
        return label_a, "Lower mean target gap across the three new-learning policies."
    if gap_b + tie_margin < gap_a:
        return label_b, "Lower mean target gap across the three new-learning policies."

    advantage_a = agg_a["mean_policy_advantage_1_10"]
    advantage_b = agg_b["mean_policy_advantage_1_10"]
    if advantage_a > advantage_b:
        return label_a, "Target-gap tie; higher mean policy advantage across the three new-learning policies."
    if advantage_b > advantage_a:
        return label_b, "Target-gap tie; higher mean policy advantage across the three new-learning policies."

    stability_a = agg_a["mean_stability"]
    stability_b = agg_b["mean_stability"]
    if stability_a <= stability_b:
        return label_a, "Target-gap and policy-advantage tie; lower mean recommendation instability."
    return label_b, "Target-gap and policy-advantage tie; lower mean recommendation instability."


def main() -> int:
    args = parse_args()
    summary_a = load_json(args.summary_a)
    summary_b = load_json(args.summary_b)
    rows_a = pd.read_csv(args.rows_a)
    rows_b = pd.read_csv(args.rows_b)
    display_a = args.display_a or args.label_a
    display_b = args.display_b or args.label_b

    shared_policies = sorted(set(summary_a["policy_summaries"]).intersection(summary_b["policy_summaries"]))
    same_rate_lookup = same_item_rates(rows_a, rows_b)

    records: list[dict] = []
    for policy_name in shared_policies:
        metrics_a = summary_a["policy_summaries"][policy_name]
        metrics_b = summary_b["policy_summaries"][policy_name]
        record = {
            "policy_name": policy_name,
            "same_recommended_item_rate": same_rate_lookup.get(policy_name, float("nan")),
        }
        for metric_name in KEY_METRICS:
            record[f"{args.label_a}_{metric_name}"] = float(metrics_a[metric_name])
            record[f"{args.label_b}_{metric_name}"] = float(metrics_b[metric_name])
            record[f"delta_{args.label_b}_minus_{args.label_a}_{metric_name}"] = float(
                metrics_b[metric_name] - metrics_a[metric_name]
            )
        records.append(record)

    comparison = pd.DataFrame(records).sort_values("policy_name", kind="mergesort")
    ensure_parent(args.output_csv)
    comparison.to_csv(args.output_csv, index=False)

    new_learning_a = aggregate_metrics(summary_a, NEW_LEARNING_POLICIES)
    new_learning_b = aggregate_metrics(summary_b, NEW_LEARNING_POLICIES)
    support_a = aggregate_metrics(summary_a, SUPPORT_POLICIES)
    support_b = aggregate_metrics(summary_b, SUPPORT_POLICIES)
    selected_label, selection_reason = choose_alpha(
        args.label_a,
        new_learning_a,
        args.label_b,
        new_learning_b,
        args.target_gap_tie_margin,
    )

    def metric_line(metric_label: str, key: str, better: str) -> str:
        return (
            f"- {metric_label}: "
            f"`{display_a}={new_learning_a[key]:.5f}`, "
            f"`{display_b}={new_learning_b[key]:.5f}` "
            f"({better} better)"
        )

    policy_blocks: list[str] = []
    for row in comparison.itertuples(index=False):
        policy_blocks.append(
            f"""### {row.policy_name}

- `{display_a}` target gap `1-10`: `{getattr(row, f"{args.label_a}_student_avg_target_gap_1_10"):.5f}`
- `{display_b}` target gap `1-10`: `{getattr(row, f"{args.label_b}_student_avg_target_gap_1_10"):.5f}`
- `{display_a}` policy advantage `1-10`: `{getattr(row, f"{args.label_a}_policy_advantage_over_actual_1_10"):.5f}`
- `{display_b}` policy advantage `1-10`: `{getattr(row, f"{args.label_b}_policy_advantage_over_actual_1_10"):.5f}`
- `{display_a}` stability: `{getattr(row, f"{args.label_a}_recommendation_stability_mean_abs_diff"):.5f}`
- `{display_b}` stability: `{getattr(row, f"{args.label_b}_recommendation_stability_mean_abs_diff"):.5f}`
- Same recommended item rate: `{row.same_recommended_item_rate:.4f}`
"""
        )

    markdown = f"""# R-PFA Alpha Policy Comparison

This note compares the **Model 2** offline policy suite for `alpha = {display_a}` and `alpha = {display_b}` on the same full public test rows.

Selection rule used here:

- primary selector: lower mean target gap across the three **new-learning** policies
- tie margin on that mean target gap: `{args.target_gap_tie_margin}`
- first tie break: higher mean policy advantage over the actual next item
- second tie break: lower mean recommendation instability

## New-learning aggregate

Policies included:

- `balanced_challenge`
- `harder_challenge`
- `confidence_building`

{metric_line("Mean target gap `1-10`", "mean_target_gap_1_10", "lower")}
{metric_line("Mean policy advantage `1-10`", "mean_policy_advantage_1_10", "higher")}
{metric_line("Mean recommendation instability", "mean_stability", "lower")}
{metric_line("Mean band-hit rate `1-10`", "mean_band_hit_rate_1_10", "higher")}

Operational alpha recommendation:

- selected alpha: `{display_a if selected_label == args.label_a else display_b}`
- reason: {selection_reason}

## Support-mode diagnostic aggregate

Policies included:

- `failure_aware_remediation`
- `spacing_aware_review`

- Mean target gap `1-10`: `{display_a}={support_a['mean_target_gap_1_10']:.5f}`, `{display_b}={support_b['mean_target_gap_1_10']:.5f}`
- Mean policy advantage `1-10`: `{display_a}={support_a['mean_policy_advantage_1_10']:.5f}`, `{display_b}={support_b['mean_policy_advantage_1_10']:.5f}`
- Mean fallback rate: `{display_a}={support_a['mean_fallback_rate']:.5f}`, `{display_b}={support_b['mean_fallback_rate']:.5f}`

These support-mode numbers are reported as diagnostics only. The operational alpha choice is anchored on the new-learning policies because those are the main unseen-item recommendation modes.

## Policy-by-policy detail

{chr(10).join(policy_blocks)}
"""

    ensure_parent(args.output_md)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Saved alpha comparison table to {args.output_csv}")
    print(f"Saved alpha comparison note to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
