from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the hybrid router v2 development path against v1 and the fixed Model 2 policy suite."
    )
    parser.add_argument("--hybrid-v1-summary", type=Path, required=True)
    parser.add_argument("--hybrid-v2-summary", type=Path, required=True)
    parser.add_argument("--hybrid-v2-tuned-summary", type=Path, required=True)
    parser.add_argument("--fixed-suite-summary", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_route_block(route_name: str, summary: dict) -> str:
    return f"""### {route_name}

- target gap `1-10`: `{summary['student_avg_target_gap_1_10']:.5f}`
- policy advantage `1-10`: `{summary['policy_advantage_over_actual_1_10']:.5f}`
- band-hit rate `1-10`: `{summary['recommended_target_band_hit_rate_1_10']:.4f}`
- stability: `{summary['recommendation_stability_mean_abs_diff']:.5f}`
- recent-failure coverage: `{summary['recent_failure_coverage_rate']:.4f}`
- due-review coverage: `{summary['due_review_coverage_rate']:.4f}`
- seen-item rate: `{summary['seen_item_recommendation_rate']:.4f}`
- fallback rate: `{summary['fallback_rate']:.4f}`
"""


def main() -> int:
    args = parse_args()
    hybrid_v1 = load_json(args.hybrid_v1_summary)
    hybrid_v2_raw = load_json(args.hybrid_v2_summary)
    hybrid_v2_tuned = load_json(args.hybrid_v2_tuned_summary)
    fixed_suite = load_json(args.fixed_suite_summary)

    overall_v1 = hybrid_v1["overall_summary"]
    overall_v2_raw = hybrid_v2_raw["overall_summary"]
    overall_v2_tuned = hybrid_v2_tuned["overall_summary"]
    route_blocks = [
        format_route_block(route_name, summary)
        for route_name, summary in sorted(hybrid_v2_tuned["route_summaries"].items())
    ]

    subgroup_blocks = []
    for subgroup_name, summary in sorted(hybrid_v2_tuned["subgroup_summaries"].items()):
        subgroup_blocks.append(
            f"""### {subgroup_name}

- target gap `1-10`: `{summary['student_avg_target_gap_1_10']:.5f}`
- policy advantage `1-10`: `{summary['policy_advantage_over_actual_1_10']:.5f}`
- band-hit rate `1-10`: `{summary['recommended_target_band_hit_rate_1_10']:.4f}`
- stability: `{summary['recommendation_stability_mean_abs_diff']:.5f}`
"""
        )

    route_share_lines = [
        f"- `{route_name}`: `{share:.2%}`"
        for route_name, share in sorted(hybrid_v2_tuned["route_shares"].items())
    ]
    route_reason_lines = [
        f"- `{reason}`: `{count}`"
        for reason, count in sorted(hybrid_v2_tuned["route_reason_counts"].items())
    ]

    fixed_balanced = fixed_suite["policy_summaries"]["balanced_challenge"]
    fixed_confidence = fixed_suite["policy_summaries"]["confidence_building"]
    fixed_harder = fixed_suite["policy_summaries"]["harder_challenge"]

    markdown = f"""# Hybrid Uncertainty Router V2

This note records the second hybrid-router generation built on top of the selected **R-PFA Model 2** scorer and **R-PFA Model 3** uncertainty signal.

## What changed from v1

Router v2 adds lagged observable proxies before each held-out recommendation step:

- failure streak
- recent attempt-level success rate
- recent hint-use rate
- recent answer-change friction rate
- response-time inflation relative to the learner's prior median
- `24`-hour due-review threshold for spacing mode

The scorer is unchanged:

- **Model 2** still provides the mean correctness estimate
- **Model 3** still provides the uncertainty signal

The router is now recorded in two forms:

- **raw v2**: the first threshold set, which over-routed into non-balanced modes
- **tuned v2**: a more conservative threshold set that is now the active v2 prototype

## Overall comparison

- v1 target gap `1-10`: `{overall_v1['student_avg_target_gap_1_10']:.5f}`
- raw v2 target gap `1-10`: `{overall_v2_raw['student_avg_target_gap_1_10']:.5f}`
- tuned v2 target gap `1-10`: `{overall_v2_tuned['student_avg_target_gap_1_10']:.5f}`
- v1 policy advantage `1-10`: `{overall_v1['policy_advantage_over_actual_1_10']:.5f}`
- raw v2 policy advantage `1-10`: `{overall_v2_raw['policy_advantage_over_actual_1_10']:.5f}`
- tuned v2 policy advantage `1-10`: `{overall_v2_tuned['policy_advantage_over_actual_1_10']:.5f}`
- v1 stability: `{overall_v1['recommendation_stability_mean_abs_diff']:.5f}`
- raw v2 stability: `{overall_v2_raw['recommendation_stability_mean_abs_diff']:.5f}`
- tuned v2 stability: `{overall_v2_tuned['recommendation_stability_mean_abs_diff']:.5f}`
- v1 recent-failure coverage: `{overall_v1['recent_failure_coverage_rate']:.4f}`
- raw v2 recent-failure coverage: `{overall_v2_raw['recent_failure_coverage_rate']:.4f}`
- tuned v2 recent-failure coverage: `{overall_v2_tuned['recent_failure_coverage_rate']:.4f}`
- v1 due-review coverage: `{overall_v1['due_review_coverage_rate']:.4f}`
- raw v2 due-review coverage: `{overall_v2_raw['due_review_coverage_rate']:.4f}`
- tuned v2 due-review coverage: `{overall_v2_tuned['due_review_coverage_rate']:.4f}`
- v1 seen-item rate: `{overall_v1['seen_item_recommendation_rate']:.4f}`
- raw v2 seen-item rate: `{overall_v2_raw['seen_item_recommendation_rate']:.4f}`
- tuned v2 seen-item rate: `{overall_v2_tuned['seen_item_recommendation_rate']:.4f}`

Comparison against the fixed Model 2 policies:

- balanced challenge target gap `1-10`: `{fixed_balanced['student_avg_target_gap_1_10']:.5f}`
- confidence-building target gap `1-10`: `{fixed_confidence['student_avg_target_gap_1_10']:.5f}`
- harder challenge target gap `1-10`: `{fixed_harder['student_avg_target_gap_1_10']:.5f}`

Interpretation:

- raw v2 made coverage and policy-advantage gains, but clearly over-routed and destabilized recommendations
- tuned v2 recovered target-gap control and kept the policy-advantage gains
- tuned v2 still remains much less stable than v1 and much less target-precise than the fixed Model 2 policies
- so tuned v2 is the current **exploratory router prototype**, not the default operational policy

This remains an **offline target-control / policy-behavior** result, not a causal learning-gain estimate.

## Tuned v2 route shares

{chr(10).join(route_share_lines)}

## Tuned v2 route reasons

{chr(10).join(route_reason_lines)}

## Tuned v2 route-level summaries

{chr(10).join(route_blocks)}

## Tuned v2 subgroup summaries

{chr(10).join(subgroup_blocks)}
"""

    ensure_parent(args.output_md)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Saved hybrid v2 note to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
