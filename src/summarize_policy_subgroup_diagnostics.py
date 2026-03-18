from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from policy_suite_common import summarize_policy_rows, write_json
from qmatrix_common import ensure_parent, load_json


DEFAULT_CONFIG_PATH = Path("config/phase1_policy_subgroup_diagnostics_model2_qmatrix_rpfa.json")


SUBGROUP_SPECS = [
    ("all_rows", "All operational replay rows"),
    ("early_steps_1_5", "Held-out steps 1-5"),
    ("later_steps_6_10", "Held-out steps 6-10"),
    ("review_eligible_context", "Context where the selected 24-hour spacing policy finds a due-review item"),
    ("high_recent_failure_context", "Top-quartile remediation recent-failure context"),
    ("low_predicted_proficiency_context", "Bottom-quartile actual-next probability context"),
    ("high_friction_context", "Hint, answer-change, or high-duration friction context"),
    ("actual_single_kc", "Actual next item has one linked KC"),
    ("actual_multi_kc", "Actual next item has multiple linked KCs"),
]

DISPLAY_POLICY_ORDER = [
    "balanced_challenge",
    "harder_challenge",
    "confidence_building",
    "failure_aware_remediation",
    "spacing_aware_review",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize operational policy performance by subgroup.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def format_metric(value: float, digits: int) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def build_attempt_context(
    operational_rows: pd.DataFrame,
    trials_context: pd.DataFrame,
    *,
    recent_failure_quantile: float,
    low_proficiency_quantile: float,
    high_duration_quantile: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    balanced_context = (
        operational_rows.loc[operational_rows["policy_name"] == "balanced_challenge", ["attempt_id", "student_id", "eval_step", "actual_next_probability"]]
        .drop_duplicates(subset=["attempt_id"])
        .rename(columns={"actual_next_probability": "actual_next_probability_context"})
    )
    remediation_context = (
        operational_rows.loc[
            operational_rows["policy_name"] == "failure_aware_remediation",
            ["attempt_id", "recent_failure_score"],
        ]
        .drop_duplicates(subset=["attempt_id"])
        .rename(columns={"recent_failure_score": "remediation_recent_failure_score"})
    )
    spacing_context = (
        operational_rows.loc[
            operational_rows["policy_name"] == "spacing_aware_review",
            ["attempt_id", "due_review_flag", "due_review_hours", "recommended_seen_item", "fallback_used"],
        ]
        .drop_duplicates(subset=["attempt_id"])
        .rename(
            columns={
                "due_review_flag": "spacing_due_review_flag",
                "due_review_hours": "spacing_due_review_hours",
                "recommended_seen_item": "spacing_recommended_seen_item",
                "fallback_used": "spacing_fallback_used",
            }
        )
    )

    attempt_context = balanced_context.merge(remediation_context, on="attempt_id", how="left").merge(
        spacing_context, on="attempt_id", how="left"
    )
    attempt_context = attempt_context.merge(trials_context, on="attempt_id", how="left", validate="one_to_one")

    positive_duration = attempt_context.loc[attempt_context["duration_seconds"] > 0, "duration_seconds"]
    duration_threshold = float(positive_duration.quantile(high_duration_quantile))
    recent_failure_threshold = float(
        attempt_context["remediation_recent_failure_score"].quantile(recent_failure_quantile)
    )
    low_proficiency_threshold = float(
        attempt_context["actual_next_probability_context"].quantile(low_proficiency_quantile)
    )

    attempt_context["all_rows"] = True
    attempt_context["early_steps_1_5"] = attempt_context["eval_step"] <= 5
    attempt_context["later_steps_6_10"] = attempt_context["eval_step"] >= 6
    attempt_context["review_eligible_context"] = attempt_context["spacing_due_review_flag"] == 1
    attempt_context["high_recent_failure_context"] = (
        attempt_context["remediation_recent_failure_score"] >= recent_failure_threshold
    )
    attempt_context["low_predicted_proficiency_context"] = (
        attempt_context["actual_next_probability_context"] <= low_proficiency_threshold
    )
    attempt_context["high_friction_context"] = (
        (attempt_context["hint_used"] == 1)
        | (attempt_context["selection_change"] > 1)
        | (attempt_context["duration_seconds"] >= duration_threshold)
    )
    attempt_context["actual_single_kc"] = attempt_context["kc_count"] == 1
    attempt_context["actual_multi_kc"] = attempt_context["kc_count"] > 1

    thresholds = {
        "recent_failure_quantile": recent_failure_quantile,
        "recent_failure_threshold": recent_failure_threshold,
        "low_proficiency_quantile": low_proficiency_quantile,
        "low_proficiency_threshold": low_proficiency_threshold,
        "high_duration_quantile": high_duration_quantile,
        "high_duration_threshold_seconds": duration_threshold,
    }
    return attempt_context, thresholds


def summarize_subgroups(
    operational_rows: pd.DataFrame,
    attempt_context: pd.DataFrame,
    *,
    max_eval_step: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subgroup_flags = attempt_context.loc[:, ["attempt_id", "student_id", *[name for name, _ in SUBGROUP_SPECS]]]
    rows = operational_rows.merge(subgroup_flags, on=["attempt_id", "student_id"], how="left", validate="many_to_one")

    summary_records: list[dict] = []
    winner_records: list[dict] = []

    for subgroup_name, subgroup_label in SUBGROUP_SPECS:
        subgroup_attempts = attempt_context.loc[attempt_context[subgroup_name]]
        subgroup_rows = rows.loc[rows[subgroup_name]]
        if subgroup_rows.empty:
            continue

        subgroup_policy_rows = []
        for policy_name in DISPLAY_POLICY_ORDER:
            policy_rows = subgroup_rows.loc[subgroup_rows["policy_name"] == policy_name].copy()
            if policy_rows.empty:
                continue
            metrics = summarize_policy_rows(policy_rows, max_eval_step=max_eval_step)
            metrics.update(
                {
                    "subgroup_name": subgroup_name,
                    "subgroup_label": subgroup_label,
                    "policy_name": policy_name,
                    "support_attempts": int(len(subgroup_attempts)),
                    "support_students": int(subgroup_attempts["student_id"].nunique()),
                }
            )
            subgroup_policy_rows.append(metrics)
            summary_records.append(metrics)

        subgroup_df = pd.DataFrame(subgroup_policy_rows)
        if subgroup_df.empty:
            continue
        best_gap = subgroup_df.sort_values(
            ["student_avg_target_gap_overall", "recommendation_stability_mean_abs_diff", "policy_name"],
            kind="mergesort",
        ).iloc[0]
        best_advantage = subgroup_df.sort_values(
            ["policy_advantage_over_actual_overall", "recommended_target_band_hit_rate_overall", "policy_name"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0]
        winner_records.extend(
            [
                {
                    "subgroup_name": subgroup_name,
                    "subgroup_label": subgroup_label,
                    "winner_metric": "student_avg_target_gap_overall",
                    "winner_policy": str(best_gap["policy_name"]),
                    "winner_value": float(best_gap["student_avg_target_gap_overall"]),
                    "support_attempts": int(len(subgroup_attempts)),
                    "support_students": int(subgroup_attempts["student_id"].nunique()),
                },
                {
                    "subgroup_name": subgroup_name,
                    "subgroup_label": subgroup_label,
                    "winner_metric": "policy_advantage_over_actual_overall",
                    "winner_policy": str(best_advantage["policy_name"]),
                    "winner_value": float(best_advantage["policy_advantage_over_actual_overall"]),
                    "support_attempts": int(len(subgroup_attempts)),
                    "support_students": int(subgroup_attempts["student_id"].nunique()),
                },
            ]
        )

    return pd.DataFrame(summary_records), pd.DataFrame(winner_records)


def build_markdown(
    summary_df: pd.DataFrame,
    winners_df: pd.DataFrame,
    thresholds: dict[str, float],
) -> str:
    threshold_lines = [
        f"- high recent failure = remediation recent-failure score >= `{thresholds['recent_failure_threshold']:.2f}` ({thresholds['recent_failure_quantile']:.0%} quantile)",
        f"- low predicted proficiency = actual-next probability <= `{thresholds['low_proficiency_threshold']:.4f}` ({thresholds['low_proficiency_quantile']:.0%} quantile)",
        f"- high friction = `hint_used == 1` or `selection_change > 1` or `duration_seconds >= {thresholds['high_duration_threshold_seconds']:.2f}` seconds ({thresholds['high_duration_quantile']:.0%} quantile of positive durations)",
        "- review-eligible context = the selected `24`-hour spacing policy identifies a due-review item",
    ]

    headline_rows = []
    for subgroup_name, subgroup_label in SUBGROUP_SPECS:
        subgroup_winners = winners_df.loc[winners_df["subgroup_name"] == subgroup_name]
        if subgroup_winners.empty:
            continue
        gap_winner = subgroup_winners.loc[
            subgroup_winners["winner_metric"] == "student_avg_target_gap_overall"
        ].iloc[0]
        advantage_winner = subgroup_winners.loc[
            subgroup_winners["winner_metric"] == "policy_advantage_over_actual_overall"
        ].iloc[0]
        headline_rows.append(
            f"""### {subgroup_name}

- support: `{int(gap_winner['support_attempts'])}` attempts, `{int(gap_winner['support_students'])}` students
- best target-gap policy: `{gap_winner['winner_policy']}` (`{float(gap_winner['winner_value']):.5f}`)
- best policy-advantage policy: `{advantage_winner['winner_policy']}` (`{float(advantage_winner['winner_value']):.5f}`)
"""
        )

    detail_blocks = []
    for subgroup_name, subgroup_label in SUBGROUP_SPECS:
        subgroup_df = summary_df.loc[summary_df["subgroup_name"] == subgroup_name].copy()
        if subgroup_df.empty:
            continue
        subgroup_df = subgroup_df.sort_values("student_avg_target_gap_overall", kind="mergesort")
        lines = []
        for row in subgroup_df.itertuples(index=False):
            lines.append(
                f"- `{row.policy_name}`: target gap `{row.student_avg_target_gap_overall:.5f}`, policy advantage `{row.policy_advantage_over_actual_overall:.5f}`, "
                f"band-hit `{row.recommended_target_band_hit_rate_overall:.4f}`, stability `{row.recommendation_stability_mean_abs_diff:.5f}`, "
                f"seen-item rate `{row.seen_item_recommendation_rate:.4f}`, fallback `{row.fallback_rate:.4f}`"
            )
        detail_blocks.append(f"### {subgroup_name}\n\n" + "\n".join(lines))

    return f"""# Operational Policy Subgroup Diagnostics

This note compares the **current operational Model 2 policy stack** by subgroup.

Important scope note:

- the four new-item policies come from the fixed Model 2 R-PFA suite
- the review policy is replaced with the selected `24`-hour `spacing_aware_review` branch
- this is still an **offline target-control / policy-behavior** analysis, not a causal learning-gain estimate

## Subgroup definitions

{chr(10).join(threshold_lines)}

## Headline winners

{chr(10).join(headline_rows)}

## Detailed policy snapshots

{chr(10).join(detail_blocks)}
"""


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    suite_rows = pd.read_csv(config["suite_rows_path"])
    spacing_rows = pd.read_csv(config["spacing_rows_path"])
    suite_rows = suite_rows.loc[suite_rows["policy_name"] != "spacing_aware_review"].copy()
    operational_rows = pd.concat([suite_rows, spacing_rows], ignore_index=True)

    trials_context = pd.read_csv(
        config["trials_path"],
        usecols=["attempt_id", "kc_count", "hint_used", "duration_seconds", "selection_change"],
    )

    attempt_context, thresholds = build_attempt_context(
        operational_rows,
        trials_context,
        recent_failure_quantile=float(config["recent_failure_quantile"]),
        low_proficiency_quantile=float(config["low_proficiency_quantile"]),
        high_duration_quantile=float(config["high_duration_quantile"]),
    )
    summary_df, winners_df = summarize_subgroups(
        operational_rows,
        attempt_context,
        max_eval_step=int(config["max_eval_step"]),
    )

    ensure_parent(Path(config["summary_output_path"]))
    summary_df.to_csv(config["summary_output_path"], index=False)
    winners_df.to_csv(config["winners_output_path"], index=False)
    write_json(
        Path(config["json_output_path"]),
        {
            "suite_rows_path": config["suite_rows_path"],
            "spacing_rows_path": config["spacing_rows_path"],
            "thresholds": thresholds,
            "summary_output_path": config["summary_output_path"],
            "winners_output_path": config["winners_output_path"],
            "subgroups": [name for name, _ in SUBGROUP_SPECS],
        },
    )

    markdown = build_markdown(summary_df, winners_df, thresholds)
    markdown_path = Path(config["markdown_output_path"])
    ensure_parent(markdown_path)
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved subgroup summary to {config['summary_output_path']}")
    print(f"Saved subgroup winners to {config['winners_output_path']}")
    print(f"Saved subgroup metadata to {config['json_output_path']}")
    print(f"Saved subgroup note to {config['markdown_output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
