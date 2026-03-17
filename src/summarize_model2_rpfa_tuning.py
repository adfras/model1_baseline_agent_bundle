from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the Model 2 R-PFA alpha tuning results.")
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    comparison = pd.read_csv(args.comparison_csv).sort_values("alpha", kind="mergesort")
    selection = load_json(args.selection_json)
    selected_alpha = float(selection["selected_alpha"])
    selected_row = comparison.loc[comparison["alpha"] == selected_alpha].iloc[0]

    table_lines = ["| alpha | log loss | Brier | AUC | calibration slope |", "|---:|---:|---:|---:|---:|"]
    for row in comparison.itertuples(index=False):
        table_lines.append(
            f"| {row.alpha:.1f} | {row.log_loss:.6f} | {row.brier_score:.6f} | {row.auc:.6f} | {row.calibration_slope:.6f} |"
        )

    interpretation = (
        "The tuning rule selected `alpha = 1.0`, so DBE did not benefit from additional recency weighting beyond plain PFA on this branch."
        if selected_alpha == 1.0
        else f"The tuning rule selected `alpha = {selected_alpha:.1f}`, so the operational R-PFA mainline uses recency weighting stronger than plain cumulative PFA."
    )

    markdown = f"""# Model 2 R-PFA Alpha Tuning

This note summarizes the explicit Q-matrix Model 2 alpha search for R-PFA history weighting.

Selection rule:

- primary selector: held-out log loss
- tie margin: `{selection['tie_margin']}`
- tie break: choose the largest alpha within the tie margin

Selected alpha:

- `{selected_alpha:.1f}`

Interpretation:

- {interpretation}

## Grid results

{chr(10).join(table_lines)}

## Selected row

- log loss `{selected_row.log_loss:.6f}`
- Brier `{selected_row.brier_score:.6f}`
- AUC `{selected_row.auc:.6f}`
- calibration slope `{selected_row.calibration_slope:.6f}`
"""
    ensure_parent(args.output_md)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Saved tuning note to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
