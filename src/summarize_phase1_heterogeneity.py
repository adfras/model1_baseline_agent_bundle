from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG_PATH = Path("config/phase1_discovery_heterogeneity_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Phase 1 heterogeneity evidence from Model 1/2/3 public discovery fits."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_indexed_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def lookup_parameter(summary: pd.DataFrame, name: str) -> dict[str, float] | None:
    if name not in summary.index:
        return None
    row = summary.loc[name]
    return {
        "mean": float(row["mean"]),
        "sd": float(row["sd"]),
        "hdi_3%": float(row.get("hdi_3%", row.get("hdi_3.0%", float("nan")))),
        "hdi_97%": float(row.get("hdi_97%", row.get("hdi_97.0%", float("nan")))),
    }


def load_structural_rows(path: Path) -> dict[str, dict[str, float]]:
    table = pd.read_csv(path)
    rows: dict[str, dict[str, float]] = {}
    for _, row in table.iterrows():
        rows[str(row["parameter"])] = {
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "hdi_3%": float(row["hdi_3%"]),
            "hdi_97%": float(row["hdi_97%"]),
        }
    return rows


def maybe_load_json(path_value: str | None) -> dict | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return load_json(path)


def build_markdown(path: Path, payload: dict) -> None:
    ensure_parent(path)
    lines = [
        "# Phase 1 Heterogeneity Summary",
        "",
        "This report applies the revised `Variance + Prediction` rule to the public discovery fits.",
        "",
        "## Decision rule",
        "",
        "- For added heterogeneity terms beyond Model 1, the default practical floor is a posterior SD above `0.03` on the logit scale.",
        "- The 94% HDI lower bound should clear that floor before the variance term is treated as substantively present.",
        "- Predictive gate: the richer model must either improve held-out log loss, or be no worse by more than `0.001` while Brier improves and calibration slope moves closer to `1.0`.",
        "",
    ]

    for model_name in ["model1", "model2", "model3"]:
        model = payload.get(model_name)
        if not model:
            continue
        predictive_metrics = (model.get("predictive") or {}).get("metrics", {})
        lines.extend(
            [
                f"## {model_name.title()}",
                "",
                f"- Held-out log loss: `{predictive_metrics.get('log_loss')}`",
                f"- Held-out Brier: `{predictive_metrics.get('brier_score')}`",
                f"- Held-out calibration slope: `{predictive_metrics.get('calibration_slope')}`",
            ]
        )
        for evidence_name, evidence in model.get("variance_evidence", {}).items():
            if evidence is None:
                continue
            lines.append(
                f"- {evidence_name}: mean `{evidence['mean']:.4f}`, 94% HDI `[{evidence['hdi_3%']:.4f}, {evidence['hdi_97%']:.4f}]`"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation template",
            "",
            "- If Model 2 adds a slope variance term that clears the practical floor and the predictive gate, growth heterogeneity is present.",
            "- If Model 3 adds a stability variance term that clears the practical floor and the predictive gate, stability heterogeneity is present.",
            "- If only Model 1 survives, baseline-level differences dominate the public discovery sample.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    model1_posterior = load_indexed_summary(Path(config["model1_posterior_summary_path"]))
    model2_posterior = load_indexed_summary(Path(config["model2_posterior_summary_path"]))
    model3_structural_path = Path(config["model3_structural_summary_path"])
    model3_structural = load_structural_rows(model3_structural_path) if model3_structural_path.exists() else None

    payload = {
        "model1": {
            "variance_evidence": {
                "student_intercept_sigma": lookup_parameter(model1_posterior, "1|student_id_sigma"),
                "item_sigma": lookup_parameter(model1_posterior, "1|item_id_sigma"),
            },
            "predictive": load_json(Path(config["model1_evaluation_summary_path"])),
        },
        "model2": {
            "variance_evidence": {
                "student_intercept_sigma": lookup_parameter(model2_posterior, "1|student_id_sigma"),
                "student_slope_sigma": lookup_parameter(model2_posterior, "practice_feature|student_id_sigma"),
                "item_sigma": lookup_parameter(model2_posterior, "1|item_id_sigma"),
            },
            "predictive": load_json(Path(config["model2_evaluation_summary_path"])),
        },
        "model3": {
            "variance_evidence": {
                "student_intercept_sigma": model3_structural.get("student_intercept_sigma") if model3_structural else None,
                "student_slope_sigma": model3_structural.get("student_slope_sigma") if model3_structural else None,
                "state_sigma_global": model3_structural.get("state_sigma_global") if model3_structural else None,
            },
            "predictive": maybe_load_json(config.get("model3_evaluation_summary_path")),
        }
        if model3_structural or maybe_load_json(config.get("model3_evaluation_summary_path"))
        else None,
    }
    payload = {key: value for key, value in payload.items() if value is not None}

    output_json_path = Path(config["output_json_path"])
    ensure_parent(output_json_path)
    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    build_markdown(Path(config["output_markdown_path"]), payload)

    print(f"Wrote heterogeneity summary to {output_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
