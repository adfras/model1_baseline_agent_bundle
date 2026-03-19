from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP_PATH = REPO_ROOT / "third_party_review_bundle_2026-03-16.zip"


STATIC_FILES = [
    "AGENTS.md",
    "PROJECT_PLAN.md",
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    ".agents/skills/model1-baseline-binary-logistic/SKILL.md",
    ".agents/skills/model2-random-slope-binary-logistic/SKILL.md",
    ".agents/skills/model3-dynamic-volatility-binary-logistic/SKILL.md",
    ".agents/skills/phase2-transfer-warm-start/SKILL.md",
    "src/fetch_dbe_kt22.py",
    "data/processed/phase1_multikc/multikc_summary.json",
    "reports/third_party_review_bundle_manifest.md",
]

REPORT_FILES = [
    "reports/current_project_status.md",
    "reports/project_pivot_and_current_focus.md",
    "reports/phase1_selection_memo.md",
    "reports/current_objective_and_failure_mode.md",
    "reports/phase1_multikc_schema_note.md",
    "reports/phase1_multikc_qmatrix_comparison.md",
    "reports/phase1_qmatrix_rpfa_tuning.md",
    "reports/phase1_qmatrix_rpfa_policy_alpha_comparison.md",
    "reports/spacing_policy_due_review_grid.md",
    "reports/phase1_qmatrix_rpfa_operational_selection.md",
    "reports/phase1_qmatrix_learner_state_profiles.md",
    "reports/policy_alignment_calibration.md",
    "reports/calibrated_policy_suite_decision.md",
    "reports/local_uncertainty_policy_suite_decision.md",
    "reports/direct_heterogeneity_policy_decision.md",
    "reports/manylabs_dbe_alignment_note.md",
    "reports/decision_native_successor_spec.md",
]

OUTPUT_FILES = [
    "outputs/phase1_multikc_qmatrix/model1/model1_evaluation_summary.json",
    "outputs/phase1_multikc_qmatrix/model1/model1_posterior_summary.csv",
    "outputs/phase1_multikc_qmatrix/model2/model2_evaluation_summary.json",
    "outputs/phase1_multikc_qmatrix/model2/model2_student_slope_summary.csv",
    "outputs/phase1_multikc_qmatrix/model3/model3_evaluation_summary.json",
    "outputs/phase1_multikc_qmatrix/model3/model3_structural_summary.csv",
    "outputs/phase1_multikc_qmatrix/model3/model3_volatility_summary.csv",
    "outputs/phase1_multikc_qmatrix_rpfa_tuning/model2_alpha_comparison.csv",
    "outputs/phase1_multikc_qmatrix_rpfa_tuning/model2_alpha_selection.json",
    "outputs/phase1_multikc_qmatrix_rpfa/model2/model2_evaluation_summary.json",
    "outputs/phase1_multikc_qmatrix_rpfa/model3/model3_evaluation_summary.json",
    "outputs/phase1_multikc_qmatrix_profiles/learner_profile_summary.json",
    "outputs/phase1_multikc_qmatrix_profiles/learner_profile_validation.json",
    "outputs/phase1_multikc_qmatrix_profiles/model2_learner_profiles.csv",
    "outputs/phase1_multikc_qmatrix_profiles/model3_learner_profiles.csv",
    "outputs/phase1_multikc_qmatrix_profiles/model3_latent_state_profiles.csv",
    "outputs/phase1_adaptive_policy/model2_qmatrix_rpfa_alpha_compare/policy_alpha_comparison.csv",
    "outputs/phase1_adaptive_policy/model2_qmatrix_rpfa/policy_suite_summary.json",
    "outputs/phase1_adaptive_policy/model2_qmatrix_rpfa_spacing_grid/spacing_due_review_grid.csv",
    "outputs/phase1_adaptive_policy/model2_qmatrix_rpfa_spacing_due24/policy_suite_summary.json",
    "outputs/phase1_adaptive_policy/policy_alignment_calibration/policy_alignment_calibration_summary.json",
    "outputs/phase1_adaptive_policy/calibrated_policy_suite_qmatrix_rpfa/policy_suite_summary.json",
    "outputs/phase1_adaptive_policy/local_uncertainty_policy_suite_qmatrix_rpfa/actual_next_calibration_summary.json",
    "outputs/phase1_adaptive_policy/local_uncertainty_policy_suite_qmatrix_rpfa/policy_suite_summary.json",
    "outputs/phase1_adaptive_policy/direct_heterogeneity_policy/direct_policy_grid_search.csv",
    "outputs/phase1_adaptive_policy/direct_heterogeneity_policy/direct_policy_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a zip bundle for third-party review.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help="Output zip path.",
    )
    return parser.parse_args()


def gather_files() -> tuple[list[Path], list[str]]:
    files: set[Path] = set()
    missing: list[str] = []

    for rel_path in STATIC_FILES + REPORT_FILES + OUTPUT_FILES:
        path = REPO_ROOT / rel_path
        if path.exists():
            files.add(path)
        else:
            missing.append(rel_path)

    for path in (REPO_ROOT / "src").glob("*.py"):
        if path.is_file():
            files.add(path)

    for path in (REPO_ROOT / "config").glob("*.json"):
        if path.is_file():
            files.add(path)

    return sorted(files), missing


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    files, missing = gather_files()

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(REPO_ROOT))

    print(f"Created review bundle: {output_path}")
    print(f"Included files: {len(files)}")
    print(f"Missing static files: {len(missing)}")
    if missing:
        for rel_path in missing:
            print(f"MISSING: {rel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
