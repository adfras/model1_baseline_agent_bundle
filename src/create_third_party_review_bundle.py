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
    "reports/third_party_review_bundle_manifest.md",
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

    for rel_path in STATIC_FILES:
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

    for path in (REPO_ROOT / "reports").glob("*.md"):
        if path.is_file():
            files.add(path)

    outputs_root = REPO_ROOT / "outputs"
    if outputs_root.exists():
        for path in outputs_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".csv", ".png"}:
                continue
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
