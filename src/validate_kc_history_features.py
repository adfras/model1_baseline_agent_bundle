from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kc_history_common import add_decay_features
from qmatrix_common import ensure_parent


OUTPUT_JSON = Path("outputs/phase1_multikc/kc_history_feature_validation.json")
OUTPUT_MD = Path("reports/kc_history_feature_validation.md")


def build_toy_sequence() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    timestamps = [
        base,
        base + pd.Timedelta(hours=12),
        base + pd.Timedelta(hours=72),
        base + pd.Timedelta(hours=84),
    ]
    return pd.DataFrame(
        {
            "attempt_id": [1, 2, 3, 4],
            "student_id": ["toy_student"] * 4,
            "item_id": ["toy_item_1", "toy_item_2", "toy_item_3", "toy_item_4"],
            "kc_id": ["kc_a"] * 4,
            "kc_name": ["KC A"] * 4,
            "kc_relationship_id": [1, 1, 1, 1],
            "kc_count": [1, 1, 1, 1],
            "timestamp": timestamps,
            "correct": [1, 0, 1, 1],
            "kc_opportunity": [0.0, 1.0, 2.0, 3.0],
            "kc_exposure_increment": [1.0, 1.0, 1.0, 1.0],
            "kc_success_increment": [1.0, 0.0, 1.0, 1.0],
            "kc_failure_increment": [0.0, 1.0, 0.0, 0.0],
            "kc_prior_success_count": [0.0, 1.0, 1.0, 2.0],
            "kc_prior_failure_count": [0.0, 0.0, 1.0, 1.0],
            "kc_practice_component": np.log1p([0.0, 1.0, 2.0, 3.0]),
            "kc_success_component": [0.0, 1.0, 1.0, 2.0],
            "kc_failure_component": [0.0, 0.0, 1.0, 1.0],
            "kc_base_indicator": [1.0, 1.0, 1.0, 1.0],
        }
    )


def assert_close(actual: np.ndarray, expected: list[float], *, atol: float = 1e-6) -> None:
    expected_array = np.asarray(expected, dtype=np.float64)
    if not np.allclose(actual, expected_array, atol=atol, equal_nan=True):
        raise AssertionError(f"Expected {expected_array.tolist()}, got {actual.tolist()}")


def main() -> int:
    toy = build_toy_sequence()

    alpha_one = add_decay_features(toy, decay_alpha=1.0, due_review_hours=48.0)
    assert_close(alpha_one["kc_prior_success_decay"].to_numpy(), [0.0, 1.0, 1.0, 2.0])
    assert_close(alpha_one["kc_prior_failure_decay"].to_numpy(), [0.0, 0.0, 1.0, 1.0])

    alpha_half = add_decay_features(toy, decay_alpha=0.5, due_review_hours=48.0)
    assert_close(alpha_half["kc_prior_success_decay"].to_numpy(), [0.0, 0.5, 0.25, 0.625])
    assert_close(alpha_half["kc_prior_failure_decay"].to_numpy(), [0.0, 0.0, 0.5, 0.25])

    expected_last_seen_hours = [np.nan, 12.0, 60.0, 12.0]
    assert_close(alpha_half["kc_last_seen_hours"].to_numpy(), expected_last_seen_hours)

    expected_due_flags = [0, 0, 1, 0]
    actual_due_flags = alpha_half["kc_due_review_default"].astype(int).tolist()
    if actual_due_flags != expected_due_flags:
        raise AssertionError(f"Expected due flags {expected_due_flags}, got {actual_due_flags}")

    payload = {
        "toy_sequence_rows": int(len(toy)),
        "alpha_one_success_matches_count": True,
        "alpha_one_failure_matches_count": True,
        "alpha_half_success_expected": [0.0, 0.5, 0.25, 0.625],
        "alpha_half_failure_expected": [0.0, 0.0, 0.5, 0.25],
        "last_seen_hours_expected": expected_last_seen_hours,
        "due_review_threshold_hours": 48.0,
        "due_review_flags_expected": expected_due_flags,
        "status": "passed",
    }
    ensure_parent(OUTPUT_JSON)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown = """# KC History Feature Validation

This note validates the new KC-history features on a toy one-student, one-KC sequence.

Checks:

- `alpha = 1.0` reproduces cumulative prior success and failure counts exactly.
- `alpha = 0.5` produces the expected recency-weighted prior success sequence `0.0, 0.5, 0.25, 0.625`.
- `alpha = 0.5` produces the expected recency-weighted prior failure sequence `0.0, 0.0, 0.5, 0.25`.
- `kc_last_seen_hours` matches the known timestamp gaps `NaN, 12, 60, 12`.
- `kc_due_review_default` flips on only when the last KC view is at least `48` hours old.

Result: all validation checks passed.
"""
    ensure_parent(OUTPUT_MD)
    OUTPUT_MD.write_text(markdown, encoding="utf-8")

    print(f"Saved KC history validation JSON to {OUTPUT_JSON}")
    print(f"Saved KC history validation note to {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
