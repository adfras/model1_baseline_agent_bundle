# KC History Feature Validation

This note validates the new KC-history features on a toy one-student, one-KC sequence.

Checks:

- `alpha = 1.0` reproduces cumulative prior success and failure counts exactly.
- `alpha = 0.5` produces the expected recency-weighted prior success sequence `0.0, 0.5, 0.25, 0.625`.
- `alpha = 0.5` produces the expected recency-weighted prior failure sequence `0.0, 0.0, 0.5, 0.25`.
- `kc_last_seen_hours` matches the known timestamp gaps `NaN, 12, 60, 12`.
- `kc_due_review_default` flips on only when the last KC view is at least `48` hours old.

Result: all validation checks passed.
