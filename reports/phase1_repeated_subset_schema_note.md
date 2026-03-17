# Phase 1 Repeated-Practice Subset Note

This note records the stronger repeated-practice discovery subset built from the single-KC public discovery table.

## Why this subset exists

- The original single-KC table still mixes many student-KC sequences with only one or two opportunities.
- That is weak support for identifying learner-specific growth.
- This subset keeps only student-KC trajectories with at least `3` opportunities.

## Audit of repeated student-KC practice

- `>= 1` opportunities: `30565` student-KC sequences, `1074` students, `39` KCs, `53614` rows
- `>= 2` opportunities: `12783` student-KC sequences, `1074` students, `33` KCs, `35832` rows
- `>= 3` opportunities: `4993` student-KC sequences, `1055` students, `18` KCs, `20252` rows
- `>= 4` opportunities: `2083` student-KC sequences, `985` students, `10` KCs, `11522` rows
- `>= 5` opportunities: `1190` student-KC sequences, `752` students, `5` KCs, `7950` rows
- `>= 6` opportunities: `529` student-KC sequences, `525` students, `2` KCs, `4645` rows
- `>= 8` opportunities: `485` student-KC sequences, `485` students, `1` KCs, `4352` rows
- `>= 10` opportunities: `8` student-KC sequences, `8` students, `1` KCs, `81` rows

## Selected subset rule

- keep only student-KC sequences with `>= 3` opportunities
- then require each student to retain at least `10` rows in the filtered table

## Filtered subset summary

- rows after sequence filter: `20252`
- rows after student-history filter: `19197`
- retained learners: `853`
- retained items: `51`
- retained KCs: `18`
- retained student-KC sequences: `4703`
- train rows: `15073`
- test rows: `4124`
- primary-eval rows: `4124`

## Validation checks

- KC opportunity monotone within retained student-KC: `True`
- chronology violations after filtering and resorting: `0`
- unseen-item test rows: `0`

## Design choice

- The selected threshold is a compromise:
  - `4+` and `5+` opportunities collapse the analysis to too few KCs
  - `3+` retains a stronger repeated-practice signal while preserving a usable learner and KC sample
