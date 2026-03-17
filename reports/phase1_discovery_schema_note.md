# Phase 1 Discovery Schema Note

This note records the heterogeneity-discovery analysis table built from DBE-KT22.

## Core restriction

- The public discovery sample keeps only items with exactly one linked knowledge component.
- Multi-KC items are excluded so that `kc_opportunity` is well defined without duplicating attempts.

## Main analysis columns

- `student_id`
- `item_id`
- `kc_id`
- `correct`
- `timestamp`
- `attempt_id`
- `trial_index_within_student`
- `overall_opportunity`
- `kc_opportunity`
- `kc_practice_feature`

Implementation note:

- `practice_feature` is retained as a compatibility alias and is set equal to `kc_practice_feature`.

## Auxiliary retained columns

- `question_difficulty`
- `kc_name`
- `hint_used`
- `trust_feedback`
- `difficulty_feedback`
- `duration_seconds`
- `selection_change`

## Discovery sample summary

- Raw visible rows before KC restriction: `158389`
- Retained single-KC rows: `54315`
- Retained learners: `1074`
- Retained items: `72`
- Retained KCs: `39`
- Dropped multi-KC rows: `104074`
- Questions with exactly one KC: `72`
- Questions with multiple KCs: `140`
- Questions with no KC link: `0`

## Validation checks

- KC opportunity monotone within student-KC: `True`
- Chronology violations after sorting: `0`
