# Phase 1 Primary-KC Sensitivity Schema Note

This note records the full-item sensitivity table for Phase 1.

## Assignment rule

- All visible items are retained.
- Each item is assigned one deterministic `primary_kc_id`.
- The assignment rule is: use the KC from the earliest relationship row in `Question_KC_Relationships.csv` for that item.

This is a sensitivity analysis, not the primary discovery dataset.

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

## Sensitivity sample summary

- Visible rows before KC assignment: `158389`
- Assigned rows after primary-KC mapping: `158389`
- Eligible learners: `1138`
- Assigned items: `212`
- Assigned KCs: `68`
- Questions with no KC link: `0`

## Validation checks

- KC opportunity monotone within student-KC: `True`
- Chronology violations after sorting: `0`
