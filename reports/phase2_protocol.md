# Phase 2 Protocol

This protocol records the revised Phase 2 design **if the operational full-dataset Phase 1 branch is accepted as the basis for scientific continuation**.

- Phase 2A: local structural replication
- Phase 2B: local warm-start application

Under the current full-dataset DBE branch, the richest supported public structure is **Model 3**. Activation remains conditional on accepting that branch despite the single-KC sensitivity collapse to Model 1.

## Local schema

Normalize the local data to:

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

- `practice_feature` remains as a compatibility alias and must equal `kc_practice_feature`

## Split design

Use a deterministic 3-way student-wise split:

- `train`
- `calibration`
- `test`

Default scaffold:

- 60% train students
- 20% calibration students
- 20% untouched test students

## Phase 2A: replication ladder

Phase 2A is a real structural replication, not just a single-model fit.

Replication rule:

- Model 1 is the hurdle benchmark
- the richest public-supported model advances as the primary structural model

Examples:

- if public Phase 1 ends at Model 2:
  - fit local Model 1 and local Model 2
- if public Phase 1 ends at Model 3:
  - fit local Models 1, 2, and 3

Questions to answer:

- does learner intercept variance replicate locally?
- does learner slope variance replicate locally, when applicable?
- does learner stability variance replicate locally, when applicable?

## Phase 2B: warm-start comparison

Run the same supported structure again with public-informed priors or hyperparameters.

General comparison rule:

- benchmark the richer supported model against local Model 1 only as a hurdle/baseline check
- the primary scientific comparison is:
  - weak-prior local chosen richer model
  - public-informed chosen richer model

This protocol should not be used to justify a Model 1-led transfer study when the public discovery dataset has not supported Model 2 or Model 3 cleanly.

## Primary outcome

Primary:

- student-averaged log loss over attempts `1-5`

Secondary:

- student-averaged log loss over attempts `1-10`
- Brier score
- calibration intercept and slope
- overall log loss after more attempts

Sparse-data fallback:

- if too few students reach 5 attempts, pre-specify a shorter primary window such as attempts `1-3`

## Current implementation status

This repo now includes:

- local schema normalization scaffold: `src/preprocess_phase2_local.py`
- deterministic local split scaffold: `src/split_phase2_local.py`
- template configs in `config/phase2_local_preprocess_template.json` and `config/phase2_local_split_template.json`

What still depends on local data being provided:

- actual local replication fits
- actual local weak-prior fit
- actual public-informed warm-start fit
- local replication and warm-start evaluation outputs

What still depends on public discovery acceptance first:

- a decision to treat the full-dataset primary-KC branch as the operational discovery result
- or an explicit decision to redesign / replace DBE because the single-KC sensitivity caveat is too strong
