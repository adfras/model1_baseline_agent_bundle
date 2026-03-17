# Phase 2 Protocol

This protocol remains in the repo as conditional scaffolding.

It is **not** the active workstream right now because no local dataset is currently available in this workspace.

Current active work is:

- public learner-model development on the full DBE dataset
- explicit Q-matrix PFA / R-PFA refinement
- offline next-question policy replay

## When this protocol activates

Use this protocol only when local data becomes available.

Then the sequence is still:

1. **Phase 2A:** local structural replication
2. **Phase 2B:** local warm-start application

## Current public carry-forward rule

Keep two ideas separate:

- richest public-supported heterogeneity model:
  - currently **Model 3** on the full-data explicit Q-matrix ladder
- best current operational policy model:
  - currently **explicit Q-matrix R-PFA Model 2**

When local data arrives, the replication decision should be made explicitly from the scientific objective at that time:

- if the goal is strict heterogeneity replication, carry forward the richest supported heterogeneity ladder
- if the goal is the best operational question-selection model, start from the best current policy model and benchmark against the richer challenger

## Local schema

Normalize the local data to:

- `student_id`
- `item_id`
- `kc_id`
- `correct`
- `timestamp`
- `attempt_id`
- `trial_index_within_student`
- KC-history fields aligned to the public representation

Preferred operational target when possible:

- KC-aware history features compatible with the current explicit Q-matrix PFA / R-PFA branch

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

Phase 2A is still a real structural replication, not just a single-model fit.

Examples:

- if public replication target is Model 2:
  - fit local Model 1 and local Model 2
- if public replication target is Model 3:
  - fit local Models 1, 2, and 3

Questions to answer:

- does learner intercept variance replicate locally?
- does learner slope variance replicate locally, when applicable?
- does learner stability variance replicate locally, when applicable?

## Phase 2B: warm-start comparison

Run the chosen supported structure again with public-informed priors or hyperparameters.

General comparison rule:

- weak-prior local fit
- public-informed fit
- evaluate on untouched local students

Primary evaluation window:

- student-averaged log loss over attempts `1-5`

Secondary:

- student-averaged log loss over attempts `1-10`
- Brier score
- calibration intercept and slope
- overall log loss after more attempts

Sparse-data fallback:

- if too few students reach 5 attempts, pre-specify a shorter primary window such as attempts `1-3`

## Current implementation status

This repo includes:

- local schema normalization scaffold:
  - [preprocess_phase2_local.py](D:/model1_baseline_agent_bundle/src/preprocess_phase2_local.py)
- deterministic local split scaffold:
  - [split_phase2_local.py](D:/model1_baseline_agent_bundle/src/split_phase2_local.py)
- template configs:
  - [phase2_local_preprocess_template.json](D:/model1_baseline_agent_bundle/config/phase2_local_preprocess_template.json)
  - [phase2_local_split_template.json](D:/model1_baseline_agent_bundle/config/phase2_local_split_template.json)

Still missing because no local data is present:

- actual local fits
- actual local replication outputs
- actual public-informed warm-start outputs
