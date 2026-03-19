# Project Plan

## Current Mainline

This repository is now organized around a smaller core:

1. **public heterogeneity discovery**
2. **learner-state estimation**
3. **decision-native successor design**

The repo is **not** currently trying to claim an adaptive next-question win on DBE.

## Main Questions

### Phase 1

Use the full visible DBE dataset to answer:

- do learners differ in baseline level?
- do learners differ in growth?
- do learners differ in stability?

### Phase 2A

Only if Phase 1 supports Model 2 or Model 3 cleanly:

- do those same learner differences replicate in a local sample?

### Phase 2B

Only after local replication:

- do public-informed priors recover those learner differences earlier than weak-prior local fitting?

## Current Public-Data Structure

The public mainline uses:

- the full visible DBE dataset
- all linked KCs per item
- the explicit Q-matrix heterogeneity ladder as the scientific source of truth

Scientific ladder:

- **Model 1**: baseline heterogeneity
- **Model 2**: growth heterogeneity
- **Model 3**: stability heterogeneity

Current scientific result:

- Model 2 survives over Model 1
- Model 3 survives over Model 2

So the public DBE data support:

- baseline heterogeneity
- growth heterogeneity
- stability heterogeneity

Reference:

- [phase1_multikc_qmatrix_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_qmatrix_comparison.md)

## Current Operational Replay Baseline

The replay branch remains in the repo only as bridge evidence.

Frozen replay baseline:

- scorer: raw explicit Q-matrix **R-PFA Model 2**
- `alpha = 0.9`
- review threshold: `24` hours
- default new-learning policy: fixed `confidence_building`
- Model 3: scientific / exploratory only

Current replay conclusion:

- DBE still does **not** support an operational adaptive-question-selection win

Reference:

- [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)
- [current_objective_and_failure_mode.md](D:/model1_baseline_agent_bundle/reports/current_objective_and_failure_mode.md)

## Current Deliverables

The outputs that matter most now are:

- full-data preprocessing
- explicit Q-matrix scientific fits and summaries
- R-PFA tuning summaries
- learner-state profile exports
- a small set of policy-failure notes documenting what did not survive operationally
- a design spec for a future decision-native system

## What Is Paused

Paused until local data exists:

- local structural replication
- local warm-start transfer

Scaffolding remains:

- [phase2_protocol.md](D:/model1_baseline_agent_bundle/reports/phase2_protocol.md)
- [preprocess_phase2_local.py](D:/model1_baseline_agent_bundle/src/preprocess_phase2_local.py)
- [split_phase2_local.py](D:/model1_baseline_agent_bundle/src/split_phase2_local.py)

## Practical Priority Order

Until local data arrives, the repo should stay focused on:

1. keeping the scientific explicit-Q ladder coherent
2. keeping learner-state exports clean and reproducible
3. keeping the replay baseline documented without letting it retake the repo
4. treating future next-item work as a decision-native redesign problem, not another bolt-on replay tweak
