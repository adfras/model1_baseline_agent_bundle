# Learner Heterogeneity and Learner-State Estimation

This repository is a public-data modelling programme for binary learner-response data.

The repo originally expanded into offline next-question replay, but the current mainline has now been tightened:

1. estimate heterogeneity on the **full visible DBE dataset**
2. export learner-level **baseline, growth, and stability** profiles from the scientific explicit Q-matrix ladder
3. keep offline policy replay in the repo only as a **bridge / negative-result track**
4. define the requirements for a future **decision-native** system instead of claiming a DBE policy win that the current data do not support

This is still **not** a live personalised learning system.

## Core docs

- [AGENTS.md](D:/model1_baseline_agent_bundle/AGENTS.md)
- [PROJECT_PLAN.md](D:/model1_baseline_agent_bundle/PROJECT_PLAN.md)
- [phase1_selection_memo.md](D:/model1_baseline_agent_bundle/reports/phase1_selection_memo.md)
- [current_project_status.md](D:/model1_baseline_agent_bundle/reports/current_project_status.md)
- [project_pivot_and_current_focus.md](D:/model1_baseline_agent_bundle/reports/project_pivot_and_current_focus.md)
- [phase1_qmatrix_learner_state_profiles.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_learner_state_profiles.md)
- [manylabs_dbe_alignment_note.md](D:/model1_baseline_agent_bundle/reports/manylabs_dbe_alignment_note.md)
- [decision_native_successor_spec.md](D:/model1_baseline_agent_bundle/reports/decision_native_successor_spec.md)
- [current_objective_and_failure_mode.md](D:/model1_baseline_agent_bundle/reports/current_objective_and_failure_mode.md)

## Current mainline answers

### Scientific heterogeneity ladder

On the full-data explicit Q-matrix ladder:

- Model 1 log loss: `0.545311`
- Model 2 log loss: `0.544366`
- Model 3 log loss: `0.543782`

Interpretation:

- baseline heterogeneity is present
- growth heterogeneity is present
- stability heterogeneity is present

So the richest supported public heterogeneity model is still **Model 3**.

Reference:

- [phase1_multikc_qmatrix_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_qmatrix_comparison.md)

### Learner-state estimation deliverable

The repo now treats learner-state estimation as a first-class DBE output.

New exported artifacts:

- Model 2 learner profiles:
  - [model2_learner_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model2_learner_profiles.csv)
- Model 3 learner profiles:
  - [model3_learner_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model3_learner_profiles.csv)
- Model 3 latent-state profiles:
  - [model3_latent_state_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model3_latent_state_profiles.csv)
- validation:
  - [learner_profile_validation.json](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/learner_profile_validation.json)

Interpretation:

- `baseline` = global intercept plus learner intercept on the logit scale
- `growth` = learner-specific practice slope on the logit scale
- `stability` = learner-specific latent state scale in Model 3

Reference:

- [phase1_qmatrix_learner_state_profiles.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_learner_state_profiles.md)

### Operational bridge baseline

The best predictive-yield branch for DBE replay remains the explicit Q-matrix **R-PFA** branch:

- selected `alpha = 0.9`
- raw R-PFA Model 2:
  - log loss `0.541470`
  - Brier `0.183001`
  - AUC `0.764493`
  - calibration slope `0.957899`
- raw R-PFA Model 3:
  - log loss `0.541660`
  - Brier `0.183103`
  - AUC `0.763996`
  - calibration slope `0.972057`

Operational freeze for offline replay:

- scorer = **raw R-PFA Model 2**
- `alpha = 0.9`
- review threshold = `24` hours
- default new-learning policy = fixed `confidence_building`
- Model 3 remains scientific / exploratory only

Reference:

- [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)

## DBE policy status

The repo keeps the offline replay work, but it is no longer the mainline claim.

Current reading:

- the replay suite is still useful as an **offline target-control / policy-behavior** bridge
- raw Model 2 remains the best operational replay input
- Model 3 uncertainty and local residual branches have not earned an operational role on DBE
- a later direct heterogeneity utility branch used Model 3 baseline/growth/stability plus latent state inside the action choice itself and still lost to the frozen spacing-or-confidence baseline
- therefore DBE does **not yet** support an adaptive-question-selection win

References:

- [local_uncertainty_policy_suite_decision.md](D:/model1_baseline_agent_bundle/reports/local_uncertainty_policy_suite_decision.md)
- [policy_alignment_calibration.md](D:/model1_baseline_agent_bundle/reports/policy_alignment_calibration.md)
- [calibrated_policy_suite_decision.md](D:/model1_baseline_agent_bundle/reports/calibrated_policy_suite_decision.md)
- [direct_heterogeneity_policy_decision.md](D:/model1_baseline_agent_bundle/reports/direct_heterogeneity_policy_decision.md)
- [current_objective_and_failure_mode.md](D:/model1_baseline_agent_bundle/reports/current_objective_and_failure_mode.md)

## Why the repo pivoted

The key lesson from comparing this repo to `C:\ManyLabsAnalyses` is:

- heterogeneity is easiest to exploit when it is inside the thing being optimized and scored directly
- DBE replay has been testing heterogeneity as a downstream policy aid instead

That is why the current repo now centers:

- **heterogeneity discovery**
- **learner-state estimation**
- **decision-native future design**

rather than continuing to treat DBE replay as the main proof target.

Reference:

- [manylabs_dbe_alignment_note.md](D:/model1_baseline_agent_bundle/reports/manylabs_dbe_alignment_note.md)
- [decision_native_successor_spec.md](D:/model1_baseline_agent_bundle/reports/decision_native_successor_spec.md)

## Public data workflow

Build the full multi-KC tables:

```powershell
py src/preprocess_phase1_multikc.py
```

Main processed outputs:

- [multikc_trials.csv](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_trials.csv)
- [multikc_attempt_kc_long.csv](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_attempt_kc_long.csv)
- [multikc_summary.json](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_summary.json)
- [phase1_multikc_schema_note.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_schema_note.md)

Current sample:

- `157,989` processed attempt rows
- `1,138` learners
- `212` items
- `93` KCs
- `300,246` attempt-KC rows

## Phase 2 status

Phase 2 scaffolding remains in the repo, but it is still paused because no local dataset is currently available in this workspace.

Available scaffolding:

- [phase2_protocol.md](D:/model1_baseline_agent_bundle/reports/phase2_protocol.md)
- [preprocess_phase2_local.py](D:/model1_baseline_agent_bundle/src/preprocess_phase2_local.py)
- [split_phase2_local.py](D:/model1_baseline_agent_bundle/src/split_phase2_local.py)

Current priority remains:

- report public heterogeneity cleanly
- export learner-state profiles cleanly
- keep DBE replay findings as bridge evidence only
