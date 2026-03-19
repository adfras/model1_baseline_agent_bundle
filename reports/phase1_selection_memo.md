# Phase 1 Selection Memo

This memo is the canonical selection note for the public DBE work.

It now separates three decisions cleanly:

1. **scientific heterogeneity selection**
2. **learner-state reporting selection**
3. **frozen replay-baseline selection**

## Scientific selection: heterogeneity ladder

On the full-data explicit Q-matrix ladder:

- Model 1 log loss `0.545311`
- Model 2 log loss `0.544366`
- Model 3 log loss `0.543782`

Interpretation:

- Model 2 survives over Model 1
- Model 3 survives over Model 2
- the public DBE data support baseline, growth, and stability heterogeneity

So the current scientific reading is:

- **Model 1**: hurdle benchmark only
- **Model 2**: supported growth model
- **Model 3**: richest supported heterogeneity model

Reference:

- [phase1_multikc_qmatrix_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_qmatrix_comparison.md)

## Learner-state reporting selection

The repo now treats learner-state estimation as a direct Phase 1 output rather than only an intermediate step toward policy replay.

Source of truth:

- scientific explicit Q-matrix posterior draws only
- no refitting
- 94% HDIs

Current exported deliverables:

- [model2_learner_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model2_learner_profiles.csv)
- [model3_learner_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model3_learner_profiles.csv)
- [model3_latent_state_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model3_latent_state_profiles.csv)
- [phase1_qmatrix_learner_state_profiles.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_learner_state_profiles.md)

Interpretation:

- `baseline` = global intercept plus learner intercept on the logit scale
- `growth` = learner-specific practice slope on the logit scale
- `stability` = learner-specific latent state scale in Model 3
- `latent_state_mean` by `state_bin` = transient deviation around each learner’s longer-run trajectory

So the current reporting decision is:

- use the scientific explicit-Q ladder as the mainline DBE reporting branch
- treat learner-state profile exports as the main public deliverable beyond aggregate model comparison

## Frozen replay-baseline selection

The replay work remains in the repo, but it is no longer the mainline claim.

Current frozen DBE replay baseline:

- scorer: raw explicit Q-matrix **R-PFA Model 2**
- `alpha = 0.9`
- review threshold: `24` hours
- default new-learning policy: fixed `confidence_building`
- Model 3: scientific / exploratory only

Why this remains frozen:

- raw Model 2 remains the best current replay input
- the uncertainty calibration layer did not survive the fixed-policy rerun
- the corrected KC-constrained local-residual restart also failed the operational gate
- a later direct heterogeneity utility branch also failed to beat the frozen spacing-or-confidence baseline
- the later router branches did not justify replacing the fixed baseline

Reference:

- [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)
- [calibrated_policy_suite_decision.md](D:/model1_baseline_agent_bundle/reports/calibrated_policy_suite_decision.md)
- [local_uncertainty_policy_suite_decision.md](D:/model1_baseline_agent_bundle/reports/local_uncertainty_policy_suite_decision.md)
- [direct_heterogeneity_policy_decision.md](D:/model1_baseline_agent_bundle/reports/direct_heterogeneity_policy_decision.md)
- [current_objective_and_failure_mode.md](D:/model1_baseline_agent_bundle/reports/current_objective_and_failure_mode.md)

## Current DBE policy conclusion

The repo now treats offline policy replay as a **bridge / negative-result track**.

That means:

- DBE policy replay remains useful for documenting what did and did not survive operationally
- DBE policy replay does **not** currently justify an adaptive-question-selection claim
- future next-item work should be framed as a **decision-native** redesign question rather than another bolt-on replay branch

Reference:

- [manylabs_dbe_alignment_note.md](D:/model1_baseline_agent_bundle/reports/manylabs_dbe_alignment_note.md)
- [decision_native_successor_spec.md](D:/model1_baseline_agent_bundle/reports/decision_native_successor_spec.md)
