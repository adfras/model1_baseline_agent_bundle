# Current Project Status

This note records the repo state after the pivot to:

- full-data multi-KC modeling
- explicit Q-matrix heterogeneity discovery
- learner-state profile exports
- offline replay retained only as a bridge / negative-result track

## Current project reading

With no local dataset currently available, the active work is now:

1. establish whether richer learner heterogeneity is present on the full public dataset
2. export learner-level baseline, growth, stability, and latent-state summaries from the scientific ladder
3. keep DBE replay results as documented bridge evidence, not as the repo’s central success criterion
4. define the requirements for a future decision-native system

Plain-language framing of the current applied objective and the current DBE failure mode:

- [current_objective_and_failure_mode.md](D:/model1_baseline_agent_bundle/reports/current_objective_and_failure_mode.md)

## Current public-data state

### Discovery sample

The operational public preprocessing has been run successfully on the full visible DBE dataset.

Current sample:

- `157,989` processed attempt rows
- `1,138` learners
- `212` items
- `93` represented KCs
- `300,246` long attempt-KC rows
- `125,877` train rows
- `32,112` test rows
- `0` chronology violations
- `0` unseen-item test rows in the primary public evaluation

Reference:

- [multikc_summary.json](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_summary.json)
- [phase1_multikc_schema_note.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_schema_note.md)

## Scientific heterogeneity ladder

### Explicit Q-matrix opportunity branch

This remains the scientific source of truth for Phase 1 DBE heterogeneity.

Current held-out results:

- Model 1 log loss `0.545311`
- Model 2 log loss `0.544366`
- Model 3 log loss `0.543782`

Interpretation:

- Model 2 survives over Model 1
- Model 3 survives over Model 2
- the full public DBE data support baseline, growth, and stability heterogeneity

Reference:

- [phase1_multikc_qmatrix_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_qmatrix_comparison.md)

## Learner-state export status

The repo now promotes learner-state estimation to a first-class Phase 1 deliverable.

Current exported artifacts:

- [model2_learner_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model2_learner_profiles.csv)
- [model3_learner_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model3_learner_profiles.csv)
- [model3_latent_state_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model3_latent_state_profiles.csv)
- [learner_profile_summary.json](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/learner_profile_summary.json)
- [learner_profile_validation.json](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/learner_profile_validation.json)
- [phase1_qmatrix_learner_state_profiles.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_learner_state_profiles.md)

Current read:

- Model 2 profiles export learner-specific `baseline` and `growth`
- Model 3 profiles export learner-specific `baseline`, `growth`, and `stability`
- Model 3 latent-state exports give one row per `student_id x state_bin`
- exports are drawn directly from the saved scientific explicit-Q posterior files with `94%` HDIs

Interpretation:

- `baseline` = global intercept plus learner intercept on the logit scale
- `growth` = learner-specific practice slope on the logit scale
- `stability` = learner-specific latent-state scale on the logit scale

## Best-yield predictive branch

The strongest predictive improvement still came from replacing opportunity-only history with KC-specific prior wins and prior fails, then tuning recency over KC-opportunity lag.

Current operational replay tuning result:

- selected alpha: `0.9`
- best raw log loss in the tuning grid: `0.541378` at `alpha = 0.8`
- selected alpha under the tie rule: `0.9`
- the later policy-facing comparison between `0.8` and `0.9` also kept `0.9`

Current selected-branch results:

- R-PFA Model 2:
  - log loss `0.541470`
  - Brier `0.183001`
  - AUC `0.764493`
  - calibration slope `0.957899`
- R-PFA Model 3:
  - log loss `0.541660`
  - Brier `0.183103`
  - AUC `0.763996`
  - calibration slope `0.972057`

Interpretation:

- the largest remaining leverage was in the history signal, not in extra policy-side complexity
- raw **R-PFA Model 2** remains the frozen replay baseline

Reference:

- [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)

## Offline replay status

The replay work remains in the repo, but it is now explicitly a **bridge / negative-result track**.

Current frozen replay baseline:

- scorer: raw explicit Q-matrix **R-PFA Model 2**
- `alpha = 0.9`
- `24`-hour `spacing_aware_review`
- default new-learning policy: fixed `confidence_building`
- Model 3: scientific / exploratory only

Why it stays frozen:

- policy-context calibration checks did not justify promoting Model 3
- the uncertainty calibration layer failed the fixed-policy gate
- the corrected KC-constrained local-residual restart also failed the operational gate
- a later direct heterogeneity utility branch, using Model 3 learner-state signals inside the action choice itself, also failed against the frozen spacing-or-confidence baseline
- the later router branches did not beat the fixed baseline cleanly enough

So the current DBE policy conclusion is:

- DBE still does **not** support an operational adaptive-question-selection win

Reference:

- [policy_alignment_calibration.md](D:/model1_baseline_agent_bundle/reports/policy_alignment_calibration.md)
- [calibrated_policy_suite_decision.md](D:/model1_baseline_agent_bundle/reports/calibrated_policy_suite_decision.md)
- [local_uncertainty_policy_suite_decision.md](D:/model1_baseline_agent_bundle/reports/local_uncertainty_policy_suite_decision.md)
- [direct_heterogeneity_policy_decision.md](D:/model1_baseline_agent_bundle/reports/direct_heterogeneity_policy_decision.md)
- [current_objective_and_failure_mode.md](D:/model1_baseline_agent_bundle/reports/current_objective_and_failure_mode.md)

## ManyLabs alignment

The repo now makes a sharper distinction between:

- heterogeneity that is directly inside the thing being optimized and evaluated
- heterogeneity that is being tested as a downstream policy aid

Current read:

- `ManyLabs` gets direct use from heterogeneity because it is inside the main likelihood and held-out objective
- `DBE` has not yet turned heterogeneity into a better next-item policy than the simpler frozen baseline

That is why the DBE mainline is now:

- heterogeneity discovery
- learner-state estimation
- decision-native successor design

Reference:

- [manylabs_dbe_alignment_note.md](D:/model1_baseline_agent_bundle/reports/manylabs_dbe_alignment_note.md)
- [decision_native_successor_spec.md](D:/model1_baseline_agent_bundle/reports/decision_native_successor_spec.md)

## Phase 2 status

Phase 2 remains scaffolded but paused.

Why:

- no local dataset is currently available in this workspace

Scaffolding still present:

- [phase2_protocol.md](D:/model1_baseline_agent_bundle/reports/phase2_protocol.md)
- [preprocess_phase2_local.py](D:/model1_baseline_agent_bundle/src/preprocess_phase2_local.py)
- [split_phase2_local.py](D:/model1_baseline_agent_bundle/src/split_phase2_local.py)

## Current repo focus

Until local data is available, the practical mainline is:

1. keep the scientific explicit Q-matrix heterogeneity ladder coherent
2. keep exporting learner-state summaries from that ladder
3. keep the frozen DBE replay baseline documented, but not central
4. treat future next-item work as a design-spec problem unless stronger decision-native data become available
