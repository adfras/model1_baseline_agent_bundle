# Current Project Status

This note records the repo state after the shift to:

- full-data multi-KC modeling
- explicit Q-matrix likelihoods
- PFA / R-PFA KC-history features
- offline next-question policy replay

## Current project reading

The repo is no longer centered on a public-model leaderboard or on a dormant warm-start study.

With no local dataset currently available, the active work is now:

1. establish whether richer learner heterogeneity is present on the full public dataset
2. improve the learner model where the data actually support it
3. test whether those learner models support useful **offline question selection**

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

See:

- [multikc_summary.json](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_summary.json)
- [phase1_multikc_schema_note.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_schema_note.md)

## Heterogeneity ladder status

### Explicit Q-matrix opportunity branch

This branch keeps all attempt rows and moves KC structure into the likelihood directly.

Current results:

- Model 1 log loss `0.545311`
- Model 2 log loss `0.544366`
- Model 3 log loss `0.543782`

Interpretation:

- Model 2 survives over Model 1
- Model 3 survives over Model 2
- on the full dataset, the public evidence supports baseline, growth, and stability heterogeneity

Reference:

- [phase1_multikc_qmatrix_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_qmatrix_comparison.md)

## Best-yield model-development result

### Explicit Q-matrix PFA / R-PFA branch

The strongest improvement so far came from replacing opportunity-only history with KC-specific prior wins and prior fails, then tuning recency weighting over KC-opportunity lag.

Current operational tuning result:

- selected alpha: `0.9`
- best raw log loss in the grid: `0.541378` at `alpha = 0.8`
- selected alpha under the tie rule: `0.9`

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

- the biggest remaining leverage was in the **history signal**
- the operational learner-model mainline is now the **R-PFA** branch with `alpha = 0.9`
- Model 2 remains the default operational learner model
- Model 3 remains a richer stability/uncertainty challenger rather than the automatic operational winner

Reference:

- [phase1_qmatrix_rpfa_tuning.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_tuning.md)
- [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)

## Sensitivity reading

### Fractional multi-KC sensitivity

The richer heterogeneity result survives the like-for-like fractional KC allocation sensitivity on the same rows and split.

### Single-KC sensitivity

The restrictive single-KC branch still collapses to Model 1.

Current interpretation:

- the single-KC branch remains useful as a construct-clean sensitivity check
- it is not the controlling mainline because it discards too much repeated structure
- the current unresolved issue is not “is there any richer signal?”
- it is “how much do the richer effect sizes move under different KC constructions?”

## Offline adaptive-policy replay status

The repo now includes an offline next-question replay suite on the explicit Q-matrix PFA / R-PFA branch.

Current policy family:

- balanced challenge
- harder challenge
- confidence-building
- failure-aware remediation
- spacing-aware review

Current high-level reading:

- `balanced_challenge`:
  - Model 3 has a tiny overall target-gap edge
  - Model 2 has slightly better policy advantage and clearly better stability
- `confidence_building`:
  - Model 2 is better on target gap, policy advantage, and stability
- `failure_aware_remediation`:
  - Model 2 is slightly better on target gap, band-hit rate, policy advantage, and stability
- `harder_challenge`:
  - Model 3 has a tiny policy-advantage edge
  - Model 2 is still better on target gap and stability
- `spacing_aware_review`:
  - Model 3 has a tiny band-hit edge
  - Model 2 is better on target gap, policy advantage, and stability

Interpretation:

- the policy layer is still an **offline target-control / policy-behavior** evaluation
- Model 2 remains the default policy model unless Model 3 wins on the policy-facing metrics
- Model 3 does not currently clear that bar

Reference:

- [adaptive_policy_suite_comparison.md](D:/model1_baseline_agent_bundle/reports/adaptive_policy_suite_comparison.md)

## Current repo focus

Until local data is available, the practical mainline is:

1. keep the heterogeneity ladder scientifically coherent on the full dataset
2. use **explicit Q-matrix R-PFA Model 2** as the operational learner-model mainline
3. keep **R-PFA Model 3** as the richer challenger
4. evaluate question-selection policies offline

## Phase 2 status

Phase 2 remains scaffolded but paused.

Why:

- no local dataset is currently available in this workspace
- the immediate work is public-data learner modeling plus offline policy evaluation

Phase 2 scaffolding still present:

- [phase2_protocol.md](D:/model1_baseline_agent_bundle/reports/phase2_protocol.md)
- [preprocess_phase2_local.py](D:/model1_baseline_agent_bundle/src/preprocess_phase2_local.py)
- [split_phase2_local.py](D:/model1_baseline_agent_bundle/src/split_phase2_local.py)
