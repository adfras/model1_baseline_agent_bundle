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
- policy-facing alpha comparison between `0.8` and `0.9` also kept `0.9`

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
- on the policy-facing comparison, the new-learning target-gap difference between `0.8` and `0.9` stayed inside the tie margin while `0.9` was slightly better on policy advantage and stability
- the overall branch calibration slope is closer to `1.0` for Model 3, but that alone is not enough to treat it as the operational model

Reference:

- [phase1_qmatrix_rpfa_tuning.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_tuning.md)
- [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)
- [phase1_qmatrix_rpfa_policy_alpha_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_policy_alpha_comparison.md)

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

## Policy-alignment calibration check

Because probability thresholds are what drive the policy layer, the repo now also checks the narrower question:

- does residual heterogeneity improve calibration on the logged actual-next items in the policy contexts that matter operationally?

Current reading:

- calibration is the right lens to check for policy alignment
- but the current logged actual-next evaluation does **not** show a policy-context calibration advantage for Model 3
- across all logged rows, Model 2 is better on Brier, log loss, and calibration slope
- the same pattern mostly holds in:
  - early-step contexts
  - confidence-trigger contexts
  - balanced-default contexts
  - high-friction contexts
- the review-due and high-failure contexts show only mixed, very small differences, not a clear Model 3 win

Interpretation:

- Model 3 remains important as the **scientific** stability / residual-heterogeneity model
- Model 3 also remains the right **exploratory** uncertainty / calibration challenger
- but the current policy-context calibration evidence still keeps **Model 2** as the operational scorer

Reference:

- [policy_alignment_calibration.md](D:/model1_baseline_agent_bundle/reports/policy_alignment_calibration.md)

## Uncertainty calibration layer

The repo now also includes the next, more useful calibration question:

- can Model 3 help if it is used as a **calibration side-channel** on top of Model 2, rather than as a replacement scorer?

Current answer:

- yes, in a small but real way
- a banded uncertainty calibrator beats:
  - raw Model 2
  - Platt-calibrated Model 2
  - context-only calibrated Model 2
- held-out evaluation-student results:
  - context-only calibrator log loss `0.516384`
  - uncertainty-side-channel calibrator log loss `0.516209`
  - context-only Brier `0.172574`
  - uncertainty-side-channel Brier `0.172524`

Interpretation:

- this is the first actual calibration win from residual heterogeneity in the repo
- the win comes from using **Model 3 uncertainty** to calibrate **Model 2**
- not from replacing Model 2 with raw Model 3
- the improvement is broad across the main policy contexts, but target-band gains are mixed and the fixed policy suite has not yet been rerun with these recalibrated probabilities

Reference:

- [uncertainty_calibration_layer.md](D:/model1_baseline_agent_bundle/reports/uncertainty_calibration_layer.md)

## Spacing review-mode tuning

Spacing review has now been tuned as its own review-mode problem on the operational Model 2 branch.

Current selected threshold:

- due-review hours: `24`

Why it won:

- best review-eligible target gap: `0.00863`
- highest review-eligible rate: `0.6539`

For comparison:

- `48` hours had review-eligible target gap `0.00995` and eligibility `0.5706`
- `72` hours had review-eligible target gap `0.00992` and eligibility `0.5135`
- `96` hours had review-eligible target gap `0.00986` and eligibility `0.4837`

Interpretation:

- the shared Model 2 vs Model 3 suite is still the main fixed-policy baseline
- the current **review-mode** setting to carry forward is `24` hours for `spacing_aware_review`

Reference:

- [spacing_policy_due_review_grid.md](D:/model1_baseline_agent_bundle/reports/spacing_policy_due_review_grid.md)

## Policy subgroup diagnostics

The repo now includes a policy heterogeneity pass on the **current operational suite**:

- the four new-item policies come from the fixed Model 2 R-PFA suite
- `spacing_aware_review` is replaced with the selected `24`-hour review branch

Current reading:

- there is **no single universal best policy**
- `confidence_building` has the smallest target gap overall, on early steps, and on multi-KC items
- `balanced_challenge` has the smallest target gap later in the sequence, on single-KC items, and in the higher-friction and lower-proficiency contexts
- `harder_challenge` most often wins on policy advantage
- `failure_aware_remediation` and `spacing_aware_review` still look like separate service modes rather than global winners

Interpretation:

- the subgroup diagnostics are useful for understanding where each fixed policy helps
- review and remediation should stay distinct modes
- any later router should stay conservative and be judged against the fixed-suite baseline

A first conservative router v3 attempt was then tried and rejected.

Why it was rejected:

- target gap `1-10`: `0.01633`
- policy advantage `1-10`: `0.17335`
- stability: `0.02701`

That was worse than:

- hybrid v1
- tuned hybrid v2
- and the fixed `balanced_challenge` baseline

Interpretation:

- the subgroup diagnostics are useful
- but the first attempt to turn them into a single conservative router still degraded the main policy metrics
- so the repo remains on the fixed suite plus subgroup diagnostics rather than carrying a v3 router branch forward

Reference:

- [policy_subgroup_diagnostics.md](D:/model1_baseline_agent_bundle/reports/policy_subgroup_diagnostics.md)
- [conservative_router_v3_attempt.md](D:/model1_baseline_agent_bundle/reports/conservative_router_v3_attempt.md)

## Simple two-mode router pass

The repo now also includes the requested simple default router:

- if review is due -> `spacing_aware_review`
- else if early step / low predicted proficiency / high recent failure / high friction -> `confidence_building`
- else -> `balanced_challenge`

This pass kept the scorer frozen at:

- **R-PFA Model 2**
- `alpha = 0.9`
- `24`-hour review threshold

Best threshold set from the small grid:

- early-step cutoff `5`
- low-proficiency threshold at the `30%` quantile (`0.71797`)
- recent-failure threshold at the `75%` quantile (`45.24461`)
- `current` friction rule

Result:

- router new-learning target gap `1-10`: `0.002871`
- fixed `confidence_building` new-learning target gap `1-10`: `0.002918`
- delta: `-0.000047`

But the router then lost on the tie-break metrics:

- policy advantage delta vs fixed `confidence_building`: `-0.004489`
- stability delta vs fixed `confidence_building`: `+0.009543`

Interpretation:

- the target-gap gain is too small to justify the stability blow-up
- the simple router is therefore **not** promoted to the operational default
- `spacing_aware_review` remains a separate review service mode
- `failure_aware_remediation` stays out of the default path
- the current frozen default new-learning choice is **fixed `confidence_building`**

Reference:

- [simple_two_mode_router_decision_memo.md](D:/model1_baseline_agent_bundle/reports/simple_two_mode_router_decision_memo.md)

## Uncertainty-aware routing prototypes

The repo now also includes a first hybrid router that uses:

- **R-PFA Model 2** for the mean correctness estimate
- **R-PFA Model 3** for a step-level uncertainty signal

Current result:

- the hybrid router improves recent-failure and due-review coverage relative to staying in balanced challenge all the time
- but it does **not** improve pure target-gap control or stability relative to the fixed-policy suite

Interpretation:

- Model 3 uncertainty is useful as a **routing signal**
- but the current hybrid is a prototype, not a new default operational policy

Reference:

- [hybrid_uncertainty_router.md](D:/model1_baseline_agent_bundle/reports/hybrid_uncertainty_router.md)

The repo now also includes a second-generation hybrid router with lagged observable proxies and the selected `24`-hour review threshold.

Current reading:

- raw v2 thresholds were too aggressive and worsened both target gap and stability
- tuned v2 improved over v1 on:
  - target gap `1-10`: `0.01108` vs `0.01223`
  - policy advantage `1-10`: `0.19473` vs `0.19082`
  - recent-failure coverage: `0.3297` vs `0.2884`
  - seen-item rate: `0.0920` vs `0.1416`
- tuned v2 still remained worse than v1 on:
  - stability: `0.03452` vs `0.01439`
  - due-review coverage: `0.2204` vs `0.2404`
- tuned v2 also remains much less target-precise than the fixed new-item policies

Interpretation:

- tuned v2 is the current **exploratory** hybrid router
- it is useful for policy-routing diagnostics and subgroup analysis
- it still does **not** replace the fixed-policy suite or the simpler hybrid v1 as the default operational baseline

Reference:

- [hybrid_uncertainty_router_v2.md](D:/model1_baseline_agent_bundle/reports/hybrid_uncertainty_router_v2.md)

## Current repo focus

Until local data is available, the practical mainline is:

1. keep the heterogeneity ladder scientifically coherent on the full dataset
2. use **explicit Q-matrix R-PFA Model 2** as the operational learner-model mainline
3. keep **R-PFA Model 3** as the richer challenger
4. use `24` hours as the current spacing-review threshold for review-mode experiments
5. evaluate question-selection policies offline
6. use uncertainty mainly for routing experiments, not as the main predictor
7. keep fixed `confidence_building` as the default new-learning policy under the frozen scorer
8. keep `balanced_challenge` and `harder_challenge` as comparators rather than the default
9. treat tuned hybrid router v2 as an exploratory policy-gating branch, not the default
10. treat the first conservative router v3 attempt and the later simple two-mode router as informative but non-promoted routing experiments

## Phase 2 status

Phase 2 remains scaffolded but paused.

Why:

- no local dataset is currently available in this workspace
- the immediate work is public-data learner modeling plus offline policy evaluation

Phase 2 scaffolding still present:

- [phase2_protocol.md](D:/model1_baseline_agent_bundle/reports/phase2_protocol.md)
- [preprocess_phase2_local.py](D:/model1_baseline_agent_bundle/src/preprocess_phase2_local.py)
- [split_phase2_local.py](D:/model1_baseline_agent_bundle/src/split_phase2_local.py)
