# Phase 1 Selection Memo

This memo is the current canonical selection note for the public DBE work.

It separates two decisions that had become tangled:

1. **scientific heterogeneity selection**
2. **operational model selection for offline question targeting**

## What Phase 1 now answers

Phase 1 still asks:

1. Do learners differ in baseline level?
2. Do learners also differ in growth rate?
3. Do learners also differ in stability around that growth?

But with no local dataset currently available, Phase 1 is also now being used to answer:

4. Which supported learner model is the best current base for offline next-question policies?

## Scientific selection: heterogeneity ladder

On the full-data explicit Q-matrix ladder:

- Model 1 log loss `0.545311`
- Model 2 log loss `0.544366`
- Model 3 log loss `0.543782`

Interpretation:

- Model 2 survives the hurdle against Model 1
- Model 3 survives the hurdle against Model 2
- the full-data public evidence supports baseline, growth, and stability heterogeneity

So the current **scientific** reading is:

- **Model 1**: hurdle benchmark only
- **Model 2**: supported growth model
- **Model 3**: richest supported heterogeneity model

Reference:

- [phase1_multikc_qmatrix_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_qmatrix_comparison.md)

## Operational selection: best-yield predictive model

The highest-yield improvement tried so far was not more latent-state tuning. It was better KC-history representation.

On the explicit Q-matrix PFA / R-PFA branch family:

- selected R-PFA alpha: `0.9`
- direct policy-facing comparison against `0.8` also kept `0.9`
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

- adding wins/fails per KC matters more than retuning the older opportunity-only structure
- recency tuning now sits on top of that PFA baseline for the operational branch
- the tie-broken operational alpha is `0.9`
- the later policy-facing alpha comparison also kept `0.9`
- Model 2 stays ahead of Model 3 on log loss, Brier, AUC, and accuracy
- Model 3 still offers a richer uncertainty/stability story and a better calibration slope, but it is not the automatic operational winner

Reference:

- [phase1_qmatrix_rpfa_tuning.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_tuning.md)
- [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)
- [phase1_qmatrix_rpfa_policy_alpha_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_policy_alpha_comparison.md)

## First policy selection result

The offline adaptive replay work is no longer limited to one simple rule.

It now supports a small modular suite:

- balanced challenge
- harder challenge
- confidence-building
- failure-aware remediation
- spacing-aware review

Current reading:

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

- the key decision is now policy-facing rather than purely predictive
- Model 2 remains the default policy model unless Model 3 clearly improves the policy metrics
- Model 3 does not currently clear that bar
- the replay remains an offline target-control / policy-behavior test, not a causal learning-gain estimate
- the fixed shared suite above used the common `48`-hour review threshold
- later spacing-only tuning on the operational Model 2 branch selected `24` hours for `spacing_aware_review`

Reference:

- [adaptive_policy_suite_comparison.md](D:/model1_baseline_agent_bundle/reports/adaptive_policy_suite_comparison.md)
- [spacing_policy_due_review_grid.md](D:/model1_baseline_agent_bundle/reports/spacing_policy_due_review_grid.md)
- [policy_subgroup_diagnostics.md](D:/model1_baseline_agent_bundle/reports/policy_subgroup_diagnostics.md)
- [hybrid_uncertainty_router_v2.md](D:/model1_baseline_agent_bundle/reports/hybrid_uncertainty_router_v2.md)

### Policy-heterogeneity note

The current operational suite has now also been compared by subgroup.

Current reading:

- there is no single universal best fixed policy
- `confidence_building` has the smallest target gap overall, on early steps, and on multi-KC items
- `balanced_challenge` has the smallest target gap later in the sequence, on single-KC items, and in the higher-friction and lower-proficiency contexts
- `harder_challenge` most often wins on policy advantage
- `failure_aware_remediation` and `spacing_aware_review` still behave more like service modes than global winners

So the policy implication is:

- keep the fixed Model 2 suite as the current baseline
- stop looking for one universal winner
- use the subgroup diagnostics as the basis for a conservative router v3

## Current decision

Keep the decisions separate and explicit.

### Scientific conclusion

On the full-data explicit-Q ladder:

- baseline heterogeneity is present
- growth heterogeneity is present
- stability heterogeneity is also present

So the richest supported public heterogeneity model is still **Model 3**.

### Operational conclusion

For current predictive yield and offline question-selection work:

- **explicit Q-matrix R-PFA Model 2** is the mainline model
- **explicit Q-matrix R-PFA Model 3** is the richer challenger
- keep `alpha = 0.9`
- use `24` hours as the current spacing-review threshold on the operational Model 2 branch

That is the current repo focus.

### Uncertainty-routing note

A first hybrid router using:

- Model 2 for the mean prediction
- Model 3 for a step-level uncertainty signal

does help route more steps into remediation, review, and diagnostic modes, but it does **not** beat the fixed-policy suite on pure target-gap control. So it remains a prototype rather than a replacement for the current operational default.

A second-generation router with lagged observable proxies and a `24`-hour review threshold is now also in the repo.

Current reading:

- raw v2 thresholds over-routed and degraded target-gap control
- tuned v2 improved target gap, policy advantage, and recent-failure coverage relative to v1
- tuned v2 still remained materially less stable than v1 and much less target-precise than the fixed-policy suite

So the current position is:

- fixed Model 2 policies remain the operational default
- hybrid v1 remains the more stable hybrid baseline
- tuned hybrid v2 is the current exploratory routing branch for later gating work

## Remaining caveats

- the restrictive single-KC sensitivity branch still collapses to Model 1
- the main robustness issue is how much effect sizes move under alternative KC handling
- the adaptive-policy replay so far is narrow:
  - target-difficulty control only
  - offline replay only
  - no causal learning-gain claim yet

## Practical reading

If the question is:

- **What is the richest public heterogeneity model currently supported?**
  - **Model 3**

- **What is the best current operational model for offline next-question targeting?**
  - **explicit Q-matrix R-PFA Model 2**
