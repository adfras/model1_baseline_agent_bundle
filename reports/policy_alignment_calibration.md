# Policy Alignment Calibration

This note checks the point that matters for policy alignment:

- residual heterogeneity may still be useful even if Model 3 is not the best mean scorer
- policies act on predicted probabilities, so calibration and uncertainty matter directly

The question here is therefore narrower than the earlier policy-suite comparison:

**Does Model 3 improve calibration on logged actual-next items in the policy contexts that the current router logic cares about?**

## Setup

- learner-model family: explicit Q-matrix **R-PFA**
- selected operational recency setting: `alpha = 0.9`
- comparison:
  - **Model 2** as the mean-scoring baseline
  - **Model 3** as the residual-heterogeneity challenger
- evaluation target:
  - logged actual-next held-out attempts only
  - one row per observed next attempt
- source policy rows:
  - fixed-suite `balanced_challenge` rows, used here only as the common logged actual-next prediction source
- policy-context definitions:
  - selected simple-router thresholds from [router_selected_summary.json](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/simple_two_mode_router_qmatrix_rpfa/router_selected_summary.json)
  - early-step cutoff `5`
  - low-proficiency threshold `0.71797`
  - recent-failure threshold `45.24461`
  - friction rule `current`

Metrics:

- Brier score
- log loss
- calibration intercept
- calibration slope

This is still an **offline policy-alignment calibration check**, not a causal policy evaluation.

## Main result

Calibration was the right lens to check, but the current evidence still does **not** show a Model 3 advantage in the policy contexts tested.

Across all logged actual-next rows:

- Model 2:
  - Brier `0.174416`
  - log loss `0.521888`
  - calibration slope `0.934142`
- Model 3:
  - Brier `0.174621`
  - log loss `0.522447`
  - calibration slope `0.929216`

So on the full logged policy-alignment sample, Model 3 is slightly worse on all three primary summary metrics.

## Context-specific reading

The same pattern mostly holds inside the policy contexts that matter to the current routing logic.

### Early steps `1-5`

- Model 2 slope `0.964497`
- Model 3 slope `0.953917`
- Model 3 also has slightly worse Brier and log loss

### Confidence-trigger context

- Model 2 slope `0.938408`
- Model 3 slope `0.932571`
- Model 3 also has slightly worse Brier and log loss

### Balanced-default context

- Model 2 slope `0.983191`
- Model 3 slope `0.973016`
- Model 3 is clearly worse on log loss here

### Review-due context

- Model 2 slope `0.935869`
- Model 3 slope `0.933000`
- Model 3 has a tiny Brier edge here
- but Model 3 still has slightly worse log loss and slightly worse slope distance to `1.0`

### Low predicted proficiency

- Model 2 slope `0.986509`
- Model 3 slope `0.982764`
- other metrics are also essentially flat-to-worse for Model 3

### High recent failure

- Model 2 slope `0.869556`
- Model 3 slope `0.864452`
- Model 3 gets a tiny log-loss edge here
- but Brier is slightly worse and calibration slope is still slightly worse

### High friction

- Model 2 slope `0.947963`
- Model 3 slope `0.941218`
- Model 3 also has slightly worse Brier and log loss

## Interpretation

The correction to the project focus is still important:

- residual heterogeneity **should** be judged partly through calibration and uncertainty
- not only through mean-prediction fit or router target gap

But the current repo evidence says:

- Model 3 is still the richest supported **scientific heterogeneity model**
- Model 3 is still the right **uncertainty / calibration challenger**
- the current logged actual-next calibration check does **not** show Model 3 improving policy alignment enough to replace Model 2 operationally

So the operational split should now be stated this way:

- **Model 2** = current mean scorer for next-question work
- **Model 3** = exploratory residual-heterogeneity layer for uncertainty-aware policy design

That is different from saying Model 3 is unimportant.

It means the current evidence supports:

- keeping residual heterogeneity central to the **question**
- but not overstating it as a current **operational win**

## Practical decision

Current operational freeze remains:

- scorer: explicit Q-matrix **R-PFA Model 2**
- `alpha = 0.9`
- review mode: `spacing_aware_review` with `24`-hour threshold
- default new-learning choice: fixed `confidence_building`
- Model 3: exploratory calibration / uncertainty layer only

Reference outputs:

- [policy_alignment_calibration_summary.json](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/policy_alignment_calibration/policy_alignment_calibration_summary.json)
- [policy_alignment_calibration_comparison.csv](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/policy_alignment_calibration/policy_alignment_calibration_comparison.csv)
