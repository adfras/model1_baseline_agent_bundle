# Local Uncertainty Policy Suite Decision

This note records the residual-heterogeneity policy-alignment restart on a KC-constrained unseen candidate slate.

The question was:

**Can policy-specific calibration, using local residual features and Model 3 uncertainty on a KC-constrained unseen slate, beat the current raw Model 2 fixed-policy baseline on the actual next-question decision problem?**

## Frozen baseline

The restart kept the learner-model stack fixed:

- scorer family: explicit Q-matrix **R-PFA Model 2**
- recency: `alpha = 0.9`
- review threshold: `24` hours
- new-learning policies:
  - `balanced_challenge`
  - `harder_challenge`
  - `confidence_building`

Review stayed outside this decision. The branch only reran the three fixed new-learning policies.

## What changed

Compared with the earlier calibration-side-channel work, this branch added three stricter pieces:

1. a deterministic **KC-constrained unseen slate**
2. **local residual / friction / self-report features** built from prior attempts only
3. **policy-specific logistic calibrators** trained on actual-next rows from calibration students only

The candidate-slate rule was:

1. unseen items sharing at least one KC with the learner's top-5 frontier KCs over the last 10 attempts
2. if that produced fewer than `15` items, relax to unseen items sharing any KC seen in the last 10 attempts
3. if that still produced fewer than `15` items, fall back to the full unseen pool

The frontier score was:

- `failure_decay + 0.5 * success_decay`

using the existing `alpha = 0.9` KC-opportunity-lag decay.

## Methods compared

On the same KC-constrained slate, the branch compared four methods:

1. `model2_raw`
2. `policy_band_calibrated`
3. `policy_band_plus_local_residuals`
4. `policy_band_plus_local_residuals_plus_model3`

The operational baseline was:

- `model2_raw`

The primary challenger was:

- `policy_band_plus_local_residuals_plus_model3`

The key ablation was:

- `policy_band_plus_local_residuals`

because it answers whether local residual alignment helps even if Model 3 adds nothing.

## Slate behavior

The slate was narrower than the full unseen pool, but still not especially tight.

Across the rerun:

- mean candidate count: `62.246`
- slate fallback rate: `0.7965`

So most decision rows still needed one of the two fallback stages. That matters when interpreting the limited operational headroom.

## Actual-next calibration reading

On logged actual-next items, the strongest calibration-loss method was the simple per-policy calibrator, not the residual-heavy branches:

- `model2_raw`:
  - log loss `0.519800`
  - Brier `0.173888`
  - calibration slope `0.943522`
- `policy_band_calibrated`:
  - log loss `0.518609`
  - Brier `0.173404`
  - calibration slope `1.019285`
- `policy_band_plus_local_residuals`:
  - log loss `0.519511`
  - Brier `0.173800`
  - calibration slope `0.997533`
- `policy_band_plus_local_residuals_plus_model3`:
  - log loss `0.519587`
  - Brier `0.173834`
  - calibration slope `0.996916`

So there is calibration information in the restart branch, but the pure policy-band recalibrator was the best actual-next calibration-loss method.

## Hard operational gate

The restart only counts as a real success if the primary challenger beats raw Model 2 on all of:

1. lower `confidence_building` target gap `1-10`
2. lower pooled mean target gap `1-10`
3. `confidence_building` stability worsening by no more than `0.001`
4. pooled mean stability worsening by no more than `0.001`

## Result

The restart **fails** the operational gate.

### Primary policy: `confidence_building`

- raw target gap `1-10`: `0.007667`
- residual + Model 3 target gap `1-10`: `0.008901`
- delta: `+0.001234`

- raw policy advantage `1-10`: `0.198166`
- residual + Model 3 policy advantage `1-10`: `0.211193`
- delta: `+0.013027`

- raw stability: `0.003332`
- residual + Model 3 stability: `0.007862`
- delta: `+0.004530`

- raw band-hit rate `1-10`: `0.990827`
- residual + Model 3 band-hit rate `1-10`: `0.978024`
- delta: `-0.012803`

So the primary policy gets better policy advantage, but it gets **worse** target precision, **worse** band-hit rate, and a much larger stability penalty than the tolerance allows.

### Pooled new-learning summary

Across `balanced_challenge`, `harder_challenge`, and `confidence_building`:

- raw mean target gap `1-10`: `0.010476`
- residual + Model 3 mean target gap `1-10`: `0.010377`
- delta: `-0.000100`

- raw mean policy advantage `1-10`: `0.190638`
- residual + Model 3 mean policy advantage `1-10`: `0.191932`
- delta: `+0.001294`

- raw mean stability: `0.004233`
- residual + Model 3 mean stability: `0.009810`
- delta: `+0.005577`

So the pooled target gap improves slightly, but the stability penalty is far too large. The branch fails even before any tie-break.

## Ablation reading

Model 3 still does not earn an operational role here.

Residual-only vs residual + Model 3:

- mean target gap delta `1-10`: `+0.000001`
- mean policy advantage delta `1-10`: `+0.000012`
- mean stability delta: `-0.000055`

That is operationally negligible. The change from local residual alignment was much larger than the change from adding Model 3 on top of it.

Policy-specific reading:

- `balanced_challenge`: residual branches are slightly worse on target gap
- `harder_challenge`: residual branches improve target gap materially
- `confidence_building`: residual branches are clearly worse on target gap and stability

Because `confidence_building` is the primary policy to beat, that is enough to keep the branch out of the operational path.

## Interpretation

This restart gave residual heterogeneity a fairer operational test than the earlier global uncertainty side-channel:

- narrower KC-aware candidate slate
- local residual features
- policy-specific calibrators

But it still does **not** produce the win the project needs.

The clean reading is:

- raw Model 2 remains the best operational fixed-policy input on DBE
- local residual alignment changes recommendations, but not in the right way for the primary policy
- Model 3 uncertainty does not add enough beyond those local residual features to justify an operational role

## Operational conclusion

Keep the frozen baseline:

- scorer: explicit Q-matrix **R-PFA Model 2**
- policy input: **raw Model 2 probabilities**
- recency: `alpha = 0.9`
- review mode: `spacing_aware_review` at `24` hours
- default new-learning choice: fixed `confidence_building`
- `balanced_challenge`: comparator / later-step reference
- `harder_challenge`: benchmark only
- `failure_aware_remediation`: not in the default path
- Model 3: scientific heterogeneity model and exploratory uncertainty source only

For DBE, residual heterogeneity remains scientifically interesting and calibration-relevant, but this restart does **not** turn it into an operational next-question-selection win.

Reference outputs:

- [policy_suite_summary.json](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/local_uncertainty_policy_suite_qmatrix_rpfa/policy_suite_summary.json)
- [policy_suite_comparison.csv](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/local_uncertainty_policy_suite_qmatrix_rpfa/policy_suite_comparison.csv)
- [actual_next_calibration_summary.json](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/local_uncertainty_policy_suite_qmatrix_rpfa/actual_next_calibration_summary.json)
- [policy_suite_rows.csv](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/local_uncertainty_policy_suite_qmatrix_rpfa/policy_suite_rows.csv)
- [policy_training_rows.csv](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/local_uncertainty_policy_suite_qmatrix_rpfa/policy_training_rows.csv)
