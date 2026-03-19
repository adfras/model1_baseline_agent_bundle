# Local Uncertainty Policy Suite Decision

This note records the **corrected** residual-heterogeneity policy-alignment restart.

It supersedes the earlier invalid version of this branch. The earlier implementation wrote the same actual-next rows under each `policy_name`, so the supposed policy-specific calibrators were mostly seeing the same training examples. The corrected rerun fixes that by selecting calibration rows inside each policy's own probability band.

This corrected rerun is the one that should be used.

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

Compared with the earlier calibration-side-channel work, this branch adds three stricter pieces:

1. a deterministic **KC-constrained unseen slate**
2. **local residual / friction / self-report features** built from prior attempts only
3. **policy-specific logistic calibrators** trained on actual-next rows from calibration students only

The candidate-slate rule is:

1. unseen items sharing at least one KC with the learner's top-5 frontier KCs over the last 10 attempts
2. if that produced fewer than `15` items, relax to unseen items sharing any KC seen in the last 10 attempts
3. if that still produced fewer than `15` items, fall back to the full unseen pool

The frontier score is:

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

## Policy-specific calibration fix

In the corrected rerun, the three policies now use genuinely different calibration subsets.

All three policies stayed in the strictest training mode:

- `balanced_challenge`: `unseen_in_slate_and_band`, calibration rows `1074`, evaluation rows `1058`
- `harder_challenge`: `unseen_in_slate_and_band`, calibration rows `569`, evaluation rows `516`
- `confidence_building`: `unseen_in_slate_and_band`, calibration rows `1093`, evaluation rows `1066`

So the policy-specific branch is now methodologically valid for the policy-band question. The earlier duplicated-row version is not.

## Actual-next calibration reading

On logged actual-next items, the strongest calibration-loss method is still the simple per-policy calibrator, not the residual-heavy branches.

By policy:

- `balanced_challenge`
  - raw log loss `0.612979`
  - policy-band calibrated `0.608794`
  - residual-only `0.610331`
  - residual + Model 3 `0.610223`
- `harder_challenge`
  - raw log loss `0.682259`
  - policy-band calibrated `0.685202`
  - residual-only `0.694567`
  - residual + Model 3 `0.692185`
- `confidence_building`
  - raw log loss `0.438630`
  - policy-band calibrated `0.438183`
  - residual-only `0.440243`
  - residual + Model 3 `0.440603`

So the corrected branch still contains calibration information, but the pure policy-band recalibrator is the best actual-next calibration-loss method in two of the three policy bands, and the residual-heavy branches are not the winner.

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
- residual + Model 3 target gap `1-10`: `0.007829`
- delta: `+0.000162`

- raw policy advantage `1-10`: `0.198166`
- residual + Model 3 policy advantage `1-10`: `0.153816`
- delta: `-0.044349`

- raw stability: `0.003332`
- residual + Model 3 stability: `0.007047`
- delta: `+0.003715`

- raw band-hit rate `1-10`: `0.990827`
- residual + Model 3 band-hit rate `1-10`: `0.974011`
- delta: `-0.016816`

So the corrected primary-policy result is still bad:

- target precision gets worse
- policy advantage gets much worse
- band-hit rate gets worse
- stability gets much worse

### Pooled new-learning summary

Across `balanced_challenge`, `harder_challenge`, and `confidence_building`:

- raw mean target gap `1-10`: `0.010476`
- residual + Model 3 mean target gap `1-10`: `0.011166`
- delta: `+0.000690`

- raw mean policy advantage `1-10`: `0.190638`
- residual + Model 3 mean policy advantage `1-10`: `0.133794`
- delta: `-0.056844`

- raw mean stability: `0.004233`
- residual + Model 3 mean stability: `0.009808`
- delta: `+0.005577`

So after fixing the policy-specific training contamination, the branch actually fails **more clearly**:

- pooled target gap is worse, not better
- pooled policy advantage is much worse
- pooled stability is still far worse than the tolerance allows

## Ablation reading

Model 3 still does not earn an operational role here.

Residual-only vs residual + Model 3:

- mean target gap delta `1-10`: `+0.000130`
- mean policy advantage delta `1-10`: `+0.000443`
- mean stability delta: `+0.000193`

That is still operationally negligible. The main effect comes from the local-residual branch itself, and it is not a good effect operationally.

Policy-specific reading:

- `balanced_challenge`: residual branches are slightly worse on target gap
- `harder_challenge`: residual branches improve target gap materially
- `confidence_building`: residual branches are clearly worse on target gap and stability

Because `confidence_building` is the primary policy to beat, that is enough to keep the branch out of the operational path.

## Interpretation

This corrected restart gives residual heterogeneity a fairer operational test than the earlier global uncertainty side-channel:

- narrower KC-aware candidate slate
- local residual features
- policy-specific calibrators

It still does **not** produce the win the project needs.

The clean reading is:

- raw Model 2 remains the best operational fixed-policy input on DBE
- the earlier duplicated-row version overstated how policy-specific this branch was
- once corrected, the branch still fails, and it fails more clearly
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
- [actual_next_calibration_summary.json](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/local_uncertainty_policy_suite_qmatrix_rpfa/actual_next_calibration_summary.json)
