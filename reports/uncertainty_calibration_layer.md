# Uncertainty Calibration Layer

This note records the next step after the direct `Model 2` vs `Model 3` policy-alignment calibration check.

That earlier check asked:

- does **raw Model 3** beat **raw Model 2** on logged actual-next calibration in policy contexts?

The answer was no.

This note asks the more useful question:

**Can Model 3 residual-heterogeneity information improve calibration when it is used as a side-channel on top of Model 2, rather than as a replacement scorer?**

## Setup

Base probability source:

- explicit Q-matrix **R-PFA Model 2**
- `alpha = 0.9`

Uncertainty side-channel:

- step-level **Model 3** uncertainty from the tuned hybrid-v2 branch
- specifically the `uncertainty_sd` field on the logged actual-next rows

Evaluation split:

- deterministic student-wise split of the replay rows
- `50%` calibration students
- `50%` evaluation students

Calibration candidates:

1. `model2_raw`
   - no extra calibration
2. `model2_platt`
   - intercept + slope on `logit(Model 2 probability)`
3. `model2_context_calibrated`
   - Platt baseline plus policy-context flags
4. `model2_plus_model3_uncertainty`
   - context calibrator plus **banded uncertainty**
   - uncertainty is split into low / mid / high bands using calibration-student tertiles
   - the model then learns separate probability slopes by uncertainty band

This is still an **offline calibration-layer check**, not a causal policy evaluation.

## Main result

This is the first actual calibration win from residual heterogeneity in the repo.

Held-out evaluation-student results:

- `model2_raw`
  - log loss `0.517910`
  - Brier `0.173076`
  - calibration slope `0.942263`
- `model2_platt`
  - log loss `0.516822`
  - Brier `0.172657`
  - calibration slope `1.017490`
- `model2_context_calibrated`
  - log loss `0.516384`
  - Brier `0.172574`
  - calibration slope `1.011692`
- `model2_plus_model3_uncertainty`
  - log loss `0.516209`
  - Brier `0.172524`
  - calibration slope `1.012617`

So relative to the strongest non-uncertainty baseline:

- log loss improves by `0.000176`
- Brier improves by `0.000050`

The calibration slope is effectively tied and is slightly farther from `1.0` than the context-only calibrator, so the win is mainly:

- **better probabilistic calibration loss**
- not a dramatic slope shift

## Context reading

The side-channel improvement is not limited to one context.

Against the context-only calibrator, the uncertainty-layer version improves log loss in:

- all rows: `-0.000176`
- early steps `1-5`: `-0.000240`
- confidence-trigger context: `-0.000146`
- balanced-default context: `-0.000486`
- review-due context: `-0.000223`
- high recent failure: `-0.000214`
- high friction: `-0.000137`

The Brier comparison shows the same direction in each of those contexts.

So the gain is small, but it is broad rather than isolated.

## Band reading

Target-band behavior is mixed, so this should be stated plainly.

Relative to the context-only calibrator:

- balanced band `[0.65, 0.80]`
  - absolute alignment gap improves from `0.02033` to `0.01743`
- harder band `[0.55, 0.65]`
  - absolute alignment gap worsens from `0.04956` to `0.05059`
- confidence band `[0.80, 0.90]`
  - absolute alignment gap worsens from `0.01255` to `0.01679`

So this is a **global calibration-loss win**, not a universal win on every policy band.

That matters because the current default new-learning policy is `confidence_building`.

## Interpretation

This is the clean split the repo should now use:

- **Model 2** remains the mean scorer
- **Model 3** should not replace Model 2 as the scorer
- but **Model 3 uncertainty now earns a real side-channel role** in the calibration layer

That means the earlier correction still stands:

- residual heterogeneity is important for policy alignment

The new evidence now adds:

- residual heterogeneity can produce a **small but real calibration win**
- but the right way to get that win is:
  - **Model 2 mean predictions**
  - plus **banded Model 3 uncertainty calibration**

not:

- replacing Model 2 with raw Model 3

## Practical reading

Current best calibration stack:

- scorer: explicit Q-matrix **R-PFA Model 2**
- recency: `alpha = 0.9`
- review threshold: `24` hours
- calibration layer: **banded Model 3 uncertainty side-channel**

Important limitation:

- the fixed-policy suite has **not** yet been rerun with these recalibrated probabilities
- so this note establishes a calibration win
- it does **not** yet establish a next-question policy win

Reference outputs:

- [uncertainty_calibration_layer_summary.json](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/uncertainty_calibration_layer/uncertainty_calibration_layer_summary.json)
- [uncertainty_calibration_layer_comparison.csv](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/uncertainty_calibration_layer/uncertainty_calibration_layer_comparison.csv)
- [uncertainty_calibration_layer_eval_rows.csv](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/uncertainty_calibration_layer/uncertainty_calibration_layer_eval_rows.csv)
