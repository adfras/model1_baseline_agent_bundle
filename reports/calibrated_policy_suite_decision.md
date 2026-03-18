# Calibrated Policy Suite Decision

This note records the hard operational test for the calibration side-channel.

The question was:

**Does the Model 3 uncertainty calibration layer make the fixed next-question policies better enough to keep operationally?**

If not, it fails for the adaptive-question objective.

## Frozen stack

The test keeps the learner-model branch fixed:

- scorer family: explicit Q-matrix **R-PFA Model 2**
- recency: `alpha = 0.9`
- review threshold: `24` hours
- policies tested:
  - `balanced_challenge`
  - `harder_challenge`
  - `confidence_building`

Methods compared on the same held-out evaluation students:

1. `model2_raw`
2. `model2_context_calibrated`
3. `model2_plus_model3_uncertainty`

The operational challenger was:

- baseline: `model2_context_calibrated`
- challenger: `model2_plus_model3_uncertainty`

The primary policy was:

- `confidence_building`

## Hard gate

The calibration side-channel only survives if all three are true:

1. `confidence_building` target gap `1-10` improves
2. mean new-learning target gap across the three fixed new-learning policies improves
3. `confidence_building` stability does not worsen by more than `0.001`

## Result

The side-channel **fails** the operational gate.

Primary policy comparison (`confidence_building`):

- context-calibrated target gap `1-10`: `0.005736`
- uncertainty-side-channel target gap `1-10`: `0.005724`
- delta: `-0.000012`

- context-calibrated policy advantage `1-10`: `0.211385`
- uncertainty-side-channel policy advantage `1-10`: `0.211338`
- delta: `-0.000047`

- context-calibrated stability: `0.003073`
- uncertainty-side-channel stability: `0.003271`
- delta: `+0.000198`

So the primary policy gets a tiny target-gap gain, but loses on policy advantage and is less stable.

Mean new-learning comparison across:

- `balanced_challenge`
- `harder_challenge`
- `confidence_building`

Results:

- context-calibrated mean target gap `1-10`: `0.005535`
- uncertainty-side-channel mean target gap `1-10`: `0.005558`
- delta: `+0.000023`

- context-calibrated mean policy advantage `1-10`: `0.194770`
- uncertainty-side-channel mean policy advantage `1-10`: `0.194773`
- delta: `+0.000003`

- context-calibrated mean stability: `0.003091`
- uncertainty-side-channel mean stability: `0.003347`
- delta: `+0.000256`

So the challenger fails the mean target-gap condition and is also less stable.

## Stronger practical reading

The harder truth is that the raw scorer is still better for the fixed-policy job.

Raw Model 2 means:

- `balanced_challenge` target gap `1-10`: `0.004932`
- `confidence_building` target gap `1-10`: `0.004811`
- `harder_challenge` target gap `1-10`: `0.006537`
- mean new-learning target gap `1-10`: `0.005427`
- mean stability: `0.000850`

Those are all better than the two calibrated policy variants on the pooled fixed-policy decision problem.

So the calibration side-channel is not just “not the winner.” For the current operational objective, it is a failure.

## Interpretation

This resolves the ambiguity:

- the uncertainty side-channel does produce a small held-out calibration-loss win on logged actual-next rows
- but that win does **not** survive the fixed-policy rerun that matters operationally

So the repo should now say plainly:

- **Model 2 raw probabilities remain the operational policy input**
- **Model 3 uncertainty remains scientifically relevant and calibration-relevant**
- but **the current uncertainty calibration layer is not kept operationally**

## Operational freeze

Current operational policy stack:

- scorer: explicit Q-matrix **R-PFA Model 2**
- policy input: **raw Model 2 probabilities**
- recency: `alpha = 0.9`
- review mode: `spacing_aware_review` at `24` hours
- default new-learning choice: fixed `confidence_building`
- `balanced_challenge`: comparator / later-step reference
- `harder_challenge`: benchmark only
- `failure_aware_remediation`: not in the default path
- Model 3: scientific heterogeneity model and exploratory uncertainty layer only

Reference outputs:

- [policy_suite_summary.json](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/calibrated_policy_suite_qmatrix_rpfa/policy_suite_summary.json)
- [policy_suite_comparison.csv](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/calibrated_policy_suite_qmatrix_rpfa/policy_suite_comparison.csv)
- [policy_suite_rows.csv](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/calibrated_policy_suite_qmatrix_rpfa/policy_suite_rows.csv)
