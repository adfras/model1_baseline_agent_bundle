# Hybrid Uncertainty Router V2

This note records the second hybrid-router generation built on top of the selected **R-PFA Model 2** scorer and **R-PFA Model 3** uncertainty signal.

## What changed from v1

Router v2 adds lagged observable proxies before each held-out recommendation step:

- failure streak
- recent attempt-level success rate
- recent hint-use rate
- recent answer-change friction rate
- response-time inflation relative to the learner's prior median
- `24`-hour due-review threshold for spacing mode

The scorer is unchanged:

- **Model 2** still provides the mean correctness estimate
- **Model 3** still provides the uncertainty signal

The router is now recorded in two forms:

- **raw v2**: the first threshold set, which over-routed into non-balanced modes
- **tuned v2**: a more conservative threshold set that is now the active v2 prototype

## Overall comparison

- v1 target gap `1-10`: `0.01223`
- raw v2 target gap `1-10`: `0.01404`
- tuned v2 target gap `1-10`: `0.01108`
- v1 policy advantage `1-10`: `0.19082`
- raw v2 policy advantage `1-10`: `0.19481`
- tuned v2 policy advantage `1-10`: `0.19473`
- v1 stability: `0.01439`
- raw v2 stability: `0.04055`
- tuned v2 stability: `0.03452`
- v1 recent-failure coverage: `0.2884`
- raw v2 recent-failure coverage: `0.3625`
- tuned v2 recent-failure coverage: `0.3297`
- v1 due-review coverage: `0.2404`
- raw v2 due-review coverage: `0.1923`
- tuned v2 due-review coverage: `0.2204`
- v1 seen-item rate: `0.1416`
- raw v2 seen-item rate: `0.0645`
- tuned v2 seen-item rate: `0.0920`

Comparison against the fixed Model 2 policies:

- balanced challenge target gap `1-10`: `0.00498`
- confidence-building target gap `1-10`: `0.00483`
- harder challenge target gap `1-10`: `0.00642`

Interpretation:

- raw v2 made coverage and policy-advantage gains, but clearly over-routed and destabilized recommendations
- tuned v2 recovered target-gap control and kept the policy-advantage gains
- tuned v2 still remains much less stable than v1 and much less target-precise than the fixed Model 2 policies
- so tuned v2 is the current **exploratory router prototype**, not the default operational policy

This remains an **offline target-control / policy-behavior** result, not a causal learning-gain estimate.

## Tuned v2 route shares

- `balanced_challenge`: `35.84%`
- `confidence_building`: `33.45%`
- `diagnostic_challenge`: `10.51%`
- `failure_aware_remediation`: `8.29%`
- `spacing_aware_review`: `11.92%`

## Tuned v2 route reasons

- `acute_failure_or_friction`: `877`
- `default_balanced`: `3793`
- `due_review_ready`: `1261`
- `high_uncertainty`: `1112`
- `mild_struggle_or_low_proficiency`: `3540`

## Tuned v2 route-level summaries

### balanced_challenge

- target gap `1-10`: `0.00482`
- policy advantage `1-10`: `0.17725`
- band-hit rate `1-10`: `1.0000`
- stability: `0.00085`
- recent-failure coverage: `0.1701`
- due-review coverage: `0.0778`
- seen-item rate: `0.0000`
- fallback rate: `0.0000`

### confidence_building

- target gap `1-10`: `0.00519`
- policy advantage `1-10`: `0.22130`
- band-hit rate `1-10`: `0.9884`
- stability: `0.00153`
- recent-failure coverage: `0.2828`
- due-review coverage: `0.1316`
- seen-item rate: `0.0000`
- fallback rate: `0.0000`

### diagnostic_challenge

- target gap `1-10`: `0.00767`
- policy advantage `1-10`: `0.24221`
- band-hit rate `1-10`: `0.9919`
- stability: `0.00081`
- recent-failure coverage: `0.1664`
- due-review coverage: `0.1079`
- seen-item rate: `0.0000`
- fallback rate: `0.0000`

### failure_aware_remediation

- target gap `1-10`: `0.05967`
- policy advantage `1-10`: `0.14823`
- band-hit rate `1-10`: `0.7959`
- stability: `0.02924`
- recent-failure coverage: `0.9852`
- due-review coverage: `0.2178`
- seen-item rate: `0.0000`
- fallback rate: `0.0148`

### spacing_aware_review

- target gap `1-10`: `0.00963`
- policy advantage `1-10`: `0.16320`
- band-hit rate `1-10`: `0.9921`
- stability: `0.00051`
- recent-failure coverage: `0.6297`
- due-review coverage: `1.0000`
- seen-item rate: `0.7724`
- fallback rate: `0.0000`


## Tuned v2 subgroup summaries

### actual_multi_kc

- target gap `1-10`: `0.01036`
- policy advantage `1-10`: `0.18650`
- band-hit rate `1-10`: `0.9855`
- stability: `0.04145`

### actual_single_kc

- target gap `1-10`: `0.01263`
- policy advantage `1-10`: `0.20525`
- band-hit rate `1-10`: `0.9671`
- stability: `0.04329`

### early_steps_1_5

- target gap `1-10`: `0.01063`
- policy advantage `1-10`: `0.19147`
- band-hit rate `1-10`: `0.9788`
- stability: `0.03407`

### high_friction_context

- target gap `1-10`: `0.02306`
- policy advantage `1-10`: `0.17915`
- band-hit rate `1-10`: `0.9356`
- stability: `0.03447`

### high_recent_failure_context

- target gap `1-10`: `0.02285`
- policy advantage `1-10`: `0.21909`
- band-hit rate `1-10`: `0.9252`
- stability: `0.03937`

### later_steps_6_10

- target gap `1-10`: `0.01188`
- policy advantage `1-10`: `0.19835`
- band-hit rate `1-10`: `0.9759`
- stability: `0.03426`

### lower_predicted_proficiency

- target gap `1-10`: `0.04568`
- policy advantage `1-10`: `0.28889`
- band-hit rate `1-10`: `0.8966`
- stability: `0.03977`

### review_eligible_context

- target gap `1-10`: `0.01292`
- policy advantage `1-10`: `0.18760`
- band-hit rate `1-10`: `0.9747`
- stability: `0.03432`

