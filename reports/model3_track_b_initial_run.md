# Model 3 Track B Initial Run

This note records the first completed Track B check for Model 3.

This note is now superseded by the stricter rerun and combined summary in [model3_followup_checks.md](/D:/model1_baseline_agent_bundle/reports/model3_followup_checks.md). It should be read only as the first provisional screen.

## Scope

Track B is the priority comparison here, so this first Model 3 implementation focuses on unseen public students.

The implementation is a tractable approximation of the latent-state idea:

- binary outcome at the trial level
- item effects
- marginal new-student intercept and practice-slope uncertainty
- a student-specific latent AR(1) state process fitted over attempt bins of width `10`

Code:

- [model3_common.py](/D:/model1_baseline_agent_bundle/src/model3_common.py)
- [fit_model3.py](/D:/model1_baseline_agent_bundle/src/fit_model3.py)
- [evaluate_model3.py](/D:/model1_baseline_agent_bundle/src/evaluate_model3.py)
- [model3_track_b_fit.json](/D:/model1_baseline_agent_bundle/config/model3_track_b_fit.json)
- [model3_track_b_evaluate_validation.json](/D:/model1_baseline_agent_bundle/config/model3_track_b_evaluate_validation.json)
- [model3_track_b_evaluate_test.json](/D:/model1_baseline_agent_bundle/config/model3_track_b_evaluate_test.json)

## Fit summary

- inference method: `vi`
- train rows: `110,434`
- train students: `796`
- train items: `212`
- attempt-bin width: `10`
- train state steps: `23`
- VI iterations: `10,000`
- posterior draws: `500`

## Validation comparison

Model 2 validation:

- log loss: `0.4302`
- Brier: `0.1381`
- accuracy: `0.7988`
- AUC: `0.7969`
- calibration intercept: `0.2975`
- calibration slope: `1.0826`

Model 3 validation:

- log loss: `0.4294`
- Brier: `0.1368`
- accuracy: `0.8048`
- AUC: `0.7959`
- calibration intercept: `-0.1072`
- calibration slope: `1.3207`

Validation reading:

- Model 3 improves log loss, Brier, and accuracy slightly
- Model 3 does not improve AUC
- calibration slope gets materially worse

## Test comparison

Model 2 test:

- log loss: `0.4506`
- Brier: `0.1466`
- accuracy: `0.7872`
- AUC: `0.7891`
- calibration intercept: `0.1963`
- calibration slope: `1.0463`

Model 3 test:

- log loss: `0.4527`
- Brier: `0.1466`
- accuracy: `0.7868`
- AUC: `0.7880`
- calibration intercept: `-0.1931`
- calibration slope: `1.2693`

Test reading:

- Model 3 does not beat Model 2 on the primary held-out test metrics
- calibration is clearly worse than Model 2

## Provisional conclusion

This first Track B Model 3 check was not strong enough to justify the volatility extension on its own.

- it was only the initial VI screen
- it did not yet have the later stricter rerun
- it should not be treated as the final Model 3 judgment
