# Model 2 Track A Initial Run

This note records the first completed Phase 1 Track A run for Model 2 on `DBE-KT22`.

## Model

Model 2 extends the Model 1 baseline by allowing learner-specific practice slopes:

```text
correct ~ practice_feature + (1 + practice_feature | student_id) + (1 | item_id)
```

Implementation files:

- [model2_common.py](/D:/model1_baseline_agent_bundle/src/model2_common.py)
- [fit_model2.py](/D:/model1_baseline_agent_bundle/src/fit_model2.py)
- [evaluate_model2.py](/D:/model1_baseline_agent_bundle/src/evaluate_model2.py)
- [model2_fit.json](/D:/model1_baseline_agent_bundle/config/model2_fit.json)
- [model2_evaluate.json](/D:/model1_baseline_agent_bundle/config/model2_evaluate.json)

## Fit summary

- inference method: `vi`
- train rows: `125,877`
- train students: `1,138`
- train items: `212`
- VI iterations: `20,000`
- posterior draws: `1,000`
- elapsed seconds: `257.97`

Outputs:

- [model2_fit_summary.json](/D:/model1_baseline_agent_bundle/outputs/model2/model2_fit_summary.json)
- [model2_posterior_summary.csv](/D:/model1_baseline_agent_bundle/outputs/model2/model2_posterior_summary.csv)
- [model2_student_slope_summary.csv](/D:/model1_baseline_agent_bundle/outputs/model2/model2_student_slope_summary.csv)

## Held-out Track A evaluation

Model 2 test metrics:

- log loss: `0.5463`
- Brier: `0.1848`
- accuracy: `0.7149`
- AUC: `0.7601`
- calibration intercept: `-0.0346`
- calibration slope: `0.9082`

Model 1 reference metrics on the same Track A holdout:

- log loss: `0.5446`
- Brier: `0.1841`
- accuracy: `0.7153`
- AUC: `0.7609`
- calibration intercept: `0.0670`
- calibration slope: `0.9253`

## Initial comparison

On this first Track A VI run, Model 2 does not improve the primary probabilistic metrics over Model 1. It is slightly worse on log loss, Brier score, accuracy, and AUC. Under the repo decision rule, that means Model 2 has not yet earned its added complexity.

This does not prove that Model 2 can never help. It does mean the current next step should be conservative:

- treat this as the first Model 2 implementation checkpoint
- do not advance to Model 3 from this result alone
- only continue with more Model 2 work if there is a clear reason to test whether inference method or split choice is masking a real gain
