# Phase 1 Explicit Q-Matrix Comparison

This note summarizes the **full explicit Q-matrix** public-data comparison for Models 1, 2, and 3 on the full multi-KC DBE branch.

This is the explicit-Q **opportunity-history ladder**. It remains the key scientific heterogeneity note, but it is no longer the last word on operational model choice because the later explicit Q-matrix PFA branch improved predictive yield further.

## Setup

- Same processed public sample as the existing full multi-KC branch:
  - `125,877` train rows
  - `32,112` held-out test rows
  - `1,138` learners
  - `212` items
  - `93` KCs
- Same held-out rows for both models
- One row per attempt
- KC structure enters the likelihood explicitly through attempt-by-KC design matrices:
  - KC incidence term
  - KC-specific prior-practice term
- Fit method:
  - PyMC VI (`advi`)
  - `20,000` iterations
  - `1,000` posterior draws

## Why this branch matters

The earlier multi-KC branch used Q-matrix information only in preprocessing, then collapsed KC practice back to a single scalar `practice_feature`.

This explicit branch keeps the Q-matrix inside the model itself:

- Model 1:
  - learner intercepts
  - item effects
  - KC intercepts
  - KC-specific shared practice effects
- Model 2:
  - Model 1
  - learner-specific deviation on the total KC-practice signal

That makes the Model 1 to Model 2 comparison much closer to the real growth-heterogeneity question.

Model 3 then adds:

- learner-specific latent volatility
- latent AR(1) learner-state deviations over attempt bins

## Results

### Explicit Q-matrix Model 1

- fit summary: [model1_fit_summary.json](/D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix/model1/model1_fit_summary.json)
- evaluation summary: [model1_evaluation_summary.json](/D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix/model1/model1_evaluation_summary.json)

Held-out metrics:

- log loss: `0.545311`
- Brier: `0.184409`
- accuracy: `0.716617`
- AUC: `0.760989`
- calibration intercept: `-0.065235`
- calibration slope: `0.931707`

Key variance terms:

- `student_intercept_sigma`: mean `0.560`, 94% HDI `[0.530, 0.588]`
- `kc_practice_sigma`: mean `0.050`, 94% HDI `[0.039, 0.061]`

### Explicit Q-matrix Model 2

- fit summary: [model2_fit_summary.json](/D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix/model2/model2_fit_summary.json)
- evaluation summary: [model2_evaluation_summary.json](/D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix/model2/model2_evaluation_summary.json)

Held-out metrics:

- log loss: `0.544366`
- Brier: `0.184051`
- accuracy: `0.716119`
- AUC: `0.761858`
- calibration intercept: `-0.069704`
- calibration slope: `0.943504`

Key variance terms:

- `student_intercept_sigma`: mean `0.518`, 94% HDI `[0.488, 0.544]`
- `student_slope_sigma`: mean `0.046`, 94% HDI `[0.040, 0.051]`
- `kc_practice_sigma`: mean `0.053`, 94% HDI `[0.041, 0.064]`

### Explicit Q-matrix Model 3

- fit summary: [model3_fit_summary.json](/D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix/model3/model3_fit_summary.json)
- evaluation summary: [model3_evaluation_summary.json](/D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix/model3/model3_evaluation_summary.json)

Held-out metrics:

- log loss: `0.543782`
- Brier: `0.183916`
- accuracy: `0.715247`
- AUC: `0.761855`
- calibration intercept: `-0.082279`
- calibration slope: `0.968148`

Key variance terms:

- `student_intercept_sigma`: mean `0.470`, 94% HDI `[0.439, 0.497]`
- `student_slope_sigma`: mean `0.0448`, 94% HDI `[0.0397, 0.0501]`
- `state_sigma_global`: mean `0.4760`, 94% HDI `[0.4488, 0.5056]`
- `rho`: mean `0.0931`, 94% HDI `[0.0653, 0.1213]`

## Model 2 vs Model 1

Held-out deltas, Model 2 minus Model 1:

- log loss: `-0.000945`
- Brier: `-0.000358`
- AUC: `+0.000869`
- calibration slope: `+0.011796`

So on the explicit Q-matrix branch:

- Model 2 clears the practical variance floor
- Model 2 also clears the predictive gate against Model 1

That is the first direct full-data result in this repo where the **growth-heterogeneity extension survives under an explicit Q-matrix likelihood**, rather than only after KC information has been collapsed in preprocessing.

## Model 3 vs Model 2

Held-out deltas, Model 3 minus Model 2:

- log loss: `-0.000584`
- Brier: `-0.000134`
- AUC: `-0.000003`
- calibration slope: `+0.024645`

So on the explicit Q-matrix branch:

- the added latent stability term is clearly non-zero
- Model 3 improves log loss and Brier relative to Model 2
- calibration slope also moves closer to `1.0`

That means the explicit Q-matrix ladder now supports:

- Model 1: baseline heterogeneity
- Model 2: growth heterogeneity
- Model 3: stability heterogeneity

## Runtime note

All three fits completed successfully, but they were slower than the earlier collapsed-feature branch because the current `.venv` PyTensor install is not linked against BLAS:

- Model 1 elapsed seconds: `945.80`
- Model 2 elapsed seconds: `988.58`
- Model 3 elapsed seconds: `1206.18`

This is a speed issue, not a validity issue.

## Current interpretation

The explicit Q-matrix branch now supports the full ladder:

- baseline heterogeneity is present
- growth heterogeneity is present
- stability heterogeneity is also present

So the current explicit-Q position is:

- explicit Q-matrix Model 1: supported baseline
- explicit Q-matrix Model 2: supported growth model
- explicit Q-matrix Model 3: supported stability model

Operational note:

- for the current operational RPFA selection, see [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)
- for the current policy-facing comparison, see [adaptive_policy_suite_comparison.md](D:/model1_baseline_agent_bundle/reports/adaptive_policy_suite_comparison.md)
