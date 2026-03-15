---
name: model3-dynamic-volatility-binary-logistic
description: Use this skill when the task is to fit the third Phase 1 model for binary learner-response data by extending the random-slope model with a latent time-varying learner state and learner-specific volatility. Use it only after Models 1 and 2 are fitted cleanly and compared. Do not use it for public-to-local transfer until it has clearly beaten Model 2 on probabilistic metrics and calibration.
---

# Model 3: dynamic logistic model with learner-specific latent volatility

## Purpose
Implement the **uncertainty/instability extension** that connects Phase 1 back to the user's previous report.

This skill exists to answer one question:

**After accounting for baseline proficiency, item difficulty, and learner-specific growth, does learner-specific trial-to-trial instability improve probabilistic forecasting and calibration?**

## Trigger conditions
Use this skill when most of the following are true:
- Model 1 and Model 2 have already been fitted.
- The user explicitly wants the third model or the volatility/uncertainty extension.
- The data are binary learner-response sequences with preserved order.
- The goal is better probabilistic forecasting, not merely descriptive growth curves.

## Do not use this skill when
Do not use this skill if:
- Model 1 and Model 2 have not yet been run on the same data.
- The task is still a simple baseline or random-slope analysis.
- The user is asking for BKT, DKT, HMMs, RNNs, transformers, or a generic knowledge-tracing benchmark.
- The user wants public-to-local transfer before public-data comparison is complete.

## Required input schema
At minimum, the dataset must support these columns or equivalents:
- `student_id`
- `item_id`
- `correct` (0/1)
- `timestamp` or a valid within-student order column

Derived columns required:
- `trial_index_within_student`
- `overall_opportunity = trial_index_within_student - 1`
- `practice_feature = log1p(overall_opportunity)`

## Model concept
Because the observed outcome is binary, this model does **not** estimate a free residual SD on the 0/1 outcome itself.

Instead, it adds a **latent learner-state deviation** that evolves over time:

\[
y_{it} \sim \text{Bernoulli}(p_{it})
\]

\[
\text{logit}(p_{it}) = \beta_0 + \alpha_i + \gamma_j + (\beta_{\text{practice}} + b_i)\,\log(1 + \text{opp}_{it}) + s_{it}
\]

\[
s_{it} = \rho s_{i,t-1} + \epsilon_{it}, \quad \epsilon_{it} \sim \mathcal{N}(0, \sigma_i^2)
\]

Where:
- `alpha_i` = learner baseline log-odds deviation
- `b_i` = learner-specific deviation in practice slope
- `gamma_j` = item difficulty term
- `s_it` = latent learner-state deviation at time `t`
- `sigma_i` = learner-specific latent volatility

## Interpretation rules
Interpret the volatility term carefully:
- it is **not** raw variance in correctness
- it is **latent instability** after accounting for level, item difficulty, and growth
- high volatility may reflect inconsistency, erratic responding, fatigue, fluctuating attention, or model misspecification

## Workflow
1. Reuse the same cleaned trial table and splits from Models 1 and 2.
2. Fit Model 1 and Model 2 first on the same analysis sample.
3. Fit Model 3 with the latent state component.
4. Compare Model 3 directly against Model 2 using the same holdout rows.
5. Carry Model 3 forward only if it clearly improves primary probabilistic metrics and calibration.

## Evaluation
Primary metrics:
- held-out log loss / mean log predictive density
- Brier score
- calibration intercept and slope
- calibration curve / reliability plot

Secondary metrics:
- accuracy
- AUC

Also report:
- direct Model 2 vs Model 3 comparison table
- learner-level summaries of volatility estimates or posterior summaries
- state-process diagnostics, convergence diagnostics, and warnings

## Done means
The task is done only when all of the following are true:
- Models 1, 2, and 3 used the same cleaned analysis sample
- chronology was preserved end to end
- the comparison with Model 2 is explicit
- the write-up makes clear that volatility improved (or failed to improve) probabilistic forecasting beyond learning-rate heterogeneity
