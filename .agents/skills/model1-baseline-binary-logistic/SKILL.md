---
name: model1-baseline-binary-logistic
description: Use this skill when the task is to fit, evaluate, or reproduce the first public-data baseline for trial-level learner-response data with a binary outcome. It is for chronological forward prediction and for unseen-public-student initialization only as a baseline, using a hierarchical logistic model with learner intercepts, item difficulty, and one shared practice term. Do not use it for volatility-aware models, personalised learning systems, or public-to-local transfer.
---

# Model 1 baseline: hierarchical logistic next-attempt forecasting

## Purpose
Implement the **smallest defensible Phase 1 baseline** for sequential learner data.

This skill exists to answer two narrow questions:
1. Does a simple hierarchical logistic model provide sensible forward predictions for later attempts from already-observed learners?
2. Does it provide a reasonable anchor for sequential prediction on unseen public students?

## Trigger conditions
Use this skill when most of the following are true:
- The data are trial-level learner interactions.
- The outcome is binary (`correct` / `incorrect`, `0` / `1`).
- The task is the first Phase 1 baseline or "Model 1".
- The goal is next-attempt probability prediction or calibration.
- The evaluation should preserve chronology.

## Do not use this skill when
Do not use this skill if the task asks for:
- learner-specific volatility or heteroskedasticity
- state-space or dynamic latent trait models
- random learner-specific slopes on practice
- public-to-local warm-start transfer
- adaptive item selection or pedagogical decision rules
- full personalised learning workflows

## Required input schema
At minimum, the dataset must support these columns or equivalents:
- `student_id`
- `item_id`
- `correct` (0/1)
- `timestamp` or a valid within-student trial order column

Derived columns required:
- `trial_index_within_student`
- `overall_opportunity = trial_index_within_student - 1`
- `practice_feature = log1p(overall_opportunity)`

## What not to use in Model 1
Do not use same-trial variables that may be unavailable at decision time or are downstream of the response process. Usually exclude:
- `time_taken`
- `hint_used`
- `num_answer_changes`
- final confidence or post-response flags

## Default model specification
\[
y_{it} \sim \text{Bernoulli}(p_{it})
\]

\[
\text{logit}(p_{it}) = \beta_0 + \alpha_i + \gamma_j + \beta_{\text{practice}}\,\log(1 + \text{opp}_{it})
\]

\[
\alpha_i \sim \mathcal{N}(0, \sigma_{\text{student}})
\]

\[
\gamma_j \sim \mathcal{N}(0, \sigma_{\text{item}})
\]

Where:
- `alpha_i` = learner baseline log-odds deviation
- `gamma_j` = item difficulty term
- `opp_it` = cumulative prior attempts for that learner before trial `t`

## Workflow
1. Audit schema and document mappings.
2. Build a clean trial table in chronological order.
3. Create `trial_index`, `overall_opportunity`, and `practice_feature = log1p(overall_opportunity)`.
4. Run Phase 1 Track A and Track B splits as defined in `AGENTS.md`.
5. Fit the baseline model.
6. Evaluate on later rows only.

## Evaluation
Primary metrics:
- held-out log loss / mean log predictive density
- Brier score
- calibration intercept and slope
- calibration curve / reliability plot

Secondary metrics:
- accuracy
- AUC

## Output checklist
Produce these outputs unless the user asks otherwise:
- processed trial table
- saved split file(s)
- modelling script
- evaluation script
- metrics table
- calibration figure
- concise methods note
- assumptions note
