---
name: model1-baseline-binary-logistic
description: Use this skill when the task is to fit, evaluate, or reproduce the first manageable baseline for trial-level learner-response data with a binary outcome. It is for already-observed learners, chronological forward prediction, and a hierarchical logistic model with learner intercepts, item difficulty, and a simple practice term. Do not use it for volatility-aware models, personalised learning controllers, knowledge tracing systems, cold-start studies, or adaptive rule-change methods.
---

# Model 1 baseline: hierarchical logistic next-attempt forecasting

## Purpose
Implement the **smallest defensible baseline** for sequential learner data.

This skill exists to answer one question:

**Does a simple hierarchical logistic model provide sensible forward predictions and calibration for later attempts from already-observed learners?**

The point is not to build a full learning system. The point is to create a reproducible, well-specified baseline that later extensions can beat.

## Trigger conditions
Use this skill when most of the following are true:
- The data are trial-level learner interactions.
- The outcome is binary (`correct` / `incorrect`, `0` / `1`).
- The task is a **first paper**, **baseline**, or **Model 1**.
- The goal is next-attempt probability prediction or calibration.
- The evaluation should preserve chronology.
- The learners in the test segment have earlier observations in training.

## Do not use this skill when
Do **not** use this skill if the task asks for:
- learner-specific volatility or heteroskedasticity
- state-space or dynamic latent trait models
- BKT, DKT, HMMs, RNNs, transformers, or other knowledge tracing architectures
- adaptive item selection or pedagogical decision rules
- new-student or cold-start generalisation as the primary claim
- joint modelling of accuracy with response time, hints, or confidence as the main outcome

## Required input schema
At minimum, the dataset must support these columns or equivalents:
- `student_id`
- `item_id`
- `correct` (0/1)
- `timestamp` **or** an already valid within-student trial order column

Preferred optional columns:
- `skill_id`
- `question_difficulty` or other static item metadata known before the attempt
- `course_week` or session/block identifiers

### What not to use in Model 1
Do not use same-trial variables that may be unavailable at decision time or are downstream of the response process. Usually exclude:
- `time_taken`
- `hint_used`
- `num_answer_changes`
- final confidence or any post-response flags

These belong in later extensions, not the baseline.

## Default model specification
Start with this model:

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
- `student` random intercept = stable between-learner differences in baseline success probability
- `item` random intercept = item difficulty term
- `opp_it` = cumulative count of prior attempts for that learner before trial `t`

### Practice term
Default to:
- `overall_opportunity = cumulative prior attempts within student`
- `practice_feature = log1p(overall_opportunity)`

Only add `skill_opportunity` as a secondary analysis if a clean `skill_id` exists and coverage is adequate.

## Recommended implementation order
Prefer this order of tooling:
1. Existing repository stack, if already established
2. `bambi` + `PyMC` + `ArviZ`
3. Native `PyMC` if formulas are not practical

If the environment makes Bayesian fitting impossible, a fallback frequentist approximation is acceptable **only if** you preserve the same substantive model structure and still evaluate forward-held-out probabilities.

## Workflow

### 1) Audit and map the schema
Before modelling:
- Confirm the outcome is binary and coded consistently.
- Confirm time order exists or can be recovered.
- Confirm `student_id` and `item_id` are stable identifiers.
- Write down any column mappings and assumptions.

If schema ambiguity exists, inspect first and document the mapping instead of guessing silently.

### 2) Build a clean trial table
Create one row per learner-item attempt.

Required steps:
- sort by `student_id`, then time
- remove duplicate rows only if they are true duplicates and document the rule
- create `trial_index_within_student`
- create `overall_opportunity = trial_index_within_student - 1`
- create `practice_feature = log1p(overall_opportunity)`
- drop rows with missing required identifiers or missing outcome

### 3) Define the primary analysis sample
This first-paper baseline should stay manageable.

Default sample rules:
- include learners with at least a documented minimum number of attempts
- split each learner chronologically
- primary evaluation should focus on later rows whose item IDs were observed in the training data
- report how many later rows were excluded because the item was unseen during training

Do **not** make claims about new-item generalisation in this baseline paper.

### 4) Split in time order
Default split:
- 80% early rows = train
- 20% later rows = test

If model tuning or prior sensitivity is required:
- 70% train
- 15% validation
- 15% test

All splits must be deterministic and saved.

### 5) Fit the model
Fit the baseline hierarchical logistic model.

Suggested weakly informative priors:
- `beta_0 ~ Normal(0, 1.5)`
- `beta_practice ~ Normal(0, 1)`
- `sigma_student ~ HalfNormal(1)`
- `sigma_item ~ HalfNormal(1)`

If you add a static pre-attempt item difficulty covariate, do **not** also add a redundant representation that creates avoidable identifiability problems. Prefer either:
- item random intercepts alone, or
- static difficulty covariate plus a smaller residual item effect if justified and stable

### 6) Evaluate only on future rows
Primary metrics:
- held-out log loss / mean log predictive density
- Brier score
- calibration intercept and slope
- calibration curve / reliability plot

Secondary metrics:
- accuracy
- AUC

Also report:
- learner-level distribution of log loss or Brier score where practical
- posterior predictive checks or equivalent model diagnostics
- convergence diagnostics if Bayesian (`R-hat`, ESS, divergences)

### 7) Write the results in the right tone
This baseline paper should conclude only what the model can support.

Safe conclusions:
- whether the model gives usable forward probability forecasts
- whether calibration is acceptable or needs improvement
- whether a simple practice term helps
- how much learners and items differ on the log-odds scale

Do **not** overclaim:
- mastery inference
- personalised decision making
- causal learning effects
- volatility differences
- new-student deployment readiness

## Output checklist
Produce these outputs unless the user asks otherwise:
- `data/processed/learner_trials.*`
- a saved split file with train/test labels
- modelling script or notebook
- evaluation script
- metrics table (CSV or parquet)
- calibration figure
- concise methods note or report
- assumptions note describing mappings, exclusions, and any deviations from default rules

## Preferred repo layout
If the repository is empty or unspecified, use a simple structure:

```text
.
├── AGENTS.md
├── .agents/
│   └── skills/
│       └── model1-baseline-binary-logistic/
│           └── SKILL.md
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocess.py
│   ├── fit_model1.py
│   └── evaluate_model1.py
├── reports/
│   └── model1_baseline_report.md
└── outputs/
    ├── metrics/
    └── figures/
```

## Done means
The task is done only when all of the following are true:
- the data mapping is explicit
- chronology is preserved end to end
- the model matches the baseline specification
- evaluation uses later rows only
- primary metrics include log loss, Brier score, and calibration
- outputs are reproducible and saved
- the write-up stays within the baseline claim

## Extension note
If the user later asks for the next step, the natural extension is **not** a full personalised learning system. The natural extension is a second model that adds learner-specific volatility or a joint accuracy-plus-time process while keeping the same forward-holdout evaluation.
