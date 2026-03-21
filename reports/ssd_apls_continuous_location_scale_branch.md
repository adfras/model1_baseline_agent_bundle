# SSD/APLS Continuous Location-Scale Branch

This note documents the SSD/APLS continuous-outcome branch that currently sits alongside the repo's DBE binary mainline.

## Why this branch exists

The DBE mainline stays focused on binary trial-level heterogeneity discovery. The SSD/APLS branch asks a different question:

- after a support request, how do learners differ in short-, medium-, and longer-horizon outcomes?
- do those differences appear only in the mean outcome, or also in residual volatility?

The branch therefore uses:

- continuous targets, not binary next-answer classification
- real elapsed-time windows, not request-count horizons
- the same held-out students for baseline and heteroskedastic comparisons

## Data source and split

Inputs:

- processed request table: [support_requests_processed.csv](D:/model1_baseline_agent_bundle/data/processed/ssd_apls/support_requests_processed.csv)
- held-out student assignments: [support_split_assignments.csv](D:/model1_baseline_agent_bundle/outputs/ssd_apls/support_scorer/support_split_assignments.csv)

Derived continuous-target table:

- [support_requests_continuous_targets.csv](D:/model1_baseline_agent_bundle/data/processed/ssd_apls/support_requests_continuous_targets.csv)

The split is reused unchanged from the earlier SSD/APLS scorer branch:

- rows are never shuffled
- train/test membership is defined at the student level
- every baseline and heteroskedastic comparison uses the same held-out students

## Time windows

The branch currently uses three real elapsed-time horizons:

- `1d` = next 24 hours
- `7d` = next 7 days
- `28d` = next 28 days

These are calendar windows measured from each support request's `created_at` timestamp. They are not:

- number of future requests
- total time spent studying
- time on task for a single problem

## Target families

Implemented targets are:

- `future_acc_1d`
- `future_acc_7d`
- `future_acc_28d`
- `delta_state_average_hint_count_1d`
- `delta_state_average_hint_count_7d`
- `delta_state_average_hint_count_28d`
- `delta_state_average_attempt_count_1d`
- `delta_state_average_attempt_count_7d`
- `delta_state_average_attempt_count_28d`
- `delta_state_average_first_action_answer_1d`
- `delta_state_average_first_action_answer_7d`
- `delta_state_average_first_action_answer_28d`
- `delta_state_average_correctness_1d`
- `delta_state_average_correctness_7d`
- `delta_state_average_correctness_28d`

Detailed formulas and support counts are in [ssd_apls_continuous_target_dictionary.md](D:/model1_baseline_agent_bundle/reports/ssd_apls_continuous_target_dictionary.md).

## What the models are trying to capture

This branch is meant to map several ways learners can differ:

- baseline level differences
- learner-specific progress trajectories
- problem-specific difficulty differences
- residual volatility that remains student-specific
- residual volatility that changes with observed learner state

That is why the scientific reference model keeps both:

- student progress random slopes
- student-specific residual-SD terms

These are not optional decorations in the scientific interpretation. They represent different forms of learner heterogeneity.

## Baseline model

The matched baseline model is in [support_location_scale_homoskedastic.stan](D:/model1_baseline_agent_bundle/stan/support_location_scale_homoskedastic.stan).

It uses:

- Gaussian outcome on the transformed target
- fixed effects from progress and state
- student random intercept and student progress random slope
- problem random intercept
- a global residual log-SD intercept
- a student-specific residual log-SD deviation

In symbols:

\[
\begin{aligned}
y_i &\sim \mathrm{Normal}(\mu_i, \sigma_i) \\
\mu_i &= \alpha + X_i \beta + a_{j[i]} + b_{j[i]} \cdot \mathrm{progress}_i + c_{\ell[i]} \\
\log \sigma_i &= \alpha_\sigma + u_{j[i]}
\end{aligned}
\]

where:

- `j[i]` is the student for row `i`
- `l[i]` is the problem for row `i`
- `a_j, b_j` are student mean effects
- `c_l` is a problem mean effect
- `u_j` is a student-specific residual-SD effect

## Heteroskedastic model

The heteroskedastic model is in [support_location_scale.stan](D:/model1_baseline_agent_bundle/stan/support_location_scale.stan).

It keeps the same mean structure and same student-specific residual-SD term, but adds state-driven residual-SD effects:

\[
\begin{aligned}
y_i &\sim \mathrm{Normal}(\mu_i, \sigma_i) \\
\mu_i &= \alpha + X_i \beta + a_{j[i]} + b_{j[i]} \cdot \mathrm{progress}_i + c_{\ell[i]} \\
\log \sigma_i &= \alpha_\sigma + Z_i \gamma + u_{j[i]}
\end{aligned}
\]

Interpretation:

- `X_i beta` explains mean outcome differences
- `Z_i gamma` explains volatility differences from observed learner state
- `u_j` captures residual between-student volatility not explained by observed state

So the baseline-vs-heteroskedastic contrast is:

- same target
- same held-out students
- same mean model
- same student-specific latent volatility term
- only the state-driven volatility term `Z_i gamma` differs

## Predictors

The branch currently uses these raw state features:

- `state_total_assignments_completed`
- `state_total_problems_completed`
- `state_assignment_completion_pct`
- `state_problem_completion_pct`
- `state_median_time_on_task`
- `state_median_first_response_time`
- `state_average_correctness`
- `state_average_attempt_count`
- `state_average_hint_count`
- `state_average_first_action_answer`

Derived mean-side features:

- `progress_z`
- `progress_z_sq`
- `progress_z_cu`
- z-scored state features
- `state_missing_any`

Variance-side features:

- z-scored state features
- `state_missing_any`

## Target transforms

Future-accuracy targets use an empirical-logit transform before fitting:

- pseudo-count = `0.5`
- denominator = number of future requests inside the window

Delta-state targets use the identity transform.

The models are fit on the transformed scale and then mapped back to the raw target scale during evaluation.

## Current evaluator outputs

For each target, the evaluator writes:

- [test_predictions.csv](D:/model1_baseline_agent_bundle/outputs/ssd_apls/continuous_location_scale/future_acc_1d/test_predictions.csv)
- [metric_comparison.csv](D:/model1_baseline_agent_bundle/outputs/ssd_apls/continuous_location_scale/future_acc_1d/metric_comparison.csv)

The metric table includes:

- overall `rmse`, `mae`, `r2`
- uncertainty metrics:
  - `mean_log_score`
  - `coverage_50_gap`
  - `coverage_80_gap`
  - `coverage_95_gap`
  - `pit_ks`
- early-fit slices for requests `1-3`, `1-5`, `1-10`
- tail-risk student error quantiles
- surprise-recovery slices
- calibration intercept, slope, and 10-bin gap
- state-bin slices over `state_average_correctness`

## Current status

As of `2026-03-21`, the branch is implemented but not complete end to end across all targets.

What exists:

- target builder
- baseline and heteroskedastic Stan models
- evaluator
- cloud runner

What has not finished yet:

- a full baseline-vs-heteroskedastic comparison for every elapsed-time target
- a completed all-target result bundle

The actively tuned run at the time of writing is:

- target: `future_acc_1d`
- model: heteroskedastic
- compute shape: `4` chains x `4` threads
- `grainsize = 3000`
- `STAN_CPP_OPTIMS = true`
- `refresh = 1`

Operational details for running and tuning this branch are in [ssd_apls_continuous_runbook.md](D:/model1_baseline_agent_bundle/reports/ssd_apls_continuous_runbook.md).
