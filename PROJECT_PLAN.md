# Project Plan: Heterogeneity Discovery, Local Replication, Then Warm-Start

## Summary

Reframe the project away from a public benchmark race and toward this question:

**Do learners differ in baseline level, growth, and stability in public longitudinal data, and do those same forms of heterogeneity replicate in a new student sample?**

Then add the applied question:

**After replication, do public-informed priors estimate those learner differences earlier for new local students than weak-prior local fitting?**

This locks in:

- public Phase 1 as **discovery of heterogeneity**
- local Phase 2A as **conditional structural replication**
- local Phase 2B as **conditional warm-start application**
- Model 1 as the hurdle benchmark, not the endpoint
- only a public-supported Model 2 or Model 3 as a trigger for the main scientific continuation
- Model 2 and Model 3 as nested heterogeneity tests, not a leaderboard

## Scientific framing

Treat the three models as nested tests:

- **Model 1:** learners differ in starting level
- **Model 2:** learners also differ in rate of improvement
- **Model 3:** learners also differ in stability around that growth

Interpretation rule:

- If Model 2 adds a practically non-trivial learner slope variance and clears the predictive gate, growth heterogeneity is present.
- If Model 3 adds a practically non-trivial learner stability variance and clears the predictive gate, stability heterogeneity is present.
- If only Model 1 survives, the public dataset has not given a stable answer to the main scientific question.

## Phase 1: public heterogeneity discovery

### Public discovery sample

Use the **full visible DBE-KT22 attempt table** as the main discovery sample, keeping all linked KCs per item.

Reason:

- the project should use the full public dataset rather than discarding most multi-KC rows
- dropping multi-KC items removes too much repeated same-skill structure for Models 2 and 3
- the mainline should preserve all linked KCs rather than collapsing each item to a single KC by default

### Public analysis schema

Main discovery columns:

- `student_id`
- `item_id`
- `kc_ids`
- `kc_count`
- `correct`
- `timestamp`
- `attempt_id`
- `trial_index_within_student`
- `overall_opportunity`
- `kc_opportunity_*` summaries derived from the long attempt-KC table
- `kc_practice_feature_sum = sum(log1p(kc_opportunity_k))`
- `practice_feature = kc_practice_feature_sum`

Implementation note:

- `practice_feature` may be retained as a compatibility alias, but the scientific signal is `kc_practice_feature`

Auxiliary columns to retain but not use in the main model ladder:

- question difficulty
- hint use
- trust / confidence feedback
- difficulty feedback
- duration / time taken
- answer changes

### Sensitivity analysis

Before freezing Phase 1, run diagnostics around the operational full-dataset branch:

- a **single-KC-only branch** for construct-clean interpretation
- a **deterministic primary-KC branch** for one-KC-per-item collapse sensitivity
- a **repeated-practice subset** to check whether thin student-KC histories are washing out slope estimation

Purpose:

- check whether the full-dataset result depends entirely on how multi-KC items are handled
- separate construct-clean sensitivity from signal-preserving operational analysis

### Public model ladder

Fit the same cleaned discovery sample in order:

1. **Model 1**
   - `correct ~ kc_practice_feature + (1 | student_id) + (1 | item_id)`

2. **Model 2**
   - `correct ~ kc_practice_feature + (1 + kc_practice_feature | student_id) + (1 | item_id)`

3. **Model 3**
   - Model 2 plus latent learner-state deviation over time
   - keep the current binned latent-state approximation if needed for tractability
   - use KC-specific practice as the growth signal

### Phase 1 evidence families

Substantive evidence:

- learner intercept variance
- learner slope variance
- learner stability / volatility variance
- posterior stability of those variance terms across reruns
- descriptive evidence that early learner estimates relate to later outcomes

Predictive evidence:

- held-out log loss
- Brier score
- calibration intercept and slope
- calibration curve

### Phase 1 decision rule

Use **Variance + Prediction**.

For added heterogeneity SD terms beyond Model 1, the default practical floor is:

- posterior SD above `0.03` on the logit scale
- with the 94% HDI lower bound also above `0.03`

Predictive gate for richer models:

- either held-out log loss improves
- or log loss is no worse by more than `0.001`, while Brier improves and calibration slope moves closer to `1.0`

Stopping logic:

- if Model 2 fails that rule, stop at Model 1
- if Model 2 passes and Model 3 fails, stop at Model 2
- if Model 3 passes, carry it forward as the richer structural model

Important interpretation:

- Model 1 is only the hurdle for asking whether growth or stability heterogeneity add anything real
- if only Model 1 survives, do **not** treat Model 1 as the main scientific result
- instead freeze the public dataset as a pilot / screening dataset for this question

## Phase 2A: local structural replication

### Local data assumption

The local sample must include usable skill / KC IDs or an equivalent concept layer.

### Local schema

Normalize the local dataset to:

- `student_id`
- `item_id`
- `kc_id`
- `correct`
- `timestamp` or valid attempt order
- `attempt_id`
- `trial_index_within_student`
- `overall_opportunity`
- `kc_opportunity`
- `kc_practice_feature`

### Local split

Use a **3-way student-wise split**:

- local `train`
- local `calibration`
- untouched local `test`

### Replication rule

Phase 2A reruns the nested ladder up to the public-supported level.

It only becomes the mainline next step if the public dataset supports Model 2 or Model 3 cleanly enough to justify a real heterogeneity replication question.

Examples:

- If public Phase 1 ends at Model 2:
  - fit local Model 1 and local Model 2
- If public Phase 1 ends at Model 3:
  - fit local Models 1, 2, and 3

This is the actual structural replication step.

## Phase 2B: local warm-start application

Fit the same Phase 2A-supported structure again with public-informed priors or hyperparameters.

Compare:

1. weak-prior local chosen richer model
2. public-informed chosen richer model

Benchmark Model 1 only as a hurdle/baseline check alongside that richer-model comparison.

So the general rule is:

- Model 1 remains the hurdle benchmark
- only a public-supported Model 2 or Model 3 activates the mainline replication and warm-start study

## Primary outcome

### Phase 1

Heterogeneity variance terms plus non-degrading predictive fit.

### Phase 2

Primary:

- student-averaged log loss over attempts `1-5`

Secondary:

- student-averaged log loss over attempts `1-10`
- Brier score
- calibration intercept and slope
- overall performance after more attempts

Sparse-data fallback:

- if too few students reach 5 attempts, pre-specify a shorter primary window such as attempts `1-3`

## Current implementation order

1. Build the full-dataset multi-KC public discovery table.
2. Fit Models 1 and 2 on that table.
3. Fit Model 3 if Model 2 survives the sharpened `Variance + Prediction` rule on that operational branch.
4. Run the deterministic primary-KC sensitivity branch.
5. Run the single-KC sensitivity branch.
6. Run the repeated-practice diagnostic branch.
7. Normalize local data to the same KC-aware schema.
8. Create the 3-way local student split.
9. If the full-dataset branch supports Model 2 or Model 3 strongly enough for scientific continuation, run local structural replication.
10. Then run the warm-start comparison.

If the full-dataset branch fails to support Model 2 or Model 3, do not skip ahead to a Model 1-led Phase 2. Treat DBE as pilot groundwork and either redesign the public discovery analysis or replace the public dataset.
