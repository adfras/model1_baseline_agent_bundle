# Phase 1 Repeated-Practice Subset Results

This note records the stronger repeated-practice redesign built from the single-KC public discovery table.

## Why this redesign was run

The single-KC discovery table still contained many student-KC sequences with only one or two opportunities.

That can wash out growth heterogeneity even when the overall row count is large.

So this redesign keeps only student-KC sequences with at least `3` opportunities and then requires each student to retain at least `10` rows after filtering.

See:

- [phase1_repeated_subset_schema_note.md](D:/model1_baseline_agent_bundle/reports/phase1_repeated_subset_schema_note.md)

## Repeated-practice subset

- `19,197` processed rows
- `853` learners
- `51` items
- `18` KCs
- `4,703` retained student-KC sequences
- `15,073` train rows
- `4,124` test rows

The audit that motivated the threshold is:

- `>= 3` opportunities: `20,252` rows across `18` KCs before the student-history filter
- `>= 4` opportunities: `11,522` rows across `10` KCs
- `>= 5` opportunities: `7,950` rows across `5` KCs

So `3+` was the strongest repeated-practice rule that still preserved a usable KC set.

## Results

### Model 1

- log loss `0.536827`
- Brier `0.181433`
- calibration slope `0.984538`

### Model 2

- log loss `0.537510`
- Brier `0.181731`
- calibration slope `0.987535`
- shared practice effect mean `0.121`
- learner slope SD mean `0.121`
- learner slope SD 94% HDI `[0.085, 0.160]`

## Paired student-level comparison

Model 2 minus Model 1:

- mean delta log loss `+0.000692`
- 95% bootstrap interval `[+0.000148, +0.001229]`
- mean delta Brier `+0.000306`
- 95% bootstrap interval `[+0.000093, +0.000527]`

Interpretation:

- Model 2 remains worse than Model 1 on the stronger repeated-practice subset
- the slope variance is clearly non-zero
- but the predictive gate still fails

## What this means

This redesign rules out the simplest criticism that the earlier result was only caused by too many one-off student-KC rows.

The stronger repeated-practice subset did produce:

- a clearly positive shared KC-practice effect
- a clearly non-zero learner slope SD

But it still did **not** make Model 2 predict better than Model 1 on held-out rows.

So the current public evidence remains:

- baseline heterogeneity is robust
- growth heterogeneity is plausible
- but growth heterogeneity still does not clear the predictive gate on the primary single-KC family, even after strengthening the repeated-practice design
