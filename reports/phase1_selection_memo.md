# Phase 1 Selection Memo

This memo is the canonical Phase 1 decision note after switching the operational discovery path to a full-data multi-KC design.

For the branch names used here, see:

- [phase1_branch_guide.md](D:/model1_baseline_agent_bundle/reports/phase1_branch_guide.md)

## What Phase 1 is for

Phase 1 is not a public leaderboard.

Its job is to answer:

1. Do learners differ in baseline level?
2. Do learners also differ in growth rate?
3. Do learners also differ in stability around that growth?

The model ladder is therefore:

- **Model 1:** level heterogeneity
- **Model 2:** growth heterogeneity
- **Model 3:** stability heterogeneity

## Sharpened decision rule

Use **Variance + Prediction**.

For added heterogeneity SD terms beyond Model 1:

- the default practical floor is `0.03` on the logit scale
- the 94% HDI lower bound should also exceed `0.03`

Predictive gate for richer models:

- either held-out log loss improves
- or log loss is no worse by more than `0.001`, while Brier improves and calibration slope moves closer to `1.0`

## Current operational discovery sample

The operational public discovery table now keeps the full visible DBE dataset and all linked KCs per item:

- `157,989` processed attempt rows
- `1,138` learners
- `212` items
- `93` represented KCs
- `300,246` long attempt-KC rows
- `125,877` train rows
- `32,112` test rows

The model-facing practice term is additive across linked KCs:

- `practice_feature = sum(log1p(kc_opportunity_k))`

See:

- [phase1_multikc_schema_note.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_schema_note.md)
- [phase1_multikc_heterogeneity_summary.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_heterogeneity_summary.md)

## Current model evidence on the operational branch

### Model 1

Held-out discovery performance:

- log loss `0.546332`
- Brier `0.184800`
- calibration slope `0.926549`

Variance evidence:

- learner intercept SD mean `0.569`
- 94% HDI `[0.538, 0.598]`

Interpretation:

- baseline level heterogeneity is clearly present
- this is the hurdle result, not the endpoint

### Model 2

Held-out discovery performance:

- log loss `0.545491`
- Brier `0.184503`
- calibration slope `0.940928`

Variance evidence:

- learner slope SD mean `0.048`
- 94% HDI `[0.042, 0.053]`

Interpretation:

- the slope variance clears the practical floor
- Model 2 clears the predictive gate
- growth heterogeneity is supported on the operational full-data branch

### Model 3

Held-out discovery performance:

- log loss `0.543726`
- Brier `0.183892`
- calibration slope `0.963244`

Variance evidence:

- learner slope SD mean `0.0485`
- learner slope SD 94% HDI `[0.0421, 0.0541]`
- latent state SD mean `0.4860`
- latent state SD 94% HDI `[0.4588, 0.5145]`
- `rho` mean `0.1180`
- `rho` 94% HDI `[0.0867, 0.1474]`

Interpretation:

- the added stability term is clearly non-zero
- Model 3 clears the predictive gate relative to Model 2
- stability heterogeneity is supported on the operational full-data branch

## Sensitivity / diagnostic branches

### Single-KC sensitivity branch

The construct-clean single-KC branch still stops at Model 1:

- Model 1 log loss `0.567414`
- Model 2 log loss `0.567742`
- Model 3 log loss `0.570940`

This branch remains useful as a sensitivity check, but it is not the operational mainline because it discards too much repeated same-skill structure.

### Deterministic primary-KC branch

The older full-data primary-KC collapse also supported richer models:

- Model 2 log loss `0.545797`
- Model 3 log loss `0.544350`

It is now treated as an intermediate sensitivity branch rather than the preferred operational method.

### Fractional multi-KC sensitivity branch

A stronger like-for-like sensitivity has now been run on the same full dataset with equal-split fractional KC exposure:

- same rows
- same split
- same model ladder
- only the KC allocation rule changes

Results:

- fractional Model 1 log loss `0.547029`
- fractional Model 2 log loss `0.546311`
- fractional Model 2 learner slope SD 94% HDI `[0.053, 0.069]`
- fractional Model 3 log loss `0.543867`
- fractional Model 3 Brier `0.183941`
- fractional Model 3 calibration slope `0.957373`
- fractional Model 3 latent state SD 94% HDI `[0.4559, 0.5113]`

Interpretation:

- growth heterogeneity still survives under fractional KC allocation
- stability heterogeneity still survives under fractional KC allocation
- the richer result weakens slightly relative to the full-credit mainline, but it does not collapse

### Explicit Q-matrix branch

An explicit Q-matrix Model 1 to Model 2 comparison has now also been run on the same full multi-KC sample:

- explicit Q-matrix Model 1 log loss `0.545311`
- explicit Q-matrix Model 2 log loss `0.544366`
- explicit Q-matrix Model 2 Brier `0.184051`
- explicit Q-matrix Model 2 calibration slope `0.943504`
- explicit Q-matrix Model 2 learner slope SD 94% HDI `[0.040, 0.051]`

Interpretation:

- when KC structure is moved directly into the likelihood, Model 2 still beats Model 1
- this strengthens the case that growth heterogeneity is not just an artifact of preprocessing collapse

See:

- [phase1_multikc_qmatrix_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_qmatrix_comparison.md)

### Repeated-practice diagnostic

The stronger repeated-practice redesign on the restrictive single-KC family:

- strengthens the slope signal
- but still does not make Model 2 beat Model 1 on that restrictive family

## Current decision

Current disciplined reading:

- the operational full-data multi-KC branch supports Model 2 and then Model 3
- a stronger fractional multi-KC sensitivity also supports Model 2 and then Model 3
- the construct-clean single-KC branch still collapses to Model 1
- therefore DBE now supports a richer heterogeneity story on the full dataset, with remaining sensitivity mainly in effect size and construct purity rather than total survival of the richer models

So the current state is:

- **Model 1** remains the hurdle benchmark
- **Model 2** survives on the operational branch
- **Model 3** is the richest currently supported model on the operational branch
- **the remaining scientific issue is robustness of the multi-KC handling, not lack of heterogeneity signal**

## Scientific implication

For this project, the real question is whether public data support:

- different rates of learning, via Model 2
- or different rates plus different stability, via Model 3

The current DBE reading is:

- using the full dataset and all linked KCs changes the answer materially
- the full-data operational branch supports richer heterogeneity
- the fractional multi-KC sensitivity also supports richer heterogeneity
- the construct-clean single-KC sensitivity branch does not reproduce that result

So the current conclusion is not “only Model 1 matters.”

The current conclusion is:

- **Model 3 is the richest supported public model family on the operational branch**
- with a clear caveat that this support is not invariant to stricter KC-handling restrictions

## Phase 2 implication

The repo should not move into a Model 1-led Phase 2.

If the project proceeds from the current operational DBE branch, the richer supported public family is **Model 3**.

That means the future Phase 2 question is:

1. can the Model 3 structure replicate locally?
2. can public-informed Model 3 estimate those differences earlier than weak-prior local fitting?

The remaining gate before treating that as the scientific mainline is robustness of the multi-KC result.
