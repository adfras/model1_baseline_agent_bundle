# Project Plan: Full-Data Heterogeneity Discovery, Then RPFA Policy Prototyping

## Summary

The original scientific question remains:

**Do learners differ in baseline level, growth, and stability in public longitudinal data, and do those same forms of heterogeneity replicate in a new student sample?**

But the current project state adds an important operational pivot:

- local data is not yet available
- the full-data public branch now supports richer learner heterogeneity
- the best predictive yield came from better KC-history features, not from warm-start work

So the **current mainline focus** is:

1. keep the public heterogeneity ladder scientifically coherent
2. use the full dataset with explicit multi-KC structure
3. upgrade the operational learner-model branch from plain PFA to **R-PFA**
4. make learner-state estimates operational for **offline next-question targeting**
5. hold local replication and warm-start as dormant scaffolding until local data exists

## Current framing

There are now two linked but distinct evaluation targets.

### 1. Scientific ladder

Use the nested models as heterogeneity tests:

- **Model 1:** baseline level heterogeneity
- **Model 2:** growth heterogeneity
- **Model 3:** stability heterogeneity

Decision rule:

- an added heterogeneity SD must clear the practical floor
- the richer model must clear the predictive gate

### 2. Operational question-selection bridge

Once a public learner model is supported, ask:

- which learner model gives the best **question-selection behavior**?
- can it pick next items at the desired difficulty for each learner?

This is currently evaluated by **offline replay**, not by live experimentation.

Operational rule:

- default deployment candidate: **Model 2**
- richer challenger: **Model 3**
- Model 3 is only preferred if it wins on policy-facing replay metrics or gives a calibration gain that matters to the chosen policy

## Phase 1: public discovery and learner-model development

### Main public data representation

Use the **full visible DBE-KT22 attempt table** and retain **all linked KCs per item**.

Current operational branches:

1. **explicit Q-matrix opportunity branch**
   - KC structure enters the likelihood directly
   - used to establish that Models 2 and 3 survive on the full dataset

2. **explicit Q-matrix PFA / R-PFA branch**
   - retains the explicit Q-matrix structure
   - replaces opportunity-only history with KC-specific prior wins and fails
   - adds KC-opportunity-lag recency weighting for the operational R-PFA branch
   - current best-yield predictive branch family

Sensitivity branches remain in the repo:

- single-KC-only
- deterministic primary-KC
- collapsed-feature multi-KC
- fractional multi-KC
- repeated-practice restrictive subset

### Current Phase 1 results

#### Explicit Q-matrix ladder

- Model 1 log loss `0.545311`
- Model 2 log loss `0.544366`
- Model 3 log loss `0.543782`

Interpretation:

- baseline heterogeneity is present
- growth heterogeneity is supported
- stability heterogeneity is also supported

#### Best-yield branch family: explicit Q-matrix PFA / R-PFA

- plain PFA established that the main predictive gain came from improving the KC-history signal
- the current operational mainline is now the **R-PFA tuning / fit branch**
- the selected R-PFA alpha controls how strongly recent KC outcomes outweigh older ones

### Current Phase 1 decision reading

Separate two conclusions clearly:

1. **heterogeneity conclusion**
   - full-data explicit Q-matrix evidence supports Model 2 and then Model 3

2. **operational model choice**
   - explicit Q-matrix R-PFA Model 2 is the default mainline model when predictive yield and downstream policy use matter most
   - explicit Q-matrix R-PFA Model 3 is the uncertainty/stability challenger

That means Model 3 is still scientifically useful, but it is no longer the default answer to every operational question.

## Phase 1.5: offline next-question policy evaluation

Because the end goal is user-specific question selection, the project now includes a narrow offline policy layer.

### Current replay task

For each held-out student:

1. update the student state after each observed attempt
2. score candidate items with the fitted learner model
3. apply a small modular policy family rather than one fixed target rule

Current v1 policy suite:

- balanced challenge
- harder challenge
- confidence-building
- failure-aware remediation
- spacing-aware review

This replay layer measures:

- target-difficulty control
- band-hit rate
- recommendation stability
- remediation and review coverage
- fallback behavior

It does **not** yet measure:

- causal learning gains
- engagement effects with ground-truth labels
- long-term retention

## Current focus

Until local data arrives, the repo should prioritize:

1. explicit Q-matrix R-PFA as the operational learner-model mainline
2. Model 2 as the default policy model
3. Model 3 as the richer challenger when stability or uncertainty matters
4. offline policy comparisons such as:
   - balanced target difficulty
   - slightly harder challenge
   - easier confidence-building
   - failure-aware remediation
   - spacing-aware review
5. engagement-proxy work only through observable history and behavior proxies, not invented labels

## Phase 2: currently paused

### Phase 2A: local structural replication

Still the correct long-term replication step:

- rerun the local ladder up to the public-supported level
- benchmark with Model 1
- evaluate whether the richer structure replicates locally

### Phase 2B: local warm-start

Still the correct later application step:

- weak-prior local fit
- public-informed fit
- early-attempt comparison on held-out local students

### Why it is paused

- no local dataset is currently available in this workspace
- the repo’s immediate value comes from getting the public learner model and offline policy logic right first

## Deliverables that now matter most

- full-data KC-aware preprocessing
- explicit Q-matrix fit/eval scripts
- PFA / R-PFA fit/eval scripts
- adaptive replay scripts
- comparison notes for:
  - heterogeneity ladder
  - improvement trials
  - R-PFA alpha tuning
  - offline policy suite replay
- a pivot note documenting what the project is focused on now
