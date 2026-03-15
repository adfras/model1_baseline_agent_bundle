---
name: phase2-transfer-warm-start
description: Use this skill when the task is to take the chosen Phase 1 public-data model and transfer it to a local dataset so new local students do not start from scratch. It applies the public-learned structure and hyperparameters as priors, adds local calibration terms, and compares a weak-prior local fit against a public-informed warm-start fit on held-out local students.
---

# Phase 2: public-to-local transfer with warm-start priors

## Purpose
Implement the **external-validation / warm-start phase** after Phase 1 is complete.

This skill exists to answer one question:

**Does using public-data-informed priors improve early prediction for new local students compared with a weak-prior local model that starts nearly from scratch?**

## Trigger conditions
Use this skill when most of the following are true:
- Phase 1 has been completed and a preferred model family has been chosen.
- The user wants to apply the public-trained model family to a local sample.
- The goal is to avoid starting from zero on new local students.
- Evaluation should be done on held-out local students.

## Do not use this skill when
Do not use this skill if:
- Phase 1 is incomplete.
- The public-data model family is still undecided.
- The user wants a live adaptive tutoring policy rather than an external-validation study.
- The user wants to transfer exact public student coefficients to local students.

## Core principle
Do **not** transfer exact public student effects to local students.

Transfer:
- the chosen model structure
- the public-learned hyperparameters / priors
- the typical spread of student baseline effects
- the typical spread of student growth effects
- for Model 3 only, the typical spread of learner volatility

Then update each local student's parameters from those priors as their local responses arrive.

## Required local data schema
At minimum, the local dataset must support these columns or equivalents:
- `student_id`
- `item_id`
- `correct` (0/1)
- `timestamp` or a valid within-student order column

Derived columns required:
- `trial_index_within_student`
- `overall_opportunity = trial_index_within_student - 1`
- `practice_feature = log1p(overall_opportunity)`

## Local split design
Create two local subsets:
1. **Local calibration subset**
   - used only to estimate local offsets and local item effects
2. **Untouched local external-test students**
   - used for the main evaluation of warm-start transfer

If the local dataset is small, use repeated student-wise cross-validation rather than a single permanent split.

## Model forms
### Weak-prior local fit
Fit the chosen model family to local data using broad or minimally informative priors.

### Public-informed warm-start fit
Fit the same model family to local data, but use public-informed priors for the student-level structure.

Examples:
- If the chosen family is Model 2:
  - local learner intercepts: `u_j ~ N(0, sigma_u_public^2)`
  - local learner slopes: `b_j ~ N(0, sigma_b_public^2)`
- If the chosen family is Model 3:
  - also use a public-informed prior for learner volatility

## Local calibration terms
When local items differ from public items, include:
- a **local overall offset**
- **local item effects**

Do not assume public item coefficients transfer directly unless the items are genuinely the same.

## Evaluation
Compare weak-prior and public-informed fits on:
- held-out local students only
- sequential prediction as each local student's responses arrive
- early attempt windows, especially:
  - attempts 1-5
  - attempts 6-10
  - attempts 11-20

Primary metrics:
- log loss / mean log predictive density
- Brier score
- calibration intercept and slope
- calibration curve / reliability plot

Secondary metrics:
- accuracy
- AUC

## Reporting
The transfer write-up must clearly distinguish:
- what came from the public data
- what was newly estimated from the local calibration subset
- what was evaluated only on untouched local students

## Done means
The task is done only when all of the following are true:
- the chosen Phase 1 model family is explicit
- the weak-prior and public-informed local fits use the same local test students
- local evaluation is student-wise, not row-wise
- early-attempt performance is reported explicitly
- the write-up makes clear whether the public-informed warm start improved local new-student prediction
