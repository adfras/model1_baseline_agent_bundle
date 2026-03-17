# AGENTS.md

## Mission

This repository runs a small modelling programme for binary learner-response data.

The project is framed around **heterogeneity discovery first, then conditional replication and warm-start transfer**.

The core questions are:

1. **Phase 1:** Do learners differ in baseline level, growth, and stability in public longitudinal data?
2. **Phase 2A:** Only if Phase 1 supports Model 2 or Model 3 cleanly, do those same forms of heterogeneity replicate in a new local sample?
3. **Phase 2B:** Only after that replication step, do public-informed priors estimate those learner differences earlier than weak-prior local fitting?

## Project scope

Stay inside these boundaries unless the user explicitly expands scope:

- Binary outcome only: `correct` / `incorrect`
- Trial-level learner-response data with preserved chronology
- Public-data discovery first
- Local replication and warm-start are conditional, not automatic
- Calibration and log-score matter more than raw accuracy
- The first paper is about heterogeneity discovery, replication, and transfer
- This is not a full personalised learning system
- Offline next-question policy replay is allowed as a bridge task when the user explicitly wants to explore question selection without new local data

## Phase structure

### Phase 1: public heterogeneity discovery

Treat the public dataset as development and discovery data.

The model ladder is a sequence of nested heterogeneity tests:

1. **Model 1: level heterogeneity**
   - learner random intercept
   - item random intercept
   - one shared KC-specific practice term

2. **Model 2: growth heterogeneity**
   - Model 1 plus learner-specific KC-practice slopes

3. **Model 3: stability heterogeneity**
   - Model 2 plus latent learner-state deviations over time
   - learner-specific latent volatility

Interpretation:

- If Model 2 adds a practically non-trivial learner slope variance without failing the predictive gate, growth heterogeneity is present.
- If Model 3 adds a practically non-trivial learner stability variance without failing the predictive gate, stability heterogeneity is present.
- If only Model 1 survives, learners differ mainly in baseline level, and the public dataset has not given a stable answer to the main scientific question.

Model 1 is therefore a **hurdle model**, not the scientific endpoint.

### Pilot rule for the public dataset

If only Model 1 survives on the **full-dataset** public discovery dataset:

- freeze that dataset as a **pilot / screening dataset**
- do **not** treat Model 1 as the Phase 2 scientific target
- do **not** move to a warm-start study led only by Model 1
- either redesign the public discovery analysis to target Model 2 cleanly
- or replace the public dataset for the main discovery question

### Phase 2A: local structural replication

Phase 2A is a real replication step, not a single-model refit.

It is only activated if Phase 1 supports **Model 2 or Model 3** cleanly enough to justify a real heterogeneity replication question.

Replication rule:

- Model 1 always advances as the benchmark.
- The richest Phase 1-supported model advances as the primary structural model.
- The local replication ladder must therefore rerun the nested models up to the public-supported level.

Examples:

- If public Phase 1 ends at Model 2:
  - fit local Model 1 and local Model 2
- If public Phase 1 ends at Model 3:
  - fit local Models 1, 2, and 3

### Phase 2B: local warm-start application

- Fit the same Phase 2A-supported structure again with public-informed priors or hyperparameters.
- Compare weak-prior local fitting against public-informed warm-start fitting.
- Evaluate only on held-out local students.
- Focus especially on early local attempts.

## Skill routing

Use the matching skill based on the task:

- `model1-baseline-binary-logistic`
  Use for the baseline level-heterogeneity model.

- `model2-random-slope-binary-logistic`
  Use for the growth-heterogeneity extension.

- `model3-dynamic-volatility-binary-logistic`
  Use for the stability-heterogeneity extension.

- `phase2-transfer-warm-start`
  Use for local replication and warm-start transfer once the public model family is chosen.

## Data rules

- Never shuffle rows before splitting.
- Never do random row-level train/test splits.
- Preserve each learner's time order end to end.
- Use only variables available before or at the point of prediction.
- For the main public discovery analysis, use the **full visible dataset** and retain **all linked KCs per item**.
- Build a long attempt-KC table internally so each attempt updates every linked KC.
- Collapse back to one row per attempt for the current model ladder using an additive multi-KC practice signal:
  - `kc_opportunity` tracked separately for each `student_id x kc_id`
  - `practice_feature = sum(log1p(kc_opportunity_k))` across the attempt's linked KCs
- For the current **operational** learner-model branch, keep the explicit multi-KC representation and use **Q-matrix-aware PFA / R-PFA history features**:
  - KC-specific prior wins
  - KC-specific prior failures
  - when recency is enabled, KC-opportunity-lag-weighted prior wins and failures
- Keep wall-clock time out of the R-PFA history features themselves.
- Use timestamps only for spacing / review eligibility in offline policy replay.
- Keep single-KC-only and deterministic primary-KC branches as sensitivity analyses, not as the default mainline branch.
- For Models 1 and 2, exclude same-trial process variables from the main model.
- For Model 3, add only latent stability/state terms; do not convert the project into a same-trial behaviour-classification task.
- Keep a **single-KC sensitivity analysis** and, when useful, a repeated-practice diagnostic subset to show how conclusions change under stricter identification choices.

## Default split strategy

### Phase 1 public discovery

- Within learner, split in time order.
- Preferred starting split: **80% train / 20% test** within learner.
- Set and document a minimum-history threshold before fitting.

### Phase 2 local replication and warm start

- Split the local dataset by `student_id`.
- Use a **3-way student-wise split**:
  - `train`
  - `calibration`
  - `test`
- Keep the `test` students untouched until evaluation.

## Evaluation

### Phase 1 primary evidence

Use two evidence families together:

1. **Substantive evidence**
   - learner intercept variance above a practical floor
   - learner slope variance above a practical floor, when applicable
   - learner stability variance above a practical floor, when applicable
   - stability of those variance terms across reruns

2. **Predictive evidence**
   - held-out log loss
   - Brier score
   - calibration intercept and slope
   - calibration curve

Default practical floor for added heterogeneity SD terms beyond Model 1:

- posterior SD above `0.03` on the logit scale
- with the 94% HDI lower bound also above `0.03`

Predictive equivalence gate for richer models:

- either held-out log loss improves
- or log loss is no worse by more than `0.001`, while Brier improves and calibration slope moves closer to `1.0`

### Phase 2 primary outcome

Primary:

- student-averaged log loss over attempts `1-5`

Secondary:

- student-averaged log loss over attempts `1-10`
- Brier score
- calibration intercept and slope
- overall performance after more attempts

Sparse-data fallback:

- if too few students reach 5 attempts, pre-specify a shorter primary window such as attempts `1-3`

## Required deliverables

Unless the user asks otherwise, produce:

- a data dictionary / schema note
- preprocessing code
- split-generation code
- fitting code for each requested model
- evaluation code
- saved metrics tables
- calibration figures
- a concise methods summary
- a comparison note when more than one model is fitted
- for Phase 2, a transfer note describing what was carried from public to local data

## Decision rules

- Fit Models 1, 2, and 3 in order on the same **full-dataset multi-KC public discovery sample**.
- Use Model 1 only to answer the narrow question:
  - do Models 2 or 3 add anything real beyond baseline level differences?
- Use the **Variance + Prediction** rule:
  - an added heterogeneity component counts as present only if its SD clears the practical floor
  - and the richer model clears the predictive gate
- If Model 2 fails that rule, stop at Model 1.
- If Model 2 passes and Model 3 fails, stop at Model 2.
- If Model 3 passes and remains numerically stable, carry it forward as the richer public model family.
- If only Model 1 survives on that full-dataset branch, the public dataset is a pilot for this question, not the basis for a Model 1-led Phase 2.
- Only a public-supported Model 2 or Model 3 can activate the main Phase 2 scientific path.
- Keep the **scientific heterogeneity ladder** separate from the **operational policy model**:
  - scientific conclusion comes from the explicit Q-matrix heterogeneity ladder
  - operational learner-model choice comes from the explicit Q-matrix PFA / R-PFA branch
- Default operational rule:
  - Model 2 is the default policy model
  - Model 3 is only preferred if it wins on the policy-facing replay metrics or gives a calibration benefit that matters to the chosen policy

## Offline policy replay

When the user explicitly wants question-selection work before local data exists:

- use the fitted learner model only as a **state estimator / probability scorer**
- compare a small family of modular policies, not one monolithic optimizer
- keep claims limited to **offline target-control / policy-behavior evaluation**
- do not claim causal learning gains without suitable logging propensities or randomized data

Current default v1 policy suite:

- `balanced_challenge`
- `harder_challenge`
- `confidence_building`
- `failure_aware_remediation`
- `spacing_aware_review`

## Preferred stack

- If the repository already has a modelling stack, stay in that stack.
- Otherwise default to **Python 3.11 + pandas + pyarrow + Bambi/PyMC + ArviZ + matplotlib**.
- Keep all code modular so the same KC-aware schema and evaluation logic can be reused in Phase 2.

## Working style

- Keep implementation simple, inspectable, and reproducible.
- Make grounded assumptions, document them briefly, and keep moving.
- Do not expand into BKT, DKT, RNNs, transformers, or a live personalised learning system unless the user explicitly changes scope.
- If the user explicitly shifts toward question selection, keep it to offline replay and policy comparison using the fitted learner models rather than inventing a full intervention framework.
- Preserve the distinction between:
  - **Phase 1:** public heterogeneity discovery
  - **Phase 2A:** conditional local structural replication
  - **Phase 2B:** conditional local warm-start application
