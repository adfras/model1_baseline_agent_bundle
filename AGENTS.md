# AGENTS.md

## Mission
This repository exists to run a **two-phase, small-scope modelling programme** for trial-level learner response data.

The project is deliberately **not** a full personalised learning system.
It has two aims only:

1. **Phase 1:** develop and compare three sequential models on a public online-learning dataset.
2. **Phase 2:** take the chosen Phase 1 model and use it as a **warm start** for a new local sample, so prediction for new local students does not begin from a blank slate.

## Project scope
Stay inside these boundaries unless the user explicitly expands scope:
- Binary outcome only: `correct` / `incorrect`.
- Trial-level learner-response data with preserved chronology.
- Public-data development first; local-data transfer second.
- Probabilistic next-attempt prediction is the primary outcome.
- Calibration and log-score matter more than raw accuracy.
- The first paper is about **forecasting and transfer**, not adaptive tutoring, rule-change control, or full personalised learning.

## Phase structure

### Phase 1: public-data model development
Fit the following model ladder in order:

1. **Model 1: baseline hierarchical logistic**
   - learner random intercept
   - item random intercept
   - one shared practice term

2. **Model 2: random-slope learner-growth model**
   - Model 1 plus learner-specific practice slopes
   - separates baseline proficiency from learning rate

3. **Model 3: dynamic learner-volatility model**
   - Model 2 plus latent learner state deviations over time
   - learner-specific latent volatility
   - used to test whether instability improves probabilistic forecasts beyond Model 2

### Phase 2: public-to-local transfer
- Freeze the Phase 1 model family after comparison.
- Transfer the **model structure** and **public-learned hyperparameters/priors** to the local dataset.
- Compare:
  - a **weak-prior local fit** that starts nearly from scratch
  - a **public-informed warm-start fit** using Phase 1 priors
- Evaluate only on **held-out local students**.
- Focus especially on **early local attempts**.

## Skill routing
Use the matching skill based on the task:

- **`model1-baseline-binary-logistic`**
  Use for the public-data baseline.

- **`model2-random-slope-binary-logistic`**
  Use for the public-data random-slope extension.

- **`model3-dynamic-volatility-binary-logistic`**
  Use for the public-data volatility/uncertainty extension.

- **`phase2-transfer-warm-start`**
  Use after Phase 1 when transferring the chosen model family to the local dataset.

## Data rules
- Never shuffle rows before splitting.
- Never do random row-level train/test splits.
- Preserve each learner's time order end to end.
- Use only variables available before or at the point of prediction.
- For Models 1 and 2, exclude same-trial process variables that are unavailable until the attempt is underway or finished.
  - Examples usually excluded from the main models: `time_taken`, `hint_used`, `num_answer_changes`, final confidence, and other in-progress/post-response signals.
- For Model 3, only add latent volatility/state terms; do not silently convert the project into a same-trial behaviour-classification task.
- If later rows contain unseen items, report them explicitly.
  - For the primary first-paper analysis, prefer a main evaluation on later rows whose item IDs were seen in the relevant training data.
  - Report the number and proportion of excluded unseen-item rows.

## Default split strategy
### Phase 1 Track A: seen-learner forward prediction
- Within learner, split in time order.
- Preferred starting split: **80% train / 20% test** within learner.
- If validation is needed: **70/15/15** within learner.
- Set and document a minimum-history threshold before fitting.

### Phase 1 Track B: unseen-public-student transfer
- Split by `student_id` into train / validation / test.
- Train on public train students only.
- For each public test student, predict sequentially from their first observed local row onward.

### Phase 2: local external validation
- Split the local dataset by `student_id`.
- Keep a small **local calibration subset** for local item effects / offsets.
- Keep an **untouched local external-test set** of students for the main transfer evaluation.

## Evaluation
Primary metrics for every phase/model:
- held-out **log loss / mean log predictive density**
- **Brier score**
- **calibration intercept and slope**
- **calibration curve / reliability plot**

Secondary metrics:
- accuracy
- AUC

For Phase 2, also report metrics by early-attempt windows such as:
- attempts 1-5
- attempts 6-10
- attempts 11-20

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
- Fit Models 1, 2, and 3 in order.
- If Model 2 does not improve meaningfully over Model 1 on primary probabilistic metrics, stop and simplify.
- If Model 3 does not improve meaningfully over Model 2 on primary probabilistic metrics and calibration, do not carry Model 3 into Phase 2 by default.
- Only transfer the most complex model that clearly earns its keep.

## Preferred stack
- If the repository already has a modelling stack, stay in that stack.
- Otherwise default to **Python 3.11 + pandas + pyarrow + Bambi/PyMC + ArviZ + matplotlib**.
- Keep all code modular so the same data schema and evaluation code can be reused in Phase 2.

## Working style
- Keep implementation simple, inspectable, and reproducible.
- Make grounded assumptions, document them briefly, and keep moving.
- Do not expand into BKT, DKT, RNNs, transformers, adaptive policy learning, or full personalised learning unless the user explicitly changes scope.
- Preserve the distinction between:
  - **Phase 1:** public-data model development
  - **Phase 2:** warm-start transfer to local students
