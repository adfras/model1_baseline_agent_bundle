# AGENTS.md

## Mission
This repository exists to implement and evaluate **Model 1**, a small-scope baseline for trial-level learner response data. The goal is **not** a full personalised learning system. The goal is a reproducible baseline hierarchical logistic model that predicts next-attempt correctness for **already-observed learners**.

## Load the baseline skill
When a task mentions **Model 1**, **baseline**, **DBE-KT22**, **learner-response data**, **next-attempt prediction**, **binary outcomes**, or **forward holdouts**, load the skill `model1-baseline-binary-logistic`.

## Scope guardrails
Stay inside these boundaries unless the user explicitly expands scope:
- Binary outcome only: `correct` / `incorrect`.
- Already-observed learners only.
- Chronological forward prediction only.
- Primary question: does the baseline model produce sensible next-attempt probability forecasts?
- No learner-specific volatility terms.
- No latent state-space models, HMMs, BKT, DKT, or neural KT.
- No adaptive decision rules, mastery controllers, or personalised learning workflows.
- No cold-start, new-student, or new-item generalisation claims in the primary analysis.
- No subgroup fairness analyses unless explicitly requested.

## Default model definition
Use this as the starting baseline:

\[
\text{logit}(P(y_{it}=1)) = \beta_0 + \alpha_i + \gamma_j + \beta_{\text{practice}}\,\log(1+\text{opp}_{it})
\]

Where:
- \(y_{it}\) is correctness for learner \(i\) on item \(j\) at trial \(t\)
- \(\alpha_i\) is a learner random intercept
- \(\gamma_j\) is an item random intercept / difficulty term
- \(\text{opp}_{it}\) is prior opportunity count within learner before trial \(t\)

If the dataset has a stable `skill_id` and good coverage, a **secondary sensitivity analysis** may replace `log(1 + overall_opportunity)` with `log(1 + skill_opportunity)`. Do **not** make that the default.

## Preferred stack
- If the repository already has a modelling stack, stay in that stack.
- Otherwise default to **Python 3.11 + pandas + pyarrow + Bambi/PyMC + ArviZ + matplotlib**.
- Keep the code easy to extend later to a volatility-aware model.

## Data rules
- Never shuffle rows before splitting.
- Never do random row-level train/test splits.
- Preserve each learner's time order.
- Use only variables available **before** the outcome is known.
- Exclude same-trial post-outcome variables from Model 1 unless explicitly asked for an auxiliary analysis. This usually means excluding variables such as final `time_taken`, `hint_used`, `num_answer_changes`, or other signals collected during/after the attempt.
- If later test rows contain items never seen in training, do not silently drop them without reporting it.
  - For the **primary** first-paper analysis, prefer restricting the main evaluation to later rows whose item IDs were observed in training.
  - Report the number and proportion of excluded new-item rows.

## Default split strategy
For each learner with enough history, fit on the early portion and evaluate on the later portion.
- Preferred starting split: **80% train / 20% test within learner**, in time order.
- If validation is needed, use **70/15/15** in time order.
- Set and document a minimum-history threshold before fitting.
- Keep split code deterministic and save the split assignments.

## Evaluation
Primary metrics:
- held-out **log loss / log predictive density**
- **Brier score**
- **calibration plot**
- **calibration intercept and slope**

Secondary metrics:
- accuracy
- AUC

Report metrics overall and, when sensible, as distributions across learners.

## Required deliverables
Unless the user asks otherwise, produce:
- a schema and feature note
- preprocessing code
- model fitting code
- evaluation code
- a concise methods summary
- saved metrics and figures
- a reproducible config or single command to rerun the pipeline

## Data retrieval record
- Primary dataset: **DBE-KT22** from ADA Dataverse.
- Source DOI: `10.26193/6DZWOH`
- Landing page: <https://doi.org/10.26193/6DZWOH>
- Scripted retrieval command: `py src/fetch_dbe_kt22.py`
- Default local destination: `data/raw/DBE-KT22/`
- Treat downloaded and extracted dataset files as local working data. Do **not** commit them to version control.
- Keep enough documentation in Markdown so another agent can re-fetch the data without relying on prior chat context.

## Working style
- Keep the implementation simple, inspectable, and reproducible.
- Make grounded assumptions, document them in a short assumptions note, and keep moving.
- Do not add extensions just because they are possible.
- Leave the repository in a state that is easy to extend later to a learner-volatility model.
