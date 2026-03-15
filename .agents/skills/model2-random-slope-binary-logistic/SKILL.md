---
name: model2-random-slope-binary-logistic
description: Use this skill when the task is to fit the second Phase 1 model for trial-level learner-response data with a binary outcome. It extends the baseline by adding learner-specific practice slopes while preserving chronological forward prediction. Do not use it for volatility models, public-to-local transfer, or adaptive decision rules.
---

# Model 2: hierarchical logistic model with learner-specific learning slopes

## Purpose
Implement the **small next step after Model 1**.

This skill exists to answer one question:

**Do learners differ not only in baseline success probability, but also in how quickly their probability of success changes with practice, and does modelling that improve forward predictions?**

## Trigger conditions
Use this skill when most of the following are true:
- The task explicitly mentions Model 2, random slopes, learner-specific learning rates, or students differ in rate of change.
- The data are trial-level learner interactions with a binary outcome.
- The goal is still Phase 1 forward next-attempt prediction.
- The user wants interpretable differences such as stronger, learning, or struggling learners.

## Do not use this skill when
Do not use this skill if the task asks for:
- learner-specific volatility / heteroskedasticity
- dynamic latent state-space models
- public-to-local transfer as the main task
- adaptive pedagogy or personalised-learning controllers
- same-trial process-variable classification as the main task

## Required input schema
At minimum, the dataset must support these columns or equivalents:
- `student_id`
- `item_id`
- `correct` (0/1)
- `timestamp` or a valid within-student order column

Derived columns required:
- `trial_index_within_student`
- `overall_opportunity = trial_index_within_student - 1`
- `practice_feature = log1p(overall_opportunity)`

## What not to use in Model 2
Keep the same exclusions as Model 1 unless the user explicitly requests an auxiliary model:
- `time_taken`
- `hint_used`
- `num_answer_changes`
- final confidence or other same-trial process variables

## Default model specification
\[
y_{it} \sim \text{Bernoulli}(p_{it})
\]

\[
\text{logit}(p_{it}) = \beta_0 + \alpha_i + \gamma_j + (\beta_{\text{practice}} + b_i)\,\log(1 + \text{opp}_{it})
\]

\[
\begin{pmatrix}
\alpha_i \\
 b_i
\end{pmatrix}
\sim \mathcal{N}\left(
\begin{pmatrix}0\\0\end{pmatrix},
\Sigma_{\text{student}}
\right)
\]

\[
\gamma_j \sim \mathcal{N}(0, \sigma_{\text{item}})
\]

In mixed-model notation, the target structure is roughly:

```text
correct ~ practice_feature + (1 + practice_feature | student_id) + (1 | item_id)
```

## Interpretation rules
Use fitted learner effects cautiously and descriptively:
- higher intercept, flatter slope = stronger from the start
- lower intercept, positive slope = learning over time
- lower intercept, flatter slope = struggling or not improving much
- strong negative slope = possible fatigue, instability, or mean-structure mismatch

## Workflow
1. Reuse the same cleaned trial table and splits from Model 1.
2. Fit Model 1 first.
3. Fit Model 2 with learner-specific practice slopes.
4. Compare Model 1 and Model 2 on the same holdout rows.
5. Report whether the added complexity improves forward probability prediction enough to justify itself.

## Evaluation
Use the same metrics as Model 1:
- held-out log loss / mean log predictive density
- Brier score
- calibration intercept and slope
- calibration curve / reliability plot

Secondary metrics:
- accuracy
- AUC

Also report:
- comparison table versus Model 1
- learner-level distribution of slope estimates or posterior summaries
- convergence diagnostics and any singular-fit warnings or divergences
