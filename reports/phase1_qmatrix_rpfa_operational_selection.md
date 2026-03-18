# Phase 1 R-PFA Operational Selection

This note records the operational model-selection result after promoting explicit Q-matrix **R-PFA** to the mainline learner-history family.

## Alpha tuning result

Model 2 was tuned over the fixed grid:

- `0.4`
- `0.6`
- `0.8`
- `0.9`
- `1.0`

Selection rule:

- primary selector: held-out log loss
- tie margin: `0.0002`
- tie break: choose the largest alpha within the tie margin

Result:

- best log loss: `0.541378` at `alpha = 0.8`
- selected alpha: `0.9`

Why `0.9` was selected:

- `alpha = 0.9` was within the pre-registered tie margin of the best log loss
- the rule then prefers the largest alpha inside that margin

Later policy-facing check:

- a direct Model 2 policy-suite comparison against `alpha = 0.8` also kept `0.9`
- the mean target-gap difference on the three new-learning policies stayed inside the tie margin
- `0.9` was slightly better on mean policy advantage and mean recommendation stability

Reference:

- [phase1_qmatrix_rpfa_tuning.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_tuning.md)
- [phase1_qmatrix_rpfa_policy_alpha_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_policy_alpha_comparison.md)

## Selected RPFA model comparison

Held-out evaluation on the same public test rows:

- RPFA Model 2:
  - log loss `0.541470`
  - Brier `0.183001`
  - AUC `0.764493`
  - calibration slope `0.957899`
- RPFA Model 3:
  - log loss `0.541660`
  - Brier `0.183103`
  - AUC `0.763996`
  - calibration slope `0.972057`

Interpretation:

- Model 3 improves calibration slope
- but Model 2 is better on log loss, Brier, AUC, and accuracy

So on predictive metrics alone, **Model 2 remains the better operational learner model**.

## Policy-facing replay reading

The five-policy offline replay suite compared:

- balanced challenge
- harder challenge
- confidence-building
- failure-aware remediation
- spacing-aware review

Main reading:

- Model 2 is more stable than Model 3 across all five policies
- Model 3 does not produce a consistent target-gap or policy-advantage win
- Model 3 changes recommendations materially, but those changes do not justify replacing Model 2 as the default policy model

Important review-mode note:

- the shared Model 2 vs Model 3 suite above used the common `48`-hour review threshold
- later spacing-only tuning on the operational Model 2 branch selected `24` hours as the current review-mode threshold

Reference:

- [adaptive_policy_suite_comparison.md](D:/model1_baseline_agent_bundle/reports/adaptive_policy_suite_comparison.md)
- [spacing_policy_due_review_grid.md](D:/model1_baseline_agent_bundle/reports/spacing_policy_due_review_grid.md)

## Operational decision

Keep the roles separate:

- **scientific heterogeneity result**:
  - the full-data explicit Q-matrix ladder still supports Model 3 as the richest public heterogeneity model
- **operational policy result**:
  - **explicit Q-matrix R-PFA Model 2** is the default deployment candidate
  - **explicit Q-matrix R-PFA Model 3** remains the uncertainty/stability challenger
  - keep `alpha = 0.9`
  - use `24` hours as the current spacing-review threshold on the Model 2 branch

That is the current repo mainline.
