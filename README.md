# Learner Heterogeneity and Adaptive Question Selection

This repository started as a public-data heterogeneity programme with conditional local replication and warm-start transfer.

That framing is still in the repo, but the **current operational focus** has shifted:

1. use the **full visible DBE dataset**
2. model multi-KC structure explicitly
3. improve learner-state estimation with better history features
4. test whether those learner models help with **offline next-question targeting**

This is still **not** a full live personalised learning system. It is the current offline bridge from heterogeneity modelling to user-specific question selection.

Core docs:

- [AGENTS.md](D:/model1_baseline_agent_bundle/AGENTS.md)
- [PROJECT_PLAN.md](D:/model1_baseline_agent_bundle/PROJECT_PLAN.md)
- [phase1_selection_memo.md](D:/model1_baseline_agent_bundle/reports/phase1_selection_memo.md)
- [current_project_status.md](D:/model1_baseline_agent_bundle/reports/current_project_status.md)
- [phase1_branch_guide.md](D:/model1_baseline_agent_bundle/reports/phase1_branch_guide.md)
- [project_pivot_and_current_focus.md](D:/model1_baseline_agent_bundle/reports/project_pivot_and_current_focus.md)
- [phase1_qmatrix_rpfa_tuning.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_tuning.md)
- [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)
- [phase1_qmatrix_rpfa_policy_alpha_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_policy_alpha_comparison.md)
- [adaptive_policy_suite_comparison.md](D:/model1_baseline_agent_bundle/reports/adaptive_policy_suite_comparison.md)
- [spacing_policy_due_review_grid.md](D:/model1_baseline_agent_bundle/reports/spacing_policy_due_review_grid.md)
- [hybrid_uncertainty_router.md](D:/model1_baseline_agent_bundle/reports/hybrid_uncertainty_router.md)
- [hybrid_uncertainty_router_v2.md](D:/model1_baseline_agent_bundle/reports/hybrid_uncertainty_router_v2.md)

## Current focus

Until local data is available, the repo is focused on three linked questions:

1. Do full-data KC-aware public models support learner heterogeneity beyond baseline level?
2. Which public learner model gives the best **operational** fit once KC history is represented properly?
3. Which learner model is better for **offline next-question targeting**?

Current answers:

- The full-data explicit Q-matrix ladder supports **Model 2** and then **Model 3**.
- The best predictive yield came from switching from opportunity-only history to **PFA / R-PFA wins/fails history**.
- The operational learner-model mainline is now the **explicit Q-matrix R-PFA branch** with selected `alpha = 0.9`.
- A direct policy-facing comparison against `alpha = 0.8` kept `0.9` in place.
- The offline policy work now uses a **modular policy suite**, not only one fixed target-`0.7` replay.
- The current selected review-mode threshold for `spacing_aware_review` is `24` hours on the operational Model 2 branch.
- On that branch, **R-PFA Model 2** remains the default policy model and **R-PFA Model 3** remains the richer challenger.

## Current mainline results

### Explicit Q-matrix heterogeneity ladder

Same held-out rows, one row per attempt, KC structure inside the likelihood:

- Model 1 log loss: `0.545311`
- Model 2 log loss: `0.544366`
- Model 3 log loss: `0.543782`

Interpretation:

- baseline heterogeneity is present
- growth heterogeneity is present
- stability heterogeneity is also present

Reference:

- [phase1_multikc_qmatrix_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_qmatrix_comparison.md)

### Best-yield improvement branch family: explicit Q-matrix PFA / R-PFA

Replacing opportunity-only history with KC-specific prior wins and fails gave the largest improvement tried so far:

- R-PFA Model 2, selected `alpha = 0.9`:
  - log loss `0.541470`
  - Brier `0.183001`
  - AUC `0.764493`
  - calibration slope `0.957899`
- R-PFA Model 3, same `alpha = 0.9`:
  - log loss `0.541660`
  - Brier `0.183103`
  - AUC `0.763996`
  - calibration slope `0.972057`

Interpretation:

- the **history representation** was the main leverage point
- the tie-broken operational alpha is `0.9`
- a direct Model 2 policy-suite comparison against `alpha = 0.8` also kept `0.9`
- R-PFA Model 2 is the best current operational predictive model
- R-PFA Model 3 remains the richer uncertainty/stability challenger

Reference:

- [phase1_qmatrix_rpfa_tuning.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_tuning.md)
- [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)
- [phase1_qmatrix_rpfa_policy_alpha_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_policy_alpha_comparison.md)

### Offline adaptive-policy replay

Current replay design:

- after each held-out attempt, score candidate items
- compare a small modular policy family:
  - balanced challenge
  - harder challenge
  - confidence-building
  - failure-aware remediation
  - spacing-aware review

Current reading:

- `balanced_challenge`: Model 2 and Model 3 are close, but Model 2 is more stable and has slightly better policy advantage over the actual historical next item.
- `confidence_building`: Model 2 is better on target gap, policy advantage, and stability.
- `failure_aware_remediation`: Model 2 is slightly better on target gap, band-hit rate, policy advantage, and stability.
- `harder_challenge`: Model 3 has a tiny policy-advantage edge, but Model 2 is still better on target gap and stability.
- `spacing_aware_review`: Model 3 has a tiny band-hit edge, but Model 2 is better on target gap, policy advantage, and stability.

Interpretation:

- Model 2 remains the default policy model
- Model 3 changes recommendations materially, but those changes do not justify replacing Model 2 under the current replay suite
- this is an **offline target-control / policy-behavior test**, not proof of causal learning gain
- the fixed shared suite above used the common `48`-hour review threshold
- later review-mode tuning on the operational Model 2 branch selected `24` hours for `spacing_aware_review`

Reference:

- [adaptive_policy_suite_comparison.md](D:/model1_baseline_agent_bundle/reports/adaptive_policy_suite_comparison.md)
- [spacing_policy_due_review_grid.md](D:/model1_baseline_agent_bundle/reports/spacing_policy_due_review_grid.md)

### Hybrid uncertainty routers

I also ran a first hybrid router that uses:

- **Model 2** for mean success probabilities
- **Model 3** for step-level uncertainty

Current reading:

- it improves **routing coverage** for recent-failure and due-review situations
- it does **not** beat the fixed-policy suite on pure target-gap control
- so it is a useful prototype for uncertainty-aware routing, not the new default policy

A second-generation router now adds lagged observable proxies such as:

- failure streak
- recent success rate
- hint-use rate
- answer-change friction
- response-time inflation

Current reading:

- raw v2 was too aggressive and degraded target-gap control
- tuned v2 improved target gap and policy advantage over v1 while reducing seen-item recommendations
- tuned v2 is still materially less stable than v1 and much less target-precise than the fixed Model 2 policies
- so the tuned v2 router is the current **exploratory** hybrid branch, not the default policy

Reference:

- [hybrid_uncertainty_router.md](D:/model1_baseline_agent_bundle/reports/hybrid_uncertainty_router.md)
- [hybrid_uncertainty_router_v2.md](D:/model1_baseline_agent_bundle/reports/hybrid_uncertainty_router_v2.md)

## Public data workflow

Build the full multi-KC tables:

```powershell
py src/preprocess_phase1_multikc.py
```

Main outputs:

- [multikc_trials.csv](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_trials.csv)
- [multikc_attempt_kc_long.csv](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_attempt_kc_long.csv)
- [multikc_summary.json](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_summary.json)
- [phase1_multikc_schema_note.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_schema_note.md)
- [kc_history_feature_validation.md](D:/model1_baseline_agent_bundle/reports/kc_history_feature_validation.md)

Current sample:

- `157,989` processed attempt rows
- `1,138` learners
- `212` items
- `93` KCs
- `300,246` attempt-KC rows

## Branch guide

The repo now has several distinct public branches. Use the guide before reading any older report:

- [phase1_branch_guide.md](D:/model1_baseline_agent_bundle/reports/phase1_branch_guide.md)

Key distinction:

- `single-KC` is now only a restrictive sensitivity check
- `explicit Q-matrix PFA / R-PFA` is the current operational modeling branch

## Phase 2 status

Phase 2 scaffolding remains in the repo, but **Phase 2 is on hold** because no local dataset is currently available in this workspace.

What remains available:

- [phase2_protocol.md](D:/model1_baseline_agent_bundle/reports/phase2_protocol.md)
- [preprocess_phase2_local.py](D:/model1_baseline_agent_bundle/src/preprocess_phase2_local.py)
- [split_phase2_local.py](D:/model1_baseline_agent_bundle/src/split_phase2_local.py)

Current practical priority is:

- strengthen public learner-state modeling
- compare learner models under offline next-question policies
- prepare the transfer path for when local data arrives
