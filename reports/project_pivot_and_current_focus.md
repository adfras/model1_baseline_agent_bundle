# Project Pivot And Current Focus

## Why this note exists

The repo moved through several real pivots, and the old notes now overstate the role of offline policy replay.

This note states the current mainline plainly.

## Original framing

The original scientific question was:

**Do learners differ in baseline performance, rate of improvement, and response stability in public longitudinal data, and do those same forms of heterogeneity replicate in a new student sample?**

The applied follow-up was:

**Can public-informed priors estimate those learner differences earlier in new students than weak-prior local fitting?**

## What changed

Three things changed the practical focus.

### 1. The data representation had been hiding signal

Earlier branches either dropped multi-KC items or flattened KC structure too aggressively.

The project improved materially once it:

- kept the full visible DBE dataset
- retained all linked KCs
- moved KC structure into the likelihood directly

### 2. Better history features mattered more than extra latent-state tuning

The biggest predictive gain came from moving from opportunity-only history to **PFA / R-PFA wins and fails per KC**.

That established the current operational replay baseline:

- explicit Q-matrix **R-PFA Model 2**
- `alpha = 0.9`
- `24`-hour review threshold

### 3. The ManyLabs comparison clarified the mismatch

In `C:\ManyLabsAnalyses`, heterogeneity is useful because it is modeled directly inside the optimized likelihood and held-out objective.

In DBE, heterogeneity has mostly been tested as a **downstream policy aid** on top of a replay setup.

That is a harder and less aligned use case.

Reference:

- [manylabs_dbe_alignment_note.md](D:/model1_baseline_agent_bundle/reports/manylabs_dbe_alignment_note.md)

## Current focus

The repo is now centered on:

1. **full-data public heterogeneity discovery**
2. **learner-state estimation from the scientific explicit Q-matrix ladder**
3. **decision-native future design requirements**

The repo is **not** currently centered on proving an adaptive-question-selection win on DBE.

## Current answers

### Scientific answer

On the full-data explicit Q-matrix ladder:

- Model 2 beats Model 1
- Model 3 beats Model 2

So the current public scientific result is:

- baseline heterogeneity is present
- growth heterogeneity is present
- stability heterogeneity is present

### Learner-state answer

The repo now treats learner-state exports as a first-class DBE deliverable.

Current exported artifacts:

- [model2_learner_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model2_learner_profiles.csv)
- [model3_learner_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model3_learner_profiles.csv)
- [model3_latent_state_profiles.csv](D:/model1_baseline_agent_bundle/outputs/phase1_multikc_qmatrix_profiles/model3_latent_state_profiles.csv)
- [phase1_qmatrix_learner_state_profiles.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_learner_state_profiles.md)

These outputs put the scientific heterogeneity result into a learner-level form that can later support replication, warm-start, or a future decision-native system.

### Operational DBE replay answer

The replay work stays in the repo, but it is now a **bridge / negative-result track**.

Current read:

- raw **R-PFA Model 2** remains the best replay scorer
- Model 3 remains the richest scientific heterogeneity model
- Model 3 has not earned an operational role on DBE replay
- the later uncertainty, local-residual, and direct heterogeneity policy branches remain negative results

So the DBE replay conclusion is:

- the repo does **not yet** have an adaptive-question-selection win on DBE

Reference:

- [direct_heterogeneity_policy_decision.md](D:/model1_baseline_agent_bundle/reports/direct_heterogeneity_policy_decision.md)
- [current_objective_and_failure_mode.md](D:/model1_baseline_agent_bundle/reports/current_objective_and_failure_mode.md)

## What is paused

These remain scaffolded but paused:

- local structural replication
- local warm-start transfer

Reason:

- no local dataset is currently available in this workspace

## What comes next

Until local data arrives, the sensible mainline is:

1. keep the scientific explicit Q-matrix heterogeneity ladder as the source of truth
2. keep exporting learner-level baseline, growth, stability, and latent-state summaries
3. keep the frozen DBE replay baseline documented, but do not treat it as the repo’s central success criterion
4. keep future next-item work in the repo at the **design-spec** level unless stronger decision-native data become available

Reference:

- [decision_native_successor_spec.md](D:/model1_baseline_agent_bundle/reports/decision_native_successor_spec.md)
- [current_objective_and_failure_mode.md](D:/model1_baseline_agent_bundle/reports/current_objective_and_failure_mode.md)

## Bottom line

The project is now focused on:

- **public heterogeneity discovery**
- **learner-state estimation**
- **future decision-native system design**

The current DBE policy work remains:

- informative bridge work
- but not the repo’s mainline claim
