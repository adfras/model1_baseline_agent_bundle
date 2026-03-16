# Current Project Status

This note maps the current repository state onto the newer two-phase project plan.

## Repo Contract

The repo now follows:

- [AGENTS.md](/D:/model1_baseline_agent_bundle/AGENTS.md)
- [PROJECT_PLAN.md](/D:/model1_baseline_agent_bundle/PROJECT_PLAN.md)

That project contract is broader than the original repo build-out. The codebase today implements Model 1 across both Phase 1 tracks, but it does not yet implement Models 2 and 3 or the Phase 2 transfer work.

## Implemented Now

### Public dataset and schema work

- DBE-KT22 retrieval via [fetch_dbe_kt22.py](/D:/model1_baseline_agent_bundle/src/fetch_dbe_kt22.py)
- schema audit via [model1_schema_audit.md](/D:/model1_baseline_agent_bundle/reports/model1_schema_audit.md)
- deterministic preprocessing and within-learner chronological splits via [preprocess_model1.py](/D:/model1_baseline_agent_bundle/src/preprocess_model1.py)

### Phase 1 Track A: seen-learner forward prediction

Implemented for Model 1:

- clean trial table
- within-learner `80/20` chronological split
- VI fit/evaluation
- multiple NUTS validation runs

Current processed sample:

- `157,989` processed rows
- `1,138` eligible learners
- `125,877` train rows
- `32,112` test rows
- `0` unseen-item test rows in the primary current split

### Phase 1 Track B: unseen-public-student initialization

Implemented for Model 1:

- deterministic student-wise public train / validation / test split via [split_model1_track_b.py](/D:/model1_baseline_agent_bundle/src/split_model1_track_b.py)
- primary evaluation restricted to items seen in the public training students
- VI fit on public train students only
- sequential evaluation on held-out public validation and test students with new-group sampling enabled for unseen learners

Current Track B sample:

- `796` train students
- `170` validation students
- `172` test students
- `110,434` train rows
- `22,891` validation rows
- `24,664` test rows
- `0` unseen-item validation rows in the primary evaluation
- `0` unseen-item test rows in the primary evaluation

### Model 1 results already available

Default VI run in [outputs/model1](/D:/model1_baseline_agent_bundle/outputs/model1):

- log loss: `0.5446`
- Brier: `0.1841`
- accuracy: `0.7153`
- AUC: `0.7609`
- calibration intercept: `0.0670`
- calibration slope: `0.9253`

Pilot MCMC run in [outputs/model1_mcmc](/D:/model1_baseline_agent_bundle/outputs/model1_mcmc):

- `2` chains, `500/500`
- useful as an early validation pass only

Fuller MCMC run in [outputs/model1_mcmc_full](/D:/model1_baseline_agent_bundle/outputs/model1_mcmc_full):

- `4` chains, `1000/1000`, `target_accept 0.95`
- divergences: `0`
- max `R-hat`: `1.05`
- min bulk ESS: `74`

Strict MCMC run in [outputs/model1_mcmc_strict](/D:/model1_baseline_agent_bundle/outputs/model1_mcmc_strict):

- `4` chains, `2000/2000`, `target_accept 0.97`
- divergences: `0`
- max `R-hat`: `1.01`
- min bulk ESS: `204`
- min tail ESS: `619`

Strict MCMC held-out evaluation in [outputs/model1_mcmc_strict](/D:/model1_baseline_agent_bundle/outputs/model1_mcmc_strict):

- log loss: `0.5455`
- Brier: `0.1845`
- accuracy: `0.7160`
- AUC: `0.7607`
- calibration intercept: `-0.0335`
- calibration slope: `0.9128`

Track B VI run in [outputs/model1_track_b](/D:/model1_baseline_agent_bundle/outputs/model1_track_b):

- fit rows: `110,434`
- validation log loss: `0.4357`
- validation Brier: `0.1402`
- validation accuracy: `0.7982`
- validation AUC: `0.7965`
- test log loss: `0.4538`
- test Brier: `0.1478`
- test accuracy: `0.7855`
- test AUC: `0.7892`

Interpretation:

- Model 1 is implemented and working.
- The stricter MCMC run is the current best posterior fit for Track A.
- Model 1 predictive behavior is stable across VI and MCMC in Track A.
- Track B is now implemented as the public unseen-student baseline.

## Added Project-Spec Assets

The full project bundle has been merged into the repo:

- [PROJECT_PLAN.md](/D:/model1_baseline_agent_bundle/PROJECT_PLAN.md)
- [model2-random-slope-binary-logistic](/D:/model1_baseline_agent_bundle/.agents/skills/model2-random-slope-binary-logistic/SKILL.md)
- [model3-dynamic-volatility-binary-logistic](/D:/model1_baseline_agent_bundle/.agents/skills/model3-dynamic-volatility-binary-logistic/SKILL.md)
- [phase2-transfer-warm-start](/D:/model1_baseline_agent_bundle/.agents/skills/phase2-transfer-warm-start/SKILL.md)

## Not Implemented Yet

The new project plan includes work that the current repo does not yet implement:

- Model 2 fit/evaluation code
- Model 3 fit/evaluation code
- Phase 2 local-data harmonization
- Phase 2 weak-prior versus public-informed warm-start comparison

## Recommended Next Order

1. Treat Model 1 as complete across Track A and Track B.
2. Implement Model 2 next and compare it against Model 1 on the same public holdout rows.
3. Only implement Model 3 if Model 2 clearly earns the extra complexity.
4. Start Phase 2 transfer only after the Phase 1 model family is frozen.
