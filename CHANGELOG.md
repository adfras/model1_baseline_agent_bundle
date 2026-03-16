# Changelog

## 2026-03-16

### Added
- deterministic Phase 1 Track B student-split generation via [src/split_model1_track_b.py](/D:/model1_baseline_agent_bundle/src/split_model1_track_b.py)
- Track B split and evaluation configs at [config/model1_track_b_split.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_split.json), [config/model1_track_b_fit.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_fit.json), [config/model1_track_b_evaluate_validation.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_evaluate_validation.json), and [config/model1_track_b_evaluate_test.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_evaluate_test.json)
- Track B run note at [reports/model1_track_b_run.md](/D:/model1_baseline_agent_bundle/reports/model1_track_b_run.md)

### Documented
- completed Phase 1 Track B implementation and results in [README.md](/D:/model1_baseline_agent_bundle/README.md) and [reports/current_project_status.md](/D:/model1_baseline_agent_bundle/reports/current_project_status.md)
- strict Model 1 MCMC held-out metrics in [reports/current_project_status.md](/D:/model1_baseline_agent_bundle/reports/current_project_status.md)

### Current Track B state
- `1,138` eligible students are split deterministically into `796` train, `170` validation, and `172` test students
- the first Track B fit uses VI on `110,434` training rows and `212` items
- primary Track B evaluation contains `0` unseen-item rows in validation and `0` unseen-item rows in test
- validation metrics: log loss `0.4357`, Brier `0.1402`, accuracy `0.7982`, AUC `0.7965`
- test metrics: log loss `0.4538`, Brier `0.1478`, accuracy `0.7855`, AUC `0.7892`

## 2026-03-15

### Added
- scripted retrieval for the DBE-KT22 dataset via [src/fetch_dbe_kt22.py](/D:/model1_baseline_agent_bundle/src/fetch_dbe_kt22.py)
- Python environment manifest via [pyproject.toml](/D:/model1_baseline_agent_bundle/pyproject.toml)
- default preprocessing config at [config/model1_preprocess.json](/D:/model1_baseline_agent_bundle/config/model1_preprocess.json)
- pure-Python preprocessing and split generation at [src/preprocess_model1.py](/D:/model1_baseline_agent_bundle/src/preprocess_model1.py)
- default fit and evaluation configs at [config/model1_fit.json](/D:/model1_baseline_agent_bundle/config/model1_fit.json) and [config/model1_evaluate.json](/D:/model1_baseline_agent_bundle/config/model1_evaluate.json)
- shared model utilities at [src/model1_common.py](/D:/model1_baseline_agent_bundle/src/model1_common.py)
- fit and evaluation scripts at [src/fit_model1.py](/D:/model1_baseline_agent_bundle/src/fit_model1.py) and [src/evaluate_model1.py](/D:/model1_baseline_agent_bundle/src/evaluate_model1.py)
- full-project plan at [PROJECT_PLAN.md](/D:/model1_baseline_agent_bundle/PROJECT_PLAN.md)
- Model 2 skill at [model2-random-slope-binary-logistic](/D:/model1_baseline_agent_bundle/.agents/skills/model2-random-slope-binary-logistic/SKILL.md)
- Model 3 skill at [model3-dynamic-volatility-binary-logistic](/D:/model1_baseline_agent_bundle/.agents/skills/model3-dynamic-volatility-binary-logistic/SKILL.md)
- Phase 2 transfer skill at [phase2-transfer-warm-start](/D:/model1_baseline_agent_bundle/.agents/skills/phase2-transfer-warm-start/SKILL.md)
- current-state note at [current_project_status.md](/D:/model1_baseline_agent_bundle/reports/current_project_status.md)
- schema audit report at [reports/model1_schema_audit.md](/D:/model1_baseline_agent_bundle/reports/model1_schema_audit.md)
- preprocessing run report at [reports/model1_preprocessing_run.md](/D:/model1_baseline_agent_bundle/reports/model1_preprocessing_run.md)
- pilot, full, and strict Model 1 MCMC configs under [config](/D:/model1_baseline_agent_bundle/config)

### Documented
- the repo as a two-phase forecasting-and-transfer project in [README.md](/D:/model1_baseline_agent_bundle/README.md) and [AGENTS.md](/D:/model1_baseline_agent_bundle/AGENTS.md)
- DBE-KT22 retrieval instructions and current Phase 1 implementation state in [README.md](/D:/model1_baseline_agent_bundle/README.md)
- dataset non-commit rule in [AGENTS.md](/D:/model1_baseline_agent_bundle/AGENTS.md) and [.gitignore](/D:/model1_baseline_agent_bundle/.gitignore)

### Current baseline-ready preprocessing state
- hidden rows are excluded from the primary sample
- learners need at least `10` visible attempts
- eligible learners are split `80%` train / `20%` test in time order
- processed sample size from the first run: `157,989` rows across `1,138` eligible learners
- current test split contains `0` unseen-item rows

### Current modeling state
- the project now has first-pass Bambi/PyMC fit and evaluation scripts
- the fit configuration explicitly points at the Rtools `g++` directory because this app session may not inherit the compiler from PATH automatically
- the default full-data fitting method is variational inference (`vi`) for practicality
- the fit/eval pipeline has been smoke-tested on a 20-student local slice and produced posterior, metrics, learner-level metrics, and a calibration figure
- a strict Model 1 MCMC fit completed with `4` chains, `2000/2000`, `target_accept 0.97`, `0` divergences, max `R-hat 1.01`, min bulk ESS `204`, and min tail ESS `619`
