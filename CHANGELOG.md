# Changelog

## 2026-03-15

### Added
- scripted retrieval for the DBE-KT22 dataset via [src/fetch_dbe_kt22.py](/D:/model1_baseline_agent_bundle/src/fetch_dbe_kt22.py)
- Python environment manifest via [pyproject.toml](/D:/model1_baseline_agent_bundle/pyproject.toml)
- default preprocessing config at [config/model1_preprocess.json](/D:/model1_baseline_agent_bundle/config/model1_preprocess.json)
- pure-Python preprocessing and split generation at [src/preprocess_model1.py](/D:/model1_baseline_agent_bundle/src/preprocess_model1.py)
- default fit and evaluation configs at [config/model1_fit.json](/D:/model1_baseline_agent_bundle/config/model1_fit.json) and [config/model1_evaluate.json](/D:/model1_baseline_agent_bundle/config/model1_evaluate.json)
- shared model utilities at [src/model1_common.py](/D:/model1_baseline_agent_bundle/src/model1_common.py)
- fit and evaluation scripts at [src/fit_model1.py](/D:/model1_baseline_agent_bundle/src/fit_model1.py) and [src/evaluate_model1.py](/D:/model1_baseline_agent_bundle/src/evaluate_model1.py)
- schema audit report at [reports/model1_schema_audit.md](/D:/model1_baseline_agent_bundle/reports/model1_schema_audit.md)
- preprocessing run report at [reports/model1_preprocessing_run.md](/D:/model1_baseline_agent_bundle/reports/model1_preprocessing_run.md)

### Documented
- DBE-KT22 retrieval instructions and preprocessing command in [README.md](/D:/model1_baseline_agent_bundle/README.md)
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
