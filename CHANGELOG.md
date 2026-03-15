# Changelog

## 2026-03-15

### Added
- scripted retrieval for the DBE-KT22 dataset via [src/fetch_dbe_kt22.py](/D:/model1_baseline_agent_bundle/src/fetch_dbe_kt22.py)
- default preprocessing config at [config/model1_preprocess.json](/D:/model1_baseline_agent_bundle/config/model1_preprocess.json)
- pure-Python preprocessing and split generation at [src/preprocess_model1.py](/D:/model1_baseline_agent_bundle/src/preprocess_model1.py)
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
