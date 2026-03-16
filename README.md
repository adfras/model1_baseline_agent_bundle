# Learner Forecasting Project Bundle

This repository now follows the two-phase project spec in [AGENTS.md](/D:/model1_baseline_agent_bundle/AGENTS.md) and [PROJECT_PLAN.md](/D:/model1_baseline_agent_bundle/PROJECT_PLAN.md).

The codebase is not a full personalised learning system. It is a small forecasting-and-transfer project:

- Phase 1: fit and compare public-data sequential models
- Phase 2: carry the chosen public model family into a local warm-start setting

## Current Status

What is implemented today:

- DBE-KT22 public-data retrieval via [fetch_dbe_kt22.py](/D:/model1_baseline_agent_bundle/src/fetch_dbe_kt22.py)
- DBE-KT22 schema audit via [model1_schema_audit.md](/D:/model1_baseline_agent_bundle/reports/model1_schema_audit.md)
- Phase 1 Track A preprocessing and deterministic within-learner chronological splits via [preprocess_model1.py](/D:/model1_baseline_agent_bundle/src/preprocess_model1.py)
- Phase 1 Track B unseen-public-student split generation via [split_model1_track_b.py](/D:/model1_baseline_agent_bundle/src/split_model1_track_b.py)
- Model 1 fit and evaluation scripts via [fit_model1.py](/D:/model1_baseline_agent_bundle/src/fit_model1.py) and [evaluate_model1.py](/D:/model1_baseline_agent_bundle/src/evaluate_model1.py)
- multiple Model 1 fitting paths: variational inference, pilot MCMC, fuller MCMC, and a stricter MCMC convergence run

What is specified but not implemented yet:

- Model 2 random-slope learner-growth model
- Model 3 dynamic learner-volatility model
- Phase 2 public-to-local warm-start transfer

A repo-level status note lives at [current_project_status.md](/D:/model1_baseline_agent_bundle/reports/current_project_status.md).

## Project Docs

- [AGENTS.md](/D:/model1_baseline_agent_bundle/AGENTS.md): current project contract and routing rules
- [PROJECT_PLAN.md](/D:/model1_baseline_agent_bundle/PROJECT_PLAN.md): staged two-phase plan
- [model1-baseline-binary-logistic](/D:/model1_baseline_agent_bundle/.agents/skills/model1-baseline-binary-logistic/SKILL.md)
- [model2-random-slope-binary-logistic](/D:/model1_baseline_agent_bundle/.agents/skills/model2-random-slope-binary-logistic/SKILL.md)
- [model3-dynamic-volatility-binary-logistic](/D:/model1_baseline_agent_bundle/.agents/skills/model3-dynamic-volatility-binary-logistic/SKILL.md)
- [phase2-transfer-warm-start](/D:/model1_baseline_agent_bundle/.agents/skills/phase2-transfer-warm-start/SKILL.md)

## Environment

The repository includes a Python environment manifest at [pyproject.toml](/D:/model1_baseline_agent_bundle/pyproject.toml).

Target interpreter:

- Python `>=3.11,<3.13`

Declared top-level dependencies:

- `requests`
- `pandas`
- `pyarrow`
- `matplotlib`
- `scikit-learn`
- `pymc`
- `bambi`
- `arviz`

Suggested setup:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Public Dataset

Current public development dataset:

- Name: `DBE-KT22`
- DOI: <https://doi.org/10.26193/6DZWOH>
- Paper: <https://arxiv.org/abs/2208.12651>

### Retrieve The Data

Run:

```powershell
py src/fetch_dbe_kt22.py
```

This downloads the dataset into:

```text
data/raw/DBE-KT22/
```

To inspect the remote manifest without downloading the files:

```powershell
py src/fetch_dbe_kt22.py --manifest-only
```

## Phase 1 Model 1 Workflow

The currently implemented path is the Phase 1 Model 1 baseline on DBE-KT22.

### Preprocess

Default preprocessing rules:

- use `Transaction.csv` as the primary attempt table
- exclude `is_hidden = true`
- use `answer_state` as the binary outcome
- order each learner by `start_time` and then transaction `id`
- require at least `10` visible attempts per learner
- split each eligible learner `80%` train / `20%` test in time order

Run:

```powershell
py src/preprocess_model1.py
```

Default config:

```text
config/model1_preprocess.json
```

Default outputs:

```text
data/processed/model1/learner_trials.csv
data/processed/model1/split_assignments.csv
data/processed/model1/preprocess_summary.json
```

### Track B Split

Phase 1 Track B keeps public train, validation, and test sets disjoint by `student_id` while reusing the cleaned attempt table from Track A.

Default Track B rules:

- deterministic hash-based student split
- `70%` train / `15%` validation / `15%` test by learner
- primary evaluation restricted to rows whose items were seen in the public train students
- unseen students handled at prediction time by sampling new learner effects from the fitted hierarchy

Run:

```powershell
py src/split_model1_track_b.py
```

Default config:

```text
config/model1_track_b_split.json
```

Default outputs:

```text
data/processed/model1_track_b/learner_trials.csv
data/processed/model1_track_b/student_split_assignments.csv
data/processed/model1_track_b/split_assignments.csv
data/processed/model1_track_b/track_b_summary.json
```

### Fit And Evaluate

Default variational fit:

```powershell
python src/fit_model1.py
python src/evaluate_model1.py
```

Configs:

- [model1_fit.json](/D:/model1_baseline_agent_bundle/config/model1_fit.json)
- [model1_evaluate.json](/D:/model1_baseline_agent_bundle/config/model1_evaluate.json)

Track B Model 1 configs:

- split: [model1_track_b_split.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_split.json)
- fit: [model1_track_b_fit.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_fit.json)
- validation evaluation: [model1_track_b_evaluate_validation.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_evaluate_validation.json)
- test evaluation: [model1_track_b_evaluate_test.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_evaluate_test.json)

Available MCMC configs:

- pilot: [model1_fit_mcmc.json](/D:/model1_baseline_agent_bundle/config/model1_fit_mcmc.json)
- fuller pass: [model1_fit_mcmc_full.json](/D:/model1_baseline_agent_bundle/config/model1_fit_mcmc_full.json)
- stricter convergence pass: [model1_fit_mcmc_strict.json](/D:/model1_baseline_agent_bundle/config/model1_fit_mcmc_strict.json)

Matching evaluation configs:

- [model1_evaluate_mcmc.json](/D:/model1_baseline_agent_bundle/config/model1_evaluate_mcmc.json)
- [model1_evaluate_mcmc_full.json](/D:/model1_baseline_agent_bundle/config/model1_evaluate_mcmc_full.json)
- [model1_evaluate_mcmc_strict.json](/D:/model1_baseline_agent_bundle/config/model1_evaluate_mcmc_strict.json)

Important runtime note:

- the fit configs explicitly point at the Rtools `g++` directory because this app session may not pick up the compiler from PATH automatically

### Current Model 1 Snapshot

Primary preprocessing sample:

- `157,989` processed rows
- `1,138` eligible learners
- `125,877` train rows
- `32,112` test rows
- `0` unseen-item test rows in the current Phase 1 Track A split

Current best completed predictive baseline:

- VI test log loss: `0.5446`
- VI test Brier: `0.1841`
- VI test AUC: `0.7609`

Current best completed posterior diagnostics:

- strict MCMC fit: `4` chains, `2000` tune, `2000` draws, `target_accept 0.97`
- divergences: `0`
- max `R-hat`: `1.01`
- min bulk ESS: `204`
- min tail ESS: `619`

Phase 1 Track B student-wise split:

- `796` train students
- `170` validation students
- `172` test students
- `110,434` train rows
- `22,891` validation rows
- `24,664` test rows
- `0` unseen-item rows in the primary validation and test evaluations

Track B VI evaluation:

- validation log loss: `0.4357`
- validation Brier: `0.1402`
- validation AUC: `0.7965`
- test log loss: `0.4538`
- test Brier: `0.1478`
- test AUC: `0.7892`

See [current_project_status.md](/D:/model1_baseline_agent_bundle/reports/current_project_status.md) for the status mapping from the old Model 1-only repo to the new two-phase plan.

## Next Implementation Steps

- implement Model 2 and compare it against Model 1 on primary probabilistic metrics
- only if Model 2 clearly earns it, implement Model 3
- only after Phase 1 model selection, start Phase 2 transfer code

## Non-Commit Rule

Downloaded datasets and generated outputs are local working artifacts and should not be committed. The repository ignore rules exclude `data/`, `outputs/`, the virtual environment, and temporary bundle-extraction directories.
