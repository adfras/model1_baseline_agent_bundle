# Model 1 Baseline Bundle

This repository is for a reproducible **Model 1** baseline on trial-level learner-response data. The current named dataset for this bundle is **DBE-KT22**.

## Environment
The repository now includes a Python environment manifest at:

- [pyproject.toml](/D:/model1_baseline_agent_bundle/pyproject.toml)

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

The manifest is explicit about the intended modeling stack, but it is not yet a fully validated lockfile. Package installation and fit-stage validation are still the next environment-dependent step.

## Dataset
- Name: `DBE-KT22`
- DOI: <https://doi.org/10.26193/6DZWOH>
- Paper: <https://arxiv.org/abs/2208.12651>

## How To Retrieve The Data
The repository includes a scripted fetch step that handles the ADA Dataverse access flow and downloads the published dataset files.

Run:

```powershell
py src/fetch_dbe_kt22.py
```

This downloads the dataset into:

```text
data/raw/DBE-KT22/
```

The script resolves and downloads these files from the Dataverse record:
- `1_DBE_KT22_file_descriptions_100102.xlsx`
- `1_Script_to_generate_sequences_100102_py.zip`
- `2_DBE_KT22_datafiles_100102_csv.zip`
- `2_DBE_KT22_Practice_Sequences_100102_json.zip`

To inspect the manifest without downloading the files:

```powershell
py src/fetch_dbe_kt22.py --manifest-only
```

## Preprocess For Model 1
The current default preprocessing stage follows the audit decisions for DBE-KT22:

- use `Transaction.csv` as the primary attempt table
- exclude rows with `is_hidden = true`
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

The schema audit that motivated these defaults is recorded in:

- [reports/model1_schema_audit.md](/D:/model1_baseline_agent_bundle/reports/model1_schema_audit.md)

## Fit And Evaluate Model 1
The repository now includes first-pass fit and evaluation scripts for the baseline hierarchical logistic model.

Fit:

```powershell
python src/fit_model1.py
```

Evaluate:

```powershell
python src/evaluate_model1.py
```

Default fit/evaluation configs:

```text
config/model1_fit.json
config/model1_evaluate.json
```

Important runtime note:

- the fit config explicitly points at the Rtools `g++` directory because this app session may not pick up the compiler from PATH automatically
- the default fitting method is variational inference (`vi`) for practicality on the full DBE-KT22 training set

Default fit/evaluation outputs:

```text
outputs/model1/model1_idata.nc
outputs/model1/model1_fit_summary.json
outputs/model1/model1_posterior_summary.csv
outputs/model1/model1_vi_history.csv
outputs/model1/model1_overall_metrics.csv
outputs/model1/model1_learner_metrics.csv
outputs/model1/model1_calibration_table.csv
outputs/model1/figures/model1_calibration.png
outputs/model1/model1_evaluation_summary.json
```

## Non-Commit Rule
Downloaded dataset files are local working data and should not be committed. The repository ignore rules exclude the local `data/` tree so the retrieval process is documented, but the actual data stays out of version control.
