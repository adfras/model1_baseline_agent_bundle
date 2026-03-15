# Model 1 Preprocessing Run

## Command

```powershell
py src/preprocess_model1.py
```

Config used:

- [model1_preprocess.json](/D:/model1_baseline_agent_bundle/config/model1_preprocess.json)

## Applied rules

- source attempt table: [Transaction.csv](/D:/model1_baseline_agent_bundle/data/raw/DBE-KT22/extracted/csv/Transaction.csv)
- static item metadata: [Questions.csv](/D:/model1_baseline_agent_bundle/data/raw/DBE-KT22/extracted/csv/Questions.csv)
- excluded `is_hidden = true` rows
- used `answer_state` as the binary outcome
- ordered each learner by `start_time`, then `id`
- minimum visible-history threshold: `10`
- within-learner split: `80%` train / `20%` test

## Output artifacts

- [learner_trials.csv](/D:/model1_baseline_agent_bundle/data/processed/model1/learner_trials.csv)
- [split_assignments.csv](/D:/model1_baseline_agent_bundle/data/processed/model1/split_assignments.csv)
- [preprocess_summary.json](/D:/model1_baseline_agent_bundle/data/processed/model1/preprocess_summary.json)

## Summary

- raw rows: `161,953`
- hidden rows excluded: `3,564`
- rows after hidden exclusion: `158,389`
- visible students: `1,261`
- eligible students with at least `10` visible attempts: `1,138`
- excluded students below threshold: `123`
- processed rows in the primary modeling sample: `157,989`
- train rows: `125,877`
- test rows: `32,112`
- primary-evaluation-eligible test rows: `32,112`
- new-item test rows: `0`

## Implications

- the current primary sample stays entirely within already-observed items at test time
- the chosen minimum-history threshold removes only a modest number of visible students
- the repository now has a reproducible preprocessing stage ready for model fitting
