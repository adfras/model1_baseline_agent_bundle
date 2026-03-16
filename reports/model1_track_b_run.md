# Model 1 Track B Run

This note records the first completed Phase 1 Track B baseline for `DBE-KT22`.

## Purpose

Track B evaluates sequential prediction for unseen public students. The train, validation, and test partitions are disjoint by `student_id`. Prediction uses the same Model 1 structure as Track A:

```text
correct ~ 1 + practice_feature + (1 | student_id) + (1 | item_id)
```

At evaluation time, unseen learner effects are handled by sampling new group levels from the fitted student hierarchy. The primary evaluation excludes rows whose item IDs were not seen in the public training students.

## Data and split

The Track B split starts from the cleaned trial table in [learner_trials.csv](/D:/model1_baseline_agent_bundle/data/processed/model1/learner_trials.csv) and reassigns students deterministically with [split_model1_track_b.py](/D:/model1_baseline_agent_bundle/src/split_model1_track_b.py).

Default config:

- [model1_track_b_split.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_split.json)

Split summary:

- `1,138` total eligible students
- `796` train students
- `170` validation students
- `172` test students
- `157,989` total rows
- `110,434` train rows
- `22,891` validation rows
- `24,664` test rows
- `212` training items
- `0` unseen-item rows in the primary validation evaluation
- `0` unseen-item rows in the primary test evaluation

## Fit

The first Track B fit uses the same variational baseline as Track A.

Fit config:

- [model1_track_b_fit.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_fit.json)

Fit summary:

- inference method: `vi`
- random seed: `20260316`
- train rows: `110,434`
- train students: `796`
- train items: `212`
- VI iterations: `20,000`
- posterior draws: `1,000`
- elapsed seconds: `213.16`

## Validation evaluation

Validation config:

- [model1_track_b_evaluate_validation.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_evaluate_validation.json)

Validation metrics:

- rows: `22,891`
- students: `170`
- log loss: `0.4357`
- Brier: `0.1402`
- accuracy: `0.7982`
- AUC: `0.7965`
- calibration intercept: `0.3886`
- calibration slope: `1.0867`

## Test evaluation

Test config:

- [model1_track_b_evaluate_test.json](/D:/model1_baseline_agent_bundle/config/model1_track_b_evaluate_test.json)

Test metrics:

- rows: `24,664`
- students: `172`
- log loss: `0.4538`
- Brier: `0.1478`
- accuracy: `0.7855`
- AUC: `0.7892`
- calibration intercept: `0.2740`
- calibration slope: `1.0525`

## Interpretation

Track B is now implemented for Model 1. The baseline handles unseen public students cleanly on the current DBE-KT22 split, with all primary validation and test rows remaining on items seen in public training. This completes the Model 1 baseline across both Phase 1 tracks and leaves Model 2 as the next substantive implementation step.
