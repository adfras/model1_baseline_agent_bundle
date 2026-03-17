# Model 1 Track B Online Warm-Start Run

This note records the true online Track B evaluation for Model 1.

## Purpose

The earlier Track B evaluation used population-level new-group prediction for unseen students.
This online run is stricter and closer to the eventual Phase 2 target:

- predict attempt 1 for a new public student from the public hierarchy
- observe the result
- update the student's latent intercept approximation
- predict attempt 2
- continue sequentially through the student's held-out history

## Configs

- [model1_track_b_online_evaluate_validation.json](../config/model1_track_b_online_evaluate_validation.json)
- [model1_track_b_online_evaluate_test.json](../config/model1_track_b_online_evaluate_test.json)

## Outputs

Expected outputs are written under `outputs/model1_track_b_online/`:

- overall metrics
- learner metrics
- calibration tables and figures
- row-level prediction files
- attempt-window summaries for attempts `1-5`, `6-10`, and `11-20`

## Status

Completed on `2026-03-16`.

## Validation Results

- rows: `22,891`
- students: `170`
- log loss: `0.4085`
- Brier: `0.1304`
- accuracy: `0.8149`
- AUC: `0.8153`
- calibration intercept: `0.0497`
- calibration slope: `1.1047`

Attempt-window summaries:

- attempts `1-5`: log loss `0.3524`, Brier `0.1092`, AUC `0.7864`
- attempts `6-10`: log loss `0.2258`, Brier `0.0667`, AUC `0.8878`
- attempts `11-20`: log loss `0.2905`, Brier `0.0856`, AUC `0.8352`

## Test Results

- rows: `24,664`
- students: `172`
- log loss: `0.4347`
- Brier: `0.1408`
- accuracy: `0.7948`
- AUC: `0.8059`
- calibration intercept: `0.0722`
- calibration slope: `1.0398`

Attempt-window summaries:

- attempts `1-5`: log loss `0.4023`, Brier `0.1278`, AUC `0.7573`
- attempts `6-10`: log loss `0.2378`, Brier `0.0709`, AUC `0.8742`
- attempts `11-20`: log loss `0.3057`, Brier `0.0922`, AUC `0.8407`

## Comparison With The Earlier Marginal Track B Baseline

Earlier Track B test evaluation from [config/model1_track_b_evaluate_test.json](../config/model1_track_b_evaluate_test.json):

- log loss: `0.4542`
- Brier: `0.1479`
- accuracy: `0.7852`
- AUC: `0.7889`

Online warm-start test evaluation:

- log loss: `0.4347`
- Brier: `0.1408`
- accuracy: `0.7948`
- AUC: `0.8059`

Interpretation:

- the true online Track B evaluation improves materially over the older marginal cold-start read
- this supports treating online Track B as the more relevant template for Phase 2
