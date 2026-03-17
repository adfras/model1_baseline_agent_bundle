# Model 2 Track B Online Warm-Start Run

This note records the true online Track B evaluation for Model 2.

## Purpose

This run uses the same public Track B split as the earlier marginal evaluation, but updates each unseen student's latent intercept and practice slope after every observed response.

## Validation Results

- rows: `22,891`
- students: `170`
- log loss: `0.4085`
- Brier: `0.1304`
- accuracy: `0.8141`
- AUC: `0.8147`
- calibration intercept: `0.0202`
- calibration slope: `1.1009`

## Test Results

- rows: `24,664`
- students: `172`
- log loss: `0.4340`
- Brier: `0.1406`
- accuracy: `0.7960`
- AUC: `0.8063`
- calibration intercept: `0.0364`
- calibration slope: `1.0357`

## Test Attempt Windows

- attempts `1-5`: log loss `0.3976`, Brier `0.1259`, AUC `0.7600`
- attempts `6-10`: log loss `0.2385`, Brier `0.0714`, AUC `0.8741`
- attempts `11-20`: log loss `0.3076`, Brier `0.0928`, AUC `0.8368`

## Interpretation

- Model 2 improves slightly over Model 1 on overall online Track B test metrics.
- The main Model 2 advantage is concentrated in attempts `1-5`.
- Later windows do not improve relative to Model 1.
