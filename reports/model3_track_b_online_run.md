# Model 3 Track B Online Warm-Start Run

This note records the true online Track B evaluation for Model 3.

## Purpose

This run applies the Model 3 screening approximation to true online Track B evaluation, updating an unseen student's latent intercept, slope, and current latent state after each observed response.

## Validation Results

- rows: `22,891`
- students: `170`
- log loss: `0.4068`
- Brier: `0.1299`
- accuracy: `0.8138`
- AUC: `0.8161`
- calibration intercept: `-0.0298`
- calibration slope: `1.0956`

## Test Results

- rows: `24,664`
- students: `172`
- log loss: `0.4327`
- Brier: `0.1403`
- accuracy: `0.7981`
- AUC: `0.8073`
- calibration intercept: `-0.0219`
- calibration slope: `1.0316`

## Test Attempt Windows

- attempts `1-5`: log loss `0.4019`, Brier `0.1274`, AUC `0.7525`
- attempts `6-10`: log loss `0.2362`, Brier `0.0709`, AUC `0.8779`
- attempts `11-20`: log loss `0.3054`, Brier `0.0926`, AUC `0.8391`

## Interpretation

- Model 3 is the best current overall online Track B model.
- Model 3 is not best on the earliest `1-5` attempts; Model 2 still leads there.
- Because this is still the screening approximation, robustness checks remain necessary before final selection.
