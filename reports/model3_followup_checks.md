# Model 3 Follow-up Checks

This note records the stricter follow-up checks for Model 3 after the first provisional Track B run.

## Current standard

Model 3 is not treated as decision-grade from a single first-pass fit. The current summary uses:

- stricter `20,000`-iteration VI fits
- saved VI diagnostics for the ELBO history
- Track A and Track B comparisons against the existing Model 2 results

Current Track B results here are still marginal cold-start evaluations, not the newer online warm-start standard.

## Track A fit diagnostics

Track A fit config:

- [model3_fit.json](../config/model3_fit.json)
- attempt-bin width: `10`
- VI iterations: `20,000`
- posterior draws: `500`

Track A VI diagnostics:

- initial loss: `235075.98`
- final loss: `52038.63`
- best loss: `51455.52`
- best iteration: `19230`
- relative improvement: `0.7786`
- tail relative change: `-0.00055`

Track A reading:

- this is stable enough to use as a screening fit
- it is not an MCMC-quality posterior check

## Track A comparison

Model 1 Track A VI:

- log loss: `0.5446`
- Brier: `0.1841`
- accuracy: `0.7153`
- AUC: `0.7609`

Model 2 Track A VI:

- log loss: `0.5463`
- Brier: `0.1848`
- accuracy: `0.7149`
- AUC: `0.7601`

Model 3 Track A VI:

- log loss: `0.5473`
- Brier: `0.1854`
- accuracy: `0.7129`
- AUC: `0.7613`
- calibration intercept: `-0.1780`
- calibration slope: `0.9569`

Track A reading:

- Model 3 does not improve the main Track A probabilistic metrics
- AUC is slightly higher, but log loss and Brier are worse

## Track B fit diagnostics

Track B fit config:

- [model3_track_b_fit.json](../config/model3_track_b_fit.json)
- attempt-bin width: `10`
- VI iterations: `20,000`
- posterior draws: `500`

Track B VI diagnostics:

- initial loss: `93939.55`
- final loss: `49032.11`
- best loss: `48656.20`
- best iteration: `19374`
- relative improvement: `0.4780`
- tail relative change: `-0.00091`

Track B reading:

- this is stable enough to use as a screening fit
- again, it is still VI rather than a full state-space MCMC posterior

## Track B Marginal Cold-Start Comparison

Model 2 validation:

- log loss: `0.4302`
- Brier: `0.1381`
- accuracy: `0.7997`
- AUC: `0.7970`
- calibration intercept: `0.2995`
- calibration slope: `1.0819`

Model 3 validation:

- log loss: `0.4223`
- Brier: `0.1353`
- accuracy: `0.8065`
- AUC: `0.7975`
- calibration intercept: `0.0869`
- calibration slope: `1.0697`

Model 2 test:

- log loss: `0.4504`
- Brier: `0.1465`
- accuracy: `0.7872`
- AUC: `0.7891`
- calibration intercept: `0.1905`
- calibration slope: `1.0454`

Model 3 test:

- log loss: `0.4460`
- Brier: `0.1450`
- accuracy: `0.7891`
- AUC: `0.7905`
- calibration intercept: `-0.0104`
- calibration slope: `1.0354`

Track B reading:

- Model 3 now improves log loss, Brier, accuracy, and AUC on both validation and test
- calibration also improves relative to Model 2
- these are still screening results under marginal cold-start evaluation rather than online per-student updating

## Track B online warm-start comparison

Model 2 online validation:

- log loss: `0.4085`
- Brier: `0.1304`
- accuracy: `0.8141`
- AUC: `0.8147`

Model 3 online validation:

- log loss: `0.4068`
- Brier: `0.1299`
- accuracy: `0.8138`
- AUC: `0.8161`

Model 2 online test:

- log loss: `0.4340`
- Brier: `0.1406`
- accuracy: `0.7960`
- AUC: `0.8063`
- calibration intercept: `0.0364`
- calibration slope: `1.0357`

Model 3 online test:

- log loss: `0.4327`
- Brier: `0.1403`
- accuracy: `0.7981`
- AUC: `0.8073`
- calibration intercept: `-0.0219`
- calibration slope: `1.0316`

Online early-attempt windows on the test split:

- attempts `1-5`:
  - Model 2 log loss `0.3976`
  - Model 3 log loss `0.4019`
- attempts `6-10`:
  - Model 2 log loss `0.2385`
  - Model 3 log loss `0.2362`
- attempts `11-20`:
  - Model 2 log loss `0.3076`
  - Model 3 log loss `0.3054`

Online Track B reading:

- Model 3 is still the best overall online Track B model
- Model 3 is better than Model 2 on validation, overall test, and the later `6-10` and `11-20` windows
- Model 2 is better on the earliest `1-5` online window

## Interim interpretation

The stricter rerun changes the picture:

- Track A still does not favor Model 3
- Track B marginal and online results both favor Model 3 over Model 2 on overall primary metrics

Because Track B is the stated priority, Model 3 now has a legitimate case to continue.

## What this still does not prove

This is still not a final selection result because:

- the current Model 3 implementation is a tractable attempt-bin approximation
- the inference is VI rather than a full state-space MCMC fit
- even after the sensitivity checks, the posterior evidence is still screening-grade rather than final

## Robustness checks completed

Two minimal robustness checks were added on the online Track B target:

1. alternate random seed with the original `state_bin_width = 10`
2. alternate `state_bin_width = 5` with the original seed
3. alternate `state_bin_width = 20` with the original seed

### Alternate-seed check

Fit diagnostics:

- random seed: `20260326`
- tail relative change: `-0.00090`

Online Track B test:

- log loss: `0.4328`
- Brier: `0.1403`
- accuracy: `0.7982`
- AUC: `0.8072`

Attempt windows:

- attempts `1-5`: `0.4020`
- attempts `6-10`: `0.2362`
- attempts `11-20`: `0.3054`

This is effectively unchanged from the original Track B online result.

### Alternate-bin-width check

Fit diagnostics:

- state-bin width: `5`
- train state steps: `46`
- tail relative change: `-0.00077`

Online Track B test:

- log loss: `0.4327`
- Brier: `0.1403`
- accuracy: `0.7976`
- AUC: `0.8074`

Attempt windows:

- attempts `1-5`: `0.4016`
- attempts `6-10`: `0.2366`
- attempts `11-20`: `0.3064`

This is also effectively unchanged from the original Track B online result.

### Second alternate-bin-width check

Fit diagnostics:

- state-bin width: `20`
- train state steps: `12`
- tail relative change: `-0.00060`

Online Track B test:

- log loss: `0.4327`
- Brier: `0.1403`
- accuracy: `0.7974`
- AUC: `0.8076`

Attempt windows:

- attempts `1-5`: `0.4023`
- attempts `6-10`: `0.2360`
- attempts `11-20`: `0.3034`

This is also effectively unchanged from the original Track B online result.

## Student-level uncertainty versus Model 1

To avoid relying only on pooled metrics, a student-level bootstrap comparison was added for the online Track B test split.

Model 3 minus Model 1 student-averaged log-loss deltas:

- overall:
  - mean delta: `-0.00123`
  - bootstrap 95% interval: `[-0.00268, 0.00020]`
- attempts `1-5`:
  - mean delta: `-0.00045`
  - bootstrap 95% interval: `[-0.00577, 0.00491]`
- attempts `1-10`:
  - mean delta: `-0.00101`
  - bootstrap 95% interval: `[-0.00385, 0.00215]`
- attempts `6-10`:
  - mean delta: `-0.00158`
  - bootstrap 95% interval: `[-0.00389, 0.00079]`
- attempts `11-20`:
  - mean delta: `-0.00032`
  - bootstrap 95% interval: `[-0.00342, 0.00297]`

Reading:

- Model 3 is consistently a little better than Model 1 on the pooled Track B point estimates
- but the student-level bootstrap intervals still overlap zero
- so the gain is real enough to keep Model 3 alive, but still too small to call decisively established

## Updated interpretation

These robustness checks materially strengthen the Model 3 case:

- the overall online Track B advantage survives a seed change
- it also survives meaningful changes in attempt-bin width
- the metric shifts are at the third-decimal level rather than changing the model ranking

So the fair current reading is now:

- Model 3 remains screening-only because it is still a binned VI approximation
- but it has passed the minimum robustness bar for a Phase 1 screening candidate
- compared with Model 2, it now has the stronger selection case on the Track B objective
- compared with Model 1, it has only a small and still statistically thin advantage on the current public Track B evidence

## Next disciplined step

Before carrying Model 3 into Phase 2, keep the framing honest:

- call it the leading screening candidate rather than a finished state-space model
- treat it as an optional challenger rather than a replacement for Model 1
- pair it with the item-shift caveat that current transfer evidence is still about new students on seen items
