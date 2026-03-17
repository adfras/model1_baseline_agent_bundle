# Model 2 R-PFA Alpha Tuning

This note summarizes the explicit Q-matrix Model 2 alpha search for R-PFA history weighting.

Selection rule:

- primary selector: held-out log loss
- tie margin: `0.0002`
- tie break: choose the largest alpha within the tie margin

Selected alpha:

- `0.9`

Interpretation:

- The tuning rule selected `alpha = 0.9`, so the operational R-PFA mainline uses recency weighting stronger than plain cumulative PFA.

## Grid results

| alpha | log loss | Brier | AUC | calibration slope |
|---:|---:|---:|---:|---:|
| 0.4 | 0.542090 | 0.183227 | 0.764023 | 0.957068 |
| 0.6 | 0.541651 | 0.183068 | 0.764440 | 0.958421 |
| 0.8 | 0.541378 | 0.182968 | 0.764634 | 0.958851 |
| 0.9 | 0.541470 | 0.183001 | 0.764493 | 0.957899 |
| 1.0 | 0.541867 | 0.183153 | 0.764064 | 0.955329 |

## Selected row

- log loss `0.541470`
- Brier `0.183001`
- AUC `0.764493`
- calibration slope `0.957899`
