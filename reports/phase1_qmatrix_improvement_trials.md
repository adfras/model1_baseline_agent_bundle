# Explicit Q-Matrix Improvement Trials

This note records targeted Phase 1 follow-up experiments run on the full-data explicit Q-matrix branch to identify the highest-yield next change without introducing new data.

It is now a **historical improvement note** rather than the current controlling selection summary. The current mainline has moved on to the explicit Q-matrix **R-PFA** branch with tuned recency weighting. For the current operational selection, use:

- [phase1_qmatrix_rpfa_tuning.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_tuning.md)
- [phase1_qmatrix_rpfa_operational_selection.md](D:/model1_baseline_agent_bundle/reports/phase1_qmatrix_rpfa_operational_selection.md)

## Baseline

Current explicit Q-matrix baseline on the held-out public test rows:

| model | log loss | Brier | AUC | calibration slope |
|---|---:|---:|---:|---:|
| Model 2 | 0.544366 | 0.184051 | 0.761858 | 0.943504 |
| Model 3 | 0.543782 | 0.183916 | 0.761855 | 0.968148 |

## Improvement branches tried

### 1. PFA-style history

Change:
- retain the explicit Q-matrix structure
- replace the shared KC-practice term with KC-specific prior success and prior failure counts
- keep the learner-specific practice slope and, for Model 3, the latent state term

Results:

| model | log loss | Brier | AUC | calibration slope |
|---|---:|---:|---:|---:|
| PFA Model 2 | 0.541867 | 0.183153 | 0.764064 | 0.955329 |
| PFA Model 3 | 0.541895 | 0.183193 | 0.763715 | 0.970979 |

Takeaway:
- this is the only branch that produced a clearly material gain
- the gain is much larger than the other tested tweaks
- after adding wins/fails history, Model 3 adds very little beyond Model 2 on predictive metrics

### 2. Fractional KC credit

Change:
- keep the explicit Q-matrix opportunity model
- rebuild the multi-KC history table with equal fractional KC credit instead of full credit

Result:

| model | log loss | Brier | AUC | calibration slope |
|---|---:|---:|---:|---:|
| Fractional-credit Model 3 | 0.543974 | 0.183979 | 0.761756 | 0.962013 |

Takeaway:
- slightly worse than the current full-credit explicit-Q Model 3
- not the best next move if the goal is predictive improvement

### 3. Finer Model 3 state bins

Change:
- keep the explicit Q-matrix opportunity model
- reduce `state_bin_width` from `10` to `5`

Result:

| model | log loss | Brier | AUC | calibration slope |
|---|---:|---:|---:|---:|
| Bin-5 Model 3 | 0.543715 | 0.183883 | 0.762173 | 0.955003 |

Takeaway:
- tiny log-loss and Brier improvement relative to the current Model 3
- calibration slope gets worse than the current bin-10 Model 3
- this is a minor tuning gain, not the main lever

## Ranking by yield

For held-out predictive return on the current public explicit-Q branch:

1. **PFA-style wins/fails history**
2. **smaller Model 3 state bins**
3. **fractional KC credit**

## Plain conclusion

The best yield came from improving the **history signal**, not from retuning the latent state and not from changing KC credit allocation.

On this branch, the strongest next model is:

- **explicit Q-matrix PFA Model 2** if the priority is the best held-out log loss and Brier
- **explicit Q-matrix PFA Model 3** only if a slightly better calibration slope is worth the extra complexity despite essentially tied predictive fit

So the next serious line of work should be:

- build the ladder around **PFA-style KC wins/fails history**
- then decide whether Model 3 still earns continuation once that better history signal is in place
