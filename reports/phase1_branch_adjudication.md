# Phase 1 Branch Adjudication Note

This note is a historical adjudication record from the earlier branch-comparison stage.

The current repo mainline has moved beyond this specific primary-KC versus single-KC comparison and now treats:

- the full multi-KC branches as operational
- the fractional multi-KC branch as the main KC-allocation sensitivity
- the single-KC branch as a restrictive construct-clean sensitivity check

This note records the stronger Phase 1 adjudication after completing the full ladder on the primary single-KC branch and comparing matched held-out students across both KC-handling branches.

It is now a **historical branch note**, not the current controlling summary. For the current repo focus, use:

- [phase1_selection_memo.md](D:/model1_baseline_agent_bundle/reports/phase1_selection_memo.md)
- [project_pivot_and_current_focus.md](D:/model1_baseline_agent_bundle/reports/project_pivot_and_current_focus.md)

## Primary single-KC branch

Held-out row-level metrics:

- Model 1 log loss `0.567414`, Brier `0.194380`, calibration slope `0.916596`
- Model 2 log loss `0.567742`, Brier `0.194524`, calibration slope `0.914719`
- Model 3 log loss `0.570940`, Brier `0.195801`, calibration slope `0.924606`

Structural terms:

- Model 2 learner slope SD mean `0.074`, 94% HDI `[0.041, 0.103]`
- Model 3 learner slope SD mean `0.077`, 94% HDI `[0.049, 0.111]`
- Model 3 latent state SD mean `0.749`, 94% HDI `[0.697, 0.792]`

Paired student-level comparison on matched held-out students:

- Model 2 minus Model 1 mean delta log loss `+0.000221`, 95% bootstrap interval `[-0.000019, 0.000466]`
- Model 2 minus Model 1 mean delta Brier `+0.000113`, 95% bootstrap interval `[0.000020, 0.000207]`
- Model 3 minus Model 1 mean delta log loss `+0.003718`, 95% bootstrap interval `[0.001686, 0.005738]`
- Model 3 minus Model 1 mean delta Brier `+0.001419`, 95% bootstrap interval `[0.000608, 0.002223]`

Interpretation:

- the primary branch does **not** support Model 2 over Model 1
- the primary branch clearly rejects Model 3 as a forecasting improvement over Model 1

## Full-item primary-KC sensitivity branch

Held-out row-level metrics:

- Model 1 log loss `0.546289`, Brier `0.184771`, calibration slope `0.925122`
- Model 2 log loss `0.545797`, Brier `0.184577`, calibration slope `0.928122`
- Model 3 log loss `0.544350`, Brier `0.184124`, calibration slope `0.949390`

Structural terms:

- Model 2 learner slope SD mean `0.059`, 94% HDI `[0.048, 0.071]`
- Model 3 learner slope SD mean `0.047`, 94% HDI `[0.038, 0.057]`
- Model 3 latent state SD mean `0.479`, 94% HDI `[0.450, 0.510]`

Paired student-level comparison on matched held-out students:

- Model 2 minus Model 1 mean delta log loss `-0.000440`, 95% bootstrap interval `[-0.000665, -0.000213]`
- Model 2 minus Model 1 mean delta Brier `-0.000173`, 95% bootstrap interval `[-0.000251, -0.000102]`
- Model 3 minus Model 1 mean delta log loss `-0.002418`, 95% bootstrap interval `[-0.003465, -0.001449]`
- Model 3 minus Model 1 mean delta Brier `-0.000685`, 95% bootstrap interval `[-0.001006, -0.000386]`
- Model 3 minus Model 2 mean delta log loss `-0.001978`, 95% bootstrap interval `[-0.003003, -0.001057]`
- Model 3 minus Model 2 mean delta Brier `-0.000511`, 95% bootstrap interval `[-0.000823, -0.000228]`

Interpretation:

- the sensitivity branch supports Model 2 over Model 1
- the sensitivity branch supports Model 3 over both Model 1 and Model 2

## Decision reading

The public Phase 1 story is now sharper:

- the **full-dataset primary-KC analysis** supports **Model 3**
- the **single-KC analysis** supports **Model 1**

That means the repo no longer has an unfinished-ladder problem. It has a **KC-handling dependence problem**.

Operational reading from this historical comparison:

- the richer result appears on the **full-data KC-aware branches**
- the restrictive **single-KC branch** likely throws away too much repeated same-skill structure to identify growth and stability robustly
- the main issue is therefore not “must use the full dataset” as a rule by itself
- the main issue is that the stricter single-KC restriction appears to remove signal needed for the richer heterogeneity terms

Under that reading:

- DBE contains real evidence for richer heterogeneity
- the main scientific caveat is multi-KC assignment ambiguity
- the next work should strengthen robustness of the full-dataset Model 3 result rather than centering Model 1
