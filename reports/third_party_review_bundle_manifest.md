# Third-Party Review Bundle Manifest

This bundle is intended for external review of the current modelling workspace state.

## Included

- project contract and plan
- the new pivot note describing what the project is focused on now
- model definitions and shared utilities
- all repository `src/*.py` scripts
- all repository `config/*.json` configs
- internal run notes and comparison reports
- the revised public-data model-development path:
  - full-dataset multi-KC preprocessing
  - collapsed-feature multi-KC reruns
  - explicit Q-matrix reruns for Models 1 / 2 / 3
  - targeted explicit-Q improvement trials:
    - PFA-style wins/fails history
    - R-PFA alpha tuning
    - fractional KC-credit sensitivity
    - finer Model 3 state bins
  - fractional multi-KC sensitivity reruns on the same rows and split
  - single-KC sensitivity reruns
  - branch guide and historical branch adjudication notes
- the new offline adaptive-policy work:
  - replay configs for R-PFA Model 2 and Model 3
  - modular policy-suite comparison note
  - policy summary outputs
- the KC-history feature validation note for recency and due-review fields
- the dormant Phase 2 scaffolding:
  - local schema normalization template
  - 3-way student split template
  - transfer protocol note
- lightweight output summaries such as:
  - fit summaries
  - evaluation summaries
  - diagnostics summaries
  - overall metrics tables
  - calibration tables
  - calibration figures
  - structural and volatility summaries where available

## Excluded

- raw data under `data/`
- processed learner-attempt tables
- large posterior artifacts such as:
  - `*.nc`
  - `*.npz`
- virtual environment files

## Important context

- The bundle reflects the current workspace state, not only the last pushed commit.
- The repo originally centered heterogeneity discovery and conditional local transfer.
- With no local dataset currently available in the workspace, the practical focus has shifted to:
  - full-data public learner-model development
  - explicit Q-matrix KC-aware modeling
  - better KC-history features
  - offline next-question policy replay
- The full-data explicit Q-matrix ladder supports Model 2 and then Model 3.
- The strongest predictive improvement came from **PFA / R-PFA wins/fails history**.
- The selected operational R-PFA alpha is `0.9`.
- On that branch, **R-PFA Model 2** beats **R-PFA Model 3** on log loss, Brier, and AUC, while **R-PFA Model 3** improves calibration slope.
- On the modular offline policy suite, **R-PFA Model 2** remains the default next-question policy model.
- The single-KC branch still collapses to Model 1 and is treated as a restrictive sensitivity analysis.
- The repo now distinguishes clearly between:
  - the richest supported heterogeneity model
  - the best current operational model for question targeting
- Phase 2 local fitting is scaffolded but not run, because no local dataset is bundled here.
