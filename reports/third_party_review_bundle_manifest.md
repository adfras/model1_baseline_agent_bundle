# Third-Party Review Bundle Manifest

This bundle is intended for external review of the current modelling workspace state.

## Included

- project contract and plan
- model definitions and shared utilities
- all repository `src/*.py` scripts
- all repository `config/*.json` configs
- internal run notes and comparison reports
- the revised Phase 1 discovery pivot:
  - full-dataset primary-KC preprocessing
  - full-dataset Model 1 / Model 2 / Model 3 discovery reruns
  - single-KC sensitivity reruns
  - branch adjudication note comparing the primary and sensitivity paths
  - heterogeneity summary scaffolding
- the revised Phase 2 scaffolding:
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
- The repo has been repivoted from a public benchmark race to a heterogeneity-discovery and local-transfer project.
- The older Track A / Track B public materials are still present as development history.
- The operational decision-grade public path now uses the full visible DBE item set with deterministic primary-KC assignment.
- The full-dataset primary-KC branch currently supports Model 3.
- The single-KC branch currently stops at Model 1 and is treated as a construct-clean sensitivity analysis.
- Phase 1 is now fully ladder-complete on both branches.
- Model 1 is now treated only as the hurdle benchmark, not the Phase 2 scientific target.
- DBE currently shows richer heterogeneity on the full dataset, but the result remains sensitive to multi-KC handling.
- The public carry-forward decision now depends on whether that full-dataset Model 3 result is robust enough to justify Phase 2.
- Phase 2 local fitting is scaffolded but not yet run, because no local dataset is bundled here.
