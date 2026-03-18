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
  - policy-facing alpha comparison (`0.8` vs `0.9`)
  - spacing review-threshold grid (`24 / 48 / 72 / 96`)
  - subgroup diagnostics for the current operational policy suite
  - a decision note on the rejected conservative router v3 attempt
  - the simple two-mode router threshold search and decision memo
  - a policy-alignment calibration note comparing Model 2 and Model 3 on logged actual-next items in policy contexts
  - an uncertainty calibration-layer note showing Model 2 plus Model 3 uncertainty as a side-channel
  - a hard decision note on the calibrated fixed-policy rerun
  - a later KC-constrained residual-heterogeneity restart with local residual features, policy-specific calibrators, and a decision note
  - policy summary outputs
  - hybrid uncertainty-router notes, including the v2 lagged-proxy branch
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
- A later policy-facing comparison against `0.8` kept `0.9`.
- The current selected spacing-review threshold on the operational Model 2 branch is `24` hours.
- On that branch, **R-PFA Model 2** beats **R-PFA Model 3** on log loss, Brier, and AUC, while **R-PFA Model 3** has the better overall branch calibration slope.
- On the modular offline policy suite, **R-PFA Model 2** remains the default next-question policy model.
- A later logged actual-next policy-alignment calibration check does **not** show a policy-context calibration advantage for Model 3, so it remains exploratory rather than operational.
- A later calibration-layer branch does show a small held-out calibration-loss win when **Model 3 uncertainty** is used as a side-channel on top of **Model 2**.
- A later fixed-policy rerun shows that this uncertainty side-channel does **not** survive the operational gate:
  - it gets a tiny `confidence_building` target-gap gain over the context-only calibrator
  - but mean new-learning target gap worsens and stability worsens
  - raw Model 2 still remains the operational policy input
- A later KC-constrained residual-heterogeneity restart then gives the idea a stricter operational test:
  - a deterministic KC-constrained unseen slate
  - local residual / friction / self-report features from prior attempts only
  - policy-specific calibrators
  - Model 3 as one extra uncertainty feature
- That restart also fails the operational gate:
  - pooled mean target gap improves only slightly
  - `confidence_building` target precision gets worse
  - stability worsens substantially
  - Model 3 adds almost nothing beyond the local residual features
- On the current operational Model 2 suite, there is no single universal best fixed policy: `confidence_building` and `balanced_challenge` split the target-gap wins, `harder_challenge` most often wins policy advantage, and remediation/review remain distinct service modes.
- A first hybrid router using **Model 2 means plus Model 3 uncertainty** is included as a prototype, but it does not replace the fixed-policy suite as the default.
- A later hybrid-router v2 branch adds lagged observable proxies and tuned thresholds.
- The tuned v2 router improves target gap and policy advantage over the original hybrid, but it is still less stable than the simpler hybrid baseline and still does not replace the fixed-policy suite as the default.
- A first conservative router v3 attempt was tried after the subgroup diagnostics, but it was worse than the existing baselines and was not kept as an active branch.
- A later simple two-mode router, built on the frozen Model 2 RPFA scorer, improved new-learning target gap only marginally and did not survive operationally because stability worsened too much.
- The single-KC branch still collapses to Model 1 and is treated as a restrictive sensitivity analysis.
- The repo now distinguishes clearly between:
  - the richest supported heterogeneity model
  - the best current operational model for question targeting
- Phase 2 local fitting is scaffolded but not run, because no local dataset is bundled here.
