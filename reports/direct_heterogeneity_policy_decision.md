# Direct Heterogeneity Policy Decision

This note records the first direct heterogeneity-informed next-item branch on DBE.

## Frozen baseline

- mean scorer: raw explicit Q-matrix `R-PFA Model 2`
- replay freeze: `alpha = 0.9`
- review threshold: `24` hours
- comparison baseline here: `operational_freeze_proxy` = `spacing_aware_review` when due, else fixed `confidence_building`

## Direct policy idea

This branch does not use a calibration side-channel.

Instead, it chooses directly among a small action slate:

- `confidence_building`
- `balanced_challenge`
- `harder_challenge`
- `spacing_aware_review`

using scientific Model 3 learner heterogeneity signals inside the decision itself:

- learner baseline rank
- learner growth rank
- learner stability rank
- current latent-state signal

## Selected parameters

- base target: `0.72`
- baseline weight: `0.04`
- growth weight: `0.04`
- stability weight: `0.04`
- state weight: `0.03`
- remediation weight: `0.015`
- review bonus: `0.0`
- seen-item penalty: `0.02`

## Evaluation result

- direct policy target gap `1-10`: `0.020983`
- freeze proxy target gap `1-10`: `0.006351`
- delta: `+0.014632`

- direct policy advantage `1-10`: `0.171326`
- freeze proxy policy advantage `1-10`: `0.189227`
- delta: `-0.017901`

- direct stability: `0.001371`
- freeze proxy stability: `0.000470`
- delta: `+0.000901`

## Interpretation

- This is the first branch in the repo where heterogeneity changes the next-item choice directly rather than entering as a calibration add-on.
- It should be read as an offline small-slate decision experiment, not a causal learning-gain result.
