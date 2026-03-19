# Phase 1 Explicit-Q Learner State Profiles

This note promotes learner-state estimation to a first-class Phase 1 DBE deliverable.

Source of truth:

- scientific explicit Q-matrix posterior draws only
- no refitting
- 94% HDIs throughout

## Exported tables

- Model 2 learner profiles: `outputs\phase1_multikc_qmatrix_profiles\model2_learner_profiles.csv`
- Model 3 learner profiles: `outputs\phase1_multikc_qmatrix_profiles\model3_learner_profiles.csv`
- Model 3 latent states: `outputs\phase1_multikc_qmatrix_profiles\model3_latent_state_profiles.csv`

- students exported: `1138`
- Model 3 state bins exported: `19`
- state-bin width: `10`

## Model 3 learner-dimension distributions

### Baseline

- mean: `1.0969`
- sd: `0.4217`
- p10 / p50 / p90: `0.5655` / `1.1088` / `1.6143`

### Growth

- mean: `0.0008`
- sd: `0.0211`
- p10 / p50 / p90: `-0.0257` / `0.0012` / `0.0266`

### Stability

- mean: `0.3571`
- sd: `0.0895`
- p10 / p50 / p90: `0.2627` / `0.3411` / `0.4625`

## Correlations Among Learner Dimensions

| left | right | pearson_r |
| --- | --- | ---: |
| baseline | baseline | 1.0000 |
| baseline | growth | 0.5565 |
| baseline | stability | -0.1127 |
| growth | baseline | 0.5565 |
| growth | growth | 1.0000 |
| growth | stability | -0.0532 |
| stability | baseline | -0.1127 |
| stability | growth | -0.0532 |
| stability | stability | 1.0000 |

## Interpretation

- `baseline` is the learner-specific starting level on the logit scale: global intercept plus learner intercept.
- `growth` is the learner-specific practice-response slope on the logit scale.
- `stability` is the learner-specific latent state scale from Model 3; larger values indicate more time-varying deviation around the learner's baseline-plus-growth trajectory.
- `latent_state_mean` by `state_bin` shows where a learner's transient state is above or below their longer-run trajectory at that practice stage.

These exports are meant to support learner-state reporting and later Phase 2 replication, not to claim a new adaptive-policy win on DBE.
