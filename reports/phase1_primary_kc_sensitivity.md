# Phase 1 Primary-KC Sensitivity Note

This note records the required full-item sensitivity analysis for Phase 1.

## Assignment rule

All visible items are retained.

Each item is assigned one deterministic `primary_kc_id` using the earliest relationship row in `Question_KC_Relationships.csv` for that item.

This is a sensitivity analysis, not the primary discovery dataset.

## Sensitivity sample

- `157,989` processed rows
- `1,138` learners
- `212` items
- `68` assigned KCs
- `125,877` train rows
- `32,112` test rows

See:

- [phase1_primary_kc_sensitivity_schema_note.md](D:/model1_baseline_agent_bundle/reports/phase1_primary_kc_sensitivity_schema_note.md)

## Sensitivity results

### Model 1

- log loss `0.546289`
- Brier `0.184771`
- calibration slope `0.925122`

### Model 2

- log loss `0.545797`
- Brier `0.184577`
- calibration slope `0.928122`
- learner slope SD mean `0.059`
- learner slope SD 94% HDI `[0.048, 0.071]`

### Model 3

- log loss `0.544350`
- Brier `0.184124`
- calibration slope `0.949390`
- learner slope SD mean `0.047`
- learner slope SD 94% HDI `[0.038, 0.057]`
- latent state SD mean `0.479`
- latent state SD 94% HDI `[0.450, 0.510]`
- AR(1) persistence mean `0.106`
- AR(1) persistence 94% HDI `[0.075, 0.135]`

## Comparison with the primary single-KC analysis

Primary single-KC analysis:

- Model 2 was slightly worse than Model 1 on held-out log loss and Brier
- so the primary branch stopped at Model 1

Primary-KC sensitivity analysis:

- Model 2 is better than Model 1 on held-out log loss
- Model 2 is better than Model 1 on Brier
- Model 2 calibration slope is also closer to `1.0`
- the slope variance clears the current practical floor
- Model 3 is better than Model 2 on held-out log loss
- Model 3 is better than Model 2 on Brier
- Model 3 calibration slope moves closer again to `1.0`
- the latent state SD also clears the current practical floor

## Interpretation

The Phase 1 conclusion is now **sensitive to KC-handling choice**.

That means:

- the single-KC primary analysis still supports Model 1
- the full-item primary-KC sensitivity analysis now supports Model 3
- Phase 1 should therefore be treated as **not yet frozen**

## Practical implication

The full-item sensitivity branch now supports the richer heterogeneity ladder through Model 3.

The next decision-grade step should be one of:

1. decide whether the single-KC primary branch or the full-item primary-KC sensitivity branch should control Phase 1 carry-forward
2. or run one stronger adjudication pass across the two KC-handling branches before freezing the public winner
