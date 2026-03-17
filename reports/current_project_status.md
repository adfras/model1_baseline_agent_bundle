# Current Project Status

This note records the repo state after switching the operational Phase 1 path to a full-data multi-KC design.

For the branch names used below, see:

- [phase1_branch_guide.md](D:/model1_baseline_agent_bundle/reports/phase1_branch_guide.md)

## Framing

The project is now:

1. public heterogeneity discovery
2. conditional local structural replication
3. conditional local warm-start application

The main public discovery path is no longer the single-KC branch or the older primary-KC collapse.

## Implemented now

### Operational Phase 1 discovery path

- multi-KC full-data preprocessing:
  - [preprocess_phase1_multikc.py](D:/model1_baseline_agent_bundle/src/preprocess_phase1_multikc.py)
  - [phase1_multikc_preprocess.json](D:/model1_baseline_agent_bundle/config/phase1_multikc_preprocess.json)
- full-data multi-KC fit/eval configs for Models 1, 2, and 3
- heterogeneity summary scaffold:
  - [summarize_phase1_heterogeneity.py](D:/model1_baseline_agent_bundle/src/summarize_phase1_heterogeneity.py)
  - [phase1_multikc_heterogeneity_summary.json](D:/model1_baseline_agent_bundle/config/phase1_multikc_heterogeneity_summary.json)

### Sensitivity / diagnostic branches still present

- deterministic primary-KC branch
- single-KC-only branch
- repeated-practice subset on the restrictive single-KC family
- explicit Q-matrix branch for Models 1 and 2

### Phase 2 scaffolding

- [preprocess_phase2_local.py](D:/model1_baseline_agent_bundle/src/preprocess_phase2_local.py)
- [split_phase2_local.py](D:/model1_baseline_agent_bundle/src/split_phase2_local.py)
- [phase2_protocol.md](D:/model1_baseline_agent_bundle/reports/phase2_protocol.md)

## Operational discovery sample

The new multi-KC preprocessing has been run successfully.

Current sample:

- `157,989` processed attempt rows
- `1,138` learners
- `212` items
- `93` represented KCs
- `300,246` long attempt-KC rows
- mean KC count per attempt `1.896`
- `125,877` train rows
- `32,112` test rows
- `0` chronology violations
- `0` unseen-item test rows in the primary evaluation

See:

- [multikc_summary.json](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_summary.json)
- [phase1_multikc_schema_note.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_schema_note.md)
- [phase1_multikc_heterogeneity_summary.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_heterogeneity_summary.md)

## Operational full-data multi-KC results

### Model 1

- log loss `0.546332`
- Brier `0.184800`
- calibration slope `0.926549`
- learner intercept SD mean `0.569`
- learner intercept SD 94% HDI `[0.538, 0.598]`

### Model 2

- log loss `0.545491`
- Brier `0.184503`
- calibration slope `0.940928`
- learner slope SD mean `0.048`
- learner slope SD 94% HDI `[0.042, 0.053]`

Interpretation:

- growth heterogeneity survives on the operational multi-KC branch
- the slope variance clears the practical floor
- Model 2 clears the predictive gate relative to Model 1

### Model 3

- log loss `0.543726`
- Brier `0.183892`
- calibration slope `0.963244`
- learner slope SD mean `0.0485`
- learner slope SD 94% HDI `[0.0421, 0.0541]`
- latent state SD mean `0.4860`
- latent state SD 94% HDI `[0.4588, 0.5145]`
- `rho` mean `0.1180`
- `rho` 94% HDI `[0.0867, 0.1474]`

Interpretation:

- the added stability term is clearly non-zero
- Model 3 improves over Model 2 on the operational branch
- the current operational full-data branch supports **Model 3**

## Sensitivity / diagnostic reading

### Fractional multi-KC sensitivity

A stronger sensitivity check has now been run on the same full dataset:

- same rows
- same split
- same model ladder
- different KC allocation rule: each multi-KC attempt contributes `1 / kc_count` exposure to each linked KC instead of full credit to each linked KC

Result:

- fractional Model 1 log loss `0.547029`
- fractional Model 2 log loss `0.546311`
- fractional Model 2 learner slope SD 94% HDI `[0.053, 0.069]`
- fractional Model 3 log loss `0.543867`
- fractional Model 3 Brier `0.183941`
- fractional Model 3 calibration slope `0.957373`
- fractional Model 3 latent state SD 94% HDI `[0.4559, 0.5113]`

Interpretation:

- the richer result survives a meaningful multi-KC allocation sensitivity
- Model 2 still survives
- Model 3 still improves over Model 2
- the robustness question is now about effect size movement across KC-handling rules, not about total collapse back to Model 1

### Explicit Q-matrix branch

The explicit Q-matrix branch keeps the same full multi-KC sample but moves KC structure into the likelihood instead of collapsing it to one scalar practice feature before fitting.

Current results:

- explicit Q-matrix Model 1 log loss `0.545311`
- explicit Q-matrix Model 2 log loss `0.544366`
- explicit Q-matrix Model 2 learner slope SD 94% HDI `[0.040, 0.051]`

Interpretation:

- Model 2 beats Model 1 on the same held-out rows under an explicit KC parameterization
- this strengthens the substantive case for growth heterogeneity on the full dataset

See:

- [phase1_multikc_qmatrix_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_qmatrix_comparison.md)

### Single-KC branch

The construct-clean single-KC branch still collapses to Model 1.

That branch is now interpreted as a sensitivity result, not the controlling mainline.

### Deterministic primary-KC branch

The older one-KC-per-item collapse also supported richer models, but it has now been superseded by the better-supported multi-KC operational branch.

### Repeated-practice subset

The repeated-practice redesign on the restrictive single-KC family strengthens the slope signal but still does not beat Model 1 on that restrictive family.

## Current scientific reading

- using the full dataset and all linked KCs changes the answer materially
- Model 2 and then Model 3 survive on the operational branch
- a meaningful multi-KC sensitivity still supports Models 2 and 3
- the remaining public-science task is robustness of effect size and interpretation across KC-handling rules, not whether richer heterogeneity exists at all on the full dataset

## Phase 2 status

Phase 2 remains conditional, but the current richer public-supported model family is now **Model 3** on the operational multi-KC branch.

What still depends on further adjudication:

- whether the multi-KC result is robust enough across alternative KC-handling schemes
- whether that robustness is strong enough to justify carrying Model 3 into local replication and warm-start
