# Learner Heterogeneity and Warm-Start Project

This repository is a small two-phase modelling project for trial-level learner-response data.

It is **not** a benchmark zoo and **not** a full personalised learning system.

The framing is:

1. discover whether learners differ in baseline level, growth, and stability in public longitudinal data
2. verify that structure in a new local sample
3. test whether public-informed priors estimate those differences earlier than weak-prior local fitting

Core docs:

- [AGENTS.md](D:/model1_baseline_agent_bundle/AGENTS.md)
- [PROJECT_PLAN.md](D:/model1_baseline_agent_bundle/PROJECT_PLAN.md)
- [phase1_selection_memo.md](D:/model1_baseline_agent_bundle/reports/phase1_selection_memo.md)
- [phase2_protocol.md](D:/model1_baseline_agent_bundle/reports/phase2_protocol.md)
- [phase1_branch_guide.md](D:/model1_baseline_agent_bundle/reports/phase1_branch_guide.md)

## Current state

Legacy public benchmarking materials are still present in the repo as development history.

The decision-grade path now uses the **full visible DBE item set with all linked KCs retained** as the operational discovery branch.

The local replication and warm-start scaffolding remains in the repo, but it is **conditional**. It is not the scientific next step unless the public discovery dataset supports Model 2 or Model 3 cleanly enough to justify a real heterogeneity replication question.

Implemented for the revised plan:

- [preprocess_phase1_discovery.py](D:/model1_baseline_agent_bundle/src/preprocess_phase1_discovery.py)
- [summarize_phase1_heterogeneity.py](D:/model1_baseline_agent_bundle/src/summarize_phase1_heterogeneity.py)
- [preprocess_phase2_local.py](D:/model1_baseline_agent_bundle/src/preprocess_phase2_local.py)
- [split_phase2_local.py](D:/model1_baseline_agent_bundle/src/split_phase2_local.py)

## Public discovery workflow

Build the full-dataset multi-KC discovery table:

```powershell
py src/preprocess_phase1_multikc.py
```

Current discovery sample:

- `157,989` processed rows
- `1,138` learners
- `212` items
- `93` represented KCs
- `300,246` long attempt-KC rows

Outputs:

- [multikc_trials.csv](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_trials.csv)
- [multikc_attempt_kc_long.csv](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_attempt_kc_long.csv)
- [multikc_summary.json](D:/model1_baseline_agent_bundle/data/processed/phase1_multikc/multikc_summary.json)
- [phase1_multikc_schema_note.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_schema_note.md)

Fit the full-dataset discovery models:

```powershell
py src/fit_model1.py --config config/phase1_multikc_model1_fit.json
py src/evaluate_model1.py --config config/phase1_multikc_model1_evaluate.json
py src/fit_model2.py --config config/phase1_multikc_model2_fit.json
py src/evaluate_model2.py --config config/phase1_multikc_model2_evaluate.json
py src/fit_model3.py --config config/phase1_multikc_model3_fit.json
py src/evaluate_model3.py --config config/phase1_multikc_model3_evaluate.json
```

Current discovery result:

- Model 1 log loss: `0.546332`
- Model 2 log loss: `0.545491`
- Model 2 learner slope SD 94% HDI: `[0.042, 0.053]`
- Model 3 log loss: `0.543726`
- Model 3 Brier: `0.183892`
- Model 3 calibration slope: `0.963244`
- Model 3 latent state SD 94% HDI: `[0.4588, 0.5145]`

Current reading:

- the full-data multi-KC branch supports Model 2 and then Model 3
- richer heterogeneity appears when the model is allowed to use the full repeated structure in DBE without collapsing each item to one KC
- the remaining issue is robustness to alternative multi-KC handling, not lack of signal

Branch naming guide:

- [phase1_branch_guide.md](D:/model1_baseline_agent_bundle/reports/phase1_branch_guide.md)

Explicit Q-matrix comparison for Models 1 and 2:

- [phase1_multikc_qmatrix_comparison.md](D:/model1_baseline_agent_bundle/reports/phase1_multikc_qmatrix_comparison.md)

Single-KC-only analysis remains as a construct-clean sensitivity check:

- [phase1_discovery_heterogeneity_summary.md](D:/model1_baseline_agent_bundle/reports/phase1_discovery_heterogeneity_summary.md)

That branch gives:

- Model 1 log loss: `0.5674`
- Model 2 log loss: `0.5677`
- Model 3 log loss: `0.5709`

Current disciplined reading:

- the **full-data multi-KC branch** is now the operational mainline
- the **single-KC branch** is a restrictive sensitivity analysis that strips out much of the repeated same-skill structure
- the repeated-practice redesign on the single-KC family is diagnostic, not the mainline decision path

Current project reading:

- the current full-dataset DBE result supports **Model 3**
- the main scientific caveat is robustness across alternative multi-KC handling choices
- the current next step is to strengthen that full-dataset branch, not to throw away data again

Updated multi-KC sensitivity:

- equal-split fractional KC exposure has now also been run on the same full dataset
- fractional Model 2 log loss: `0.546311`
- fractional Model 3 log loss: `0.543867`
- fractional Model 3 Brier: `0.183941`
- fractional Model 3 calibration slope: `0.957373`

So the richer result survives a meaningful KC-allocation sensitivity. The remaining uncertainty is no longer “does the result disappear as soon as KC handling changes?” It is now about how much the effect size moves under alternative multi-KC constructions.

A stronger repeated-practice redesign has also now been run:

- [phase1_repeated_subset_results.md](D:/model1_baseline_agent_bundle/reports/phase1_repeated_subset_results.md)

That redesign keeps only student-KC sequences with `>= 3` opportunities within the single-KC family.

Result:

- Model 1 log loss: `0.536827`
- Model 2 log loss: `0.537510`
- Model 2 learner slope SD 94% HDI: `[0.085, 0.160]`

So the redesign shows why the single-KC family can wash out richer heterogeneity, but it does not replace the full-dataset mainline.

Current rule:

- added heterogeneity SD terms beyond Model 1 must clear a practical floor of `0.03`
- the richer model must either improve log loss, or be no worse by more than `0.001` while Brier improves and calibration slope moves closer to `1.0`

Current unresolved adjudication point:

- how much the Model 3 full-data multi-KC gain moves across alternative KC-handling choices, not whether it vanishes entirely

## Phase 2 scaffolding

Normalize local data:

```powershell
py src/preprocess_phase2_local.py --config config/phase2_local_preprocess_template.json
```

Create the local 3-way student split:

```powershell
py src/split_phase2_local.py --config config/phase2_local_split_template.json
```

Phase 2A is now explicitly a replication ladder:

- Model 1 is the hurdle benchmark
- the richest public-supported model would advance as the primary structural model

Under the current DBE result, the richest currently supported operational model family is **Model 3** on the full-data multi-KC branch.

The Phase 2 protocol remains in the repo as conditional scaffolding for the case where:

- a richer public model survives cleanly, or
- a new discovery dataset replaces DBE as the main public source

So the current main task is still public discovery adjudication, not a Model 1-led warm-start study.

Primary outcome:

- student-averaged log loss over attempts `1-5`

Secondary:

- student-averaged log loss over attempts `1-10`

Sparse-data fallback:

- if too few students reach 5 attempts, use a shorter primary window such as attempts `1-3`
