# Simple Two-Mode Router Decision Memo

This note records the next policy pass after the subgroup diagnostics and rejected conservative router v3 attempt.

## Frozen operational stack

The scorer is now frozen for policy work:

- **operational scorer:** explicit Q-matrix **R-PFA Model 2**
- **decay alpha:** `0.9`
- **review threshold:** `24` hours for `spacing_aware_review`
- **Model 3 role:** exploratory uncertainty signal only

This pass did **not** change the learner model.

## Router tested

The tested default router was the requested simple split:

- if review is due -> `spacing_aware_review`
- else if any of:
  - early step
  - low predicted proficiency
  - high recent failure
  - high friction
  then -> `confidence_building`
- else -> `balanced_challenge`

`harder_challenge` stayed a benchmark only.

`failure_aware_remediation` was removed from the default path and kept only as a service-mode comparator.

## Threshold search

The search grid used:

- early-step cutoff: `3` vs `5`
- low-proficiency threshold: `20% / 25% / 30%` quantiles of the balanced-reference probability
- recent-failure threshold: `70% / 75% / 80%` quantiles of recent failure total
- friction rule: `current` vs `stricter`

Selected threshold set under the pre-specified router ranking rule:

- early-step cutoff: `5`
- low-proficiency quantile: `0.30`
- low-proficiency threshold: `0.71797`
- recent-failure quantile: `0.75`
- recent-failure threshold: `45.24461`
- friction rule: `current`

Reference outputs:

- [router_threshold_search.csv](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/simple_two_mode_router_qmatrix_rpfa/router_threshold_search.csv)
- [router_selected_summary.json](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/simple_two_mode_router_qmatrix_rpfa/router_selected_summary.json)
- [router_selected_rows.csv](D:/model1_baseline_agent_bundle/outputs/phase1_adaptive_policy/simple_two_mode_router_qmatrix_rpfa/router_selected_rows.csv)

## Result

The best simple router only helped by a **tiny** amount on the primary new-learning target-gap metric:

- router new-learning target gap `1-10`: `0.002871`
- best fixed new-learning baseline (`confidence_building`) target gap `1-10`: `0.002918`
- delta: `-0.000047`

But it lost on the tie-break metrics:

- router new-learning policy advantage `1-10`: `0.205625`
- fixed `confidence_building`: `0.210114`
- delta: `-0.004489`

- router new-learning stability: `0.009920`
- fixed `confidence_building`: `0.000378`
- delta: `+0.009543`

So the gain in target gap was too small to justify the stability blow-up.

## Review-mode reading

Treating spacing as a separate review service remains the right move.

Selected review-mode result:

- review target gap `1-10`: `0.008632`
- review seen-item rate: `0.8279`
- review fallback rate: `0.0000`
- due-review coverage: `1.0000`

This should be reported separately from new-item targeting, not blended into the same winner-takes-all policy claim.

## Decision

The simple router does **not** survive as the new operational default.

Current operational decision:

- **scorer:** explicit Q-matrix R-PFA Model 2
- **review mode:** `spacing_aware_review` with `24`-hour threshold
- **default new-learning choice:** fixed `confidence_building`
- `balanced_challenge`: keep as the safer comparator / later-sequence reference
- `harder_challenge`: keep as a benchmark for policy advantage
- `failure_aware_remediation`: do not include in the default path
- **Model 3:** keep exploratory only

So the repo should now stop trying to discover another complex router and carry forward this simpler operational freeze until stronger data or evaluation becomes available.
