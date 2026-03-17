# Phase 1 Multi-KC Discovery Schema Note

This note records the full-data multi-KC Phase 1 discovery table built from DBE-KT22.

## Main design

- All visible attempts are retained if their item has at least one linked KC.
- Multi-KC items are not dropped.
- A long attempt-KC table is built internally so each attempt contributes one prior-opportunity update to every linked KC.
- The model-facing attempt table stays one row per attempt.

## Practice construction

- KC update mode: `full_credit`
- For each student-KC pair, `kc_opportunity` tracks prior exposure before the current attempt.
- For multi-KC items, the attempt contributes to every linked KC after the response using the configured update rule.
- The main model-facing practice term is:
  - `practice_feature = kc_practice_feature_sum`
  - where `kc_practice_feature_sum = sum(log1p(kc_opportunity_k))` across the attempt's linked KCs

Current update rule summary:

- KC increment per linked KC: `1.0 per linked KC`
- Practice aggregation: `sum(log1p(kc_opportunity_k)) across linked KCs`
- Recency decay alpha currently materialized in the long table: `0.9`
- Due-review threshold for default long-table flags: `48.0` hours

Additional summaries retained on each attempt:

- `kc_count`
- `kc_ids`
- `kc_names`
- `kc_opportunity_sum`
- `kc_opportunity_mean`
- `kc_opportunity_max`
- `kc_practice_feature_mean`
- `kc_practice_feature_weighted`
- `any_first_kc`
- `all_first_kc`

## Discovery sample summary

- Raw rows before hidden exclusion: `161953`
- Visible rows after hidden exclusion: `158389`
- Attempt rows with at least one KC: `158389`
- Long attempt-KC rows: `300246`
- Eligible learners: `1138`
- Eligible attempt rows: `157989`
- Items with at least one KC: `212`
- KCs represented: `93`
- Single-KC attempt rows: `54315`
- Multi-KC attempt rows: `104074`
- Questions with no KC link: `0`
- Mean KC count per attempt: `1.89562406480248`
- Median KC count per attempt: `2.0`

## Validation checks

- KC opportunity monotone within student-KC: `True`
- Chronology violations after sorting: `0`
- `alpha = 1.0` decays match cumulative prior success counts: `True`
- `alpha = 1.0` decays match cumulative prior failure counts: `True`

## Held-out profile

- Mean KC count in test rows: `1.7101395117090183`
- Share of test rows with any first-seen KC: `0.37484429496761335`
- Share of test rows with all linked KCs first-seen: `0.21904583956153462`
- Mean `kc_opportunity_mean` in test rows: `3.813501702375021`
- Mean `practice_feature` in test rows: `1.9922518738712571`
