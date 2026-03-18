# Conservative Router V3 Attempt

This note records a first **conservative router v3** attempt that was tried after the policy subgroup diagnostics.

The goal was to build a narrower router that:

- kept **R-PFA Model 2** as the scorer
- used the subgroup diagnostics to switch more conservatively between:
  - `balanced_challenge`
  - `confidence_building`
  - `harder_challenge`
- kept remediation and review as more clearly separated service modes

## Result

The first v3 attempt was worse than the current baselines and was **not kept** in the repo.

Held-out replay summary:

- v3 target gap `1-10`: `0.01633`
- v3 policy advantage `1-10`: `0.17335`
- v3 stability: `0.02701`
- v3 recent-failure coverage: `0.35274`
- v3 due-review coverage: `0.15166`
- v3 seen-item rate: `0.02287`

Comparison:

- hybrid v1 target gap `1-10`: `0.01223`
- tuned hybrid v2 target gap `1-10`: `0.01108`
- fixed `balanced_challenge` target gap `1-10`: `0.00498`

Interpretation:

- v3 became too remediation-heavy
- that pushed target-gap performance down too far
- even with a lower seen-item rate, it was not good enough to justify replacing the fixed-policy suite or the existing exploratory hybrid branches

## Decision

- do **not** keep the v3 router code as an active repo branch
- do **not** treat v3 as the next default path
- keep the current operational baseline as:
  - fixed Model 2 policy suite
  - selected `24`-hour spacing review mode
  - subgroup diagnostics
  - hybrid v1 and tuned hybrid v2 only as exploratory routing references

So the current repo state remains:

- **operational scorer:** explicit Q-matrix R-PFA Model 2
- **operational baseline:** fixed policy suite
- **exploratory routers:** hybrid v1 and tuned hybrid v2
- **rejected exploratory attempt:** conservative router v3
