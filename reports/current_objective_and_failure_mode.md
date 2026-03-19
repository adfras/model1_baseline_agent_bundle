# Current Objective And Why DBE Replay Is Failing

## Current objective

The current applied objective is no longer just:

- detect heterogeneity in public learner data

It is now:

- use learner heterogeneity to decide **which item should come next for this student**

That means the practical question is:

**Can learner differences in baseline, growth, and stability change the next-item decision in a way that beats the simpler frozen DBE baseline?**

The current frozen replay baseline is:

- scorer: raw explicit Q-matrix `R-PFA Model 2`
- history decay: `alpha = 0.9`
- review rule: `24`-hour `spacing_aware_review`
- default new-learning action: fixed `confidence_building`

## What has already been shown

### What succeeded

The public DBE data do support scientific heterogeneity.

On the full-data explicit Q-matrix ladder:

- Model 1 log loss `0.545311`
- Model 2 log loss `0.544366`
- Model 3 log loss `0.543782`

So the scientific read is:

- baseline heterogeneity is present
- growth heterogeneity is present
- stability heterogeneity is present

### What did not succeed

The repo has not yet converted that heterogeneity into a better next-item policy than the frozen raw-Model-2 baseline.

Failed policy-side attempts now include:

- Model 3 as a richer replay scorer
- Model 3 as a calibration / uncertainty side-channel
- KC-constrained local-residual policy-specific calibration
- hybrid routers using uncertainty and lagged proxies
- a direct heterogeneity utility branch where Model 3 learner-state signals changed the action choice itself

The latest direct branch is the clearest version of the current applied goal:

- it chose directly among:
  - `confidence_building`
  - `balanced_challenge`
  - `harder_challenge`
  - `spacing_aware_review`
- it used:
  - learner baseline rank
  - learner growth rank
  - learner stability rank
  - current latent-state signal

Result:

- direct policy target gap `1-10`: `0.020983`
- frozen spacing-or-confidence baseline: `0.006351`
- direct policy advantage `1-10`: `0.171326`
- frozen baseline: `0.189227`
- direct stability: `0.001371`
- frozen baseline: `0.000470`

So even when heterogeneity entered the action utility directly, the branch still lost.

Reference:

- [direct_heterogeneity_policy_decision.md](D:/model1_baseline_agent_bundle/reports/direct_heterogeneity_policy_decision.md)

## Why DBE replay is failing operationally

This does **not** mean heterogeneity is absent.

It means the current DBE replay setup is a weak place to exploit it.

### 1. The baseline is already strong

Raw `R-PFA Model 2` already finds near-target items very well.

That leaves little room for extra heterogeneity signals to improve the action choice.

### 2. The extra signal changes probabilities more than decisions

The heterogeneity terms are scientifically real, but they have not been strong enough to improve the final item ranking consistently.

In practice, the extra information has usually:

- nudged probabilities
- changed routing a little
- or shifted calibration a little

without improving the chosen action enough to beat the baseline.

### 3. The replay action problem is still not decision-native

DBE replay is built from logged attempts, not from a native recommendation system.

What is missing:

- the actual candidate slate shown at recommendation time
- logging propensities or randomized action selection
- an explicit reward for alternative item choices
- a direct measure of learning gain from the chosen action

So the project is still approximating the decision problem after the fact.

### 4. Heterogeneity is easier to exploit in ManyLabs than in DBE

In `C:\ManyLabsAnalyses`, heterogeneity sits directly inside the thing being optimized and evaluated.

In DBE, heterogeneity has mostly been tested as a downstream policy aid.

That makes DBE the harder, less aligned setting.

### 5. The current replay utilities keep collapsing toward simpler fixed actions

The direct heterogeneity branch was supposed to express a richer student-specific action rule.

Instead, the selected branch mostly collapsed to `balanced_challenge`.

That is a sign that the replay setup is not giving the heterogeneity terms enough clean leverage to justify more complex action logic.

## The current read

The most accurate current statement is:

- DBE is a good public dataset for **heterogeneity discovery**
- DBE is a useful public dataset for **learner-state estimation**
- DBE has **not yet** supported a convincing operational next-item win from those heterogeneity signals

So the project is currently trying to answer:

**How do we turn real learner heterogeneity into a decision-native next-item system, instead of bolting it onto replay after the fact?**

## What this implies

Right now the repo should be read in two layers.

### Mainline layer

- scientific heterogeneity discovery
- learner-state profile export
- decision-native system requirements

### Bridge / failure-analysis layer

- replay branches that tested whether heterogeneity could already produce a next-item win on DBE
- documented negative results showing that the current DBE setup has not yet achieved that

## Bottom line

The project is now trying to move from:

- “heterogeneity exists”

to:

- “heterogeneity changes what item we should choose next for this student”

That second step is the one that is still failing on DBE replay.

The reason is not that heterogeneity is fake.

The reason is that the current DBE replay setup has not been able to turn that heterogeneity into enough decision leverage to beat the simpler frozen baseline.
