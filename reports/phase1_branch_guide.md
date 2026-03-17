# Phase 1 Branch Guide

This note defines the main Phase 1 branch names used in the repo.

## Why this note exists

Several different KC-handling branches now exist in the public DBE analysis.

Those branches answer different questions:

- some are restrictive construct checks
- some use the full dataset but simplify KC structure before fitting
- some keep KC structure directly inside the likelihood

Without a naming guide, it is easy to confuse them.

## 1. Single-KC branch

Definition:

- keep only items linked to exactly one KC
- drop all multi-KC items

Purpose:

- construct-clean sensitivity analysis
- makes `kc_opportunity` unambiguous without extra allocation rules

What it costs:

- throws away a large share of the visible DBE rows
- strips out much of the repeated same-skill structure

Current role:

- sensitivity only
- not the operational mainline

## 2. Primary-KC branch

Definition:

- keep all items
- assign each item one deterministic `primary_kc_id`
- in this repo, use the earliest row in `Question_KC_Relationships.csv`

Purpose:

- full-data sensitivity branch
- checks what happens if every item is forced to one KC

What it costs:

- keeps all rows
- but collapses multi-KC items to one chosen KC

Current role:

- older full-data sensitivity branch
- no longer the preferred operational branch

## 3. Collapsed-feature multi-KC branch

Definition:

- keep all items
- keep all linked KCs in preprocessing
- compute KC-specific opportunity per linked KC
- then collapse that KC information to one scalar attempt-level feature before fitting

In this repo, the main collapsed feature was:

- `practice_feature = sum(log1p(kc_opportunity_k))`

Purpose:

- first full-data multi-KC operational branch
- lets the model use all linked KCs without dropping rows
- still keeps the fitted likelihood relatively simple

What it costs:

- KC structure affects the model only through a single summarized practice term
- KC effects are not modeled directly inside the likelihood

Current role:

- important historical full-data branch
- still useful as a comparison baseline
- now superseded by the explicit Q-matrix branch for Models 1 and 2

## 4. Fractional multi-KC sensitivity branch

Definition:

- same rows as the full multi-KC branch
- same split
- same model ladder
- only the KC allocation rule changes

In this repo:

- each linked KC gets fractional exposure `1 / kc_count` instead of full credit

Purpose:

- sensitivity test for whether the richer heterogeneity result depends on full-credit KC accumulation

Current role:

- main like-for-like multi-KC sensitivity check

## 5. Explicit Q-matrix branch

Definition:

- keep all items
- keep all linked KCs
- build attempt-by-KC design matrices
- put KC structure directly inside the model likelihood

For the current explicit branch:

- Model 1 includes:
  - learner intercepts
  - item effects
  - KC intercepts
  - KC-specific practice effects
- Model 2 adds:
  - learner-specific growth deviations on the KC-practice signal

Purpose:

- direct test of growth heterogeneity without collapsing KC structure to one scalar before fitting

What it changes relative to the collapsed-feature branch:

- KC information is modeled explicitly, not only summarized in preprocessing

Current role:

- newest operational comparison branch for the full ladder
- current best place to test whether Models 2 and 3 add real heterogeneity on the full dataset

## 6. Explicit Q-matrix PFA / R-PFA branch

Definition:

- keep all items
- keep all linked KCs
- keep KC structure directly in the likelihood
- replace opportunity-only KC history with:
  - prior KC wins
  - prior KC fails
- for the operational R-PFA path, optionally weight those histories by KC-opportunity lag

Purpose:

- improve the learner-history signal without dropping data
- test whether better KC history matters more than extra latent-state tuning

Current role:

- best-yield predictive branch family in the repo
- current operational mainline for learner modeling and offline policy replay

## 7. Adaptive-policy replay branch

Definition:

- take the fitted PFA / R-PFA learner models
- replay held-out students sequentially
- score candidate items after each observed attempt
- choose items under a small modular policy family

Current v1 policy suite:

- balanced challenge
- harder challenge
- confidence-building
- failure-aware remediation
- spacing-aware review

Purpose:

- bridge learner modeling to question selection
- compare Model 2 and Model 3 as policy models rather than only as forecasters

Current role:

- first narrow offline policy test
- does not estimate causal learning gain

## Short translation table

- `single-KC branch`:
  - only one-KC items, many rows dropped
- `primary-KC branch`:
  - all items kept, one KC assigned per item
- `collapsed-feature multi-KC branch`:
  - all items kept, all linked KCs used in preprocessing, then summarized to one scalar feature
- `fractional multi-KC branch`:
  - same full dataset, only KC credit allocation changes
- `explicit Q-matrix branch`:
  - all items kept, KC terms enter the likelihood directly
- `explicit Q-matrix PFA / R-PFA branch`:
  - explicit-Q branch plus wins/fails history, with optional recency weighting
- `adaptive-policy replay branch`:
  - sequential offline question-selection test built on the fitted learner models

## Current operational reading

Right now:

- the old single-KC branch is a restrictive sensitivity check
- the collapsed-feature and fractional multi-KC branches support richer heterogeneity
- the explicit Q-matrix branch shows that Model 2 beats Model 1 and Model 3 beats Model 2 on the same full-data held-out rows
- the explicit Q-matrix PFA / R-PFA branch is now the best-yield predictive branch family
- the selected operational RPFA alpha is `0.9`
- the adaptive-policy replay branch is where Model 2 and Model 3 are compared as question-selection models
- the current operational reading is:
  - richest scientific model = Model 3 on the explicit Q-matrix ladder
  - default policy model = R-PFA Model 2

So when the repo says a result came from the `collapsed-feature branch`, that does **not** mean the old single-KC branch.
