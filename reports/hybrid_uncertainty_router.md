# Hybrid Uncertainty Router

This note records the first uncertainty-aware policy router built on top of the selected explicit Q-matrix RPFA models.

## Goal

Use:

- **R-PFA Model 2** for the mean correctness estimate
- **R-PFA Model 3** for a step-level uncertainty signal

Then let that uncertainty signal help choose **which policy mode** to use, rather than asking Model 3 to replace Model 2 as the main mean predictor.

## Router design

At each held-out step, the router:

1. scores candidate items with **Model 2**
2. computes a step-level uncertainty SD from **Model 3**
3. routes into one of these modes:
   - `failure_aware_remediation`
   - `spacing_aware_review`
   - `diagnostic_challenge`
   - `harder_challenge`
   - `balanced_challenge`

Current thresholds:

- high uncertainty if uncertainty SD `>= 0.35`
- low uncertainty if uncertainty SD `< 0.20`
- remediation if recent failure rate `>= 0.45`
- harder progression if recent success rate `>= 0.75` and uncertainty is low

The current heuristic did not route any rows into `confidence_building`.

## Overall result

Held-out replay summary:

- student-averaged target gap `1-10`: `0.012233`
- band-hit rate `1-10`: `0.981669`
- policy advantage over actual historical next item `1-10`: `0.190817`
- recommendation stability mean absolute difference: `0.014391`
- recent-failure coverage rate: `0.288387`
- due-review coverage rate: `0.240386`
- seen-item recommendation rate: `0.141642`

## Route shares

- balanced challenge: `46.95%`
- diagnostic challenge: `19.52%`
- spacing-aware review: `18.52%`
- harder challenge: `10.67%`
- failure-aware remediation: `4.34%`

Route reasons:

- default balanced: `4,969`
- high uncertainty: `2,066`
- due review ready: `1,960`
- confident progression: `1,129`
- recent failure rate: `459`

## What improved

Compared with staying in `balanced_challenge` all the time, the hybrid router covers more of the situations it was designed to notice:

- recent-failure coverage:
  - balanced challenge alone: `0.182840`
  - hybrid router overall: `0.288387`
- due-review coverage:
  - balanced challenge alone: `0.084286`
  - hybrid router overall: `0.240386`

So the uncertainty-aware router does succeed as a **mode switcher**:

- it sends a meaningful share of steps into diagnostic mode when uncertainty is high
- it activates review mode when due-review items are available and the learner is not currently in a strong remediation state

## What did not improve

This first hybrid router does **not** beat the simple fixed policies on pure target-difficulty control:

- balanced challenge alone has target gap `1-10` of `0.004983`
- confidence-building alone has target gap `1-10` of `0.004829`
- hybrid router overall has target gap `1-10` of `0.012233`

That is expected to some extent:

- the hybrid router is not always aiming at the same target probability
- it intentionally trades some target precision for remediation, review, and diagnostic routing

But the current router is also clearly less stable than the simple fixed policies, so it should be treated as a **first prototype**, not the new default.

## Reading

The useful conclusion is:

- **uncertainty is worth using for policy routing**
- but the current Model 3 uncertainty signal works better as a **secondary controller** than as a reason to replace Model 2

So the practical position is now:

- keep **R-PFA Model 2** as the default question-selection model
- use Model 3 uncertainty only to build and refine **routing heuristics**
- improve the router next by:
  - tuning thresholds
  - adding engagement / frustration proxies from observable behavior
  - developing item-level diagnostic criteria rather than only step-level uncertainty

This remains an **offline policy-behavior result**, not a causal learning-gain estimate.
