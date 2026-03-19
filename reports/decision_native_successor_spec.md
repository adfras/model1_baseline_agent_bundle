# Decision-Native Successor Spec

This memo defines how a future system would answer:

**What item should we choose next for this student?**

It is a design note only. It is **not** implemented in this repo.

## Core shift

The next system should not bolt heterogeneity onto a replay policy after the fact.

Instead, heterogeneity must enter the next-item decision **inside the objective** being optimized.

## Decision problem

For student `i` at time `t`, choose item `j` from a small feasible candidate slate to maximize:

`utility(i, j, t) = challenge_fit + remediation_value + review_value + information_value - risk_penalty - repetition_penalty`

The chosen item is the feasible candidate with the highest utility.

## Student-state layers

The future system should separate two layers of learner state.

### 1. Mean proficiency / growth state

This layer estimates:

- current expected correctness for a student-item pair
- learner baseline level
- learner growth tendency

### 2. Risk / trust / instability state

This layer estimates:

- how much to trust the mean prediction
- whether the learner is unstable, noisy, or friction-prone right now
- whether a riskier or more diagnostic item is justified

## How heterogeneity should enter the decision directly

Heterogeneity should affect the utility terms directly:

- **baseline / proficiency heterogeneity**
  - changes target difficulty
- **growth heterogeneity**
  - changes how aggressively the system advances challenge
- **stability / volatility heterogeneity**
  - changes risk tolerance
  - changes whether the system prefers diagnostic or safer items
- **review-related heterogeneity**
  - changes spacing and review value

## Candidate slate assumption

The future system should not rank the entire item universe.

It should score a small feasible slate such as:

- curriculum-eligible items
- recent-KC-neighborhood items
- due-review items
- a bounded set of plausible next items

That is the scale at which uncertainty and heterogeneity can plausibly change the winning action.

## Required logging for a real policy system

Future policy work needs decision-native logs that DBE replay does not currently provide.

Required fields:

- candidate slate shown at decision time
- chosen action
- allowed action set
- timestamps
- response time
- hint use
- selection changes
- trust / difficulty feedback
- randomized exploration markers or logging propensities
- short-horizon reward labels
  - next-attempt correctness
  - short-window mastery change
  - review success / recovery after failure

## Evaluation contract

This repo should not claim future policy wins from DBE replay alone.

Future policy claims require at least one of:

- randomized logging
- known propensities
- valid off-policy evaluation support

Without that, the repo should keep claims limited to:

- learner-state estimation
- calibration analysis
- offline target-control / policy-behavior diagnostics

## Immediate implication for this repo

This repo now freezes:

- scientific heterogeneity discovery on DBE
- learner-state exports from the scientific explicit-Q ladder
- offline replay as bridge evidence only

It does **not** open a new active policy branch here.
