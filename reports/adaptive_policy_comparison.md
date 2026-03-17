# Adaptive Policy Comparison

This note records the first offline replay comparison for adaptive next-question scoring on the explicit Q-matrix PFA branch.

It is now a **historical single-policy note**. The current mainline uses the explicit Q-matrix **R-PFA** branch and a five-policy replay suite. For the current policy comparison, use:

- [adaptive_policy_suite_comparison.md](D:/model1_baseline_agent_bundle/reports/adaptive_policy_suite_comparison.md)

## Setup

- model candidates:
  - explicit Q-matrix PFA Model 2
  - explicit Q-matrix PFA Model 3
- recommendation rule:
  - choose the unseen item with predicted correctness closest to `0.7`
- target band:
  - `[0.6, 0.8]`
- evaluation window:
  - first `10` held-out primary-evaluation steps per student
- replay type:
  - offline scoring against the logged held-out sequence
  - this measures target-difficulty control, not causal learning impact

## Main result

Both models can steer recommendations very close to the target band in this replay.

### Model 2

- recommendation rows: `10,583`
- students covered: `1,138`
- student-averaged target gap:
  - attempts `1-5`: `0.00445`
  - attempts `1-10`: `0.00482`
- recommended target-band hit rate:
  - attempts `1-5`: `1.000`
  - attempts `1-10`: `1.000`
- mean advantage over the actual historical next item:
  - attempts `1-5`: `0.18567`
  - attempts `1-10`: `0.18329`
- recommendation stability mean absolute difference: `0.00074`

### Model 3

- recommendation rows: `10,583`
- students covered: `1,138`
- student-averaged target gap:
  - attempts `1-5`: `0.00475`
  - attempts `1-10`: `0.00500`
- recommended target-band hit rate:
  - attempts `1-5`: `1.000`
  - attempts `1-10`: `1.000`
- mean advantage over the actual historical next item:
  - attempts `1-5`: `0.18593`
  - attempts `1-10`: `0.18299`
- recommendation stability mean absolute difference: `0.00123`

## Direct comparison

- Mean Model 3 minus Model 2 target-gap difference:
  - attempts `1-5`: `+0.000316`
  - attempts `1-10`: `+0.000201`
- Same recommended item rate across the two models: `0.3048`

## Interpretation

- Both models are able to recommend items very close to the desired target difficulty in this first replay.
- Model 2 is slightly better on target-gap control and is more stable.
- Model 3 recommends different items from Model 2 on most steps, but those differences do not currently improve target-difficulty control.
- So on this first offline adaptive evaluation, **Model 2 is the better next-question policy model**.

## Caveat

This replay uses posterior-mean scoring and evaluates target-difficulty control against the logged sequence. It does **not** yet estimate the causal learning gain from asking the recommended item instead of the observed historical item.
