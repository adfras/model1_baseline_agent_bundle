# ManyLabs And DBE Alignment Note

This note records the lesson from comparing this DBE workspace to `C:\ManyLabsAnalyses`.

## Why heterogeneity pays off in ManyLabs

In `C:\ManyLabsAnalyses`, heterogeneity is part of the thing being optimized directly:

- person-level residual variance is inside the main likelihood
- held-out evaluation asks whether that richer variance model improves prediction and calibration over a homoskedastic baseline
- the main research question is already aligned to that variance term

So heterogeneity is not being bolted on there. It is inside the primary model-and-evaluate loop.

## Why DBE has been harder

In DBE, the scientific heterogeneity result is real:

- baseline heterogeneity is supported
- growth heterogeneity is supported
- stability heterogeneity is supported

But the replay work asks a harder downstream question:

- can those heterogeneity signals improve the next-item choice beyond a strong raw Model 2 baseline?

That is harder because:

- the current replay setup has limited headroom
- the policy problem is downstream of the learner model
- the logged DBE data do not provide true decision-native policy data such as candidate slates, propensities, or counterfactual rewards

## Repo decision

So the repo now treats the two problems differently:

- **ManyLabs logic**:
  - heterogeneity is useful because it is directly inside the optimized objective
- **DBE logic**:
  - heterogeneity is scientifically important and now exported as learner-state profiles
  - replay-based policy claims remain exploratory only

## Practical implication

The DBE mainline is now:

1. scientific heterogeneity discovery
2. learner-state estimation and reporting
3. future decision-native system design requirements

The DBE replay work stays in the repo as:

- useful bridge work
- useful negative-result evidence
- but not the repo’s central success claim
