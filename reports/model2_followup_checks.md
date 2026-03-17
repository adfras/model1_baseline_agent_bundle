# Model 2 Follow-up Checks

This note records the next-round validation checks for Model 2 after the initial Track A VI comparison.

## Checks completed

1. Phase 1 Track B VI fit and evaluation on unseen public students.
   This is the marginal cold-start version, not the newer online warm-start path.
2. Phase 1 Track A pilot MCMC fit and evaluation on the original within-student forward holdout.
3. Phase 1 Track A stricter MCMC fit and evaluation with higher `target_accept`, more tuning, and four chains.
4. Phase 1 Track B true online warm-start evaluation on unseen public students.

Small note:

- exact Track B marginal numbers can drift slightly across reruns because new unseen-student effects are sampled stochastically during prediction

## Track A reference points

Model 1 Track A VI:

- log loss: `0.5446`
- Brier: `0.1841`
- accuracy: `0.7153`
- AUC: `0.7609`

Model 2 Track A VI:

- log loss: `0.5463`
- Brier: `0.1848`
- accuracy: `0.7149`
- AUC: `0.7601`

Initial reading:

- the first VI comparison favored Model 1 on Track A

## Track B VI Marginal Cold-Start Results

Model 1 Track B validation:

- log loss: `0.4355`
- Brier: `0.1401`
- accuracy: `0.7979`
- AUC: `0.7964`

Model 2 Track B validation:

- log loss: `0.4302`
- Brier: `0.1381`
- accuracy: `0.7997`
- AUC: `0.7970`

Model 1 Track B test:

- log loss: `0.4542`
- Brier: `0.1479`
- accuracy: `0.7852`
- AUC: `0.7889`

Model 2 Track B test:

- log loss: `0.4504`
- Brier: `0.1465`
- accuracy: `0.7872`
- AUC: `0.7891`

Track B reading:

- Model 2 is slightly better on the primary probabilistic metrics used by the project
- the AUC change is negligible
- this means Model 2 is not a clean failure once unseen-student initialization is considered
- these results are still from marginal cold-start evaluation rather than true online warm-start updating

## Track B online warm-start results

Model 1 online validation:

- log loss: `0.4085`
- Brier: `0.1304`
- accuracy: `0.8149`
- AUC: `0.8153`

Model 2 online validation:

- log loss: `0.4085`
- Brier: `0.1304`
- accuracy: `0.8141`
- AUC: `0.8147`

Model 1 online test:

- log loss: `0.4347`
- Brier: `0.1408`
- accuracy: `0.7948`
- AUC: `0.8059`

Model 2 online test:

- log loss: `0.4340`
- Brier: `0.1406`
- accuracy: `0.7960`
- AUC: `0.8063`

Early-attempt windows on the online test split:

- attempts `1-5`:
  - Model 1 log loss `0.4023`
  - Model 2 log loss `0.3976`
- attempts `6-10`:
  - Model 1 log loss `0.2378`
  - Model 2 log loss `0.2385`
- attempts `11-20`:
  - Model 1 log loss `0.3057`
  - Model 2 log loss `0.3076`

Online Track B reading:

- Model 2 still improves slightly over Model 1 on overall test log loss, Brier, accuracy, and AUC
- the improvement is small overall
- the strongest Model 2 gain is on the earliest `1-5` attempts, which is relevant to Phase 2
- Model 1 remains slightly better in the later `6-10` and `11-20` windows

## Track A pilot MCMC results

Pilot fit config:

- `2` chains
- `500` tune
- `500` draws
- `1` core
- `target_accept = 0.9`

Model 2 Track A pilot MCMC metrics:

- log loss: `0.5457`
- Brier: `0.1845`
- accuracy: `0.7158`
- AUC: `0.7607`
- calibration intercept: `-0.0293`
- calibration slope: `0.9053`

Pilot MCMC diagnostics:

- divergences: `18`
- max `R-hat`: `1.11`
- min bulk ESS: `22`
- min tail ESS: `55`

Pilot MCMC reading:

- predictive performance is still not clearly better than Model 1 on Track A
- the diagnostics are poor enough that this run should not be treated as a clean inferential result

## Track A strict MCMC results

Strict fit config:

- `4` chains
- `1500` tune
- `1000` draws
- `4` cores
- `target_accept = 0.97`

Model 2 Track A strict MCMC metrics:

- log loss: `0.5458`
- Brier: `0.1846`
- accuracy: `0.7157`
- AUC: `0.7606`
- calibration intercept: `-0.0291`
- calibration slope: `0.9043`

Strict MCMC diagnostics:

- divergences: `0`
- max `R-hat`: `1.08`
- min bulk ESS: `42`
- min tail ESS: `106`

Strict MCMC reading:

- the divergences are gone, which is a real improvement over the pilot run
- the remaining `R-hat` and ESS values are still weaker than the finished Model 1 strict fit
- predictive performance on Track A still does not beat Model 1

## Interim interpretation

The follow-up checks move Model 2 from "probably not worth it" to "mixed evidence":

- Track A still does not show a clear gain
- Track B shows a small but consistent gain on log loss and Brier
- online Track B shows that Model 2's main advantage is concentrated in the earliest attempts
- the strict MCMC rerun removes divergences but still does not deliver a Track A predictive win for Model 2

## Track B full-rank ADVI trust check

This was the minimum next trustworthiness check for the online Track B result.

Fit config:

- full-rank ADVI
- `20,000` iterations
- `1,000` posterior draws
- elapsed seconds: `2605.97`

Full-rank VI diagnostics:

- initial loss: `277924.66`
- final loss: `53694.19`
- best loss: `49198.95`
- tail mean: `51460.31`
- tail SD: `3092.87`
- tail relative change: `-0.00344`

That is materially less stable than the original mean-field VI run, whose tail SD was only `18.44` and whose tail relative change was `-0.00017`.

Key posterior shifts:

- student intercept SD:
  - mean-field Track B VI: `0.122`
  - full-rank Track B VI: `0.378`
  - Track A strict MCMC: `0.574`
- student slope SD:
  - mean-field Track B VI: `0.116`
  - full-rank Track B VI: `0.054`
  - Track A strict MCMC: `0.060`

So the full-rank check moves the posterior geometry materially away from the original mean-field Track B fit and toward the stricter Track A MCMC story on the slope SD term.

Track B online test metrics under full-rank ADVI:

- log loss: `0.4547`
- Brier: `0.1456`
- accuracy: `0.7931`
- AUC: `0.8015`
- calibration intercept: `0.0559`
- calibration slope: `1.4063`

Early-attempt windows under full-rank ADVI:

- attempts `1-5`: log loss `0.4391`
- attempts `6-10`: log loss `0.2838`
- attempts `11-20`: log loss `0.3464`

Trust-check reading:

- the original mean-field online gain does not survive the stronger VI approximation
- the earliest-attempt advantage disappears under the full-rank rerun
- calibration also gets materially worse

So the current Model 2 Track B win should now be treated as inference-sensitive rather than selection-grade.

## Practical implication

If the project emphasis is strictly Track A forward prediction for already-observed learners, Model 1 still looks safer.

If the project emphasis includes unseen-student initialization as a serious precursor to Phase 2 transfer, Model 2 still matters historically because it motivated the Track B comparison.

But after the full-rank trust check, the current disciplined reading is:

- mean-field Model 2 was promising
- stronger inference did not confirm that promise
- Model 2 should no longer be the default Phase 2 choice from the current evidence

## Next disciplined step

Do this next:

- treat Model 2 as an inference-sensitive negative/ambiguous result, not the current leader
- keep it in the repo as a documented comparison point
- make the main Phase 1 selection decision between Model 1 and the now-robust Model 3 screening path
