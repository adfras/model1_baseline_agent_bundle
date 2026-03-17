# Phase 1 Heterogeneity Summary

This report applies the revised `Variance + Prediction` rule to the public discovery fits.

## Decision rule

- For added heterogeneity terms beyond Model 1, the default practical floor is a posterior SD above `0.03` on the logit scale.
- The 94% HDI lower bound should clear that floor before the variance term is treated as substantively present.
- Predictive gate: the richer model must either improve held-out log loss, or be no worse by more than `0.001` while Brier improves and calibration slope moves closer to `1.0`.

## Model1

- Held-out log loss: `0.5470285290059463`
- Held-out Brier: `0.18506239252774762`
- Held-out calibration slope: `0.9261824400260369`
- student_intercept_sigma: mean `0.5670`, 94% HDI `[0.5370, 0.5960]`
- item_sigma: mean `1.1410`, 94% HDI `[1.0910, 1.1890]`

## Model2

- Held-out log loss: `0.5463110981141837`
- Held-out Brier: `0.18480925690789698`
- Held-out calibration slope: `0.9384043265407697`
- student_intercept_sigma: mean `0.5280`, 94% HDI `[0.4990, 0.5580]`
- student_slope_sigma: mean `0.0610`, 94% HDI `[0.0530, 0.0690]`
- item_sigma: mean `1.1430`, 94% HDI `[1.0960, 1.1900]`

## Model3

- Held-out log loss: `0.5438667678067205`
- Held-out Brier: `0.1839411959632485`
- Held-out calibration slope: `0.9573733031258924`
- student_intercept_sigma: mean `0.4705`, 94% HDI `[0.4382, 0.5027]`
- student_slope_sigma: mean `0.0577`, 94% HDI `[0.0499, 0.0647]`
- state_sigma_global: mean `0.4830`, 94% HDI `[0.4559, 0.5113]`

## Interpretation template

- If Model 2 adds a slope variance term that clears the practical floor and the predictive gate, growth heterogeneity is present.
- If Model 3 adds a stability variance term that clears the practical floor and the predictive gate, stability heterogeneity is present.
- If only Model 1 survives, baseline-level differences dominate the public discovery sample.
