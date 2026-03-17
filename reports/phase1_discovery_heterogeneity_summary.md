# Phase 1 Heterogeneity Summary

This report applies the revised `Variance + Prediction` rule to the public discovery fits.

## Decision rule

- For added heterogeneity terms beyond Model 1, the default practical floor is a posterior SD above `0.03` on the logit scale.
- The 94% HDI lower bound should clear that floor before the variance term is treated as substantively present.
- Predictive gate: the richer model must either improve held-out log loss, or be no worse by more than `0.001` while Brier improves and calibration slope moves closer to `1.0`.

## Model1

- Held-out log loss: `0.5674143611764336`
- Held-out Brier: `0.19438026031279787`
- Held-out calibration slope: `0.9165957481640151`
- student_intercept_sigma: mean `0.5050`, 94% HDI `[0.4740, 0.5410]`
- item_sigma: mean `1.2220`, 94% HDI `[1.1730, 1.2820]`

## Model2

- Held-out log loss: `0.5677420881846381`
- Held-out Brier: `0.19452397066130908`
- Held-out calibration slope: `0.9147186202717862`
- student_intercept_sigma: mean `0.4980`, 94% HDI `[0.4670, 0.5320]`
- student_slope_sigma: mean `0.0740`, 94% HDI `[0.0410, 0.1030]`
- item_sigma: mean `1.2250`, 94% HDI `[1.1680, 1.2850]`

## Interpretation template

- If Model 2 adds a slope variance term that clears the practical floor and the predictive gate, growth heterogeneity is present.
- If Model 3 adds a stability variance term that clears the practical floor and the predictive gate, stability heterogeneity is present.
- If only Model 1 survives, baseline-level differences dominate the public discovery sample.
