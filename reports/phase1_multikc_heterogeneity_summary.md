# Phase 1 Heterogeneity Summary

This report applies the revised `Variance + Prediction` rule to the public discovery fits.

## Decision rule

- For added heterogeneity terms beyond Model 1, the default practical floor is a posterior SD above `0.03` on the logit scale.
- The 94% HDI lower bound should clear that floor before the variance term is treated as substantively present.
- Predictive gate: the richer model must either improve held-out log loss, or be no worse by more than `0.001` while Brier improves and calibration slope moves closer to `1.0`.

## Model1

- Held-out log loss: `0.5463318804775981`
- Held-out Brier: `0.18479950712802393`
- Held-out calibration slope: `0.9265485467459662`
- student_intercept_sigma: mean `0.5690`, 94% HDI `[0.5380, 0.5980]`
- item_sigma: mean `1.1370`, 94% HDI `[1.0870, 1.1860]`

## Model2

- Held-out log loss: `0.5454913620715228`
- Held-out Brier: `0.1845026388522046`
- Held-out calibration slope: `0.9409283357112193`
- student_intercept_sigma: mean `0.5230`, 94% HDI `[0.4930, 0.5540]`
- student_slope_sigma: mean `0.0480`, 94% HDI `[0.0420, 0.0530]`
- item_sigma: mean `1.1390`, 94% HDI `[1.0910, 1.1870]`

## Model3

- Held-out log loss: `0.543726209010983`
- Held-out Brier: `0.18389153993470994`
- Held-out calibration slope: `0.9632444600725458`
- student_intercept_sigma: mean `0.4558`, 94% HDI `[0.4212, 0.4905]`
- student_slope_sigma: mean `0.0485`, 94% HDI `[0.0421, 0.0541]`
- state_sigma_global: mean `0.4860`, 94% HDI `[0.4588, 0.5145]`

## Interpretation template

- If Model 2 adds a slope variance term that clears the practical floor and the predictive gate, growth heterogeneity is present.
- If Model 3 adds a stability variance term that clears the practical floor and the predictive gate, stability heterogeneity is present.
- If only Model 1 survives, baseline-level differences dominate the public discovery sample.
