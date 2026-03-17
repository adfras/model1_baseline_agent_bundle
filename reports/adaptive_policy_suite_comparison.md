# Adaptive Policy Suite Comparison

This note compares the offline policy suite for explicit Q-matrix R-PFA Model 2 and Model 3.

Model metadata:

- Model 2 history mode: `rpfa`, decay alpha `0.9`
- Model 3 history mode: `rpfa`, decay alpha `0.9`
- Evaluation window: first `10` held-out primary-evaluation steps per student
- Due-review threshold: `48.0` hours

Current reading:

- Model 2 remains the default policy model unless Model 3 clearly wins on policy-facing metrics.
- This report is an offline target-control / policy-behavior comparison, not a causal learning-gain estimate.

### balanced_challenge

- Model 2 target gap `1-5`: `0.00457`
- Model 3 target gap `1-5`: `0.00461`
- Model 2 target gap `1-10`: `0.00498`
- Model 3 target gap `1-10`: `0.00496`
- Model 2 band-hit rate `1-10`: `0.9983`
- Model 3 band-hit rate `1-10`: `0.9987`
- Model 2 policy advantage over actual `1-10`: `0.18130`
- Model 3 policy advantage over actual `1-10`: `0.18108`
- Model 2 stability mean abs diff: `0.00085`
- Model 3 stability mean abs diff: `0.00133`
- Model 2 recent-failure coverage: `0.1828`
- Model 3 recent-failure coverage: `0.1805`
- Model 2 due-review coverage: `0.0843`
- Model 3 due-review coverage: `0.0718`
- Model 2 fallback rate: `0.0000`
- Model 3 fallback rate: `0.0000`
- Model 2 seen-item recommendation rate: `0.0000`
- Model 3 seen-item recommendation rate: `0.0000`
- Model 2 mean candidate count: `212.00`
- Model 3 mean candidate count: `212.00`
- Same recommended item rate: `0.3118`

### confidence_building

- Model 2 target gap `1-5`: `0.00407`
- Model 3 target gap `1-5`: `0.00409`
- Model 2 target gap `1-10`: `0.00483`
- Model 3 target gap `1-10`: `0.00486`
- Model 2 band-hit rate `1-10`: `0.9935`
- Model 3 band-hit rate `1-10`: `0.9932`
- Model 2 policy advantage over actual `1-10`: `0.20113`
- Model 3 policy advantage over actual `1-10`: `0.19917`
- Model 2 stability mean abs diff: `0.00090`
- Model 3 stability mean abs diff: `0.00130`
- Model 2 recent-failure coverage: `0.2561`
- Model 3 recent-failure coverage: `0.2458`
- Model 2 due-review coverage: `0.1015`
- Model 3 due-review coverage: `0.0983`
- Model 2 fallback rate: `0.0000`
- Model 3 fallback rate: `0.0000`
- Model 2 seen-item recommendation rate: `0.0000`
- Model 3 seen-item recommendation rate: `0.0000`
- Model 2 mean candidate count: `212.00`
- Model 3 mean candidate count: `212.00`
- Same recommended item rate: `0.3746`

### failure_aware_remediation

- Model 2 target gap `1-5`: `0.05609`
- Model 3 target gap `1-5`: `0.05653`
- Model 2 target gap `1-10`: `0.05829`
- Model 3 target gap `1-10`: `0.05851`
- Model 2 band-hit rate `1-10`: `0.8223`
- Model 3 band-hit rate `1-10`: `0.8202`
- Model 2 policy advantage over actual `1-10`: `0.12833`
- Model 3 policy advantage over actual `1-10`: `0.12775`
- Model 2 stability mean abs diff: `0.01885`
- Model 3 stability mean abs diff: `0.01979`
- Model 2 recent-failure coverage: `0.9453`
- Model 3 recent-failure coverage: `0.9453`
- Model 2 due-review coverage: `0.1878`
- Model 3 due-review coverage: `0.1894`
- Model 2 fallback rate: `0.0547`
- Model 3 fallback rate: `0.0547`
- Model 2 seen-item recommendation rate: `0.0000`
- Model 3 seen-item recommendation rate: `0.0000`
- Model 2 mean candidate count: `212.00`
- Model 3 mean candidate count: `212.00`
- Same recommended item rate: `0.8168`

### harder_challenge

- Model 2 target gap `1-5`: `0.00605`
- Model 3 target gap `1-5`: `0.00627`
- Model 2 target gap `1-10`: `0.00642`
- Model 3 target gap `1-10`: `0.00647`
- Model 2 band-hit rate `1-10`: `0.9899`
- Model 3 band-hit rate `1-10`: `0.9899`
- Model 2 policy advantage over actual `1-10`: `0.20399`
- Model 3 policy advantage over actual `1-10`: `0.20521`
- Model 2 stability mean abs diff: `0.00089`
- Model 3 stability mean abs diff: `0.00154`
- Model 2 recent-failure coverage: `0.1762`
- Model 3 recent-failure coverage: `0.1834`
- Model 2 due-review coverage: `0.0823`
- Model 3 due-review coverage: `0.0750`
- Model 2 fallback rate: `0.0000`
- Model 3 fallback rate: `0.0000`
- Model 2 seen-item recommendation rate: `0.0000`
- Model 3 seen-item recommendation rate: `0.0000`
- Model 2 mean candidate count: `212.00`
- Model 3 mean candidate count: `212.00`
- Same recommended item rate: `0.3866`

### spacing_aware_review

- Model 2 target gap `1-5`: `0.04230`
- Model 3 target gap `1-5`: `0.04235`
- Model 2 target gap `1-10`: `0.04189`
- Model 3 target gap `1-10`: `0.04192`
- Model 2 band-hit rate `1-10`: `0.9908`
- Model 3 band-hit rate `1-10`: `0.9921`
- Model 2 policy advantage over actual `1-10`: `0.15203`
- Model 3 policy advantage over actual `1-10`: `0.15089`
- Model 2 stability mean abs diff: `0.00075`
- Model 3 stability mean abs diff: `0.00131`
- Model 2 recent-failure coverage: `0.4377`
- Model 3 recent-failure coverage: `0.4358`
- Model 2 due-review coverage: `0.5706`
- Model 3 due-review coverage: `0.5706`
- Model 2 fallback rate: `0.4294`
- Model 3 fallback rate: `0.4294`
- Model 2 seen-item recommendation rate: `0.4699`
- Model 3 seen-item recommendation rate: `0.4785`
- Model 2 mean candidate count: `212.00`
- Model 3 mean candidate count: `212.00`
- Same recommended item rate: `0.3241`

