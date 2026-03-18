# Spacing Policy Due-Review Threshold Grid

This note compares `spacing_aware_review` across a due-review-hours grid on the **operational Model 2 R-PFA branch**.

Selection rule used here:

- primary selector: lower student-averaged target gap on the **review-eligible subset**
- tie margin on that target gap: `0.001`
- first tie break: higher review-eligible rate
- second tie break: higher review-eligible policy advantage over the actual next item

Selected threshold:

- due-review hours: `24`
- label: `due24`
- reason: Best review-eligible target gap, with the broadest review eligibility among the compared thresholds.

Important interpretation rule:

- this is a **review-mode** comparison
- the eligible-subset metrics are the main evidence
- the overall fallback rate is reported separately so review-mode coverage is visible

### due24

- Due-review threshold: `24` hours
- Review-eligible rate: `0.6539`
- Review-eligible target gap: `0.00863`
- Review-eligible policy advantage: `0.17891`
- Review-eligible band-hit rate: `0.9897`
- Review-eligible seen-item rate: `0.8279`
- Overall fallback rate: `0.3461`
- Overall seen-item rate: `0.5413`

### due48

- Due-review threshold: `48` hours
- Review-eligible rate: `0.5706`
- Review-eligible target gap: `0.00995`
- Review-eligible policy advantage: `0.17357`
- Review-eligible band-hit rate: `0.9856`
- Review-eligible seen-item rate: `0.8235`
- Overall fallback rate: `0.4294`
- Overall seen-item rate: `0.4699`

### due72

- Due-review threshold: `72` hours
- Review-eligible rate: `0.5135`
- Review-eligible target gap: `0.00992`
- Review-eligible policy advantage: `0.17265`
- Review-eligible band-hit rate: `0.9851`
- Review-eligible seen-item rate: `0.8358`
- Overall fallback rate: `0.4865`
- Overall seen-item rate: `0.4292`

### due96

- Due-review threshold: `96` hours
- Review-eligible rate: `0.4837`
- Review-eligible target gap: `0.00986`
- Review-eligible policy advantage: `0.17283`
- Review-eligible band-hit rate: `0.9861`
- Review-eligible seen-item rate: `0.8353`
- Overall fallback rate: `0.5163`
- Overall seen-item rate: `0.4040`

