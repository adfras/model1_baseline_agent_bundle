# Model 1 Item-Shift Stress Test

This note records the compact unseen-item stress test that was added after the main Track B online runs.

## Design

Starting point:

- existing public Track B student split
- same unseen-student online evaluation path as the main Model 1 Track B run

Stress change:

- deterministically hold out `10%` of the original Track B training items
- remove the corresponding training-student rows for those items
- keep validation and test students unchanged
- allow unseen test items to use a zero-centered item effect during online prediction

Implementation:

- split script: [src/split_track_b_item_stress.py](../src/split_track_b_item_stress.py)
- split config: [config/model1_track_b_item_stress_split.json](../config/model1_track_b_item_stress_split.json)
- fit config: [config/model1_item_stress_fit.json](../config/model1_item_stress_fit.json)
- eval config: [config/model1_item_stress_online_evaluate_test.json](../config/model1_item_stress_online_evaluate_test.json)

## Stress dataset summary

- held-out items: `21`
- held-out item share of original Track B training items: `0.0991`
- removed training rows: `10,951`
- stress-test rows:
  - train: `99,483`
  - validation: `22,891`
  - test: `24,664`
- unseen-item rows in the test split: `2,433` of `24,664` (`9.86%`)

## Main comparison

Baseline Model 1 online Track B test on seen items only:

- log loss: `0.4347`
- Brier: `0.1408`
- accuracy: `0.7948`
- AUC: `0.8059`

Item-shift stress Model 1 online Track B test:

- log loss: `0.4439`
- Brier: `0.1439`
- accuracy: `0.7940`
- AUC: `0.7921`

## Seen versus unseen item slices

Seen-item rows within the stress test:

- rows: `22,231`
- log loss: `0.4339`
- Brier: `0.1404`
- accuracy: `0.7969`
- AUC: `0.8068`

New-item rows within the stress test:

- rows: `2,433`
- log loss: `0.5355`
- Brier: `0.1759`
- accuracy: `0.7670`
- AUC: `0.5988`

## Attempt-window view

Stress-test online windows:

- attempts `1-5`: log loss `0.4036`
- attempts `6-10`: log loss `0.2772`
- attempts `11-20`: log loss `0.3060`

## Reading

The important result is not just that the overall stress test is worse.

The important result is where the degradation sits:

- on seen items, Model 1 is still very close to its original online Track B performance
- on genuinely unseen items, performance drops sharply

So the current Phase 1 evidence still supports:

- warm-start transfer to new students on already-seen public items

It does not yet support:

- warm-start transfer to new students plus new items without local item calibration

That matters directly for Phase 2:

- if local items differ from public items, the transfer story should focus on student-side priors and local item estimation rather than pretending item transfer is already solved
