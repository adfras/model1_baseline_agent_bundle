# SSD/APLS Continuous Target Dictionary

This note defines the continuous elapsed-time targets used by the SSD/APLS location-scale branch.

## Source table and ordering

All targets are built from:

- [support_requests_processed.csv](D:/model1_baseline_agent_bundle/data/processed/ssd_apls/support_requests_processed.csv)

Rows are ordered within each `user_xid` by:

- `created_at`
- then `request_id`

The derived table is:

- [support_requests_continuous_targets.csv](D:/model1_baseline_agent_bundle/data/processed/ssd_apls/support_requests_continuous_targets.csv)

## Time windows

The branch currently uses three real future windows:

- `1d` = 24 hours
- `7d` = 168 hours
- `28d` = 672 hours

These windows mean:

- look forward from the current support request
- include later requests for the same student whose elapsed time is within the window

They do not mean:

- the student spent 1, 7, or 28 days on one problem
- the student studied for 1, 7, or 28 days in total

## Future-accuracy targets

Let `y_{s,j}` be `next_problem_correctness` for student `s` on later request `j`.

Let `W_H(s,t)` be the set of later requests for student `s` that occur within horizon `H` after request `t`.

Then:

\[
\mathrm{future\_acc\_H}(s,t)=\frac{1}{|W_H(s,t)|}\sum_{j \in W_H(s,t)} y_{s,j}
\]

Defined targets:

- `future_acc_1d`
- `future_acc_7d`
- `future_acc_28d`

Support counts are also saved:

- `future_acc_1d_count`
- `future_acc_7d_count`
- `future_acc_28d_count`

These counts are the number of future requests included in the mean.

## Delta-state targets

For each state metric `m`, let `t_H^*(s,t)` be the last later request for student `s` that still falls inside horizon `H`.

Then:

\[
\Delta m_H(s,t)=m_{s,t_H^*(s,t)}-m_{s,t}
\]

Defined target families:

- `delta_state_average_hint_count_H`
- `delta_state_average_attempt_count_H`
- `delta_state_average_first_action_answer_H`
- `delta_state_average_correctness_H`

for `H in {1d, 7d, 28d}`.

If no later request exists inside the window, the target is undefined and the row is dropped for that target.

## Plain-English interpretation

`future_acc_*`

- higher is better
- asks whether the learner performs better after the support request over the next calendar window

`delta_state_average_hint_count_*`

- lower is better
- negative values mean later help dependence is lower than it is now

`delta_state_average_attempt_count_*`

- lower is better
- negative values mean later work becomes less effortful or less error-prone

`delta_state_average_first_action_answer_*`

- higher is better
- positive values mean the learner becomes more likely to answer cleanly on the first action

`delta_state_average_correctness_*`

- higher is better
- positive values mean the learner's running correctness trend improves

## Important feature wording

`state_median_time_on_task` does not mean total study time.

Plain meaning:

- the learner's typical time spent per earlier problem
- specifically the median of earlier time-on-task values

`state_median_first_response_time` means:

- the learner's typical delay before the first response on earlier work

## Feature missingness

In the continuous-target table, every raw `state_*` feature currently has the same missingness rate:

- `18.85%`

During fitting:

- missing state values are imputed with the train-set median
- a binary `state_missing_any` flag is added

## Current support by target

Support counts below come from [support_requests_continuous_targets.csv](D:/model1_baseline_agent_bundle/data/processed/ssd_apls/support_requests_continuous_targets.csv).

### Future-accuracy targets

| Target | Rows | Train rows | Test rows | Test students |
|---|---:|---:|---:|---:|
| `future_acc_1d` | 51,303 | 41,119 | 10,184 | 1,967 |
| `future_acc_7d` | 60,049 | 48,081 | 11,968 | 2,068 |
| `future_acc_28d` | 66,512 | 53,237 | 13,275 | 2,184 |

### Hint-count delta targets

| Target | Rows | Train rows | Test rows | Test students |
|---|---:|---:|---:|---:|
| `delta_state_average_hint_count_1d` | 41,804 | 33,482 | 8,322 | 1,631 |
| `delta_state_average_hint_count_7d` | 48,316 | 38,639 | 9,677 | 1,717 |
| `delta_state_average_hint_count_28d` | 53,311 | 42,619 | 10,692 | 1,809 |

### Attempt-count delta targets

| Target | Rows | Train rows | Test rows | Test students |
|---|---:|---:|---:|---:|
| `delta_state_average_attempt_count_1d` | 41,804 | 33,482 | 8,322 | 1,631 |
| `delta_state_average_attempt_count_7d` | 48,316 | 38,639 | 9,677 | 1,717 |
| `delta_state_average_attempt_count_28d` | 53,311 | 42,619 | 10,692 | 1,809 |

### First-action-answer delta targets

| Target | Rows | Train rows | Test rows | Test students |
|---|---:|---:|---:|---:|
| `delta_state_average_first_action_answer_1d` | 41,804 | 33,482 | 8,322 | 1,631 |
| `delta_state_average_first_action_answer_7d` | 48,316 | 38,639 | 9,677 | 1,717 |
| `delta_state_average_first_action_answer_28d` | 53,311 | 42,619 | 10,692 | 1,809 |

### Running-correctness delta targets

| Target | Rows | Train rows | Test rows | Test students |
|---|---:|---:|---:|---:|
| `delta_state_average_correctness_1d` | 41,804 | 33,482 | 8,322 | 1,631 |
| `delta_state_average_correctness_7d` | 48,316 | 38,639 | 9,677 | 1,717 |
| `delta_state_average_correctness_28d` | 53,311 | 42,619 | 10,692 | 1,809 |

## How the targets are used in fitting

For `future_acc_*` targets:

- raw target is in `[0, 1]`
- the fitter applies an empirical-logit transform with pseudo-count `0.5`
- the transformed target is then standardized on the train set

For `delta_*` targets:

- the identity transform is used
- the transformed target is then standardized on the train set

The evaluator maps predictions back to the raw target scale before writing:

- [test_predictions.csv](D:/model1_baseline_agent_bundle/outputs/ssd_apls/continuous_location_scale/future_acc_1d/test_predictions.csv)
- [metric_comparison.csv](D:/model1_baseline_agent_bundle/outputs/ssd_apls/continuous_location_scale/future_acc_1d/metric_comparison.csv)
