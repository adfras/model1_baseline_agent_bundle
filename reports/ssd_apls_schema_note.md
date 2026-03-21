# SSD/APLS Support-Request Schema Note

This note records the canonical row used for the SSD/APLS support-selection branch.

The same processed request table now also feeds the repo's SSD/APLS continuous elapsed-time branch. That newer branch keeps the same one-row-per-request table, but derives continuous future-window targets from it instead of training directly on the one-step binary reward.

Related docs:

- [ssd_apls_continuous_location_scale_branch.md](D:/model1_baseline_agent_bundle/reports/ssd_apls_continuous_location_scale_branch.md)
- [ssd_apls_continuous_target_dictionary.md](D:/model1_baseline_agent_bundle/reports/ssd_apls_continuous_target_dictionary.md)

## Canonical decision row

- one row per support request
- reward is `next_problem_correctness`
- the action is the chosen support strategy
- the family gate sits above the exact-support scorer

Canonical columns written to the processed table:

- `request_id`
- `user_xid`
- `assignment_log_id`
- `problem_id`
- `skill_id`
- `chosen_strategy_id`
- `alternative_strategy_ids`
- `candidate_strategy_ids`
- `candidate_count`
- `chosen_family`
- `propensity`
- `next_problem_correctness`
- learner-state feature columns prefixed with `state_`
- chosen-strategy feature columns prefixed with `strategy_`

## Family definition

- `enable_multimedia_family = False`
- `scaffold`: support marked as a hint
- `reveal`: support marked as an explanation

## Summary

- processed requests: `82184`
- students: `15420`
- problems: `3312`
- skills: `0`
- chosen strategies: `7475`
- mean candidate count: `2.675`

## Continuous-branch note

For the continuous elapsed-time branch, this same processed row is used only as the starting point. The final training targets are derived per request, for example:

- `future_acc_1d`
- `future_acc_7d`
- `future_acc_28d`
- `delta_state_average_hint_count_1d`
- `delta_state_average_attempt_count_7d`

Those targets are built in:

- [01_build_support_continuous_targets.R](D:/model1_baseline_agent_bundle/R/01_build_support_continuous_targets.R)

and written to:

- [support_requests_continuous_targets.csv](D:/model1_baseline_agent_bundle/data/processed/ssd_apls/support_requests_continuous_targets.csv)
