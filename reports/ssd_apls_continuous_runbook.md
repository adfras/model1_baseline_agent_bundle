# SSD/APLS Continuous Branch Runbook

This note is the practical runbook for the SSD/APLS continuous elapsed-time branch.

## Main entrypoints

Target builder:

- [01_build_support_continuous_targets.R](D:/model1_baseline_agent_bundle/R/01_build_support_continuous_targets.R)

Model fitter:

- [02_fit_support_location_scale.R](D:/model1_baseline_agent_bundle/R/02_fit_support_location_scale.R)

Evaluator:

- [03_eval_support_location_scale.R](D:/model1_baseline_agent_bundle/R/03_eval_support_location_scale.R)

All-target cloud runner:

- [run_all_targets_time_windows_t2d16.sh](D:/model1_baseline_agent_bundle/scripts/run_all_targets_time_windows_t2d16.sh)

Compute benchmark helper:

- [benchmark_support_location_scale_t2d16.sh](D:/model1_baseline_agent_bundle/scripts/benchmark_support_location_scale_t2d16.sh)

## Build the continuous-target table

```powershell
Rscript R/01_build_support_continuous_targets.R `
  --in data/processed/ssd_apls/support_requests_processed.csv `
  --split outputs/ssd_apls/support_scorer/support_split_assignments.csv `
  --out data/processed/ssd_apls/support_requests_continuous_targets.csv
```

This writes:

- [support_requests_continuous_targets.csv](D:/model1_baseline_agent_bundle/data/processed/ssd_apls/support_requests_continuous_targets.csv)

## Fit one target

### Heteroskedastic model

```powershell
Rscript R/02_fit_support_location_scale.R `
  --in data/processed/ssd_apls/support_requests_continuous_targets.csv `
  --target future_acc_1d `
  --model hetero `
  --chains 4 `
  --iter 1000 `
  --parallel_chains 4 `
  --threads_per_chain 4 `
  --grainsize 3000 `
  --adapt_delta 0.97 `
  --max_treedepth 13 `
  --refresh 1 `
  --save_warmup true `
  --show_messages true `
  --show_exceptions true `
  --init small `
  --cpp_optims true `
  --out_dir outputs/ssd_apls/continuous_location_scale/future_acc_1d
```

### Matched baseline model

```powershell
Rscript R/02_fit_support_location_scale.R `
  --in data/processed/ssd_apls/support_requests_continuous_targets.csv `
  --target future_acc_1d `
  --model homo `
  --chains 4 `
  --iter 1000 `
  --parallel_chains 4 `
  --threads_per_chain 4 `
  --grainsize 3000 `
  --adapt_delta 0.97 `
  --max_treedepth 13 `
  --refresh 1 `
  --save_warmup true `
  --show_messages true `
  --show_exceptions true `
  --init small `
  --cpp_optims true `
  --out_dir outputs/ssd_apls/continuous_location_scale/future_acc_1d
```

## Evaluate one target

```powershell
Rscript R/03_eval_support_location_scale.R `
  --in data/processed/ssd_apls/support_requests_continuous_targets.csv `
  --target future_acc_1d `
  --out_dir outputs/ssd_apls/continuous_location_scale/future_acc_1d
```

This writes:

- [test_predictions.csv](D:/model1_baseline_agent_bundle/outputs/ssd_apls/continuous_location_scale/future_acc_1d/test_predictions.csv)
- [metric_comparison.csv](D:/model1_baseline_agent_bundle/outputs/ssd_apls/continuous_location_scale/future_acc_1d/metric_comparison.csv)

## All-target cloud run

The current cloud runner:

- builds the continuous target table
- loops over all elapsed-time targets
- fits heteroskedastic then baseline
- evaluates each target
- archives outputs to Google Cloud Storage
- shuts the VM down

Current script:

- [run_all_targets_time_windows_t2d16.sh](D:/model1_baseline_agent_bundle/scripts/run_all_targets_time_windows_t2d16.sh)

## Current compute settings

The actively used production-direction settings are:

- VM family: Google Cloud `t2d-standard-16` Spot
- `4` chains
- `4` parallel chains
- `4` threads per chain
- `grainsize = 3000`
- `STAN_CPP_OPTIMS = true`
- `refresh = 1`
- `save_warmup = true`
- `adapt_delta = 0.97`
- `max_treedepth = 13`
- explicit `small` init files instead of `init = 0`

## Why the current tuning choices were made

`refresh = 1`

- used for operational visibility
- earlier buffered or infrequent refresh settings created too much ambiguity about progress

`save_warmup = true`

- used so the chain CSVs visibly grow during warmup
- this was necessary to rule out the earlier "alive but not writing" failure mode

`init = small`

- used to avoid bad default initialization behavior on the hierarchical scale parameters
- explicit small JSON inits are generated inside [02_fit_support_location_scale.R](D:/model1_baseline_agent_bundle/R/02_fit_support_location_scale.R)

`STAN_CPP_OPTIMS = true`

- enabled during compilation as a compute-only optimization
- it does not change the scientific model

`grainsize = 3000`

- the branch previously ran with `grainsize = 64`, which was too small
- a later short benchmark showed `grainsize = 1` was also too slow here
- `3000` was therefore chosen as the current operational setting for the real run
- it is a practical tuning choice, not a proven global optimum

## Benchmark script

The compute benchmark helper currently compares:

- `4 threads` with `grainsize = 1`
- `4 threads` with `grainsize = 3000`
- `3 threads` with `grainsize = 1`
- `3 threads` with `grainsize = 3000`

Script:

- [benchmark_support_location_scale_t2d16.sh](D:/model1_baseline_agent_bundle/scripts/benchmark_support_location_scale_t2d16.sh)

The benchmark is intentionally compute-only. It should never be read as a scientific model change.

## Output structure

Per-target output directory:

- [outputs/ssd_apls/continuous_location_scale/future_acc_1d](D:/model1_baseline_agent_bundle/outputs/ssd_apls/continuous_location_scale/future_acc_1d)

Common files:

- `homo_summary.csv`
- `hetero_summary.csv`
- `homo_metadata.rds`
- `hetero_metadata.rds`
- `test_predictions.csv`
- `metric_comparison.csv`
- `cmdstan_homo/*.csv`
- `cmdstan_hetero/*.csv`
- `support_location_scale-profile-*.csv`

Compiled model cache:

- [outputs/ssd_apls/continuous_location_scale/cmdstan_compiled](D:/model1_baseline_agent_bundle/outputs/ssd_apls/continuous_location_scale/cmdstan_compiled)

## What to read when a run looks wrong

First:

- per-target CmdStan logs on the VM
- `cmdstan_homo/*.csv`
- `cmdstan_hetero/*.csv`

Then:

- `support_location_scale-profile-*.csv`
- `homo_summary.csv`
- `hetero_summary.csv`

The profiling CSVs are the right place to inspect hot sections before spending more on hardware changes.

## Hardware notes

The branch is currently tuned first on `T2D`, not immediately migrated to another family.

Reason:

- `T2D` exposes one vCPU per physical core
- that makes it a clean baseline for within-chain threading
- compute tuning should be checked there before moving to `C4D`, `N4D`, or `C3D`

If faster silicon is needed after local tuning:

- first paid benchmark: `c4d-highcpu-32` Spot
- second paid benchmark: `n4d-highcpu-32` Spot
- easiest migration/value benchmark: `c3d-highcpu-30` Spot

## Current interpretation

The branch is implemented and runnable, but it is still an active modeling programme rather than a frozen deliverable.

What is already true:

- the target definitions are real elapsed-time windows
- the baseline and heteroskedastic models are matched correctly
- the fitter, evaluator, and cloud runner exist

What still remains:

- complete all target fits
- compare the full target panel
- decide which target families best separate learner-level mean differences from learner-level volatility differences
