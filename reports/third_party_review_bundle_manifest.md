# Third-Party Review Bundle Manifest

This bundle is intended for external review of the current modelling workspace state.

It is now a **slim review bundle**, not a full exploration archive.

Use this zip when you want a reviewer to see the current state without dragging through every superseded branch.

## Included

- project contract and plan
- the current mainline repo notes:
  - focus / status / selection memo
  - the plain-language current-objective and failure-mode note
  - the ManyLabs-vs-DBE alignment note
  - the decision-native successor spec
- all repository `src/*.py` scripts
- all repository `config/*.json` configs
- the core public-data scientific ladder notes:
  - full multi-KC schema note
  - explicit Q-matrix comparison note
  - RPFA tuning and operational-selection notes
- the learner-state export note and exported profile tables
- only the policy notes needed to explain the current negative result:
  - policy-alignment calibration
  - calibrated fixed-policy decision
  - corrected local-uncertainty restart decision
  - direct heterogeneity policy decision
- only the summary outputs needed to support those notes
- the public preprocessing summary JSON

## Excluded

- raw data under `data/`
- processed learner-attempt tables except the top-level sample summary JSON
- large posterior artifacts such as:
  - `*.nc`
  - `*.npz`
- row-level policy replay CSVs
- superseded router branches and their outputs
- exploratory historical reports that are no longer needed to understand the current state
- virtual environment files

If a reviewer later needs the full exploration trail, keep using the workspace or the separate full archive bundle rather than overloading this slim zip again.

## Important context

- The bundle reflects the current workspace state, not only the last pushed commit.
- The repo originally centered heterogeneity discovery and conditional local transfer.
- With no local dataset currently available, the practical focus is now:
  - full-data public heterogeneity discovery
  - learner-state estimation from the scientific explicit-Q ladder
  - decision-native successor design
  - offline next-question replay only as a bridge / negative-result track
- The full-data explicit Q-matrix ladder supports Model 2 and then Model 3 scientifically.
- The strongest predictive improvement came from **PFA / R-PFA wins/fails history**.
- The selected operational R-PFA alpha is `0.9`.
- The selected spacing-review threshold on the operational Model 2 branch is `24` hours.
- The repo now exports learner-level baseline, growth, stability, and latent-state tables directly from the scientific explicit-Q posterior files.
- The current replay conclusion remains negative:
  - raw `R-PFA Model 2` stays the operational baseline
  - Model 3 remains the richer scientific model
  - calibration-side and local-residual branches did not survive operationally
  - a later direct heterogeneity utility branch also failed to beat the frozen baseline
