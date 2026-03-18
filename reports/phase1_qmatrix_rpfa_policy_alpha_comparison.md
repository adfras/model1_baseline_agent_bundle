# R-PFA Alpha Policy Comparison

This note compares the **Model 2** offline policy suite for `alpha = 0.8` and `alpha = 0.9` on the same full public test rows.

Selection rule used here:

- primary selector: lower mean target gap across the three **new-learning** policies
- tie margin on that mean target gap: `0.0002`
- first tie break: higher mean policy advantage over the actual next item
- second tie break: lower mean recommendation instability

## New-learning aggregate

Policies included:

- `balanced_challenge`
- `harder_challenge`
- `confidence_building`

- Mean target gap `1-10`: `0.8=0.00542`, `0.9=0.00541` (lower better)
- Mean policy advantage `1-10`: `0.8=0.19532`, `0.9=0.19547` (higher better)
- Mean recommendation instability: `0.8=0.00098`, `0.9=0.00088` (lower better)
- Mean band-hit rate `1-10`: `0.8=0.99411`, `0.9=0.99389` (higher better)

Operational alpha recommendation:

- selected alpha: `0.9`
- reason: Target-gap tie; higher mean policy advantage across the three new-learning policies.

## Support-mode diagnostic aggregate

Policies included:

- `failure_aware_remediation`
- `spacing_aware_review`

- Mean target gap `1-10`: `0.8=0.05060`, `0.9=0.05009`
- Mean policy advantage `1-10`: `0.8=0.13940`, `0.9=0.14018`
- Mean fallback rate: `0.8=0.24204`, `0.9=0.24204`

These support-mode numbers are reported as diagnostics only. The operational alpha choice is anchored on the new-learning policies because those are the main unseen-item recommendation modes.

## Policy-by-policy detail

### balanced_challenge

- `0.8` target gap `1-10`: `0.00490`
- `0.9` target gap `1-10`: `0.00498`
- `0.8` policy advantage `1-10`: `0.18117`
- `0.9` policy advantage `1-10`: `0.18130`
- `0.8` stability: `0.00096`
- `0.9` stability: `0.00085`
- Same recommended item rate: `0.6341`

### confidence_building

- `0.8` target gap `1-10`: `0.00479`
- `0.9` target gap `1-10`: `0.00483`
- `0.8` policy advantage `1-10`: `0.20095`
- `0.9` policy advantage `1-10`: `0.20113`
- `0.8` stability: `0.00099`
- `0.9` stability: `0.00090`
- Same recommended item rate: `0.6509`

### failure_aware_remediation

- `0.8` target gap `1-10`: `0.05927`
- `0.9` target gap `1-10`: `0.05829`
- `0.8` policy advantage `1-10`: `0.12707`
- `0.9` policy advantage `1-10`: `0.12833`
- `0.8` stability: `0.02003`
- `0.9` stability: `0.01885`
- Same recommended item rate: `0.8152`

### harder_challenge

- `0.8` target gap `1-10`: `0.00658`
- `0.9` target gap `1-10`: `0.00642`
- `0.8` policy advantage `1-10`: `0.20385`
- `0.9` policy advantage `1-10`: `0.20399`
- `0.8` stability: `0.00100`
- `0.9` stability: `0.00089`
- Same recommended item rate: `0.7085`

### spacing_aware_review

- `0.8` target gap `1-10`: `0.04193`
- `0.9` target gap `1-10`: `0.04189`
- `0.8` policy advantage `1-10`: `0.15174`
- `0.9` policy advantage `1-10`: `0.15203`
- `0.8` stability: `0.00090`
- `0.9` stability: `0.00075`
- Same recommended item rate: `0.5724`

