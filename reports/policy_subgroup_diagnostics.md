# Operational Policy Subgroup Diagnostics

This note compares the **current operational Model 2 policy stack** by subgroup.

Important scope note:

- the four new-item policies come from the fixed Model 2 R-PFA suite
- the review policy is replaced with the selected `24`-hour `spacing_aware_review` branch
- this is still an **offline target-control / policy-behavior** analysis, not a causal learning-gain estimate

## Subgroup definitions

- high recent failure = remediation recent-failure score >= `3.25` (75% quantile)
- low predicted proficiency = actual-next probability <= `0.5225` (25% quantile)
- high friction = `hint_used == 1` or `selection_change > 1` or `duration_seconds >= 77.91` seconds (75% quantile of positive durations)
- review-eligible context = the selected `24`-hour spacing policy identifies a due-review item

## Headline winners

### all_rows

- support: `10583` attempts, `1138` students
- best target-gap policy: `confidence_building` (`0.00483`)
- best policy-advantage policy: `harder_challenge` (`0.20399`)

### early_steps_1_5

- support: `5566` attempts, `1138` students
- best target-gap policy: `confidence_building` (`0.00407`)
- best policy-advantage policy: `harder_challenge` (`0.21158`)

### later_steps_6_10

- support: `5017` attempts, `1043` students
- best target-gap policy: `balanced_challenge` (`0.00574`)
- best policy-advantage policy: `confidence_building` (`0.20722`)

### review_eligible_context

- support: `6920` attempts, `722` students
- best target-gap policy: `balanced_challenge` (`0.00564`)
- best policy-advantage policy: `harder_challenge` (`0.20190`)

### high_recent_failure_context

- support: `2653` attempts, `457` students
- best target-gap policy: `balanced_challenge` (`0.00552`)
- best policy-advantage policy: `confidence_building` (`0.24527`)

### low_predicted_proficiency_context

- support: `2646` attempts, `889` students
- best target-gap policy: `balanced_challenge` (`0.00522`)
- best policy-advantage policy: `confidence_building` (`0.48207`)

### high_friction_context

- support: `3616` attempts, `1010` students
- best target-gap policy: `balanced_challenge` (`0.00491`)
- best policy-advantage policy: `confidence_building` (`0.22752`)

### actual_single_kc

- support: `4645` attempts, `962` students
- best target-gap policy: `balanced_challenge` (`0.00534`)
- best policy-advantage policy: `confidence_building` (`0.23483`)

### actual_multi_kc

- support: `5938` attempts, `1122` students
- best target-gap policy: `confidence_building` (`0.00442`)
- best policy-advantage policy: `harder_challenge` (`0.22451`)


## Detailed policy snapshots

### all_rows

- `confidence_building`: target gap `0.00483`, policy advantage `0.20113`, band-hit `0.9935`, stability `0.00090`, seen-item rate `0.0000`, fallback `0.0000`
- `balanced_challenge`: target gap `0.00498`, policy advantage `0.18130`, band-hit `0.9983`, stability `0.00085`, seen-item rate `0.0000`, fallback `0.0000`
- `harder_challenge`: target gap `0.00642`, policy advantage `0.20399`, band-hit `0.9899`, stability `0.00089`, seen-item rate `0.0000`, fallback `0.0000`
- `spacing_aware_review`: target gap `0.03554`, policy advantage `0.15861`, band-hit `0.9923`, stability `0.00069`, seen-item rate `0.5413`, fallback `0.3461`
- `failure_aware_remediation`: target gap `0.05829`, policy advantage `0.12833`, band-hit `0.8223`, stability `0.01885`, seen-item rate `0.0000`, fallback `0.0547`
### early_steps_1_5

- `confidence_building`: target gap `0.00407`, policy advantage `0.19564`, band-hit `0.9946`, stability `0.00064`, seen-item rate `0.0000`, fallback `0.0000`
- `balanced_challenge`: target gap `0.00457`, policy advantage `0.18244`, band-hit `0.9991`, stability `0.00077`, seen-item rate `0.0000`, fallback `0.0000`
- `harder_challenge`: target gap `0.00605`, policy advantage `0.21158`, band-hit `0.9912`, stability `0.00082`, seen-item rate `0.0000`, fallback `0.0000`
- `spacing_aware_review`: target gap `0.03598`, policy advantage `0.15265`, band-hit `0.9925`, stability `0.00074`, seen-item rate `0.5158`, fallback `0.3715`
- `failure_aware_remediation`: target gap `0.05609`, policy advantage `0.12922`, band-hit `0.8255`, stability `0.02165`, seen-item rate `0.0000`, fallback `0.0692`
### later_steps_6_10

- `balanced_challenge`: target gap `0.00574`, policy advantage `0.18003`, band-hit `0.9974`, stability `0.00102`, seen-item rate `0.0000`, fallback `0.0000`
- `confidence_building`: target gap `0.00599`, policy advantage `0.20722`, band-hit `0.9922`, stability `0.00131`, seen-item rate `0.0000`, fallback `0.0000`
- `harder_challenge`: target gap `0.00722`, policy advantage `0.19557`, band-hit `0.9884`, stability `0.00103`, seen-item rate `0.0000`, fallback `0.0000`
- `spacing_aware_review`: target gap `0.03217`, policy advantage `0.16522`, band-hit `0.9922`, stability `0.00057`, seen-item rate `0.5697`, fallback `0.3179`
- `failure_aware_remediation`: target gap `0.06073`, policy advantage `0.12734`, band-hit `0.8186`, stability `0.01671`, seen-item rate `0.0000`, fallback `0.0387`
### review_eligible_context

- `balanced_challenge`: target gap `0.00564`, policy advantage `0.17716`, band-hit `0.9988`, stability `0.00104`, seen-item rate `0.0000`, fallback `0.0000`
- `confidence_building`: target gap `0.00590`, policy advantage `0.19637`, band-hit `0.9915`, stability `0.00121`, seen-item rate `0.0000`, fallback `0.0000`
- `harder_challenge`: target gap `0.00703`, policy advantage `0.20190`, band-hit `0.9879`, stability `0.00099`, seen-item rate `0.0000`, fallback `0.0000`
- `spacing_aware_review`: target gap `0.00863`, policy advantage `0.17916`, band-hit `0.9897`, stability `0.00039`, seen-item rate `0.8279`, fallback `0.0000`
- `failure_aware_remediation`: target gap `0.05927`, policy advantage `0.12317`, band-hit `0.8251`, stability `0.02058`, seen-item rate `0.0000`, fallback `0.0496`
### high_recent_failure_context

- `balanced_challenge`: target gap `0.00552`, policy advantage `0.20135`, band-hit `1.0000`, stability `0.00113`, seen-item rate `0.0000`, fallback `0.0000`
- `confidence_building`: target gap `0.00559`, policy advantage `0.24527`, band-hit `0.9913`, stability `0.00081`, seen-item rate `0.0000`, fallback `0.0000`
- `harder_challenge`: target gap `0.00579`, policy advantage `0.20300`, band-hit `0.9974`, stability `0.00139`, seen-item rate `0.0000`, fallback `0.0000`
- `spacing_aware_review`: target gap `0.02849`, policy advantage `0.19789`, band-hit `0.9947`, stability `0.00069`, seen-item rate `0.5786`, fallback `0.3129`
- `failure_aware_remediation`: target gap `0.05829`, policy advantage `0.16245`, band-hit `0.8737`, stability `0.01736`, seen-item rate `0.0000`, fallback `0.0000`
### low_predicted_proficiency_context

- `balanced_challenge`: target gap `0.00522`, policy advantage `0.35353`, band-hit `0.9974`, stability `0.00196`, seen-item rate `0.0000`, fallback `0.0000`
- `harder_challenge`: target gap `0.00573`, policy advantage `0.23372`, band-hit `0.9992`, stability `0.00132`, seen-item rate `0.0000`, fallback `0.0000`
- `confidence_building`: target gap `0.00616`, policy advantage `0.48207`, band-hit `0.9849`, stability `0.00298`, seen-item rate `0.0000`, fallback `0.0000`
- `spacing_aware_review`: target gap `0.03370`, policy advantage `0.40524`, band-hit `0.9917`, stability `0.00101`, seen-item rate `0.5325`, fallback `0.3534`
- `failure_aware_remediation`: target gap `0.06101`, policy advantage `0.32433`, band-hit `0.8012`, stability `0.03061`, seen-item rate `0.0000`, fallback `0.0756`
### high_friction_context

- `balanced_challenge`: target gap `0.00491`, policy advantage `0.18411`, band-hit `0.9989`, stability `0.00134`, seen-item rate `0.0000`, fallback `0.0000`
- `confidence_building`: target gap `0.00513`, policy advantage `0.22752`, band-hit `0.9945`, stability `0.00195`, seen-item rate `0.0000`, fallback `0.0000`
- `harder_challenge`: target gap `0.00631`, policy advantage `0.18799`, band-hit `0.9920`, stability `0.00148`, seen-item rate `0.0000`, fallback `0.0000`
- `spacing_aware_review`: target gap `0.03324`, policy advantage `0.18042`, band-hit `0.9928`, stability `0.00169`, seen-item rate `0.5899`, fallback `0.2992`
- `failure_aware_remediation`: target gap `0.05759`, policy advantage `0.13465`, band-hit `0.8103`, stability `0.03001`, seen-item rate `0.0000`, fallback `0.0622`
### actual_single_kc

- `balanced_challenge`: target gap `0.00534`, policy advantage `0.18238`, band-hit `0.9981`, stability `0.00100`, seen-item rate `0.0000`, fallback `0.0000`
- `confidence_building`: target gap `0.00556`, policy advantage `0.23483`, band-hit `0.9925`, stability `0.00133`, seen-item rate `0.0000`, fallback `0.0000`
- `harder_challenge`: target gap `0.00674`, policy advantage `0.17776`, band-hit `0.9910`, stability `0.00084`, seen-item rate `0.0000`, fallback `0.0000`
- `spacing_aware_review`: target gap `0.03384`, policy advantage `0.18236`, band-hit `0.9899`, stability `0.00064`, seen-item rate `0.5692`, fallback `0.3236`
- `failure_aware_remediation`: target gap `0.06133`, policy advantage `0.12951`, band-hit `0.7767`, stability `0.02208`, seen-item rate `0.0000`, fallback `0.0424`
### actual_multi_kc

- `confidence_building`: target gap `0.00442`, policy advantage `0.17476`, band-hit `0.9943`, stability `0.00122`, seen-item rate `0.0000`, fallback `0.0000`
- `balanced_challenge`: target gap `0.00479`, policy advantage `0.18045`, band-hit `0.9985`, stability `0.00109`, seen-item rate `0.0000`, fallback `0.0000`
- `harder_challenge`: target gap `0.00642`, policy advantage `0.22451`, band-hit `0.9891`, stability `0.00103`, seen-item rate `0.0000`, fallback `0.0000`
- `spacing_aware_review`: target gap `0.03552`, policy advantage `0.14004`, band-hit `0.9943`, stability `0.00099`, seen-item rate `0.5195`, fallback `0.3638`
- `failure_aware_remediation`: target gap `0.05517`, policy advantage `0.12741`, band-hit `0.8579`, stability `0.02955`, seen-item rate `0.0000`, fallback `0.0643`
