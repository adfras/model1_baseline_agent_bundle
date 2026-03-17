# Project Pivot And Current Focus

## Why this note exists

The repo has moved through several real pivots, and older notes can make the current focus hard to see.

This note states plainly what changed and what the project is focusing on now.

## Original framing

The original scientific question was:

**Do learners differ in baseline performance, rate of improvement, and response stability in public longitudinal data, and do those same forms of heterogeneity replicate in a new student sample?**

The applied follow-up was:

**Can public-informed priors estimate those learner differences earlier in new students than weak-prior local fitting?**

## What changed

Three things shifted the project.

### 1. The data representation had been hiding signal

Earlier branches either:

- dropped multi-KC items
- or flattened multi-KC structure too aggressively

That made richer heterogeneity hard to detect reliably.

The project improved materially once it:

- kept the full visible DBE dataset
- retained all linked KCs
- moved KC structure into the likelihood directly

### 2. Better history features mattered more than extra tuning

The largest gain did not come from more latent-state tuning.

It came from replacing opportunity-only history with **PFA-style prior wins and fails per KC**, and then promoting **R-PFA** as the operational history family when recent KC outcomes matter more than older ones.

That changed the practical model ranking:

- explicit Q-matrix **R-PFA Model 2** is now the operational learner-model mainline
- explicit Q-matrix **R-PFA Model 3** remains the richer challenger
- the selected operational recency weight is `alpha = 0.9`

### 3. Local replication is not currently available

There is no local dataset in this workspace yet.

So the project cannot honestly stay centered on local replication and warm-start right now.

Instead, the useful work is to:

- get the public learner model right
- then test whether it supports sensible **offline next-question policies**

## What the project is focusing on now

Current practical focus:

1. use the **full dataset**
2. use **explicit Q-matrix** learner models
3. use **PFA / R-PFA wins/fails history**
4. compare learner models under a **modular offline policy suite**

That means the repo now distinguishes between:

- the **richest supported heterogeneity model**
- the **best current operational model for question targeting**

## Current answers

### Scientific heterogeneity answer

On the full-data explicit Q-matrix ladder:

- Model 2 beats Model 1
- Model 3 beats Model 2

So the full public data now support:

- baseline heterogeneity
- growth heterogeneity
- stability heterogeneity

### Operational model answer

On the improved explicit Q-matrix PFA / R-PFA branch:

- the current operational default is **Model 2**
- **Model 3** remains the uncertainty/stability challenger
- the operational choice is now judged by **policy-facing replay metrics**, not just held-out fit
- the selected RPFA alpha is `0.9`

So the best current operational learner model is:

- **explicit Q-matrix R-PFA Model 2**

Model 3 is still useful as:

- a richer stability / uncertainty challenger

### Policy answer

The repo no longer relies on one fixed target-`0.7` replay only.

It now compares a small suite of interpretable policy types:

- balanced challenge
- harder challenge
- confidence-building
- failure-aware remediation
- spacing-aware review

The current policy decision rule is:

- keep **Model 2** as the default policy model
- promote **Model 3** only if it clearly improves the offline policy metrics or offers a calibration gain that matters to the chosen rule

Current answer:

- Model 3 does **not** currently clear that bar

## What is paused

These are still in the repo, but they are not the active workstream right now:

- local structural replication
- local warm-start transfer

Reason:

- no local dataset is currently available in this workspace

## What comes next

Until local data arrives, the sensible next work is:

1. keep the R-PFA learner-model branch as mainline
2. compare a small number of offline policy rules:
   - balanced
   - harder challenge
   - easier confidence-building
   - failure-aware remediation
   - spacing-aware review
3. add engagement or frustration **proxies** only from observable data, not invented labels
4. keep Model 3 as a challenger when stability or uncertainty is the reason to use it

## Bottom line

The project is now focused on:

- **public full-data learner modeling**
- **better KC-history representation**
- **offline user-specific question targeting**

The current best operational path is:

- **explicit Q-matrix R-PFA Model 2**
