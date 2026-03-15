# Project plan: Phase 1 public-data development + Phase 2 warm-start transfer

## Core aim
Build a small, staged modelling programme that starts with public learner-response data and then transfers the chosen model family to a local dataset so new local students do not start from scratch.

This project is **not** a full personalised learning system.
It is a forecasting-and-transfer project.

---

## Phase 1: public-data model development

### Data source
Use a public trial-level learner-response dataset with preserved chronology, such as DBE-KT22.

### Analysis table
Create one row per student item attempt with at least:
- `student_id`
- `item_id`
- `correct`
- `timestamp` or valid within-student order
- `trial_index_within_student`
- `overall_opportunity`
- `practice_feature = log1p(overall_opportunity)`

Keep auxiliary fields for later notes or descriptive analysis, but do not use them in Models 1 and 2 by default:
- `gt_difficulty`
- `difficulty_feedback`
- `answer_confidence`
- `hint_used`
- `time_taken`
- `num_ans_changes`

### Evaluation tracks
Use both:

**Track A: seen-learner forward prediction**
- split within learner by time
- fit on early attempts, test on later attempts

**Track B: unseen-public-student initialization**
- split by `student_id`
- train on public train students
- predict sequentially for public test students

### Model 1
Hierarchical logistic model with:
- learner intercepts
- item intercepts
- one shared practice term

Question answered:
- can a standard hierarchical logistic model give sensible next-attempt probabilities?

### Model 2
Model 1 plus learner-specific practice slopes.

Question answered:
- do learners differ not just in starting level, but in rate of change?

### Model 3
Model 2 plus a latent learner-state deviation over time with learner-specific latent volatility.

Question answered:
- after accounting for level and growth, does learner-specific instability improve probabilistic forecasting?

### Phase 1 primary metrics
- log loss / mean log predictive density
- Brier score
- calibration intercept and slope
- calibration curve / reliability plot

Secondary metrics:
- accuracy
- AUC

### Phase 1 decision rule
- If Model 2 does not improve meaningfully over Model 1, stop and simplify.
- If Model 3 does not improve meaningfully over Model 2 on primary probabilistic metrics and calibration, do not carry Model 3 into Phase 2 by default.
- Carry forward only the most complex model that clearly earns its keep.

### Phase 1 outputs
- cleaned public analysis table
- saved public split files
- fitted Models 1, 2, and 3
- model-comparison table
- calibration figures
- concise methods and assumptions note
- frozen public-informed priors / hyperparameters for the chosen model family

---

## Phase 2: public-to-local warm-start transfer

### Goal
Use the chosen Phase 1 model family on a local dataset so that new local students start from a public-informed prior rather than from a blank slate.

### What transfers
Transfer:
- the chosen model structure
- the public-learned hyperparameters / priors
- the typical spread of student intercepts
- the typical spread of student learning slopes
- if Model 3 is chosen, the typical spread of learner volatility

Do not transfer exact public student coefficients to local students.

### Harmonize local data
Convert the local dataset to the same schema as Phase 1:
- `student_id`
- `item_id`
- `correct`
- `timestamp` or valid order
- `trial_index_within_student`
- `overall_opportunity`
- `practice_feature`

### Local split design
Create:
- a **local calibration subset** to estimate local offsets and local item effects
- an **untouched local external-test set** of students for the main evaluation

If the local sample is small, use repeated student-wise cross-validation.

### Two local fits to compare
1. **Weak-prior local fit**
   - same chosen model family
   - broad priors
   - close to starting from scratch

2. **Public-informed warm-start fit**
   - same chosen model family
   - public-informed priors for student-level effects
   - local offset and local item effects estimated locally

### Same-item vs new-item rule
- If local items are the same as public items, direct item transfer is more plausible.
- If local items differ, transfer the student-side priors and estimate local item effects rather than assuming public item coefficients carry over.

### Phase 2 primary question
Does the public-informed warm-start fit improve early prediction for held-out local students compared with the weak-prior local fit?

### Phase 2 evaluation focus
Evaluate on untouched local students only.

Report overall performance, plus early-attempt windows such as:
- attempts 1-5
- attempts 6-10
- attempts 11-20

Metrics:
- log loss / mean log predictive density
- Brier score
- calibration intercept and slope
- calibration curve / reliability plot

Secondary metrics:
- accuracy
- AUC

### Derived student summaries
Do not train separate class labels.
Instead derive summaries from the fitted student parameters:
- **current proficiency** = predicted success on a reference medium-difficulty item
- **learning rate** = learner-specific practice slope
- **stability** = learner-specific volatility, if Model 3 is used
- **uncertainty** = posterior width around these summaries

### Phase 2 outputs
- harmonized local trial table
- saved student-wise local split file(s)
- weak-prior local fit
- public-informed warm-start fit
- transfer comparison table
- early-attempt calibration figures
- short transfer note describing what was carried from public to local data

---

## Minimum publishable version
If scope has to stay especially tight:
- complete Phase 1 with Models 1 and 2 first
- only add Model 3 if it clearly earns its keep
- in Phase 2, compare weak-prior vs public-informed transfer using the chosen model family

This keeps the project small while preserving the larger trajectory.
