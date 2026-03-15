# Model 1 Schema Audit: DBE-KT22

## Scope
This note audits the downloaded **DBE-KT22** dataset for the baseline **Model 1** use case:

- binary next-attempt prediction
- already-observed learners only
- chronological forward holdouts
- no same-trial post-outcome predictors in the primary model

Source record:
- Dataset DOI: <https://doi.org/10.26193/6DZWOH>
- Paper: <https://arxiv.org/abs/2208.12651>
- Local fetch script: [src/fetch_dbe_kt22.py](/D:/model1_baseline_agent_bundle/src/fetch_dbe_kt22.py)

## Files present
The dataset-provided workbook `1_DBE_KT22_file_descriptions_100102.xlsx` describes the extracted files as follows.

- `Questions.csv`: question details such as id, text, and difficulty
- `Question_Choices.csv`: answer choices for each question
- `KCs.csv`: knowledge components and descriptions
- `KC_Relationships.csv`: relationships among knowledge components
- `Question_KC_Relationships.csv`: links between questions and knowledge components
- `Transaction.csv`: exercise practice attempts with correctness and auxiliary fields
- `Specialization.csv`: specialization lookup table
- `Student_Specialization.csv`: links students to specializations
- `Sequencer.py`: supplementary sequence-generation script
- `Practice_Sequences.json`: example sequence output

## Candidate primary table
The primary trial table for Model 1 should be built from:

- [Transaction.csv](/D:/model1_baseline_agent_bundle/data/raw/DBE-KT22/extracted/csv/Transaction.csv)
- optionally joined to [Questions.csv](/D:/model1_baseline_agent_bundle/data/raw/DBE-KT22/extracted/csv/Questions.csv) for static item metadata

`Transaction.csv` columns:

- `id`
- `selection_change`
- `start_time`
- `end_time`
- `difficulty_feedback`
- `trust_feedback`
- `answer_state`
- `answer_text`
- `student_id`
- `hint_used`
- `question_id`
- `answer_choice_id`
- `is_hidden`

`Questions.csv` columns:

- `id`
- `question_rich_text`
- `question_title`
- `explanation`
- `hint_text`
- `question_text`
- `difficulty`

## Model 1 mapping
Recommended baseline mapping:

- `student_id` <- `Transaction.student_id`
- `item_id` <- `Transaction.question_id`
- `correct` <- `Transaction.answer_state`
- `timestamp` <- `Transaction.start_time`
- `attempt_id` <- `Transaction.id`

Recommended deterministic ordering:

1. sort by `student_id`
2. sort by parsed `start_time`
3. break timestamp ties with `id`

The dataset authors' supplementary [Sequencer.py](/D:/model1_baseline_agent_bundle/data/raw/DBE-KT22/extracted/script/Sequencer.py) also sorts practice rows by `student_id` and `start_time`, which supports using `start_time` as the primary chronology field.

## Integrity summary
From `Transaction.csv`:

- rows: `161,953`
- unique students: `1,264`
- unique questions: `212`
- `answer_state` values: `true`, `false`
- missing `student_id`: `0`
- missing `question_id`: `0`
- missing `answer_state`: `0`
- missing `start_time`: `0`
- missing `end_time`: `0`
- duplicate transaction ids: `0`
- date range of `start_time`: `2019-08-07T08:05:46.019-07:00` to `2022-05-14T17:26:37.777-07:00`

From `Questions.csv`:

- rows: `212`
- unique question ids: `212`
- `difficulty` values observed: `1`, `2`, `3`

Foreign-key check:

- `Transaction.question_id` missing from `Questions.id`: `0`

Per-student attempt counts before any filtering:

- min: `1`
- median: `148`
- max: `1,171`
- students with at least `10` attempts: `1,140`
- students with at least `20` attempts: `1,074`

Per-question attempt counts:

- min: `466`
- median: `792`
- max: `1,090`

Chronology/timing checks:

- duplicated `student_id` + `start_time` pairs: `489`
- maximum duplicate count for one `student_id` + `start_time` pair: `31`
- negative durations (`end_time < start_time`): `4`
- zero durations: `7`
- positive durations: `161,942`
- maximum observed duration: `499,399` seconds

Interpretation:

- `start_time` is complete and usable for chronology.
- It is not unique within student, so `id` should be used as a deterministic tiebreaker.
- Duration-related anomalies are rare and do not block Model 1 because same-trial timing fields should be excluded from the primary model anyway.

## Outcome audit
Use `Transaction.answer_state` directly as the binary outcome.

Why:

- `answer_state` is already binary and complete.
- Reconstructing correctness from `Question_Choices.is_correct` is not reliable in this dataset.

Observed issues when comparing `Transaction.answer_choice_id` to `Question_Choices.is_correct`:

- `705` rows disagree with `answer_state`
- `1` transaction row uses `answer_choice_id = 619`, which does not appear in `Question_Choices.csv`
- the same `question_id` + `answer_choice_id` pair can appear with both `true` and `false` `answer_state`

Examples:

- for `question_id = 168`, choice `556` appears with both `answer_state = true` and `answer_state = false`
- for `question_id = 209`, choice `725` appears with both `answer_state = true` and `answer_state = false`
- for `question_id = 216`, choice `753` appears with both `answer_state = true` and `answer_state = false`

Conclusion:

- `answer_choice_id` is not sufficient to reconstruct correctness
- `answer_state` should be treated as the authoritative outcome label for Model 1

## Primary-model feature decisions
Keep for the baseline trial table:

- `id` as a deterministic attempt key
- `student_id`
- `question_id`
- `start_time`
- `answer_state`

Optional static item metadata:

- `Questions.difficulty` exists and is complete on a `1` to `3` scale
- do not include it in the default Model 1 because the baseline already includes an item random intercept
- it may be useful later as a sensitivity analysis if handled carefully

Exclude from the primary Model 1 predictors:

- `end_time`
- `selection_change`
- `difficulty_feedback`
- `trust_feedback`
- `answer_text`
- `hint_used`
- `answer_choice_id`

Reason:

- these are same-trial response-process or post-attempt fields
- they are not valid baseline predictors for next-attempt forecasting under the repo guardrails

## Skill / KC structure
There is no single stable `skill_id` column.

Available concept structure:

- [Question_KC_Relationships.csv](/D:/model1_baseline_agent_bundle/data/raw/DBE-KT22/extracted/csv/Question_KC_Relationships.csv)
- [KCs.csv](/D:/model1_baseline_agent_bundle/data/raw/DBE-KT22/extracted/csv/KCs.csv)

Coverage summary:

- questions with at least one KC link: `212`
- KCs used in question mapping: `93`
- total KC rows: `98`
- KC links per question: min `1`, median `2`, max `4`

Implication for Model 1:

- the dataset does not expose a simple one-skill-per-row field
- using skill-level opportunity counts would require a many-to-many design decision
- the default practice feature should remain overall learner opportunity, not skill opportunity

## `is_hidden` audit
`Transaction.is_hidden` is not a minor random flag.

Observed pattern:

- hidden rows: `3,564`
- students with any hidden rows: `21`
- students with all rows hidden: `3`
- students with hidden share at least `50%`: `15`
- hidden rows span all `212` questions

If rows with `is_hidden = true` are excluded:

- retained rows: `158,389`
- retained students: `1,261`
- retained questions: `212`
- dropped rows: `3,564`
- students fully removed: `3`

Recommendation:

- for the primary baseline analysis, exclude rows where `is_hidden = true`
- document this explicitly as a sample restriction because the flag is concentrated in a small set of atypical students and likely reflects special visibility/test conditions rather than ordinary learner practice

## Recommended primary analysis sample
Primary sample recommendation for preprocessing:

1. start from `Transaction.csv`
2. exclude rows with `is_hidden = true`
3. parse `start_time` as timezone-aware datetime
4. map `answer_state` to `correct in {0,1}`
5. sort by `student_id`, `start_time`, `id`
6. create `trial_index_within_student`
7. create `overall_opportunity = trial_index_within_student - 1`
8. create `practice_feature = log1p(overall_opportunity)`
9. optionally join `Questions.difficulty` for reporting only, not the default model

## Open issues to carry into preprocessing
- confirm whether to drop learners with very short histories before splitting; a minimum-history threshold is still to be chosen
- verify whether the rare nonpositive durations should simply be ignored or separately flagged in the assumptions note
- preserve a record of excluded hidden rows and later excluded unseen-item test rows in saved artifacts
- note that the observed timestamp range extends into May 2022 even though the dataset description mentions 2018-2021

## Bottom line
DBE-KT22 is usable for the baseline Model 1 task.

The strongest current decisions are:

- use `Transaction.csv` as the primary attempt table
- use `answer_state` directly as the outcome
- order rows by `student_id`, `start_time`, `id`
- default to overall opportunity counts
- exclude `is_hidden = true` rows from the primary analysis sample
- exclude same-trial process fields from the primary baseline predictors
