from __future__ import annotations

import bambi as bmb
import pandas as pd


MODEL2_FORMULA = "correct ~ practice_feature + (1 + practice_feature|student_id) + (1|item_id)"


def build_model(df: pd.DataFrame) -> bmb.Model:
    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1.5),
        "practice_feature": bmb.Prior("Normal", mu=0, sigma=1.0),
        "1|student_id": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1.0)),
        "practice_feature|student_id": bmb.Prior(
            "Normal",
            mu=0,
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
        ),
        "1|item_id": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1.0)),
    }
    return bmb.Model(
        MODEL2_FORMULA,
        df,
        family="bernoulli",
        priors=priors,
        auto_scale=False,
    )
