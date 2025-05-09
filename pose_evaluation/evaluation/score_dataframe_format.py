import warnings
from pathlib import Path
import pandas as pd
import numpy as np


class ScoreDFCol:
    GLOSS_A_PATH = "GLOSS_A_PATH"
    GLOSS_B_PATH = "GLOSS_B_PATH"
    GLOSS_A = "GLOSS_A"
    GLOSS_B = "GLOSS_B"
    SCORE = "SCORE"
    METRIC = "METRIC"
    SIGNATURE = "SIGNATURE"
    TIME = "TIME"


def load_score_csv(csv_file: Path):
    scores_csv_df = pd.read_csv(
        csv_file,
        dtype={
            ScoreDFCol.METRIC: str,
            ScoreDFCol.GLOSS_A: str,
            ScoreDFCol.GLOSS_B: str,
            ScoreDFCol.SIGNATURE: str,
            ScoreDFCol.GLOSS_A_PATH: str,
            ScoreDFCol.GLOSS_B_PATH: str,
            ScoreDFCol.SCORE: np.float64,
        },
        float_precision="high",
    )

    # Warn and drop rows missing required columns
    required_cols = [ScoreDFCol.METRIC, ScoreDFCol.SIGNATURE, ScoreDFCol.SCORE, ScoreDFCol.GLOSS_A, ScoreDFCol.GLOSS_B]
    missing_rows = scores_csv_df[scores_csv_df[required_cols].isnull().any(axis=1)]
    if not missing_rows.empty:
        warnings.warn(f"{len(missing_rows)} malformed row(s) dropped from {csv_file}: missing required fields")
        print(missing_rows)
        scores_csv_df = scores_csv_df.drop(missing_rows.index)

    scores_csv_df[ScoreDFCol.SIGNATURE] = scores_csv_df[ScoreDFCol.SIGNATURE].apply(
        lambda x: x.split("=")[0].strip() if "=" in x else x.strip()
    )
    assert (
        len(scores_csv_df[ScoreDFCol.METRIC].unique()) == 1
    ), f"csv_file {csv_file} has multiple metric names: {scores_csv_df[ScoreDFCol.METRIC].unique()} "
    assert (
        len(scores_csv_df[ScoreDFCol.SIGNATURE].unique()) == 1
    ), f"More than one signature! {csv_file}, {scores_csv_df[ScoreDFCol.SIGNATURE].unique()}"
    return scores_csv_df
