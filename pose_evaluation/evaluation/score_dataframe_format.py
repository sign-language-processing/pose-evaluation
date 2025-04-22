from pathlib import Path
import pandas as pd


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
        index_col=0,
        dtype={
            ScoreDFCol.METRIC: str,
            ScoreDFCol.GLOSS_A: str,
            ScoreDFCol.GLOSS_B: str,
            ScoreDFCol.SIGNATURE: str,
            ScoreDFCol.GLOSS_A_PATH: str,
            ScoreDFCol.GLOSS_B_PATH: str,
        },
        float_precision="high",
    )

    scores_csv_df[ScoreDFCol.SIGNATURE] = scores_csv_df[ScoreDFCol.SIGNATURE].apply(
        lambda x: x.split("=")[0].strip() if "=" in x else x.strip()
    )
    assert len(scores_csv_df[ScoreDFCol.METRIC].unique()) == 1
    assert (
        len(scores_csv_df[ScoreDFCol.SIGNATURE].unique()) == 1
    ), f"{csv_file}, {scores_csv_df[ScoreDFCol.SIGNATURE].unique()}"
    return scores_csv_df
