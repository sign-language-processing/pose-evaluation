from collections import defaultdict
import time
from typing import List, Optional, Annotated
from pathlib import Path

from tqdm import tqdm
from pose_format import Pose
import pandas as pd
import numpy as np
import random
import typer

from pose_evaluation.evaluation.create_metrics import get_metrics
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import STANDARDIZED_GLOSS_COL_NAME

app = typer.Typer()


def combine_dataset_dfs(dataset_df_files: List[Path], splits: List[str], filter_en_vocab: bool = False):
    dfs = []
    for file_path in dataset_df_files:
        if file_path.exists():
            typer.echo(f"✅ Found: {file_path}")
            df = pd.read_csv(file_path)
            df = df[df["SPLIT"].isin(splits)]
            df["DATASET"] = file_path.stem
            typer.echo(f"Loaded {len(df)} rows from splits: {splits}")
            dfs.append(df)
        else:
            typer.echo(f"❌ Missing: {file_path}")

    df = pd.concat(dfs)
    df = df.drop(columns=["VIDEO_FILE_PATH"])

    df = df.dropna()

    if filter_en_vocab:
        df = df[~df["GLOSS"].str.contains("EN:", na=False)]

    return df


def load_pose_files(df, path_col="POSE_FILE_PATH"):
    paths = df[path_col].unique()
    return {path: Pose.read(Path(path).read_bytes()) for path in paths}


def run_metrics(
    df: pd.DataFrame,
    out_path: Path,
    metrics: List[DistanceMetric],
    gloss_count: Optional[int] = 5,
    out_gloss_multiplier=4,
    shuffle_metrics=True,
):

    gloss_vocabulary = df[STANDARDIZED_GLOSS_COL_NAME].unique().tolist()

    if gloss_count:
        gloss_vocabulary = gloss_vocabulary[:gloss_count]

    if shuffle_metrics:
        random.shuffle(metrics)

    for g_index, gloss in enumerate(gloss_vocabulary):
        in_gloss_df = df[df[STANDARDIZED_GLOSS_COL_NAME] == gloss]
        out_gloss_df = df[df[STANDARDIZED_GLOSS_COL_NAME] != gloss]

        other_class_count = len(in_gloss_df) * out_gloss_multiplier
        out_gloss_df = out_gloss_df.sample(n=other_class_count, random_state=42)

        pose_data = load_pose_files(in_gloss_df)
        pose_data.update(load_pose_files(out_gloss_df))

        typer.echo(
            f"For gloss {gloss} We have {len(in_gloss_df)} in, {len(out_gloss_df)} out, and we have loaded {len(pose_data.keys())} poses total"
        )

        for i, metric in enumerate(metrics):
            typer.echo("*" * 60)
            typer.echo(f"Gloss #{g_index}/{len(gloss_vocabulary)}: {gloss} ")
            typer.echo(f"Metric #{i}/{len(metrics)}: {metric.name}")
            typer.echo(f"Metric #{i}/{len(metrics)} Signature: {metric.get_signature().format()}")
            results = defaultdict(list)
            results_path = out_path / f"{gloss}_{metric.name}_outgloss_{out_gloss_multiplier}x_score_results.csv"
            if results_path.exists():
                print(f"Results for {results_path} already exist. Skipping!")
                continue

            for _, hyp_row in tqdm(in_gloss_df.iterrows()):
                hyp_path = hyp_row["POSE_FILE_PATH"]
                hyp_pose = pose_data[hyp_path].copy()

                for _, ref_row in out_gloss_df.iterrows():
                    ref_path = ref_row["POSE_FILE_PATH"]
                    ref_pose = pose_data[ref_path].copy()

                    start_time = time.perf_counter()
                    score = metric.score_with_signature(hyp_pose, ref_pose)

                    if score is None or score.score is None or np.isnan(score.score):

                        typer.echo(f"⚠️ Got invalid score {score.score} for {hyp_path} vs {ref_path}")
                        if score.score is None:
                            print(f"None score")
                        elif np.isnan(score.score):
                            print("NaN score")

                        print(metric.get_signature().format())
                        print(metric.pose_preprocessors)
                        print(hyp_pose.body.data.shape)
                        print(ref_pose.body.data.shape)

                        exit()

                        # print(hyp_pose.body.data)

                        # print(ref_pose.body.data)

                        # exit()

                    end_time = time.perf_counter()
                    results["METRIC"].append(metric.name)
                    results["SCORE"].append(score.score)
                    results["GLOSS_A"].append(hyp_row[STANDARDIZED_GLOSS_COL_NAME])
                    results["GLOSS_B"].append(ref_row[STANDARDIZED_GLOSS_COL_NAME])
                    results["SIGNATURE"].append(score.format())
                    results["GLOSS_A_PATH"].append(hyp_path)
                    results["GLOSS_B_PATH"].append(ref_path)
                    results["TIME"].append(end_time - start_time)

            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(results_path)
            print(f"Wrote {len(results_df)} scores to {results_path}")
            print("\n")


@app.command()
def main(
    dataset_df_files: Annotated[
        List[Path], typer.Argument(help="List of dataset csvs, with columns POSE_FILE_PATH, SPLIT, and VIDEO_ID")
    ],
    splits: Optional[str] = typer.Option(
        None, help="Comma-separated list of splits to process (e.g., 'train,val,test'), default is 'test' only"
    ),
    gloss_count: Optional[int] = typer.Option(5, help="Number of glosses to select"),
    out_gloss_multiplier: Optional[int] = typer.Option(4, help="Number of out-of-gloss items to sample"),
    filter_en_vocab: Annotated[
        bool,
        typer.Option(
            help="Whether to filter out 'gloss' values starting with 'EN:', which are actually English, not ASL."
        ),
    ] = False,
    out: Path = typer.Option("metric_results", exists=False, file_okay=False, help="Folder to save the results"),
):
    """
    Accept a list of dataset DataFrame file paths.
    """
    if splits is None:
        splits = ["test"]
    else:
        splits = [s.strip() for s in splits.split(",") if s.strip()]

    df = combine_dataset_dfs(dataset_df_files=dataset_df_files, splits=splits, filter_en_vocab=filter_en_vocab)

    typer.echo("\nCOMBINED DATAFRAME")
    typer.echo(df.info())
    typer.echo(df.describe())

    typer.echo(f"* Vocabulary: {len(df['GLOSS'].unique())}")
    typer.echo(f"* Pose Files: {len(df['POSE_FILE_PATH'].unique())}")

    metrics = get_metrics()
    typer.echo(f"Metrics: {[m.name for m in metrics]}")
    typer.echo(f"We have a total of {len(metrics)} metrics")

    typer.echo(f"Saving results to {out}")
    out.mkdir(parents=True, exist_ok=True)
    run_metrics(df, out_path=out, metrics=metrics, gloss_count=gloss_count, out_gloss_multiplier=out_gloss_multiplier)


if __name__ == "__main__":
    app()
