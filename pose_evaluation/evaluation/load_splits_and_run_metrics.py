import concurrent.futures
from collections import defaultdict
import time
from typing import List, Optional, Annotated
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from functools import partial
import random

from tqdm import tqdm
from pose_format import Pose
import pandas as pd
import numpy as np
import random
import typer

from pose_evaluation.evaluation.create_metrics import get_metrics
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import DatasetDFCol
from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol

app = typer.Typer()


def combine_dataset_dfs(dataset_df_files: List[Path], splits: List[str], filter_en_vocab: bool = False):
    dfs = []
    for file_path in dataset_df_files:
        if file_path.exists():
            typer.echo(f"✅ Found: {file_path}")
            df = pd.read_csv(
                file_path,
                dtype={
                    DatasetDFCol.GLOSS: str,
                    DatasetDFCol.SPLIT: str,
                    DatasetDFCol.VIDEO_ID: str,
                    DatasetDFCol.POSE_FILE_PATH: str,
                },
            )

            df = df[df[DatasetDFCol.SPLIT].isin(splits)]
            df[DatasetDFCol.DATASET] = file_path.stem
            typer.echo(f"Loaded {len(df)} rows from splits: {splits}")
            dfs.append(df)
        else:
            typer.echo(f"❌ Missing: {file_path}")

    df = pd.concat(dfs)
    df = df.drop(columns=[DatasetDFCol.VIDEO_FILE_PATH])

    df = df.dropna()

    if filter_en_vocab:
        df = df[~df[DatasetDFCol.GLOSS].str.contains("EN:", na=False)]

    return df


def load_pose_files(df, path_col=DatasetDFCol.POSE_FILE_PATH, progress=False):
    paths = df[path_col].unique()
    return {
        path: Pose.read(Path(path).read_bytes()) for path in tqdm(paths, desc="Loading poses", disable=not progress)
    }


def run_metrics_in_out_trials(
    df: pd.DataFrame,
    out_path: Path,
    metrics: List[DistanceMetric],
    gloss_count: Optional[int] = 5,
    out_gloss_multiplier=4,
    shuffle_metrics=True,
    metric_count: Optional[int] = 10,
    additional_glosses: Optional[List[str]] = None,
    shuffle_query_glosses=True,
):

    query_gloss_vocabulary = df[DatasetDFCol.GLOSS].unique().tolist()

    if gloss_count:
        query_gloss_vocabulary = query_gloss_vocabulary[:gloss_count]

    if additional_glosses:
        # prepend them
        print(f"Prepending additional query glosses {additional_glosses}")
        combined = additional_glosses + query_gloss_vocabulary
        # Use dict.fromkeys() to remove duplicates while keeping order
        query_gloss_vocabulary = list(dict.fromkeys(combined))
        print(f"Query gloss vocabulary is now length {len(query_gloss_vocabulary)}")
        print(f"Query gloss vocabulary: {query_gloss_vocabulary}")

    if shuffle_query_glosses:
        random.shuffle(query_gloss_vocabulary)

    if shuffle_metrics:
        random.shuffle(metrics)

    if metric_count is not None:
        typer.echo(f"Selecting {metric_count} of {len(metrics)} metrics")
        metrics = metrics[:metric_count]

    gloss_dfs_folder = out_path / "gloss_dfs"
    gloss_dfs_folder.mkdir(exist_ok=True, parents=True)

    scores_path = out_path / "scores"
    scores_path.mkdir(exist_ok=True, parents=True)

    for g_index, gloss in enumerate(
        tqdm(
            query_gloss_vocabulary,
            total=len(query_gloss_vocabulary),
            desc=f"Running evaluations for all {len(query_gloss_vocabulary)} glosses",
        )
    ):

        in_gloss_df_path = gloss_dfs_folder / f"{gloss}_in.csv"
        if in_gloss_df_path.is_file():
            print(f"Reading in-gloss df from {in_gloss_df_path}")
            in_gloss_df = pd.read_csv(in_gloss_df_path, index_col=0, dtype={DatasetDFCol.GLOSS: str})
        else:
            print(f"Writing out-gloss df to {in_gloss_df_path}")
            in_gloss_df = df[df[DatasetDFCol.GLOSS] == gloss]
            in_gloss_df.to_csv(in_gloss_df_path)

        out_gloss_df_path = gloss_dfs_folder / f"{gloss}_out.csv"
        if out_gloss_df_path.is_file():
            print(f"Reading in out-gloss-df from {out_gloss_df_path}")
            out_gloss_df = pd.read_csv(out_gloss_df_path, index_col=0, dtype={DatasetDFCol.GLOSS: str})
        else:
            print(f"Writing out-gloss df to {out_gloss_df_path}")
            out_gloss_df = df[df[DatasetDFCol.GLOSS] != gloss]
            other_class_count = len(in_gloss_df) * out_gloss_multiplier
            out_gloss_df = out_gloss_df.sample(n=other_class_count, random_state=42)
            out_gloss_df.to_csv(out_gloss_df_path)

        ref_df = pd.concat([in_gloss_df, out_gloss_df], ignore_index=True)

        pose_data = load_pose_files(in_gloss_df)
        pose_data.update(load_pose_files(out_gloss_df))

        typer.echo(
            f"For gloss {gloss} We have {len(in_gloss_df)} in, {len(out_gloss_df)} out, and we have loaded {len(pose_data.keys())} poses total"
        )

        for i, metric in enumerate(metrics):
            typer.echo("*" * 60)
            typer.echo(f"Gloss #{g_index}/{len(query_gloss_vocabulary)}: {gloss}")
            typer.echo(f"Metric #{i}/{len(metrics)}: {metric.name}")
            typer.echo(f"Metric #{i}/{len(metrics)} Signature: {metric.get_signature().format()}")
            typer.echo(f"Testing with {len(in_gloss_df)} in-gloss files, and {len(out_gloss_df)} out-gloss")

            results = defaultdict(list)
            results_path = scores_path / f"{gloss}_{metric.name}_outgloss_{out_gloss_multiplier}x_score_results.csv"
            if results_path.exists():
                print(f"Results for {results_path} already exist. Skipping!")
                continue

            for _, hyp_row in tqdm(
                in_gloss_df.iterrows(),
                total=len(in_gloss_df),
                desc=f"Calculating all {len(in_gloss_df)*len(ref_df)} distances for gloss #{g_index}/{len(query_gloss_vocabulary)}({gloss}) against all {len(ref_df)} references",
            ):
                hyp_path = hyp_row[DatasetDFCol.POSE_FILE_PATH]
                hyp_pose = pose_data[hyp_path].copy()

                for _, ref_row in ref_df.iterrows():
                    ref_path = ref_row[DatasetDFCol.POSE_FILE_PATH]
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

                    end_time = time.perf_counter()
                    results[ScoreDFCol.METRIC].append(metric.name)
                    results[ScoreDFCol.SCORE].append(score.score)
                    results[ScoreDFCol.GLOSS_A].append(hyp_row[DatasetDFCol.GLOSS])
                    results[ScoreDFCol.GLOSS_B].append(ref_row[DatasetDFCol.GLOSS])
                    results[ScoreDFCol.SIGNATURE].append(metric.get_signature().format())
                    results[ScoreDFCol.GLOSS_A_PATH].append(hyp_path)
                    results[ScoreDFCol.GLOSS_B_PATH].append(ref_path)
                    results[ScoreDFCol.TIME].append(end_time - start_time)

            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(results_path)
            print(f"Wrote {len(results_df)} scores to {results_path}")
            print("\n")


def compute_batch_pairs(hyp_chunk, ref_chunk, metric, signature):
    hyp_pose_data = load_pose_files(hyp_chunk)
    ref_pose_data = load_pose_files(ref_chunk)
    batch_results = []

    for _, hyp_row in hyp_chunk.iterrows():
        hyp_path = hyp_row[DatasetDFCol.POSE_FILE_PATH]
        hyp_pose = hyp_pose_data[hyp_path].copy()

        for _, ref_row in ref_chunk.iterrows():
            ref_path = ref_row[DatasetDFCol.POSE_FILE_PATH]
            ref_pose = ref_pose_data[ref_path].copy()

            start_time = time.perf_counter()
            score = metric.score_with_signature(hyp_pose, ref_pose)
            end_time = time.perf_counter()

            score_val = score.score if score and score.score is not None else np.nan

            batch_results.append(
                {
                    ScoreDFCol.METRIC: metric.name,
                    ScoreDFCol.SCORE: score_val,
                    ScoreDFCol.GLOSS_A: hyp_row[DatasetDFCol.GLOSS],
                    ScoreDFCol.GLOSS_B: ref_row[DatasetDFCol.GLOSS],
                    ScoreDFCol.SIGNATURE: signature,
                    ScoreDFCol.GLOSS_A_PATH: hyp_path,
                    ScoreDFCol.GLOSS_B_PATH: ref_path,
                    ScoreDFCol.TIME: end_time - start_time,
                }
            )
    
    return batch_results


def run_metrics_full_distance_matrix_batched_parallel(
    df: pd.DataFrame,
    out_path: Path,
    metrics: list,
    batch_size: int = 500,
    max_workers: int = 4,
):
    print(
        f"Calculating full distance matrix on {len(df)} poses from {len(df[DatasetDFCol.DATASET].unique())} datasets, for {len(metrics)} metrics"
    )
    print(f"Batch size {batch_size}, max workers {max_workers}")
    print(f"Splits: {df[DatasetDFCol.SPLIT].unique()}")
    print(f"Results will be saved to {out_path}")

    how_many = 1000
    print(f"TODO REMOVE THIS: HARDCODED TAKING FIRST {how_many}")
    df = df.head(how_many)

    scores_path = out_path / "scores"
    scores_path.mkdir(exist_ok=True, parents=True)

    dataset_names = "+".join(df[DatasetDFCol.DATASET].unique().tolist())
    split_names = "+".join(df[DatasetDFCol.SPLIT].unique().tolist())

    for i, metric in tqdm(enumerate(metrics), total=len(metrics), desc="Iterating over metrics"):
        typer.echo("*" * 60)
        typer.echo(f"Metric #{i + 1}/{len(metrics)}: {metric.name}")
        signature = metric.get_signature().format()
        typer.echo(f"Metric Signature: {signature}")
        typer.echo(f"Batch Size: {batch_size}, so that's {batch_size*batch_size} distances per.")

        metric_results_path = scores_path / f"batches_{metric.name}_{dataset_names}_{split_names}"
        metric_results_path.mkdir(parents=True, exist_ok=True)

        futures = {}
        batch_id = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for hyp_start in tqdm(range(0, len(df), batch_size), desc=f"Hyp batching for Metric {i}"):
                hyp_chunk = df.iloc[hyp_start : hyp_start + batch_size]

                for ref_start in range(0, len(df), batch_size):
                    ref_chunk = df.iloc[ref_start : ref_start + batch_size]
                    batch_filename = metric_results_path / f"batch_{batch_id:06d}_hyp{hyp_start}_ref{ref_start}.parquet"

                    if batch_filename.exists():
                        typer.echo(f"✅ Skipping batch {batch_id} (already exists)")
                    else:
                        future = executor.submit(compute_batch_pairs, hyp_chunk, ref_chunk, metric, signature)
                        futures[future] = (batch_id, batch_filename)

                    batch_id += 1

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Saving batches"):
            batch_id, batch_filename = futures[future]
            result_rows = future.result()
            if result_rows:
                pd.DataFrame(result_rows).to_parquet(batch_filename, index=False, compression="snappy")
                typer.echo(f"💾 Saved batch {batch_id} to {batch_filename.name}")

        # Final merge
        typer.echo("🔄 Merging all batch files...")
        all_batch_files = sorted(metric_results_path.glob("batch_*.parquet"))
        all_dfs = [pd.read_parquet(f) for f in all_batch_files]
        merged_df = pd.concat(all_dfs, ignore_index=True)

        final_path = scores_path / f"full_matrix_{metric.name}_on_{dataset_names}_{split_names}_score_results.parquet"
        merged_df.to_parquet(final_path, index=False, compression="snappy")
        typer.echo(f"✅ Final results written to {final_path}\n")


@app.command()
def main(
    dataset_df_files: Annotated[
        List[Path], typer.Argument(help="List of dataset csvs, with columns POSE_FILE_PATH, SPLIT, and VIDEO_ID")
    ],
    splits: Optional[str] = typer.Option(
        None, help="Comma-separated list of splits to process (e.g., 'train,val,test'), default is 'test' only"
    ),
    full: Optional[bool] = typer.Option(
        False, help="Whether to run FULL distance matrix with the specified dataset dfs"
    ),
    gloss_count: Optional[int] = typer.Option(83, help="Number of glosses to select"),
    additional_glosses: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of additional glosses to use for testing in addition to the ones selected by gloss_count",
    ),
    out_gloss_multiplier: Optional[int] = typer.Option(4, help="Number of out-of-gloss items to sample"),
    metric_count: Optional[int] = typer.Option(None, help="Number of metrics to sample"),
    filter_en_vocab: Annotated[
        bool,
        typer.Option(
            help="Whether to filter out 'gloss' values starting with 'EN:', which are actually English, not ASL."
        ),
    ] = False,
    out: Path = typer.Option(
        "metric_results_full_matrix", exists=False, file_okay=False, help="Folder to save the results"
    ),
):
    """
    Accept a list of dataset DataFrame file paths.
    """
    if splits is None:
        splits = ["test"]
    else:
        splits = [s.strip() for s in splits.split(",") if s.strip()]

    if additional_glosses is not None:
        additional_glosses = additional_glosses.split(",")

    df = combine_dataset_dfs(dataset_df_files=dataset_df_files, splits=splits, filter_en_vocab=filter_en_vocab)

    typer.echo("\nCOMBINED DATAFRAME")
    typer.echo(df.info())
    typer.echo(df.describe())

    typer.echo(f"* Vocabulary: {len(df[DatasetDFCol.GLOSS].unique())}")
    typer.echo(f"* Pose Files: {len(df[DatasetDFCol.POSE_FILE_PATH].unique())}")

    metrics = get_metrics()
    # typer.echo(f"Metrics: {[m.name for m in metrics]}")
    typer.echo(f"We have a total of {len(metrics)} metrics")

    # untrimmed on the 48 cpu machine please
    metrics_to_use = [
        "untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        # "untrimmed_unnormalized_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        # "untrimmed_normalizedbyshoulders_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        # "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        # "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        # "untrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        # "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        # "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        # "startendtrimmed_unnormalized_hands_defaultdist0.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        # "startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        # "Return4Metric",
    ]
    # Get a set of target names for efficient lookup
    metrics_to_use_set = set(metrics_to_use)

    # Filter metrics based on .name
    filtered_metrics = [m for m in metrics if m.name in metrics_to_use_set]

    # Find which metrics_to_use were unmatched
    metric_names = {m.name for m in metrics}
    unmatched_metrics = [m for m in metrics_to_use if m not in metric_names]

    print("Filtered metrics:", [m.name for m in filtered_metrics])
    print("Unmatched metrics_to_use:", unmatched_metrics)
    metrics = filtered_metrics
    random.shuffle(metrics)

    typer.echo(f"Saving results to {out}")
    out.mkdir(parents=True, exist_ok=True)
    if full:
        run_metrics_full_distance_matrix_batched_parallel(
            # df, out_path=out, metrics=metrics, batch_size=100, max_workers=10 # 12-cpu machine
            # df, out_path=out, metrics=metrics, batch_size=20, max_workers=30 # 12-cpu machine
            df,
            out_path=out,
            metrics=metrics,
            batch_size=8,
            max_workers=30,
        )

    else:
        run_metrics_in_out_trials(
            df,
            out_path=out,
            metrics=metrics,
            gloss_count=gloss_count,
            out_gloss_multiplier=out_gloss_multiplier,
            shuffle_metrics=True,
            metric_count=metric_count,
            additional_glosses=additional_glosses,
        )


if __name__ == "__main__":
    app()
# with 30 glosses total including really big glosses, BLACK, HOME, HUNGRY, REFRIGERATOR, UNCLE
# results in ['RUSSIA', 'BRAG', 'HOUSE', 'HOME', 'WORM', 'REFRIGERATOR', 'BLACK', 'SUMMER', 'SICK', 'REALSICK', 'WEATHER', 'MEETING', 'COLD', 'WINTER', 'THANKSGIVING', 'THANKYOU', 'HUNGRY', 'FULL', 'LIBRARY', 'CELERY', 'CANDY1', 'CASTLE2', 'GOVERNMENT', 'OPINION1', 'PJS', 'LIVE2', 'WORKSHOP', 'UNCLE', 'EASTER', 'BAG2']
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 12 dataset_dfs/*.csv --additional-glosses "RUSSIA,BRAG,HOUSE,HOME,WORM,REFRIGERATOR,BLACK,SUMMER,SICK,REALSICK,WEATHER,MEETING,COLD,WINTER,THANKSGIVING,THANKYOU,HUNGRY,FULL" 2>&1|tee out/$(date +%s).txt

# Add more glosses, 100 total.
# Results in this list: ['ACCENT', 'ADULT', 'AIRPLANE', 'APPEAR', 'BAG2', 'BANANA2', 'BEAK', 'BERRY', 'BIG', 'BINOCULARS', 'BLACK', 'BRAG', 'BRAINWASH', 'CAFETERIA', 'CALM', 'CANDY1', 'CASTLE2', 'CELERY', 'CHEW1', 'COLD', 'CONVINCE2', 'COUNSELOR', 'DART', 'DEAF2', 'DEAFSCHOOL', 'DECIDE2', 'DIP3', 'DOLPHIN2', 'DRINK2', 'DRIP', 'DRUG', 'EACH', 'EARN', 'EASTER', 'ERASE1', 'EVERYTHING', 'FINGERSPELL', 'FISHING2', 'FORK4', 'FULL', 'GOTHROUGH', 'GOVERNMENT', 'HIDE', 'HOME', 'HOUSE', 'HUNGRY', 'HURRY', 'KNITTING3', 'LEAF1', 'LEND', 'LIBRARY', 'LIVE2', 'MACHINE', 'MAIL1', 'MEETING', 'NECKLACE4', 'NEWSTOME', 'OPINION1', 'ORGANIZATION', 'PAIR', 'PEPSI', 'PERFUME1', 'PIG', 'PILL', 'PIPE2', 'PJS', 'REALSICK', 'RECORDING', 'REFRIGERATOR', 'REPLACE', 'RESTAURANT', 'ROCKINGCHAIR1', 'RUIN', 'RUSSIA', 'SCREWDRIVER3', 'SENATE', 'SHAME', 'SHARK2', 'SHAVE5', 'SICK', 'SNOWSUIT', 'SPECIALIST', 'STADIUM', 'SUMMER', 'TAKEOFF1', 'THANKSGIVING', 'THANKYOU', 'TIE1', 'TOP', 'TOSS', 'TURBAN', 'UNCLE', 'VAMPIRE', 'WASHDISHES', 'WEAR', 'WEATHER', 'WINTER', 'WORKSHOP', 'WORM', 'YESTERDAY']
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 83 dataset_dfs/*.csv --additional-glosses "RUSSIA,BRAG,HOUSE,HOME,WORM,REFRIGERATOR,BLACK,SUMMER,SICK,REALSICK,WEATHER,MEETING,COLD,WINTER,THANKSGIVING,THANKYOU,HUNGRY,FULL" 2>&1|tee out/$(date +%s).txt


# with all the known-similar glosses included:
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 83 dataset_dfs/*.csv --additional-glosses "RUSSIA,BRAG,HOUSE,HOME,WORM,REFRIGERATOR,BLACK,SUMMER,SICK,REALSICK,WEATHER,MEETING,COLD,WINTER,THANKSGIVING,THANKYOU,HUNGRY,FULL" 2>&1|tee out/$(date +%s).txt


# stat -c "%y" metric_results/scores/* | cut -d':' -f1 | sort | uniq -c # get the timestamps/hour
# cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/count_files_by_hour.py
