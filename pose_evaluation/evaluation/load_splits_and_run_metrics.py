import concurrent.futures
from collections import defaultdict
from itertools import product
import math
import time
from typing import List, Optional, Annotated
from pathlib import Path
import random

from tqdm import tqdm
from pose_format import Pose
import pandas as pd
import numpy as np

import typer

from pose_evaluation.evaluation.create_metrics import get_metrics, get_embedding_metrics
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import DatasetDFCol
from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol
from pose_evaluation.metrics.embedding_distance_metric import EmbeddingDistanceMetric

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
            typer.echo(f"Loaded {len(df)} rows")
            df = df[df[DatasetDFCol.SPLIT].isin(splits)]
            df[DatasetDFCol.DATASET] = file_path.stem
            typer.echo(f"Loaded {len(df)} rows from splits: {splits}")
            typer.echo(f"There are {len(df[DatasetDFCol.GLOSS].unique())} unique glosses")
            dfs.append(df)
        else:
            typer.echo(f"❌ Missing: {file_path}")
    df = pd.concat(dfs)
    if DatasetDFCol.VIDEO_FILE_PATH in df.columns:
        df = df.drop(columns=[DatasetDFCol.VIDEO_FILE_PATH])

    df = df.dropna()

    if filter_en_vocab:
        df = df[~df[DatasetDFCol.GLOSS].str.contains("EN:", na=False)]
    typer.echo(f"Loaded {len(df)} rows total, from {len(dataset_df_files)} files")
    return df


def load_pose_files(df, path_col=DatasetDFCol.POSE_FILE_PATH, progress=False):
    paths = df[path_col].unique()
    return {
        path: Pose.read(Path(path).read_bytes()) for path in tqdm(paths, desc="Loading poses", disable=not progress)
    }


def load_embedding(file_path: Path) -> np.ndarray:
    """
    Load a SignCLIP embedding from a .npy file, ensuring it has the correct shape.

    Args:
        file_path (Path): Path to the .npy file.

    Returns:
        np.ndarray: The embedding with shape (768,).
    """
    embedding = np.load(file_path)
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding[0]  # Reduce shape from (1, 768) to (768,)
    return embedding


def load_embedding_for_pose(df, model, pose_file_path):
    row = df[(df[DatasetDFCol.POSE_FILE_PATH] == pose_file_path) & (df[DatasetDFCol.EMBEDDING_MODEL] == model)]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one match for {pose_file_path}, model{model}: found {len(row)} rows.")
    embedding_path = row.iloc[0][DatasetDFCol.EMBEDDING_FILE_PATH]
    return load_embedding(embedding_path)


def run_metrics_in_out_trials(
    df: pd.DataFrame,
    out_path: Path,
    metrics: List["DistanceMetric"],
    gloss_count: Optional[int] = 5,
    out_gloss_multiplier=4,
    shuffle_metrics=True,
    metric_count: Optional[int] = 10,
    additional_glosses: Optional[List[str]] = None,
    shuffle_query_glosses=False,
    skip_glosses_with_more_than_this_many: Optional[int] = None,
    gloss_dfs_folder: Optional[Path] = None,
):
    query_gloss_vocabulary = df[DatasetDFCol.GLOSS].unique().tolist()
    typer.echo(f"The number of possible unique glosses to pick from is {len(query_gloss_vocabulary)}")

    if gloss_dfs_folder is None:
        gloss_dfs_folder = out_path.parent / "gloss_dfs"
        if gloss_dfs_folder.is_dir():
            typer.echo(f"Using existing gloss_dfs_folder: {gloss_dfs_folder}")
        else:
            gloss_dfs_folder = out_path / "gloss_dfs"
            gloss_dfs_folder.mkdir(exist_ok=True, parents=True)

    if gloss_count:
        query_gloss_vocabulary = query_gloss_vocabulary[:gloss_count]

    if additional_glosses:
        typer.echo(f"Prepending additional query glosses {additional_glosses}")
        combined = additional_glosses + query_gloss_vocabulary
        query_gloss_vocabulary = list(dict.fromkeys(combined))
        typer.echo(f"Query gloss vocabulary is now length {len(query_gloss_vocabulary)}")
        typer.echo(f"Query gloss vocabulary: {query_gloss_vocabulary}")

    if shuffle_metrics:
        random.shuffle(metrics)

    if shuffle_query_glosses:
        random.shuffle(query_gloss_vocabulary)

    if skip_glosses_with_more_than_this_many is not None:
        filtered_gloss_vocabulary = []
        print(f"GLOSS,SAMPLES")
        for gloss in query_gloss_vocabulary:
            in_gloss_df_path = gloss_dfs_folder / f"{gloss}_in.csv"
            # TODO: this crashes if it's a new gloss without an in_gloss CSV. BUT there's some metric_inputs_df filtering with embeddings to think about later... a new function perhaps?
            in_gloss_df = pd.read_csv(in_gloss_df_path, index_col=0, dtype={DatasetDFCol.GLOSS: str})
            if len(in_gloss_df) <= skip_glosses_with_more_than_this_many:
                typer.echo(
                    f"{gloss}, {len(in_gloss_df)}: less than or equal to the threshold {skip_glosses_with_more_than_this_many}. Adding to the gloss vocabulary"
                    # f"{gloss},{len(in_gloss_df)}"
                )
                filtered_gloss_vocabulary.append(gloss)

        filtered_gloss_vocabulary = list(set(filtered_gloss_vocabulary))
        typer.echo(
            f"{len(filtered_gloss_vocabulary)} Glosses with items less than or equal to {skip_glosses_with_more_than_this_many}: "
        )

        query_gloss_vocabulary = filtered_gloss_vocabulary

    # filter glosses that will cause issues #TODO: A better solution upstream, and maybe don't use glosses in filenames
    # Also, the load_parquets script can handle the weird filenames no problem, and convert to pyarrow.
    removed_glosses = []
    for punc in ["/", " ", "\\", ".", "_"]:
        query_gloss_vocabulary = [g for g in query_gloss_vocabulary if punc not in g]
        print(f"Filtered out glosses with {punc}: there are now {len(query_gloss_vocabulary)}")

    typer.echo(f"{query_gloss_vocabulary}")
    # All gloss–metric combinations
    gloss_metric_combos = list(product(query_gloss_vocabulary, metrics))

    if shuffle_metrics and shuffle_query_glosses:
        random.shuffle(gloss_metric_combos)

    if metric_count is not None:
        typer.echo(f"Selecting {metric_count} of {len(gloss_metric_combos)} gloss-metric combinations")
        gloss_metric_combos = gloss_metric_combos[:metric_count]

    scores_path = out_path / "scores"
    scores_path.mkdir(exist_ok=True, parents=True)

    for g_index, (gloss, metric) in enumerate(
        tqdm(gloss_metric_combos, desc="Running evaluations for gloss–metric combinations")
    ):
        if isinstance(metric, EmbeddingDistanceMetric):
            typer.echo(f"{metric} is an Embedding Metric")
            if DatasetDFCol.EMBEDDING_MODEL in df.columns:
                metric_inputs_df = df[df[DatasetDFCol.EMBEDDING_MODEL] == metric.model]
                typer.echo(f"Found {len(metric_inputs_df)} embedding rows matching model {metric.model}")
            else:
                typer.echo(f"No {DatasetDFCol.EMBEDDING_MODEL} in dataframe. Skipping!")
                continue
        else:
            if any(col in df.columns for col in [DatasetDFCol.EMBEDDING_MODEL, DatasetDFCol.EMBEDDING_FILE_PATH]):
                metric_inputs_df = df.drop(
                    columns=[DatasetDFCol.EMBEDDING_MODEL, DatasetDFCol.EMBEDDING_FILE_PATH]
                ).drop_duplicates()
            else:
                metric_inputs_df = df.drop_duplicates()

        # if not Path("gloss_counts.csv").is_file():
        #     gloss_count_df = metric_inputs_df.groupby("GLOSS").size().reset_index(name="SAMPLE_COUNT")
        #     gloss_count_df = gloss_count_df.sort_values(by="SAMPLE_COUNT")
        #     gloss_count_df.to_csv("gloss_counts.csv", index=False)
        #     print(Path("gloss_counts.csv").resolve())
        #     exit()
        results = defaultdict(list)
        results_path = scores_path / f"{gloss}_{metric.name}_outgloss_{out_gloss_multiplier}x_score_results.parquet"
        if results_path.exists():
            typer.echo(f"*** Results for {results_path} already exist. Skipping! ***")
            continue

        in_gloss_df_path = gloss_dfs_folder / f"{gloss}_in.csv"
        if in_gloss_df_path.is_file():
            typer.echo(f"Reading in-gloss df from {in_gloss_df_path}")
            in_gloss_df = pd.read_csv(in_gloss_df_path, index_col=0, dtype={DatasetDFCol.GLOSS: str})
        else:
            typer.echo(f"Writing in-gloss df to {in_gloss_df_path}")
            in_gloss_df = metric_inputs_df[metric_inputs_df[DatasetDFCol.GLOSS] == gloss]
            in_gloss_df.to_csv(in_gloss_df_path)

        if (
            skip_glosses_with_more_than_this_many is not None
            and len(in_gloss_df) > skip_glosses_with_more_than_this_many
        ):
            typer.echo(
                f"*** {gloss} has {len(in_gloss_df)} items, greater than the max: {skip_glosses_with_more_than_this_many}, skipping! ***"
            )
            continue

        out_gloss_df_path = gloss_dfs_folder / f"{gloss}_out.csv"
        if out_gloss_df_path.is_file():
            typer.echo(f"Reading out-gloss df from {out_gloss_df_path}")
            out_gloss_df = pd.read_csv(out_gloss_df_path, index_col=0, dtype={DatasetDFCol.GLOSS: str})
        else:
            typer.echo(f"Writing out-gloss df to {out_gloss_df_path}")
            out_gloss_df = metric_inputs_df[metric_inputs_df[DatasetDFCol.GLOSS] != gloss]
            other_class_count = len(in_gloss_df) * out_gloss_multiplier
            out_gloss_df = out_gloss_df.sample(n=other_class_count, random_state=42)
            out_gloss_df.to_csv(out_gloss_df_path)

        ref_df = pd.concat([in_gloss_df, out_gloss_df], ignore_index=True)

        pose_data = load_pose_files(in_gloss_df)
        pose_data.update(load_pose_files(out_gloss_df))

        typer.echo(
            f"For trial #{g_index}/{len(gloss_metric_combos)} ({len(query_gloss_vocabulary)}gX{len(metrics)}m) {gloss}+{metric.name}, we have {len(in_gloss_df)} in, "
            f"{len(out_gloss_df)} out, and {len(pose_data.keys())} poses total"
        )

        for _, hyp_row in tqdm(
            in_gloss_df.iterrows(),
            total=len(in_gloss_df),
            desc=f"{gloss}/{metric.name}",
        ):
            hyp_path = hyp_row[DatasetDFCol.POSE_FILE_PATH]

            for _, ref_row in ref_df.iterrows():
                ref_path = ref_row[DatasetDFCol.POSE_FILE_PATH]

                if isinstance(metric, EmbeddingDistanceMetric):
                    try:
                        hyp = load_embedding_for_pose(metric_inputs_df, model=metric.model, pose_file_path=hyp_path)
                        ref = load_embedding_for_pose(metric_inputs_df, model=metric.model, pose_file_path=ref_path)
                    except ValueError as e:
                        # typer.echo(f"ValueError on hyp_path {hyp_path}, ref_path {ref_path}: {e}")
                        continue

                else:
                    hyp = pose_data[hyp_path].copy()
                    ref = pose_data[ref_path].copy()

                start_time = time.perf_counter()
                score = metric.score_with_signature(hyp, ref)
                end_time = time.perf_counter()

                if score is None or score.score is None or np.isnan(score.score):
                    typer.echo(f"⚠️ Invalid score for {hyp_path} vs {ref_path}: {score.score}")
                    typer.echo(metric.get_signature().format())
                    typer.echo(metric.pose_preprocessors)

                results[ScoreDFCol.METRIC].append(metric.name)
                results[ScoreDFCol.SCORE].append(score.score)
                results[ScoreDFCol.GLOSS_A].append(hyp_row[DatasetDFCol.GLOSS])
                results[ScoreDFCol.GLOSS_B].append(ref_row[DatasetDFCol.GLOSS])
                results[ScoreDFCol.SIGNATURE].append(metric.get_signature().format())
                results[ScoreDFCol.GLOSS_A_PATH].append(hyp_path)
                results[ScoreDFCol.GLOSS_B_PATH].append(ref_path)
                results[ScoreDFCol.TIME].append(end_time - start_time)

        results_df = pd.DataFrame.from_dict(results)
        results_df.to_parquet(results_path, index=False, compression="snappy")
        typer.echo(f"Wrote {len(results_df)} scores to {results_path}")

        typer.echo("\n")
        typer.echo("*" * 50)
    typer.echo(f"Query glosses had {len(query_gloss_vocabulary)}: {query_gloss_vocabulary}")
    typer.echo(f"Scores saved to {scores_path}")


def compute_batch_pairs(hyp_chunk, ref_chunk, metric, signature, batch_filename):
    hyp_pose_data = load_pose_files(hyp_chunk)
    ref_pose_data = load_pose_files(ref_chunk)

    result_rows = []

    for _, hyp_row in hyp_chunk.iterrows():
        hyp_path = hyp_row[DatasetDFCol.POSE_FILE_PATH]
        hyp_pose = hyp_pose_data[hyp_path].copy()

        for _, ref_row in ref_chunk.iterrows():
            ref_path = ref_row[DatasetDFCol.POSE_FILE_PATH]
            ref_pose = ref_pose_data[ref_path].copy()

            start = time.perf_counter()
            score = metric.score_with_signature(hyp_pose, ref_pose)
            end = time.perf_counter()

            result_rows.append(
                {
                    ScoreDFCol.METRIC: metric.name,
                    ScoreDFCol.SCORE: score.score if score and score.score is not None else np.nan,
                    ScoreDFCol.GLOSS_A: hyp_row[DatasetDFCol.GLOSS],
                    ScoreDFCol.GLOSS_B: ref_row[DatasetDFCol.GLOSS],
                    ScoreDFCol.SIGNATURE: signature,
                    ScoreDFCol.GLOSS_A_PATH: hyp_path,
                    ScoreDFCol.GLOSS_B_PATH: ref_path,
                    ScoreDFCol.TIME: end - start,
                }
            )

    pd.DataFrame(result_rows).to_parquet(batch_filename, index=False, compression="snappy")
    del hyp_pose_data
    del ref_pose_data
    del hyp_chunk
    del ref_chunk

    return batch_filename


def run_metrics_full_distance_matrix_batched_parallel(
    df: pd.DataFrame,
    out_path: Path,
    metrics: list,
    batch_size: int = 100,
    max_workers: int = 4,
    merge=False,
    intersplit: bool = True,
    max_hyp: Optional[int] = None,
):
    typer.echo(
        f"Calculating {'intersplit' if intersplit else 'full'} distance matrix on {len(df)} poses "
        f"from {len(df[DatasetDFCol.DATASET].unique())} datasets, for {len(metrics)} metrics"
    )

    if intersplit:
        splits = df[DatasetDFCol.SPLIT].unique().tolist()
        if len(splits) != 2:
            raise ValueError(f"Expected exactly two splits for intersplit comparison, got: {splits}")
        else:
            typer.echo(f"Using {splits} from {df[DatasetDFCol.DATASET].unique()} datasets")

        split_a, split_b = splits
        df_a = df[df[DatasetDFCol.SPLIT] == split_a].reset_index(drop=True)
        df_b = df[df[DatasetDFCol.SPLIT] == split_b].reset_index(drop=True)
        n_a, n_b = len(df_a), len(df_b)
        typer.echo(f"Intersplit mode: comparing {split_a} ({n_a}) vs {split_b} ({n_b})")
        batches_hyp = math.ceil(n_a / batch_size)
        batches_ref = math.ceil(n_b / batch_size)

    else:
        df_a = df_b = df.reset_index(drop=True)
        n = len(df)
        batches_hyp = batches_ref = math.ceil(n / batch_size)

    total_batches = batches_hyp * batches_ref
    typer.echo(f"Batch size {batch_size}, max workers {max_workers}")
    typer.echo(f"Splits: {df[DatasetDFCol.SPLIT].unique()}")
    typer.echo(f"Results will be saved to {out_path}")
    typer.echo(f"Expecting {total_batches} total batches ({batches_hyp}x{batches_ref})")

    scores_path = out_path / "scores"
    typer.echo(f"Scores will be saved in batches to {scores_path}")
    scores_path.mkdir(exist_ok=True, parents=True)

    dataset_names = "+".join(df[DatasetDFCol.DATASET].unique().tolist())
    if intersplit:
        split_names = "vs".join(df[DatasetDFCol.SPLIT].unique().tolist())
    else:
        split_names = "+".join(df[DatasetDFCol.SPLIT].unique().tolist())

    for i, metric in tqdm(enumerate(metrics), total=len(metrics), desc="Iterating over metrics"):
        typer.echo("*" * 60)
        typer.echo(f"Metric #{i + 1}/{len(metrics)}: {metric.name}")
        signature = metric.get_signature().format()
        typer.echo(f"Metric Signature: {signature}")
        typer.echo(f"Batch Size: {batch_size}, so that's {batch_size*batch_size} distances per.")
        typer.echo(f"For metric {i}: {total_batches} total batches expected ({batches_hyp}x{batches_ref})")

        metric_results_path = scores_path / f"batches_{metric.name}_{dataset_names}_{split_names}"
        metric_results_path.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Saving batches to {metric_results_path}")

        existing_results = list(metric_results_path.glob("*.parquet"))
        typer.echo(f"{len(existing_results)} batches already exist; {total_batches - len(existing_results)} remaining.")

        futures = {}
        batch_id = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for hyp_start in tqdm(range(0, len(df_a), batch_size), desc=f"Hyp batching for Metric {i}"):
                hyp_chunk = df_a.iloc[hyp_start : hyp_start + batch_size].copy()

                for ref_start in range(0, len(df_b), batch_size):
                    ref_chunk = df_b.iloc[ref_start : ref_start + batch_size].copy()
                    batch_filename = metric_results_path / f"batch_{batch_id:06d}_hyp{hyp_start}_ref{ref_start}.parquet"

                    if not batch_filename.exists():
                        future = executor.submit(
                            compute_batch_pairs, hyp_chunk, ref_chunk, metric, signature, batch_filename
                        )
                        futures[future] = batch_filename

                    batch_id += 1
        typer.echo(f"Expecting {total_batches} total batches ({batches_hyp}x{batches_ref})")
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Waiting for workers"):
            batch_filename = futures.pop(future)
            result_path = future.result()
            typer.echo(f"💾 Worker saved: {Path(result_path).name}")

        if merge:
            typer.echo("🔄 Merging all batch files...")
            all_batch_files = sorted(metric_results_path.glob("batch_*.parquet"))
            all_dfs = [pd.read_parquet(f) for f in all_batch_files]
            merged_df = pd.concat(all_dfs, ignore_index=True)

            final_path = (
                scores_path / f"full_matrix_{metric.name}_on_{dataset_names}_{split_names}_score_results.parquet"
            )
            merged_df.to_parquet(final_path, index=False, compression="snappy")
            typer.echo(f"✅ Final results written to {final_path}\n")


def get_filtered_metrics(
    metrics,
    top10=False,
    top10_nohands_nointerp=False,
    top10_nohands_nodtw_nointerp=False,
    top50_nointerp=False,
    get_top_10_nointerp_default10_fillmasked10=False,
    return4=False,
    include_keywords: List[str] | None = None,
    exclude_keywords: List[str] | None = None,
    specific_metrics: str | list[str] | None = None,
    csv_path: Path | None = None,
):
    if isinstance(specific_metrics, str):
        specific_metrics = [specific_metrics]

    print(f"Filtering {len(metrics)} Metrics:")
    print(f"Include: {include_keywords}")
    print(f"Exclude: {exclude_keywords}")

    # top 10
    top_10_metrics_by_map = [
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist0.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        # "Return4Metric",
    ]

    # top_10_by_map excluding hands,interp15,interp120
    top_10_by_map_excluding_hands_interp = [
        "startendtrimmed_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_youtubeaslkeypoints_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_reduceholistic_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_youtubeaslkeypoints_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
    ]

    # top 10 excluding hands, dtw, interp15,interp120
    top_10_by_map_excluding_hands_interp_and_dtw = [
        "startendtrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_zeropad_fillmasked0.0_AggregatedPowerDistanceMetric",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_zeropad_fillmasked10.0_AggregatedPowerDistanceMetric",
        "startendtrimmed_unnormalized_youtubeaslkeypoints_defaultdist0.0_nointerp_padwithfirstframe_fillmasked1.0_AggregatedPowerDistanceMetric",
        "startendtrimmed_unnormalized_youtubeaslkeypoints_defaultdist0.0_nointerp_padwithfirstframe_fillmasked0.0_AggregatedPowerDistanceMetric",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist10.0_nointerp_zeropad_fillmasked0.0_AggregatedPowerDistanceMetric",
        "startendtrimmed_unnormalized_removelegsandworld_defaultdist1.0_nointerp_zeropad_fillmasked0.0_AggregatedPowerDistanceMetric",
        "startendtrimmed_unnormalized_removelegsandworld_defaultdist10.0_nointerp_zeropad_fillmasked10.0_AggregatedPowerDistanceMetric",
        "untrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_zeropad_fillmasked0.0_AggregatedPowerDistanceMetric",
        "untrimmed_unnormalized_reduceholistic_defaultdist1.0_nointerp_zeropad_fillmasked10.0_AggregatedPowerDistanceMetric",
        "startendtrimmed_unnormalized_removelegsandworld_defaultdist10.0_nointerp_zeropad_fillmasked0.0_AggregatedPowerDistanceMetric",
    ]

    # top 50 by MAP
    top_50_by_map = [
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist0.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_interp120_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_interp120_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist10.0_interp15_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_interp120_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist10.0_interp120_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist10.0_interp120_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist0.0_interp15_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist10.0_interp120_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist1.0_interp120_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist0.0_interp120_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist1.0_interp15_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist1.0_interp120_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist10.0_interp15_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist0.0_interp120_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist1.0_interp15_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist10.0_interp15_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist1.0_interp15_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist1.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist0.0_interp15_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_interp15_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist1.0_interp120_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_interp120_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist1.0_interp15_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_reduceholistic_defaultdist10.0_interp15_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist10.0_interp120_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist0.0_interp15_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist0.0_interp120_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist0.0_interp120_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist0.0_interp120_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist10.0_interp120_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist1.0_interp120_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist1.0_interp120_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist10.0_interp15_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist1.0_interp120_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
    ]

    # top 50 of 677 metrics excluding interp15,interp120
    top_50_by_map_excluding_interp = [
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist0.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist1.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist0.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_youtubeaslkeypoints_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist1.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_reduceholistic_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_youtubeaslkeypoints_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_reduceholistic_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist0.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist1.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_zeropad_fillmasked1.0_AggregatedPowerDistanceMetric",
        "untrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
    ]

    # top 10 from round 1 and 2 by MAP, with the same default and fillmasked as the top metric and without interp
    # aka excluding defaultdist1.0,defaultdist0.0,fillmasked1.0,fillmasked0.0,interp120,interp15,
    top_10_nointerp_default10_fillmasked10 = [
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_unnormalized_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_zeropad_fillmasked10.0_AggregatedPowerDistanceMetric",
        "startendtrimmed_unnormalized_removelegsandworld_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_unnormalized_removelegsandworld_defaultdist10.0_nointerp_zeropad_fillmasked10.0_AggregatedPowerDistanceMetric",
        "untrimmed_unnormalized_removelegsandworld_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "untrimmed_normalizedbyshoulders_removelegsandworld_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
        "startendtrimmed_normalizedbyshoulders_removelegsandworld_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast",
    ]

    # Get a set of target names for efficient lookup
    metrics_to_use = []
    if specific_metrics is not None:
        typer.echo(f"Adding specific metric: {specific_metrics}")
        metrics_to_use.extend(specific_metrics)

    if top10:
        metrics_to_use.extend(top_10_metrics_by_map)

    if top10_nohands_nointerp:
        metrics_to_use.extend(top_10_by_map_excluding_hands_interp)
    if top10_nohands_nodtw_nointerp:
        metrics_to_use.extend(top_10_by_map_excluding_hands_interp_and_dtw)

    if top50_nointerp:
        metrics_to_use.extend(top_50_by_map_excluding_interp)

    if get_top_10_nointerp_default10_fillmasked10:
        typer.echo(f"Adding {len(top_10_nointerp_default10_fillmasked10)} from top_10_nointerp_default10_fillmasked10")
        metrics_to_use.extend(top_10_nointerp_default10_fillmasked10)

    if return4:
        typer.echo(f"Adding Return4 Metric")
        metrics_to_use.append("Return4Metric_defaultdist4.0")

    if csv_path is not None:
        df = pd.read_csv(csv_path)
        csv_metrics = df["METRIC"].unique().tolist()
        typer.echo(f"Adding {len(csv_metrics)} from {csv_path}")
        metrics_to_use.extend(csv_metrics)

    if include_keywords is not None:
        include_list = [m.name for m in metrics if all(k.lower() in m.name.lower() for k in include_keywords)]
        typer.echo(f"Adding {len(include_list)} metrics that include all of: {include_keywords}")
        metrics_to_use.extend(include_list)
        typer.echo(f"There are now {len(metrics_to_use)} metrics")

    if exclude_keywords is not None:
        typer.echo(f"Filtering metrics to those that exclude all of: {exclude_keywords}")
        metrics_to_use = [m for m in metrics_to_use if not any(k.lower() in m.lower() for k in exclude_keywords)]
        typer.echo(f"There are now {len(metrics_to_use)} metrics")

    metrics_to_use_set = set(metrics_to_use)

    # Filter metrics based on .name

    filtered_metrics = [m for m in metrics if m.name in metrics_to_use_set]

    # Find which metrics_to_use were unmatched
    metric_names = {m.name for m in metrics}
    unmatched_metrics = [m for m in metrics_to_use if m not in metric_names]

    typer.echo(specific_metrics)
    typer.echo(f"{len(unmatched_metrics)} Unmatched metrics_to_use: {unmatched_metrics}")
    typer.echo(
        f"{len(filtered_metrics)} Filtered metrics, here are the first few: {[m.name for m in filtered_metrics[:10]]}"
    )

    return filtered_metrics


@app.command()
def main(
    dataset_df_files: Annotated[
        List[Path], typer.Argument(help="List of dataset csvs, with columns POSE_FILE_PATH, SPLIT, and VIDEO_ID")
    ],
    splits: Optional[str] = typer.Option(
        None, help="Comma-separated list of splits to process (e.g., 'train,val,test'), default is 'test' only"
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
    full: Optional[bool] = typer.Option(
        False, help="Whether to run FULL distance matrix with the specified dataset dfs"
    ),
    full_intersplit: Optional[bool] = typer.Option(
        True, help="Whether to run FULL distance matrix with the specified dataset dfs, but between two splits"
    ),
    max_workers: Optional[int] = typer.Option(4, help="How many workers to use for the full distance matrix?"),
    batch_size: Optional[int] = typer.Option(
        100, help="Batch size for the workers. This is the number of hyps, so distances per batch will be this squared"
    ),
    filter_metrics: bool = typer.Option(True, help="whether to use the filtered set of metrics"),
    embedding_metrics: bool = typer.Option(False, help="whether to add in embedding metric"),
    specific_metrics: List[str] = typer.Option(
        None, help="If specified, will add these metrics to the list of filtered metrics"
    ),
    specific_metrics_csv: Path = typer.Option(
        None, help="If specified, will read the metrics from this CSV and add those"
    ),
    include_keywords: List[str] = typer.Option(
        None, help="Will filter metrics to only those that include any of these"
    ),
    exclude_keywords: List[str] = typer.Option(
        None, help="Will filter metrics to only those that include none of these"
    ),
    skip_glosses_with_more_than_this_many: int = typer.Option(
        None, help="skip long glosses with more than this many items/samples"
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

    metrics = get_metrics()  # generate all possible metrics
    # typer.echo(f"Metrics: {[m.name for m in metrics]}")
    typer.echo(f"We have a total of {len(metrics)} metrics")

    if specific_metrics is not None or specific_metrics_csv is not None:
        typer.echo(specific_metrics)
        metrics = get_filtered_metrics(
            metrics,
            top10=False,
            top10_nohands_nointerp=False,
            top10_nohands_nodtw_nointerp=False,
            top50_nointerp=False,
            return4=False,
            specific_metrics=specific_metrics,
            csv_path=specific_metrics_csv,
            include_keywords=include_keywords,
            exclude_keywords=exclude_keywords,
        )
    else:

        if filter_metrics:

            metrics = get_filtered_metrics(
                metrics,
                top10=False,
                top10_nohands_nointerp=False,
                top10_nohands_nodtw_nointerp=False,
                top50_nointerp=False,
                return4=False,
                include_keywords=include_keywords,
                exclude_keywords=exclude_keywords,
            )

    if embedding_metrics:
        try:
            embed_metrics = list(get_embedding_metrics(df))
            # typer.echo(f"Created embedding_metrics:{embed_metrics}")
            metrics.extend(embed_metrics)
        except ValueError as e:
            typer.echo(f"Unable to add embedding metrics due to ValueError: {e}")

    random.shuffle(metrics)
    print(f"{len(metrics)} metrics after filtering:")
    for metric in metrics:
        typer.echo(metric.name)

    typer.echo(f"Saving results to {out}")
    out.mkdir(parents=True, exist_ok=True)
    if full:
        run_metrics_full_distance_matrix_batched_parallel(
            df,
            out_path=out,
            metrics=metrics,
            batch_size=batch_size,
            max_workers=max_workers,
            intersplit=full_intersplit,
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
            shuffle_query_glosses=True,
            skip_glosses_with_more_than_this_many=skip_glosses_with_more_than_this_many,
        )


if __name__ == "__main__":
    app()
# with 30 glosses total including really big glosses, BLACK, HOME, HUNGRY, REFRIGERATOR, UNCLE
# results in ['RUSSIA', 'BRAG', 'HOUSE', 'HOME', 'WORM', 'REFRIGERATOR', 'BLACK', 'SUMMER', 'SICK', 'REALSICK', 'WEATHER', 'MEETING', 'COLD', 'WINTER', 'THANKSGIVING', 'THANKYOU', 'HUNGRY', 'FULL', 'LIBRARY', 'CELERY', 'CANDY1', 'CASTLE2', 'GOVERNMENT', 'OPINION1', 'PJS', 'LIVE2', 'WORKSHOP', 'UNCLE', 'EASTER', 'BAG2']
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 12 dataset_dfs/*.csv --additional-glosses "RUSSIA,BRAG,HOUSE,HOME,WORM,REFRIGERATOR,BLACK,SUMMER,SICK,REALSICK,WEATHER,MEETING,COLD,WINTER,THANKSGIVING,THANKYOU,HUNGRY,FULL" 2>&1|tee out/$(date +%s).txt

# round 2
# Add more glosses, 100 total.
# Results in this list: ['ACCENT', 'ADULT', 'AIRPLANE', 'APPEAR', 'BAG2', 'BANANA2', 'BEAK', 'BERRY', 'BIG', 'BINOCULARS', 'BLACK', 'BRAG', 'BRAINWASH', 'CAFETERIA', 'CALM', 'CANDY1', 'CASTLE2', 'CELERY', 'CHEW1', 'COLD', 'CONVINCE2', 'COUNSELOR', 'DART', 'DEAF2', 'DEAFSCHOOL', 'DECIDE2', 'DIP3', 'DOLPHIN2', 'DRINK2', 'DRIP', 'DRUG', 'EACH', 'EARN', 'EASTER', 'ERASE1', 'EVERYTHING', 'FINGERSPELL', 'FISHING2', 'FORK4', 'FULL', 'GOTHROUGH', 'GOVERNMENT', 'HIDE', 'HOME', 'HOUSE', 'HUNGRY', 'HURRY', 'KNITTING3', 'LEAF1', 'LEND', 'LIBRARY', 'LIVE2', 'MACHINE', 'MAIL1', 'MEETING', 'NECKLACE4', 'NEWSTOME', 'OPINION1', 'ORGANIZATION', 'PAIR', 'PEPSI', 'PERFUME1', 'PIG', 'PILL', 'PIPE2', 'PJS', 'REALSICK', 'RECORDING', 'REFRIGERATOR', 'REPLACE', 'RESTAURANT', 'ROCKINGCHAIR1', 'RUIN', 'RUSSIA', 'SCREWDRIVER3', 'SENATE', 'SHAME', 'SHARK2', 'SHAVE5', 'SICK', 'SNOWSUIT', 'SPECIALIST', 'STADIUM', 'SUMMER', 'TAKEOFF1', 'THANKSGIVING', 'THANKYOU', 'TIE1', 'TOP', 'TOSS', 'TURBAN', 'UNCLE', 'VAMPIRE', 'WASHDISHES', 'WEAR', 'WEATHER', 'WINTER', 'WORKSHOP', 'WORM', 'YESTERDAY']
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 83 dataset_dfs/*.csv --additional-glosses "RUSSIA,BRAG,HOUSE,HOME,WORM,REFRIGERATOR,BLACK,SUMMER,SICK,REALSICK,WEATHER,MEETING,COLD,WINTER,THANKSGIVING,THANKYOU,HUNGRY,FULL" 2>&1|tee out/$(date +%s).txt


# Round 3
# with all the known-similar glosses included, and saving to metric_results_round_3
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 84 dataset_dfs/*.csv --additional-glosses "MEETING,WEATHER,COLD,WINTER,CHICAGO,PERCENT,PERCENT,PHILADELPHIA,CHICAGO,PHILADELPHIA,TRADITION,WORK,GIFT,GIVE,SANTA,THANKSGIVING,SANTA,THANKYOU,FULL,SANTA,THANKSGIVING,THANKYOU,FULL,THANKSGIVING,FULL,THANKYOU,ANIMAL,HOLIDAY,HAVE,HOLIDAY,ANIMAL,HAVE,COW,MOOSE,BUFFALO,MOOSE,MOOSE,PRESIDENT,BUFFALO,COW,COW,PRESIDENT,BUFFALO,PRESIDENT,FAMILY,TEAM,FAMILY,GROUP,CLASS,FAMILY,GROUP,TEAM,CLASS,TEAM,CLASS,GROUP,DIRTY,PIG,GRASS,PIG,DIRTY,GRASS,DUTY,PUMPKIN,HERE,SALAD,HUNGRY,THIRSTY,FULL,HUNGRY,FULL,THIRSTY,ANIMAL,VACATION,HAVE,VACATION,COW,HORSE,COW,DEER,BUFFALO,DEER,DEER,PRESIDENT,DEER,MOOSE,MOUSE,RAT,MOUSE,ROSE,RAT,ROSE,LION,TIGER,BEAR,HUG,BEAR,LOVE,HUG,LOVE,SNAKE,SPICY,ALASKA,HAWAII,ALASKA,PRETTY,ALASKA,FACE,HAWAII,PRETTY,FACE,HAWAII,FACE,PRETTY,ARIZONA,RESTAURANT,ARIZONA,CAFETERIA,CAFETERIA,RESTAURANT,CALIFORNIA,GOLD,CALIFORNIA,SILVER,CALIFORNIA,PHONE,CALIFORNIA,WHY,GOLD,SILVER,GOLD,PHONE,GOLD,WHY,PHONE,SILVER,SILVER,WHY,PHONE,WHY,COLOR,FRIENDLY,NEWYORK,PRACTICE,WEDNESDAY,WEST,TEA,VOTE,APPLE,ONION,WATER,WINE,COOKIE,PIE,FAVORITE,TASTE,FAVORITE,LUCKY,LUCKY,TASTE,AUNT,GIRL,CHILD,CHILDREN,PLEASE,SORRY,ENJOY,PLEASE,DONTKNOW,KNOW,LEARN,STUDENT,DAY,TODAY,TOMORROW,YESTERDAY,TUESDAY,WEDNESDAY,FRIDAY,TUESDAY,SATURDAY,TUESDAY,FRIDAY,WEDNESDAY,SATURDAY,WEDNESDAY,FRIDAY,SATURDAY,HIGHSCHOOL,THURSDAY,SIX,THREE,NINE,THREE,NINE,SIX,SEVEN,SIX,EIGHT,SIX,EIGHT,SEVEN,NINE,SEVEN,EIGHT,NINE,NAME,WEIGHT,MY,YOUR,BAD,GOOD,PENCIL,WRITE,ICECREAM,MICROPHONE,ADVERTISE,SHOES,GAME,RACE,EXCITED,THRILLED,NEWSPAPER,PRINT,FEW,SEVERAL,INTRODUCE,INVITE,SOCKS,STARS,SEE,WATCH,ENOUGH,FULL,CHAIR,SIT,TELL,TRUE,BUT,DIFFERENT,BATHROOM,TUESDAY,HUSBAND,WIFE,MOTHER,VOMIT,OHISEE,YELLOW,HARDOFHEARING,HISTORY,PREFER,TASTE,CHALLENGE,GAME,CLOSE,OPEN,BLACK,SUMMER,PAPER,SCHOOL,NAME,WEIGH,HOUSE,ROOF,BEER,BROWN,DANCE,READ,COMMITTEE,SENATE,EXPERIMENT,SCIENCE,ATTENTION,FOCUS,BRAG,RUSSIA,DONTCARE,DONTMIND,GALLAUDET,GLASSES,FRIENDLY,SAD" --out metric_results_round_3 2>&1|tee out/$(date +%s).txt

# Use round-2 metrics and glosses, put results in scores
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 83 dataset_dfs/*.csv --additional-glosses "RUSSIA,BRAG,HOUSE,HOME,WORM,REFRIGERATOR,BLACK,SUMMER,SICK,REALSICK,WEATHER,MEETING,COLD,WINTER,THANKSGIVING,THANKYOU,HUNGRY,FULL" --out foo --specific-metrics-csv metric_results_round_2/4_23_2025_score_analysis_3300_trials/stats_by_metric.csv 2>&1|tee out/$(date +%s).txt

# z-stretching (before I added it to create_metrics)
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 84 dataset_dfs/*.csv --additional-glosses "MEETING,WEATHER,COLD,WINTER,CHICAGO,PERCENT,PERCENT,PHILADELPHIA,CHICAGO,PHILADELPHIA,TRADITION,WORK,GIFT,GIVE,SANTA,THANKSGIVING,SANTA,THANKYOU,FULL,SANTA,THANKSGIVING,THANKYOU,FULL,THANKSGIVING,FULL,THANKYOU,ANIMAL,HOLIDAY,HAVE,HOLIDAY,ANIMAL,HAVE,COW,MOOSE,BUFFALO,MOOSE,MOOSE,PRESIDENT,BUFFALO,COW,COW,PRESIDENT,BUFFALO,PRESIDENT,FAMILY,TEAM,FAMILY,GROUP,CLASS,FAMILY,GROUP,TEAM,CLASS,TEAM,CLASS,GROUP,DIRTY,PIG,GRASS,PIG,DIRTY,GRASS,DUTY,PUMPKIN,HERE,SALAD,HUNGRY,THIRSTY,FULL,HUNGRY,FULL,THIRSTY,ANIMAL,VACATION,HAVE,VACATION,COW,HORSE,COW,DEER,BUFFALO,DEER,DEER,PRESIDENT,DEER,MOOSE,MOUSE,RAT,MOUSE,ROSE,RAT,ROSE,LION,TIGER,BEAR,HUG,BEAR,LOVE,HUG,LOVE,SNAKE,SPICY,ALASKA,HAWAII,ALASKA,PRETTY,ALASKA,FACE,HAWAII,PRETTY,FACE,HAWAII,FACE,PRETTY,ARIZONA,RESTAURANT,ARIZONA,CAFETERIA,CAFETERIA,RESTAURANT,CALIFORNIA,GOLD,CALIFORNIA,SILVER,CALIFORNIA,PHONE,CALIFORNIA,WHY,GOLD,SILVER,GOLD,PHONE,GOLD,WHY,PHONE,SILVER,SILVER,WHY,PHONE,WHY,COLOR,FRIENDLY,NEWYORK,PRACTICE,WEDNESDAY,WEST,TEA,VOTE,APPLE,ONION,WATER,WINE,COOKIE,PIE,FAVORITE,TASTE,FAVORITE,LUCKY,LUCKY,TASTE,AUNT,GIRL,CHILD,CHILDREN,PLEASE,SORRY,ENJOY,PLEASE,DONTKNOW,KNOW,LEARN,STUDENT,DAY,TODAY,TOMORROW,YESTERDAY,TUESDAY,WEDNESDAY,FRIDAY,TUESDAY,SATURDAY,TUESDAY,FRIDAY,WEDNESDAY,SATURDAY,WEDNESDAY,FRIDAY,SATURDAY,HIGHSCHOOL,THURSDAY,SIX,THREE,NINE,THREE,NINE,SIX,SEVEN,SIX,EIGHT,SIX,EIGHT,SEVEN,NINE,SEVEN,EIGHT,NINE,NAME,WEIGHT,MY,YOUR,BAD,GOOD,PENCIL,WRITE,ICECREAM,MICROPHONE,ADVERTISE,SHOES,GAME,RACE,EXCITED,THRILLED,NEWSPAPER,PRINT,FEW,SEVERAL,INTRODUCE,INVITE,SOCKS,STARS,SEE,WATCH,ENOUGH,FULL,CHAIR,SIT,TELL,TRUE,BUT,DIFFERENT,BATHROOM,TUESDAY,HUSBAND,WIFE,MOTHER,VOMIT,OHISEE,YELLOW,HARDOFHEARING,HISTORY,PREFER,TASTE,CHALLENGE,GAME,CLOSE,OPEN,BLACK,SUMMER,PAPER,SCHOOL,NAME,WEIGH,HOUSE,ROOF,BEER,BROWN,DANCE,READ,COMMITTEE,SENATE,EXPERIMENT,SCIENCE,ATTENTION,FOCUS,BRAG,RUSSIA,DONTCARE,DONTMIND,GALLAUDET,GLASSES,FRIENDLY,SAD" --out metric_results_z_offsets --no-filter-metrics 2>&1|tee out/$(date +%s).txt

# round 4: 250 glosses, random metrics, both shuffled
# so we get a lot of single-gloss metrics
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 84 dataset_dfs/*.csv --additional-glosses "MEETING,WEATHER,COLD,WINTER,CHICAGO,PERCENT,PERCENT,PHILADELPHIA,CHICAGO,PHILADELPHIA,TRADITION,WORK,GIFT,GIVE,SANTA,THANKSGIVING,SANTA,THANKYOU,FULL,SANTA,THANKSGIVING,THANKYOU,FULL,THANKSGIVING,FULL,THANKYOU,ANIMAL,HOLIDAY,HAVE,HOLIDAY,ANIMAL,HAVE,COW,MOOSE,BUFFALO,MOOSE,MOOSE,PRESIDENT,BUFFALO,COW,COW,PRESIDENT,BUFFALO,PRESIDENT,FAMILY,TEAM,FAMILY,GROUP,CLASS,FAMILY,GROUP,TEAM,CLASS,TEAM,CLASS,GROUP,DIRTY,PIG,GRASS,PIG,DIRTY,GRASS,DUTY,PUMPKIN,HERE,SALAD,HUNGRY,THIRSTY,FULL,HUNGRY,FULL,THIRSTY,ANIMAL,VACATION,HAVE,VACATION,COW,HORSE,COW,DEER,BUFFALO,DEER,DEER,PRESIDENT,DEER,MOOSE,MOUSE,RAT,MOUSE,ROSE,RAT,ROSE,LION,TIGER,BEAR,HUG,BEAR,LOVE,HUG,LOVE,SNAKE,SPICY,ALASKA,HAWAII,ALASKA,PRETTY,ALASKA,FACE,HAWAII,PRETTY,FACE,HAWAII,FACE,PRETTY,ARIZONA,RESTAURANT,ARIZONA,CAFETERIA,CAFETERIA,RESTAURANT,CALIFORNIA,GOLD,CALIFORNIA,SILVER,CALIFORNIA,PHONE,CALIFORNIA,WHY,GOLD,SILVER,GOLD,PHONE,GOLD,WHY,PHONE,SILVER,SILVER,WHY,PHONE,WHY,COLOR,FRIENDLY,NEWYORK,PRACTICE,WEDNESDAY,WEST,TEA,VOTE,APPLE,ONION,WATER,WINE,COOKIE,PIE,FAVORITE,TASTE,FAVORITE,LUCKY,LUCKY,TASTE,AUNT,GIRL,CHILD,CHILDREN,PLEASE,SORRY,ENJOY,PLEASE,DONTKNOW,KNOW,LEARN,STUDENT,DAY,TODAY,TOMORROW,YESTERDAY,TUESDAY,WEDNESDAY,FRIDAY,TUESDAY,SATURDAY,TUESDAY,FRIDAY,WEDNESDAY,SATURDAY,WEDNESDAY,FRIDAY,SATURDAY,HIGHSCHOOL,THURSDAY,SIX,THREE,NINE,THREE,NINE,SIX,SEVEN,SIX,EIGHT,SIX,EIGHT,SEVEN,NINE,SEVEN,EIGHT,NINE,NAME,WEIGHT,MY,YOUR,BAD,GOOD,PENCIL,WRITE,ICECREAM,MICROPHONE,ADVERTISE,SHOES,GAME,RACE,EXCITED,THRILLED,NEWSPAPER,PRINT,FEW,SEVERAL,INTRODUCE,INVITE,SOCKS,STARS,SEE,WATCH,ENOUGH,FULL,CHAIR,SIT,TELL,TRUE,BUT,DIFFERENT,BATHROOM,TUESDAY,HUSBAND,WIFE,MOTHER,VOMIT,OHISEE,YELLOW,HARDOFHEARING,HISTORY,PREFER,TASTE,CHALLENGE,GAME,CLOSE,OPEN,BLACK,SUMMER,PAPER,SCHOOL,NAME,WEIGH,HOUSE,ROOF,BEER,BROWN,DANCE,READ,COMMITTEE,SENATE,EXPERIMENT,SCIENCE,ATTENTION,FOCUS,BRAG,RUSSIA,DONTCARE,DONTMIND,GALLAUDET,GLASSES,FRIENDLY,SAD" --no-filter-metrics --out metric_results_round_4 2>&1|tee out/$(date +%s).txt

# Backfilling after default_dist issue: metric_results_1_2_z_combined_818_metrics/
# using 250 gloss vocabulary
# and the specific 818 metrics from before
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 84 dataset_dfs/*.csv --additional-glosses "MEETING,WEATHER,COLD,WINTER,CHICAGO,PERCENT,PERCENT,PHILADELPHIA,CHICAGO,PHILADELPHIA,TRADITION,WORK,GIFT,GIVE,SANTA,THANKSGIVING,SANTA,THANKYOU,FULL,SANTA,THANKSGIVING,THANKYOU,FULL,THANKSGIVING,FULL,THANKYOU,ANIMAL,HOLIDAY,HAVE,HOLIDAY,ANIMAL,HAVE,COW,MOOSE,BUFFALO,MOOSE,MOOSE,PRESIDENT,BUFFALO,COW,COW,PRESIDENT,BUFFALO,PRESIDENT,FAMILY,TEAM,FAMILY,GROUP,CLASS,FAMILY,GROUP,TEAM,CLASS,TEAM,CLASS,GROUP,DIRTY,PIG,GRASS,PIG,DIRTY,GRASS,DUTY,PUMPKIN,HERE,SALAD,HUNGRY,THIRSTY,FULL,HUNGRY,FULL,THIRSTY,ANIMAL,VACATION,HAVE,VACATION,COW,HORSE,COW,DEER,BUFFALO,DEER,DEER,PRESIDENT,DEER,MOOSE,MOUSE,RAT,MOUSE,ROSE,RAT,ROSE,LION,TIGER,BEAR,HUG,BEAR,LOVE,HUG,LOVE,SNAKE,SPICY,ALASKA,HAWAII,ALASKA,PRETTY,ALASKA,FACE,HAWAII,PRETTY,FACE,HAWAII,FACE,PRETTY,ARIZONA,RESTAURANT,ARIZONA,CAFETERIA,CAFETERIA,RESTAURANT,CALIFORNIA,GOLD,CALIFORNIA,SILVER,CALIFORNIA,PHONE,CALIFORNIA,WHY,GOLD,SILVER,GOLD,PHONE,GOLD,WHY,PHONE,SILVER,SILVER,WHY,PHONE,WHY,COLOR,FRIENDLY,NEWYORK,PRACTICE,WEDNESDAY,WEST,TEA,VOTE,APPLE,ONION,WATER,WINE,COOKIE,PIE,FAVORITE,TASTE,FAVORITE,LUCKY,LUCKY,TASTE,AUNT,GIRL,CHILD,CHILDREN,PLEASE,SORRY,ENJOY,PLEASE,DONTKNOW,KNOW,LEARN,STUDENT,DAY,TODAY,TOMORROW,YESTERDAY,TUESDAY,WEDNESDAY,FRIDAY,TUESDAY,SATURDAY,TUESDAY,FRIDAY,WEDNESDAY,SATURDAY,WEDNESDAY,FRIDAY,SATURDAY,HIGHSCHOOL,THURSDAY,SIX,THREE,NINE,THREE,NINE,SIX,SEVEN,SIX,EIGHT,SIX,EIGHT,SEVEN,NINE,SEVEN,EIGHT,NINE,NAME,WEIGHT,MY,YOUR,BAD,GOOD,PENCIL,WRITE,ICECREAM,MICROPHONE,ADVERTISE,SHOES,GAME,RACE,EXCITED,THRILLED,NEWSPAPER,PRINT,FEW,SEVERAL,INTRODUCE,INVITE,SOCKS,STARS,SEE,WATCH,ENOUGH,FULL,CHAIR,SIT,TELL,TRUE,BUT,DIFFERENT,BATHROOM,TUESDAY,HUSBAND,WIFE,MOTHER,VOMIT,OHISEE,YELLOW,HARDOFHEARING,HISTORY,PREFER,TASTE,CHALLENGE,GAME,CLOSE,OPEN,BLACK,SUMMER,PAPER,SCHOOL,NAME,WEIGH,HOUSE,ROOF,BEER,BROWN,DANCE,READ,COMMITTEE,SENATE,EXPERIMENT,SCIENCE,ATTENTION,FOCUS,BRAG,RUSSIA,DONTCARE,DONTMIND,GALLAUDET,GLASSES,FRIENDLY,SAD" --out metric_results_1_2_z_combined_818_metrics/ --specific-metrics-csv metric_results_1_2_z_combined_818_metrics/round1_2_z_combined_metric_stats_with_default_distance_issue.csv 2>&1|tee out/in_out_trials_$(date +%s).txt


# embed metrics only, 250 glosses, skipping glosses over 30 long
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py --gloss-count 84 dataset_dfs_with_embed/*.csv --additional-glosses "MEETING,WEATHER,COLD,WINTER,CHICAGO,PERCENT,PERCENT,PHILADELPHIA,CHICAGO,PHILADELPHIA,TRADITION,WORK,GIFT,GIVE,SANTA,THANKSGIVING,SANTA,THANKYOU,FULL,SANTA,THANKSGIVING,THANKYOU,FULL,THANKSGIVING,FULL,THANKYOU,ANIMAL,HOLIDAY,HAVE,HOLIDAY,ANIMAL,HAVE,COW,MOOSE,BUFFALO,MOOSE,MOOSE,PRESIDENT,BUFFALO,COW,COW,PRESIDENT,BUFFALO,PRESIDENT,FAMILY,TEAM,FAMILY,GROUP,CLASS,FAMILY,GROUP,TEAM,CLASS,TEAM,CLASS,GROUP,DIRTY,PIG,GRASS,PIG,DIRTY,GRASS,DUTY,PUMPKIN,HERE,SALAD,HUNGRY,THIRSTY,FULL,HUNGRY,FULL,THIRSTY,ANIMAL,VACATION,HAVE,VACATION,COW,HORSE,COW,DEER,BUFFALO,DEER,DEER,PRESIDENT,DEER,MOOSE,MOUSE,RAT,MOUSE,ROSE,RAT,ROSE,LION,TIGER,BEAR,HUG,BEAR,LOVE,HUG,LOVE,SNAKE,SPICY,ALASKA,HAWAII,ALASKA,PRETTY,ALASKA,FACE,HAWAII,PRETTY,FACE,HAWAII,FACE,PRETTY,ARIZONA,RESTAURANT,ARIZONA,CAFETERIA,CAFETERIA,RESTAURANT,CALIFORNIA,GOLD,CALIFORNIA,SILVER,CALIFORNIA,PHONE,CALIFORNIA,WHY,GOLD,SILVER,GOLD,PHONE,GOLD,WHY,PHONE,SILVER,SILVER,WHY,PHONE,WHY,COLOR,FRIENDLY,NEWYORK,PRACTICE,WEDNESDAY,WEST,TEA,VOTE,APPLE,ONION,WATER,WINE,COOKIE,PIE,FAVORITE,TASTE,FAVORITE,LUCKY,LUCKY,TASTE,AUNT,GIRL,CHILD,CHILDREN,PLEASE,SORRY,ENJOY,PLEASE,DONTKNOW,KNOW,LEARN,STUDENT,DAY,TODAY,TOMORROW,YESTERDAY,TUESDAY,WEDNESDAY,FRIDAY,TUESDAY,SATURDAY,TUESDAY,FRIDAY,WEDNESDAY,SATURDAY,WEDNESDAY,FRIDAY,SATURDAY,HIGHSCHOOL,THURSDAY,SIX,THREE,NINE,THREE,NINE,SIX,SEVEN,SIX,EIGHT,SIX,EIGHT,SEVEN,NINE,SEVEN,EIGHT,NINE,NAME,WEIGHT,MY,YOUR,BAD,GOOD,PENCIL,WRITE,ICECREAM,MICROPHONE,ADVERTISE,SHOES,GAME,RACE,EXCITED,THRILLED,NEWSPAPER,PRINT,FEW,SEVERAL,INTRODUCE,INVITE,SOCKS,STARS,SEE,WATCH,ENOUGH,FULL,CHAIR,SIT,TELL,TRUE,BUT,DIFFERENT,BATHROOM,TUESDAY,HUSBAND,WIFE,MOTHER,VOMIT,OHISEE,YELLOW,HARDOFHEARING,HISTORY,PREFER,TASTE,CHALLENGE,GAME,CLOSE,OPEN,BLACK,SUMMER,PAPER,SCHOOL,NAME,WEIGH,HOUSE,ROOF,BEER,BROWN,DANCE,READ,COMMITTEE,SENATE,EXPERIMENT,SCIENCE,ATTENTION,FOCUS,BRAG,RUSSIA,DONTCARE,DONTMIND,GALLAUDET,GLASSES,FRIENDLY,SAD" --out metric_results_embeddings/ --specific-metrics-csv metric_results_embeddings/no_metrics.csv --embedding-metrics --skip-glosses-with-more-than-this-many 30 2>&1|tee embed_out/in_out_trials_$(date +%s).txt


########################################
# COUNTING
# stat -c "%y" metric_results/scores/* | cut -d':' -f1 | sort | uniq -c # get the timestamps/hour
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/count_files_by_hour.py


###########################################################################
# Full Matrix
# Batch size 100, max workers 8 seems to level off at 8 workers working, 29 GB memory usage
# Batch size 100, max workers 40 seems to level off at 40 workers working, 62 GB memory usage
# batch size 50, workers 60 seems to level off at 30 workers working and 11 GB memory usage
# batch size 80, workers 60 seems to level off at 43 workers working and 15 GB memory usage... creeping up to 44 and 21 after 45 minutes
# find metric_results_full_matrix/scores/batches_startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast_asl-citizen_test/ |wc -l

# run with 40 workers, 100 batch size, and do specific metric on asl citizen test set, specifying some specific metrics
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py dataset_dfs/asl-citizen.csv --splits test --full --max-workers 40 --batch-size 100 --specific-metrics "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast" --specific-metrics "untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast" 2>&1|tee out/full_matrix$(date +%s).txt

# same as above, but instead just the "filtered" metrics with ends up being 58 + return4
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py dataset_dfs/asl-citizen.csv --splits test --full --max-workers 40 --batch-size 100 --specific-metrics "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast" --specific-metrics "untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast" 2>&1|tee out/full_matrix$(date +%s).txt

# Top metric that isn't just a defaultdistance difference
# untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py dataset_dfs/asl-citizen.csv --splits test --full --max-workers 40 --batch-size 100 --specific-metrics "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast" 2>&1|tee out/full_matrix$(date +%s).txt
# same, but training set
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py dataset_dfs/asl-citizen.csv --splits train --full --max-workers 40 --batch-size 100 --specific-metrics "untrimmed_normalizedbyshoulders_hands_defaultdist1.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast" 2>&1|tee out/full_matrix$(date +%s).txt

# #2 metric, on asl citizen test, with only 2 workers
# startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py dataset_dfs/asl-citizen.csv --splits test --full --max-workers 2 --batch-size 100 --specific-metrics "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast" 2>&1|tee out/full_matrix$(date +%s).txt
# same, on train
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py dataset_dfs/asl-citizen.csv --splits train --full --max-workers 2 --batch-size 100 --specific-metrics "startendtrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast" 2>&1|tee out/full_matrix$(date +%s).txt


# full matrix for fastest z-stretching within the top 10 by MAP, train
# 38 trials results: 0.0226587702833351 mean score time, 0.9956021308898926 MAP
# untrimmed_zspeed1.0_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast
# On 48-cpu, 96g machine, 44 workers, batch size 100, memory usage rises to about 21 GB after a few minutes a and then rises more slowly
# After 11 minutes it had done 132/160k batches. Killed it and ran again with Batch Size: 200, so that's 40000 distances per, about 40401 total batches (201x201).
# Then it levelled off around 32 GB
# train+test intersplit
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py dataset_dfs/asl-citizen.csv --splits "train,test" --full --max-workers 44 --batch-size 100 --specific-metrics "untrimmed_zspeed1.0_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast" --full-intersplit --out metric_results_full_matrix/ 2>&1|tee out/full_matrix$(date +%s).txt

# top Precision@10 metric
# Of 33450
# with about 1433 w/ > 5 trials, after 48k trials
# untrimmed_zspeed1.0_normalizedbyshoulders_removelegsandworld_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py dataset_dfs/asl-citizen.csv --splits "train,test" --full --max-workers 8 --batch-size 100 --specific-metrics "untrimmed_zspeed1.0_normalizedbyshoulders_removelegsandworld_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast" --full-intersplit --out metric_results_full_matrix/ 2>&1|tee out/full_matrix$(date +%s).txt

# top MAP metric that isn't dtw (but is zspeed)
# Of about 1433 w/ > 5 trials, after 48k trials
# excl. 'defaultdist1000.0,defaultdist100.0,defaultdist10.0,defaultdist1.0,dtw,embedding', leaving about 197
# untrimmed_zspeed100.0_normalizedbyshoulders_removelegsandworld_defaultdist0.0_interp15_zeropad_fillmasked10.0_AggregatedPowerDistanceMetric

# top MAP that isn't dtw or zspeed
# Of about 1433 w/ > 5 trials, after 48k trials
# excl. 'defaultdist1000.0,defaultdist100.0,defaultdist10.0,defaultdist1.0,dtw,zspeed,embedding',
# startendtrimmed_unnormalized_hands_defaultdist0.0_interp120_zeropad_fillmasked1.0_AggregatedPowerDistanceMetric
# running it on the 64-cpu, 30g machine: 8 workers 100 batch seems
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/load_splits_and_run_metrics.py dataset_dfs/asl-citizen.csv --splits "train,test" --full --max-workers 8 --batch-size 100 --specific-metrics "startendtrimmed_unnormalized_hands_defaultdist0.0_interp120_zeropad_fillmasked1.0_AggregatedPowerDistanceMetric" --full-intersplit --out metric_results_full_matrix/ 2>&1|tee out/full_matrix$(date +%s).txt

# monitor progress:
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && watch python pose_evaluation/evaluation/count_files_by_hour.py metric_results_full_matrix/scores/batches_untrimmed_zspeed1.0_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast_asl-citizen_train/ --target-count 40401
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && watch python pose_evaluation/evaluation/count_files_by_hour.py metric_results_full_matrix/scores/batches_untrimmed_zspeed1.0_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast_asl-citizen_testvstrain --target-count 132660
