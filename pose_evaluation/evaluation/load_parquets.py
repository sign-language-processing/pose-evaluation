#!/usr/bin/env python3
import argparse
from typing import List, Optional, Union
from pathlib import Path
import random

import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc


def merge_parquet_files(
    input_paths: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
    partition_columns: Optional[List[str]] = None,
) -> None:
    """
    Merge one or more Parquet files into a new dataset directory,
    optionally partitioned by given column names.

    Args:
        input_paths (str | Path | list): Parquet file path(s) or directory.
        output_dir (str | Path): Output directory for merged dataset.
        partition_columns (list, optional): Column names to partition by.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Merging {len(input_paths)} to {output_dir} with partitions: {partition_columns}")

    if isinstance(input_paths, (str, Path)):
        input_path = Path(input_paths)
        if input_path.is_dir():
            dataset = ds.dataset(str(input_path), format="parquet")
        else:
            dataset = ds.dataset([str(input_path)], format="parquet")
    else:
        input_paths = [str(Path(p)) for p in input_paths]

        dataset = ds.dataset(input_paths, format="parquet")

    # pull the splits out

    ds.write_dataset(
        data=dataset,
        base_dir=str(output_dir),
        format="parquet",
        partitioning=partition_columns or None,
        existing_data_behavior="overwrite_or_ignore",
    )

    print(f"Merged {len(dataset.files)} files into {output_dir.resolve()}")


def count_rows(parquets: list[Path]):
    row_count = 0
    for i, p in enumerate(tqdm(parquets, desc="loading files")):
        row_count += pq.ParquetFile(p).metadata.num_rows
        if (i + 1) % 1000 == 0 and i > 0:
            print(f"Read {i+1:,} files, counted {row_count:,} rows so far, or {row_count/(i+1):,} on average")

    print(f"Total row count: {row_count:,}, or on average {row_count/len(parquets)}")


def load_and_summarize(df):
    # print("\nDataFrame Info:")
    # df.info()

    # print("\nDataFrame Describe:")
    # print(df.describe(include="all"))

    print("\nDataFrame Head:")
    print(df.head())

    unique_metrics = df["METRIC"].unique()
    print(f"{len(unique_metrics)} METRICS: {unique_metrics}")
    print(f"METRICS: {len(df['METRIC'].unique())}")
    out_folder = Path("metric_distances_partial")
    out_folder.mkdir(exist_ok=True)

    for metric in unique_metrics:
        metric_df = df[df["METRIC"] == metric]
        print(f"{metric}")
        print(f"*\tSCORES: {len(metric_df['SCORE']):,}")

        print(f"*\tQUERY GLOSSES: {len(metric_df['GLOSS_A'].unique())}")
        # print(df["GLOSS_A"].unique())

        print(f"*\tREF GLOSSES: {len(metric_df['GLOSS_B'].unique())}")

        # print(df["GLOSS_B"].unique())

        # print(f"LOWEST SCORES")
        # metric_df = metric_df.sort_values(by="SCORE", ascending=True)
        # print(metric_df[["SCORE","GLOSS_A","GLOSS_B"]].head())


def get_unique_values(input_paths: List[Union[str, Path]], column_names: List[str]) -> dict:
    """
    Returns a dictionary of unique values for each specified column across the given Parquet files.

    Parameters:
        input_paths (List[Union[str, Path]]): List of paths to Parquet files.
        column_names (List[str]): List of column names to extract unique values from.

    Returns:
        dict: A dictionary where each key is a column name and the value is a set of unique values.
    """
    unique_values = {col: set() for col in column_names}
    dataset = ds.dataset([str(p) for p in input_paths], format="parquet")

    for batch in tqdm(dataset.to_batches(columns=column_names), total=len(input_paths), desc="Loading batches"):
        table = batch.to_pandas()
        for col in column_names:
            unique_values[col].update(table[col].dropna().unique())

    return {k: sorted(v) for k, v in unique_values.items()}


def summarize_columns(input_paths):

    unique_values = get_unique_values(input_paths, ["METRIC", "GLOSS_A", "GLOSS_A_PATH", "GLOSS_B", "GLOSS_B_PATH"])
    for column, values in unique_values.items():
        print(f"{column} has {len(values)} unique values")
        for value in values[:10]:
            print(f"*\t{value}")


def main():
    parser = argparse.ArgumentParser(description="Inspect a metric results file.")
    parser.add_argument(
        "path",
        type=Path,
        # default="full_matrix_Return4Metric_on_asl-citizen_test_score_results.parquet",
        help="Filename to load, or folder to search for parquet files",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="How many files to load if loading from a dir",
    )

    parser.add_argument(
        "--count",
        action="store_true",
        help="whether to count",
    )

    parser.add_argument(
        "--count-unique",
        action="store_true",
        help="whether to count unique values for the columns",
    )

    parser.add_argument(
        "--merge-dir",
        type=Path,
        default=None,
        help="If given, will merge parquets and put them here",
    )

    args = parser.parse_args()
    filepath = args.path

    if not filepath.exists():
        print(f"Error: File not found at {filepath}")
        return

    row_count = 0
    parquets = []
    print(f"Loading Path: {filepath} with name {filepath.name}")
    if filepath.name.endswith(".parquet"):
        parquets.append(filepath)
    elif filepath.is_dir():
        parquets = list(filepath.rglob("*.parquet"))
        print(f"Found {len(parquets)} parquet files")
        if args.max_files is not None:
            random.shuffle(parquets)
            print(f"loading {args.max_files} at random")

            parquets = parquets[: args.max_files]

    else:
        print(f"Unsupported file extension: {filepath}")
        return

    if args.count:
        count_rows(parquets)

    if args.count_unique:
        summarize_columns(parquets)

    if args.merge_dir is not None:
        merge_dir = args.merge_dir
        merge_dir.mkdir(exist_ok=True, parents=True)
        merge_parquet_files(parquets, merge_dir, ["METRIC", "GLOSS_A"])


if __name__ == "__main__":
    main()
# python pose_evaluation/evaluation/load_parquets.py metric_results_full_matrix/scores/batches_untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast_asl-citizen_test/
