#!/usr/bin/env python3

import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

from tqdm import tqdm

from typing import Optional, List
from collections import defaultdict


def load_dataset(dataset_dir: Path):
    # partitioning = ds.partitioning(pa.schema([("METRIC", pa.string())]))  # Add your partition column here
    dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive")
    # print(dataset.partitioning)  # pyarrow._dataset.HivePartitioning object

    # the following outputs:
    # METRIC: string
    # -- schema metadata --
    # pandas: '{"index_columns": [{"kind": "range", "name": null, "start": 0, "' + 1191
    # print(dataset.partitioning.schema)

    # the following prints a list with one list of strings in it. [[metric1, ...,metric2]]
    # print(dataset.partitioning.dictionaries)
    # print(type(dataset.partitioning.dictionaries)) # a list
    # for item in dataset.partitioning.dictionaries:
    #     print(len(item))
    # table = dataset.to_table()
    return dataset


def get_metric_partition_names(dataset):
    for partition in dataset.partitioning.dictionaries:
        # partition should be a list of metrics, for example ['metric1', 'metric2', ...]
        for metric in partition:
            yield metric


def load_metric_df(dataset, metric):
    filter_expr = pc.equal(ds.field("METRIC"), metric)
    # Filter the dataset by this metric (partition)
    metric_df = dataset.to_table(filter=filter_expr).to_pandas()
    return metric_df


def yield_metric_batches(dataset, metric):
    filter_expr = pc.equal(ds.field("METRIC"), metric)
    # Filter the dataset by this metric (partition)
    for record_batch in dataset.to_batches(filter=filter_expr):
        yield record_batch


def load_metric_dfs(dataset):
    # Load the dataset using the Hive partitioning
    # print(dataset.partitioning)  # pyarrow._dataset.HivePartitioning object

    # Print the schema of the partitioning to understand its structure
    # print(dataset.partitioning.schema)

    # Iterate over the partition dictionaries (which are the metrics in this case)
    for metric in get_metric_partition_names(dataset):
        # Create a filter expression using pyarrow.compute
        metric_df = load_metric_df(dataset, metric)

        yield metric, metric_df


def summarize_df(df):
    print("First 5 rows:")
    print(df.head())

    # Print the number of unique values per column
    print("\nUnique values per column:")
    for col in df.columns:
        # Convert column to pandas series and calculate unique values
        column_data = df[col]
        unique_count = column_data.nunique()  # Unique values count
        print(f"  - {col}: {unique_count} unique values")


def summarize_dataset_with_batches(dataset, columns: Optional[List] = None, sample_count=5):

    if columns is None:
        columns = ["METRIC", "GLOSS_A", "GLOSS_B", "GLOSS_A_PATH", "GLOSS_B_PATH"]

    print(f"Iterating over batches to find unique values for {columns}")

    # Collect unique values across all batches
    unique_values = defaultdict(set)  # key: column name, value: set of unique entries

    for i, record_batch in enumerate(tqdm(dataset.to_batches(), desc="iterating over batches")):
        df = record_batch.to_pandas()

        for col in columns:
            uniques = df[col].unique()
            # Add unique values for this column to the set
            unique_values[col].update(uniques)
        if i % 1000 == 0:
            print("*" * 60)
            for col in columns:
                col_uniques = unique_values[col]
                print(f"Column '{col}': {len(col_uniques)} unique value(s), here are the first {sample_count}")
                for val in sorted(list(col_uniques)[:sample_count]):
                    print(f"  - {val}")

    # Print summary of unique values per column
    print("$" * 60)
    for col in columns:
        col_uniques = unique_values[col]
        print(f"Column '{col}': {len(col_uniques)} unique value(s), here are the first {sample_count}")
        for val in sorted(list(col_uniques)[:sample_count]):
            print(f"  - {val}")


def summarize_metric_partitions(dataset_path):
    """
    Summarizes number of rows per METRIC partition using pyarrow's count_rows().
    Size is not included unless specifically required, as row count is most efficient.

    Parameters:
        dataset_path (str or Path): Path to the dataset root.

    Returns:
        dict: {metric_value: {'num_rows': int}}
    """
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")

    if "METRIC" not in dataset.partitioning.schema.names:
        raise ValueError("No METRIC partition found in dataset.")

    partition_stats = {}

    metric_index = dataset.partitioning.schema.names.index("METRIC")
    metric_values = dataset.partitioning.dictionaries[metric_index].to_pylist()

    for metric in metric_values:
        filter_expr = ds.field("METRIC") == metric
        num_rows = dataset.count_rows(filter=filter_expr)
        partition_stats[metric] = {"num_rows": num_rows}

    return partition_stats


def main():
    parser = argparse.ArgumentParser(description="Load and summarize a partitioned Parquet dataset.")
    parser.add_argument("dataset_dir", type=Path, help="Path to the root directory of the dataset")
    parser.add_argument("--count-metrics", action="store_true", help="Summarize by column")
    parser.add_argument("--summarize_cols", action="store_true", help="Summarize by column")
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f"Error: directory {args.dataset_dir} does not exist.")
        return

    dataset = load_dataset(args.dataset_dir)

    metric_names = list(get_metric_partition_names(dataset))

    if args.count_metrics:
        metric_stats = summarize_metric_partitions(args.dataset_dir)
        for metric, metric_stats in metric_stats.items():
            print(f"{metric}: {metric_stats['num_rows']} rows")

    # OOM KILLED
    # for metric, metric_df in load_metric_dfs(dataset):
    #     print(f"Metric {metric} has {len(metric_df):,} rows")
    #     summarize_df(metric_df)
    if args.summarize_cols:
        summarize_dataset_with_batches(dataset)


if __name__ == "__main__":
    main()


# conda activate /opt/home/cleong/envs/pose_eval_src && python pose_evaluation/evaluation/load_pyarrow_dataset.py metric_results_full_matrix/pyarrow_dataset/semlex/
