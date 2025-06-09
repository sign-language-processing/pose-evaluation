#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm


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
        yield from partition


def load_metric_df(dataset, metric):
    filter_expr = pc.equal(ds.field("METRIC"), metric)
    # Filter the dataset by this metric (partition)
    metric_df = dataset.to_table(filter=filter_expr).to_pandas()
    return metric_df


def yield_metric_batches(dataset, metric):
    filter_expr = pc.equal(ds.field("METRIC"), metric)
    # Filter the dataset by this metric (partition)
    yield from dataset.to_batches(filter=filter_expr)


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
    # print("First 5 rows:")
    # print(df.head())

    # Print the number of unique values per column
    print("\nUnique values per column:")
    for col in df.columns:
        # Convert column to pandas series and calculate unique values
        column_data = df[col]
        unique_count = column_data.nunique()  # Unique values count
        print(f"  - {col}: {unique_count} unique values")
    total_bytes = df.memory_usage(deep=True).sum()
    print(f"{total_bytes:,} b")
    print(f"{total_bytes / 1024:,} kb")
    print(f"{total_bytes / (1024**2):,} mb")


def summarize_dataset_with_batches(dataset, columns: list | None = None, sample_count=5):
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
    Summarizes number of rows per METRIC partition using pyarrow's
    count_rows(). Size is not included unless specifically required, as row
    count is most efficient.

    Parameters
    ----------
        dataset_path (str or Path): Path to the dataset root.

    Returns
    -------
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


def get_unique_values_for_metric_column(dataset, metric: str, column: str) -> set:
    """
    Efficiently loads unique values from a single column for a given metric.

    Parameters
    ----------
    - dataset: a pyarrow.dataset.Dataset object, already opened
    - metric: the metric name (partition value) to filter on
    - column: the name of the column to extract unique values from

    Returns
    -------
    - A Python set of unique values in that column for the given metric

    """
    filter_expr = pc.equal(ds.field("METRIC"), metric)
    scanner = dataset.scanner(filter=filter_expr, columns=[column])
    table = scanner.to_table()
    return set(table[column].to_pylist())


def get_common_gloss_paths_across_metrics(
    dataset, column_a: str = "GLOSS_A_PATH", column_b: str = "GLOSS_B_PATH"
) -> tuple[set[str], set[str]]:
    """
    Computes the intersection of unique GLOSS_A_PATH and GLOSS_B_PATH values
    across all metrics in the hive-partitioned dataset.

    Parameters
    ----------
    - dataset: pyarrow.dataset.Dataset object
    - column_a: name of the Gloss A column (default: 'GLOSS_A_PATH')
    - column_b: name of the Gloss B column (default: 'GLOSS_B_PATH')

    Returns
    -------
    - (common_gloss_a_paths, common_gloss_b_paths): tuple of sets

    """
    common_a_paths = None
    common_b_paths = None
    metric_names = list(get_metric_partition_names(dataset))

    for i, metric in tqdm(enumerate(metric_names), desc="Finding common a-paths, b-paths"):
        a_paths = get_unique_values_for_metric_column(dataset, metric, column_a)
        b_paths = get_unique_values_for_metric_column(dataset, metric, column_b)

        if common_a_paths is None:
            common_a_paths = a_paths
            common_b_paths = b_paths
        else:
            common_a_paths &= a_paths
            common_b_paths &= b_paths

        print(f"Metric #{i}/{len(metric_names)} {metric} → {len(a_paths):,} A-paths, {len(b_paths):,} B-paths")

    print(f"\nCommon A-paths: {len(common_a_paths):,}")
    print(f"Common B-paths: {len(common_b_paths):,}")
    return common_a_paths, common_b_paths


def load_filtered_metric_df(
    dataset,
    metric: str,
    common_a_paths: set,
    common_b_paths: set,
    column_a: str = "GLOSS_A_PATH",
    column_b: str = "GLOSS_B_PATH",
) -> pd.DataFrame:
    """
    Loads a filtered DataFrame for a given metric, including only rows where
    both GLOSS_A_PATH and GLOSS_B_PATH are in the common sets.

    Parameters
    ----------
    - dataset: pyarrow.dataset.Dataset object
    - metric: metric partition name
    - common_a_paths: set of allowed GLOSS_A_PATH values
    - common_b_paths: set of allowed GLOSS_B_PATH values
    - column_a: name of the Gloss A column (default: 'GLOSS_A_PATH')
    - column_b: name of the Gloss B column (default: 'GLOSS_B_PATH')

    Returns
    -------
    - Filtered pandas DataFrame

    """
    # Build filter expressions
    metric_filter = pc.equal(ds.field("METRIC"), metric)
    gloss_a_filter = ds.field(column_a).isin(pa.array(list(common_a_paths)))
    gloss_b_filter = ds.field(column_b).isin(pa.array(list(common_b_paths)))

    combined_filter = metric_filter & gloss_a_filter & gloss_b_filter

    # Create scanner and convert to pandas DataFrame
    scanner = dataset.scanner(filter=combined_filter)
    return scanner.to_table().to_pandas()


def save_filtered_metrics(dataset, metric_names, out_path, common_a, common_b):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, metric in enumerate(metric_names):
        print(f"[{i + 1}/{len(metric_names)}] Processing {metric}...")

        # Load filtered DataFrame
        metric_df = load_filtered_metric_df(dataset, metric, common_a, common_b)

        if metric_df.empty:
            print(f"  -> Skipping {metric} (no rows after filtering)")
            continue

        # Ensure 'METRIC' column exists (in case it's dropped during filtering)
        if "METRIC" not in metric_df.columns:
            metric_df["METRIC"] = metric

        # Convert to PyArrow Table
        metric_table = pa.Table.from_pandas(metric_df, preserve_index=False)

        # Write as partitioned Parquet (Hive-style: partition by METRIC)
        pq.write_to_dataset(
            metric_table,
            root_path=str(out_path),
            partition_cols=["METRIC"],
            existing_data_behavior="overwrite_or_ignore",
        )

        print(f"  -> Wrote {len(metric_df)} rows to {out_path / f'METRIC={metric}'}")


def main():
    parser = argparse.ArgumentParser(description="Load and summarize a partitioned Parquet dataset.")
    parser.add_argument("dataset_dir", type=Path, help="Path to the root directory of the dataset")
    parser.add_argument(
        "--comparable-out",
        type=Path,
        default=Path(
            "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4_pruned_to_match_embeddings/pyarrow_scores_comparable/"
        ),
        help="If supplied, will find commmon gloss a/b paths and output only scores that are common to all metrics, here",
    )
    parser.add_argument("--count-metrics", action="store_true", help="Summarize by column")
    parser.add_argument("--summarize_cols", action="store_true", help="Summarize by column")

    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f"Error: directory {args.dataset_dir} does not exist.")
        return

    dataset = load_dataset(args.dataset_dir)

    metric_names = list(get_metric_partition_names(dataset))

    if args.count_metrics:
        metric_stats_summary = summarize_metric_partitions(args.dataset_dir)
        for metric, metric_stats in metric_stats_summary.items():
            print(f"{metric}: {metric_stats['num_rows']} rows")

    # for i, metric in enumerate(metric_names):
    # print(i, metric)
    # gloss_a_path_set = get_unique_values_for_metric_column(dataset, metric, "GLOSS_A_PATH")
    # gloss_b_path_set = get_unique_values_for_metric_column(dataset, metric, "GLOSS_B_PATH")
    # score_set = get_unique_values_for_metric_column(dataset, metric, "SCORE")
    # print(f"GLOSS_A_PATH has {len(gloss_a_path_set)}")
    # print(f"GLOSS_B_PATH has {len(gloss_b_path_set)}")

    # Common A: 2325
    # Common B: 3413
    if args.comparable_out is not None:
        print(f"Comparable dataset will be output to {args.comparable_out}")
        common_a, common_b = get_common_gloss_paths_across_metrics(dataset)
        print(f"Common A: {len(common_a)}")
        print(f"Common B: {len(common_b)}")

        # out_path = "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4_pruned_to_match_embeddings/pyarrow_scores_comparable/"
        out_path = args.comparable_out

        save_filtered_metrics(dataset, metric_names, out_path, common_a, common_b)

    # for i, metric in enumerate(metric_names):
    #     filtered_metric_df = load_filtered_metric_df(dataset, metric, common_a, common_b)
    #     # summarize_df(filtered_metric_df)
    #     print(f"{len(filtered_metric_df['GLOSS_A_PATH'].unique()):,} Gloss A paths")
    #     print(f"{len(filtered_metric_df['GLOSS_B_PATH'].unique()):,} Gloss B paths")

    # save back out to a pyarrow dataset, partitioned by "METRIC", e.g. /opt/home/cleong/projects/pose-evaluation/metric_results_round_4_pruned_to_match_embeddings/scores/pyarrow_scores_comparable/'METRIC=EmbeddingDistanceMetric_pop_sign_finetune_checkpoint_best_cosine'/

    # OOM KILLED
    # for metric, metric_df in load_metric_dfs(dataset):
    #     print(f"Metric {metric} has {len(metric_df):,} rows")
    #     print(f"{len(metric_df['GLOSS_A_PATH'].unique()):,} Gloss A paths")
    #     print(f"{len(metric_df['GLOSS_B_PATH'].unique()):,} Gloss B paths")
    #     print(f"{len(metric_df['SCORE'].unique()):,} Scores")

    # summarize_df(metric_df)
    if args.summarize_cols:
        summarize_dataset_with_batches(dataset)


if __name__ == "__main__":
    main()


# conda activate /opt/home/cleong/envs/pose_eval_src && python pose_evaluation/evaluation/load_pyarrow_dataset.py metric_results_full_matrix/pyarrow_dataset/semlex/
