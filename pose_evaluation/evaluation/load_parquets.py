#!/usr/bin/env python3
import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol


def parse_dataset_split(folder_name: str):
    parts = folder_name.split("_")
    if len(parts) >= 2:
        split = parts[-1]
        dataset = parts[-2]
        # dataset = "_".join(parts[:-1])
    else:
        dataset = folder_name
        split = "unknown"
    return dataset, split


def merge_parquet_files(
    input_paths: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
    partition_columns: Optional[List[str]] = None,
    parse_parent_for_datasets_and_splits: bool = True,
) -> None:
    """
    Merge one or more Parquet files into a new dataset directory,
    optionally partitioned by given column names.

    Args:
        input_paths (str | Path | list): Parquet file path(s) or directory.
        output_dir (str | Path): Output directory for merged dataset.
        partition_columns (list, optional): Column names to partition by.
    """
    print(f"** MERGE TO PYARROW DATASET **")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize input paths
    if isinstance(input_paths, (str, Path)):
        input_path = Path(input_paths)
        if input_path.is_dir():
            input_paths = list(input_path.rglob("*.parquet"))
        else:
            input_paths = [input_path]
    else:
        input_paths = [Path(p) for p in input_paths]

    print(f"Merging {len(input_paths)} files to {output_dir} with partitions: {partition_columns}")

    # Group by dataset/split if enabled
    grouped_paths = defaultdict(list)
    for path in input_paths:
        if parse_parent_for_datasets_and_splits:
            parent = path.parent.name
            dataset, split = parse_dataset_split(parent)
        else:
            dataset, split = "unknown", "unknown"
        grouped_paths[(dataset, split)].append(path)

    # Iterate over grouped splits and write each
    for (dataset, split), files in grouped_paths.items():

        print(f"Preparing dataset object for dataset={dataset}, split={split}, ({len(files)} files...)")
        dataset_obj = ds.dataset(files, format="parquet")

        target_dir = output_dir / f"{dataset}" / f"{split}"
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"Writing data from {len(files):,} files for dataset={dataset}, split={split}")
        if partition_columns:
            for col in partition_columns:
                if col not in dataset_obj.schema.names:
                    raise ValueError(f"Partition column '{col}' missing from dataset.")

            partition_schema = pa.schema([(col, dataset_obj.schema.field(col).type) for col in partition_columns])
            partitioning = ds.partitioning(partition_schema, flavor="hive")
            # partitioning = ds.partitioning(pa.schema([(ScoreDFCol.METRIC, pa.string())]), flavor="hive")
        else:
            partitioning = None

        ds.write_dataset(
            data=dataset_obj,
            base_dir=target_dir,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
            file_options=ds.ParquetFileFormat().make_write_options(compression="snappy"),
        )

        print(f"✓ Wrote data from {len(files)} files to {target_dir.resolve()}")
        parquet_files = list(Path(target_dir).rglob("*.parquet"))
        print(f"Number of Parquet files written: {len(parquet_files)}")
        partition_dirs = [p for p in Path(target_dir).rglob("*") if p.is_dir()]
        print(f"Number of partition directories: {len(partition_dirs)}")


def count_rows(parquets: list[Path]):
    print(f"** COUNTING ROWS **")
    row_count = 0
    for i, p in enumerate(tqdm(parquets, desc="loading files")):
        row_count += pq.ParquetFile(p).metadata.num_rows
        if (i + 1) % 1000 == 0 and i > 0:
            print(f"Read {i+1:,} files, counted {row_count:,} rows so far, or {row_count/(i+1):,} on average")

    print(f"Total row count: {row_count:,}, or on average {row_count/len(parquets):,} across {len(parquets)} files")


def load_and_merge(
    parquets: List[Path],
    dedupe: bool = True,
    output_path: Optional[Path] = None,
) -> pa.Table:
    """
    Load and merge multiple Parquet files into a single in-memory pyarrow Table.
    Optionally deduplicate the rows and save the result to a Parquet file.

    Args:
        parquets (List[Path]): List of Parquet file paths.
        dedupe (bool): Whether to deduplicate rows after merging.
        output_path (Optional[Path]): If provided, saves the merged table to this path.

    Returns:
        pyarrow.Table: Merged (and optionally deduplicated) in-memory table.
    """
    # NOTE: dies if you do 6k parquet files...
    print(f"** LOAD AND MERGE **")
    print(f"[INFO] Loading {len(parquets)} parquet file(s)...")
    tables = []

    for i, path in tqdm(enumerate(parquets), desc="Reading parquet files", total=len(parquets)):
        table = pq.read_table(path)
        # cast to float for consistency
        if ScoreDFCol.SCORE in table.column_names:
            table = table.set_column(
                table.schema.get_field_index(ScoreDFCol.SCORE),
                ScoreDFCol.SCORE,
                table.column(ScoreDFCol.SCORE).cast(pa.float64()),
            )
        tables.append(table)

    merged_table = pa.concat_tables(tables, promote_options="default")
    print(f"[INFO] Merged table has {merged_table.num_rows:,} rows before deduplication")

    if dedupe:
        df = merged_table.to_pandas()
        before = len(df)
        df = df.drop_duplicates(subset=[ScoreDFCol.METRIC, ScoreDFCol.GLOSS_A_PATH, ScoreDFCol.GLOSS_B_PATH])
        after = len(df)
        print(f"[INFO] Dropped {before - after:,} duplicate rows")
        merged_table = pa.Table.from_pandas(df)
        print(f"[INFO] Table has {merged_table.num_rows:,} rows after deduplication")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partition_schema = pa.schema([(ScoreDFCol.METRIC, pa.string())])

        ds.write_dataset(
            data=merged_table,
            base_dir=output_path,
            format="parquet",
            partitioning=ds.partitioning(partition_schema, flavor="hive"),
            existing_data_behavior="overwrite_or_ignore",
            file_options=ds.ParquetFileFormat().make_write_options(compression="snappy"),
        )
        print(f"[INFO] Saved merged table to {output_path.resolve()}")

    return merged_table


def get_unique_values(input_paths: List[Union[str, Path]], column_names: Optional[List[str]] = None) -> dict:
    """
    Returns a dictionary of unique values for each specified column across the given Parquet files.

    Parameters:
        input_paths (List[Union[str, Path]]): List of paths to Parquet files.
        column_names (List[str]): List of column names to extract unique values from.

    Returns:
        dict: A dictionary where each key is a column name and the value is a set of unique values.
    """
    unique_values = defaultdict(set)
    dataset = ds.dataset([str(p) for p in input_paths], format="parquet")

    for batch in tqdm(dataset.to_batches(columns=column_names), total=len(input_paths), desc="Loading batches"):
        table = batch.to_pandas()

        for col in table.columns:
            if column_names is None or col in column_names:
                unique_values[col].update(table[col].dropna().unique())
    return {k: sorted(v) for k, v in unique_values.items()}


def summarize_columns(input_paths, sample_count=10, column_names=None):
    print("** COUNT UNIQUES BY COLUMN **")
    unique_values = get_unique_values(
        input_paths,
        column_names=column_names,
    )
    for column, values in unique_values.items():
        random.shuffle(values)
        print(f"{column} has {len(values)} unique values, here are a few at random")
        for value in values[:sample_count]:
            print(f"*\t{value}")


def print_sampled_preview(table, num_rows=5, max_val_len=500):
    print("Sampled rows preview:")
    for i in range(min(num_rows, table.num_rows)):
        row = table.slice(i, 1)
        row_dict = {col: row.column(col_idx)[0].as_py() for col_idx, col in enumerate(table.column_names)}
        print(f"\nRow {i}:")
        for key, val in row_dict.items():
            val_str = str(val)
            if len(val_str) > max_val_len:
                val_str = val_str[: max_val_len - 3] + "..."
            print(f"  {key:<20} : {val_str}")


# https://medium.com/pythoneers/dipping-into-data-streams-the-magic-of-reservoir-sampling-762f41b78781
def sample_rows(input_paths, sample_count=100, column_names=None, shuffle=True):
    print("** Sample Rows **")
    if shuffle:
        random.shuffle(input_paths)
    dataset = ds.dataset([str(p) for p in input_paths], format="parquet")

    reservoir = []
    total_seen = 0

    for batch in tqdm(dataset.to_batches(columns=column_names), desc="Sampling rows"):
        for i in range(batch.num_rows):
            row_batch = batch.slice(i, 1)
            row_table = pa.Table.from_batches([row_batch])
            total_seen += 1
            if len(reservoir) < sample_count:
                reservoir.append(row_table)
            else:
                j = random.randint(0, total_seen - 1)
                if j < sample_count:
                    reservoir[j] = row_table

    sampled_table = pa.concat_tables(reservoir)

    return sampled_table


def head(input_paths, n=100, column_names=None):
    print(f"** Head: n={n} **")
    dataset = ds.dataset([str(p) for p in input_paths], format="parquet")
    scanner = dataset.scanner(columns=column_names)
    return scanner.head(n)


def main():

    default_score_columns = ",".join(
        [value for name, value in vars(ScoreDFCol).items() if not name.startswith("__") and not callable(value)]
    )
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
        "--sample-columns",
        type=str,
        default=default_score_columns,
        help=f"comma-separated list of columns to sample, or '' for all. Default:{default_score_columns}",
    )

    parser.add_argument(
        "--count-unique",
        action="store_true",
        help="whether to count unique values for the columns",
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        help="Whether to print a few random rows",
    )
    parser.add_argument(
        "--head",
        type=int,
        help="print this many rows",
    )

    parser.add_argument(
        "--merge-dir",
        type=Path,
        default=None,
        help="If given, will merge parquets and put them here",
    )

    parser.add_argument(
        "--load-and-merge",
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

    if args.sample_columns:
        sample_columns = args.sample_columns.split(",")
    else:
        sample_columns = None

    if args.count:
        count_rows(parquets)

    if args.count_unique:
        summarize_columns(parquets, column_names=sample_columns)

    if args.sample_rows:
        sampled_table = sample_rows(parquets, sample_count=args.sample_rows, column_names=sample_columns)
        print_sampled_preview(sampled_table, num_rows=args.sample_rows)

    if args.head:
        sampled_table = head(parquets, n=args.head, column_names=sample_columns)
        print_sampled_preview(sampled_table, num_rows=args.head)

    if args.merge_dir is not None:
        merge_dir = args.merge_dir
        merge_dir.mkdir(exist_ok=True, parents=True)
        # don't do GLOSS_A, results in millions of files.
        merge_parquet_files(parquets, merge_dir, [ScoreDFCol.METRIC])

    if args.load_and_merge is not None:
        load_and_merge(parquets, dedupe=True, output_path=args.load_and_merge)


if __name__ == "__main__":
    main()
# python pose_evaluation/evaluation/load_parquets.py metric_results_full_matrix/scores/batches_untrimmed_unnormalized_hands_defaultdist10.0_nointerp_dtw_fillmasked1.0_dtaiDTWAggregatedDistanceMetricFast_asl-citizen_test/

# merge full_matrix_scores to datasets
# cd /opt/home/cleong/projects/pose-evaluation/ && conda activate /opt/home/cleong/envs/pose_eval_src && python pose_evaluation/evaluation/load_parquets.py /opt/home/cleong/projects/pose-evaluation/metric_results_full_matrix/scores/ --merge-dir /opt/home/cleong/projects/pose-evaluation/metric_results_full_matrix/pyarrow_dataset
