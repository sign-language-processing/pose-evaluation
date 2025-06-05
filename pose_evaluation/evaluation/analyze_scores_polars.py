import polars as pl
from pathlib import Path
import argparse
import polars as pl
from pathlib import Path

import polars as pl


import time
import psutil
import os
import statistics


def run_experiments(path: Path, limit_rows: int = 100_000_000, runs: int = 10):
    streaming_stats = []
    non_streaming_stats = []

    print(f"\n=== 🚀 Running {runs} streaming runs ===")
    for i in range(runs):
        print(f"\n🔁 Streaming Run {i + 1}/{runs}")
        _, mem_used, duration = compute_map_by_metric_with_profiling(path, limit_rows=limit_rows, streaming=True)
        streaming_stats.append((mem_used, duration))

    print(f"\n=== 🚀 Running {runs} non-streaming runs ===")
    for i in range(runs):
        print(f"\n🔁 Non-Streaming Run {i + 1}/{runs}")
        _, mem_used, duration = compute_map_by_metric_with_profiling(path, limit_rows=limit_rows, streaming=False)
        non_streaming_stats.append((mem_used, duration))

    def summarize(name, stats):
        mems, times = zip(*stats)
        print(f"\n📊 Summary: {name}")
        print(
            f"  → Memory (MB): mean = {statistics.mean(mems):.2f}, std = {statistics.stdev(mems):.2f}, min = {min(mems):.2f}, max = {max(mems):.2f}"
        )
        print(
            f"  → Time (s):    mean = {statistics.mean(times):.2f}, std = {statistics.stdev(times):.2f}, min = {min(times):.2f}, max = {max(times):.2f}"
        )

    summarize("Streaming", streaming_stats)
    summarize("Non-Streaming", non_streaming_stats)


def compute_map_by_metric_with_profiling(
    path: Path, limit_rows: int = 10_000, streaming: bool = False
) -> pl.DataFrame | None:
    """
    Load a limited number of rows from a LazyFrame and compute mean average precision (mAP) per METRIC.
    Includes profiling for memory usage and execution time.

    Parameters:
    - path: Path to a Parquet file.
    - limit_rows: Number of rows to load (default 10,000).
    - streaming: Whether to use Polars streaming engine.

    Returns:
    - Polars DataFrame with columns: METRIC, mAP
    """
    print(f"⏳ Scanning first {limit_rows:,} rows from {path}...")

    lf = pl.scan_parquet(path).limit(limit_rows)

    # Step 1: Add relevance column
    lf = lf.with_columns([(pl.col("GLOSS_A") == pl.col("GLOSS_B")).cast(pl.Int8).alias("relevant")])

    # Step 2: Sort by METRIC, GLOSS_A_PATH, SCORE
    lf = lf.sort(["METRIC", "GLOSS_A_PATH", "SCORE"])

    # Step 3: Rank candidates per query within each METRIC
    lf = lf.with_columns([pl.col("SCORE").rank("dense").over(["METRIC", "GLOSS_A_PATH"]).alias("rank")])

    # Step 4: Compute precision@k for relevant items
    lf = lf.with_columns(
        [(pl.col("relevant").cum_sum().over(["METRIC", "GLOSS_A_PATH"]) / pl.col("rank")).alias("precision_at_k")]
    )

    # Step 5: Compute average precision per query
    avg_precision = (
        lf.filter(pl.col("relevant") == 1)
        .group_by(["METRIC", "GLOSS_A_PATH"])
        .agg(pl.col("precision_at_k").mean().alias("average_precision"))
    )

    # Step 6: Compute mean average precision per METRIC
    map_by_metric = avg_precision.group_by("METRIC").agg(pl.col("average_precision").mean().alias("mAP"))

    # Print plans
    # print("🧠 === Logical Plan ===")
    # print(map_by_metric.explain(optimized=False))

    engine = "streaming" if streaming else "auto"
    # print(f"\n⚙️ === Optimized Plan (engine: {engine}) ===")
    # print(map_by_metric.explain(engine=engine, optimized=True))

    # Profile memory and execution time using psutil
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1e6
    start_time = time.perf_counter()

    try:
        result = map_by_metric.collect(engine=engine)
    except Exception as e:
        print(f"❌ Streaming execution failed: {e}")
        return None

    mem_after = process.memory_info().rss / 1e6
    duration = time.perf_counter() - start_time

    print(f"\n✅ Result:\n{result}")
    print(f"\n📊 Memory used (RSS): {mem_after - mem_before:.2f} MB")
    print(f"⏱️ Execution time: {duration:.2f} seconds")

    return result, mem_after - mem_before, duration


def show_head(lf: pl.LazyFrame, n=5):
    head = lf.head(5)

    print("=== head Logical Plan ===")
    print(head.explain(optimized=False))

    print("\n=== head Optimized Plan ===")
    print(head.explain(engine="streaming", optimized=True))

    print("\n=== head ===")
    print(head.collect(engine="streaming"))


def compute_score_by_metric(lf: pl.LazyFrame):

    print(lf.schema)

    lazy_result = lf.group_by("METRIC").agg(pl.col("SCORE").mean().alias("mean_SCORE"))

    # Print logical and optimized plans
    print("=== Logical Plan ===")
    print(lazy_result.explain(optimized=False))

    print("\n=== Optimized Plan ===")
    print(lazy_result.explain(engine="streaming", optimized=True))

    # Now trigger execution
    result = lazy_result.collect(engine="streaming").sort("METRIC")

    print("\n=== Result ===")
    print(result)


def compute_map_by_metric_safe(lf: pl.LazyFrame, limit_rows: None | int = None) -> pl.LazyFrame:
    """
    Computes mean average precision (mAP) per METRIC from a Polars LazyFrame.
    Lower scores are better (ascending rank order).
    Fixed to match TorchMetrics exactly - replicates TorchMetrics' internal ranking logic.
    """
    start_time = time.perf_counter()
    if limit_rows is not None:
        lf = lf.limit(limit_rows)
        print(f"⏳ Scanning first {limit_rows:,} rows")

    # Filter out self-scores
    lf = lf.filter(pl.col("GLOSS_A_PATH") != pl.col("GLOSS_B_PATH"))

    # Add relevant flag
    lf = lf.with_columns([(pl.col("GLOSS_A") == pl.col("GLOSS_B")).cast(pl.Int8).alias("relevant")])

    # Add random column for shuffling (to match the sample(frac=1, random_state=42))
    # Then sort by shuffle order first, then by score
    lf = (
        lf.with_columns([pl.int_range(pl.len()).shuffle(seed=42).alias("shuffle_order")])
        .sort(["METRIC", "GLOSS_A_PATH", "shuffle_order"])
        .sort(["METRIC", "GLOSS_A_PATH", "SCORE"], maintain_order=True)
    )

    # Now we need to replicate TorchMetrics' ranking behavior
    # TorchMetrics expects higher scores = better, so it would rank -SCORE in descending order
    # Since we have lower scores = better, we rank SCORE in ascending order
    # BUT we need to use TorchMetrics' dense ranking logic

    # The key difference: TorchMetrics uses the ORIGINAL order after shuffling as tiebreaker
    # Let's add a position column to handle ties properly
    lf = lf.with_columns([pl.int_range(pl.len()).over(["METRIC", "GLOSS_A_PATH"]).alias("position")])

    # Rank by score, with position as tiebreaker (mimicking TorchMetrics' behavior)
    lf = lf.with_columns(
        [
            pl.col("SCORE").rank("dense", descending=False).over(["METRIC", "GLOSS_A_PATH"]).alias("rank_score"),
            # Also need ordinal rank for precision calculation
            pl.col("SCORE").rank("ordinal", descending=False).over(["METRIC", "GLOSS_A_PATH"]).alias("rank_ordinal"),
        ]
    )

    # Sort by rank to ensure proper order for cumulative sum
    lf = lf.sort(["METRIC", "GLOSS_A_PATH", "rank_ordinal"])

    # Calculate cumulative sum of relevant items up to each position
    lf = lf.with_columns([pl.col("relevant").cum_sum().over(["METRIC", "GLOSS_A_PATH"]).alias("relevant_count")])

    # Calculate precision_at_k only for relevant items
    # Use the ordinal rank for precision calculation (this matches TorchMetrics)
    lf = lf.with_columns(
        [
            pl.when(pl.col("relevant") == 1)
            .then(pl.col("relevant_count") / pl.col("rank_ordinal"))
            .otherwise(None)
            .alias("precision_at_k")
        ]
    )

    # Average precision per query
    avg_precision = (
        lf.filter(pl.col("precision_at_k").is_not_null())
        .group_by(["METRIC", "GLOSS_A_PATH"])
        .agg(pl.col("precision_at_k").mean().alias("average_precision"))
    )

    # Mean average precision per metric
    result = (
        avg_precision.group_by("METRIC")
        .agg(pl.col("average_precision").mean().alias("mAP"))
        .collect(engine="streaming")
    )

    duration = time.perf_counter() - start_time
    print(f"⏱️ Execution time: {duration:.2f} seconds")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean SCORE for each METRIC partition.")
    parser.add_argument("path", type=Path, help="Path to the METRIC=hive-partitioned Parquet dataset.")
    args = parser.parse_args()

    lf = pl.scan_parquet(args.path)
    show_head(lf)

    # run_experiments(args.path)

    compute_score_by_metric(lf)

    result = compute_map_by_metric_memory_efficient(pl.scan_parquet(args.path), limit_rows=300_000_000)
    print(result)

    for row in result.iter_rows():
        print(row)

    # https://realpython.com/polars-lazyframe/#how-lazyframes-cope-with-large-volumes-of-data
    # map_query_plan = compute_map_by_metric(lf)

    # # result_df = map_lazy_result.collect(engine="streaming")
    # preview_df = lf.head(1_000_000).collect(engine="streaming")

    # print(preview_df)
    # print(type(preview_df))
# conda activate /opt/home/cleong/envs/pose_eval_src_polars && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/analyze_scores_polars.py metric_results_full_matrix/pyarrow_200_small/asl-citizen/testvstrain/
# /opt/home/cleong/projects/pose-evaluation/metric_results_full_matrix/pyarrow_200_small/asl-citizen/testvstrain/METRIC=startendtrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast/part-0.parquet
