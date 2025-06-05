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


def compute_map_by_metric_safe_chunked(
    lf: pl.LazyFrame, limit_rows: None | int = None, chunk_size: int = 1000000
) -> pl.LazyFrame:
    """
    Memory-efficient version that processes data in chunks by METRIC to avoid OOM.
    """
    import time

    start_time = time.perf_counter()
    if limit_rows is not None:
        lf = lf.limit(limit_rows)
        print(f"⏳ Scanning first {limit_rows:,} rows")

    # Get unique metrics first
    metrics = lf.select("METRIC").unique().collect(engine="streaming")["METRIC"].to_list()
    print(f"📊 Processing {len(metrics)} metrics")

    results = []

    for metric in metrics:
        print(f"🔄 Processing metric: {metric}")

        # Process one metric at a time to control memory usage
        metric_lf = lf.filter(pl.col("METRIC") == metric)

        # Filter out self-scores
        metric_lf = metric_lf.filter(pl.col("GLOSS_A_PATH") != pl.col("GLOSS_B_PATH"))

        # Add relevant flag
        metric_lf = metric_lf.with_columns([(pl.col("GLOSS_A") == pl.col("GLOSS_B")).cast(pl.Int8).alias("relevant")])

        # For very large datasets, we might need to process by GLOSS_A_PATH chunks
        # Get unique GLOSS_A_PATH values for this metric
        gloss_a_paths = metric_lf.select("GLOSS_A_PATH").unique().collect(engine="streaming")["GLOSS_A_PATH"].to_list()

        avg_precisions = []

        # Process in batches of GLOSS_A_PATH to control memory
        batch_size = max(1, chunk_size // 10000)  # Adjust based on expected comparisons per path

        for i in range(0, len(gloss_a_paths), batch_size):
            batch_paths = gloss_a_paths[i : i + batch_size]
            print(
                f"  📁 Processing batch {i//batch_size + 1}/{(len(gloss_a_paths) + batch_size - 1)//batch_size} ({len(batch_paths)} paths)"
            )

            batch_lf = metric_lf.filter(pl.col("GLOSS_A_PATH").is_in(batch_paths))

            # Add shuffling and sort
            batch_lf = (
                batch_lf.with_columns([pl.int_range(pl.len()).shuffle(seed=42).alias("shuffle_order")])
                .sort(["GLOSS_A_PATH", "shuffle_order"])
                .sort(["GLOSS_A_PATH", "SCORE"], maintain_order=True)
            )

            # Add ranking
            batch_lf = batch_lf.with_columns(
                [pl.col("SCORE").rank("ordinal", descending=False).over(["GLOSS_A_PATH"]).alias("rank_ordinal")]
            )

            # Sort by rank for cumsum
            batch_lf = batch_lf.sort(["GLOSS_A_PATH", "rank_ordinal"])

            # Calculate cumulative sum and precision
            batch_lf = batch_lf.with_columns(
                [pl.col("relevant").cum_sum().over(["GLOSS_A_PATH"]).alias("relevant_count")]
            ).with_columns(
                [
                    pl.when(pl.col("relevant") == 1)
                    .then(pl.col("relevant_count") / pl.col("rank_ordinal"))
                    .otherwise(None)
                    .alias("precision_at_k")
                ]
            )

            # Calculate average precision for this batch
            batch_avg_precision = (
                batch_lf.filter(pl.col("precision_at_k").is_not_null())
                .group_by(["GLOSS_A_PATH"])
                .agg(pl.col("precision_at_k").mean().alias("average_precision"))
                .collect(engine="streaming")
            )

            avg_precisions.append(batch_avg_precision)

        # Combine all average precisions for this metric
        if avg_precisions:
            combined_avg_precision = pl.concat(avg_precisions)
            map_value = combined_avg_precision["average_precision"].mean()
            results.append({"METRIC": metric, "mAP": map_value})

    # Convert results to DataFrame
    result_df = pl.DataFrame(results)

    duration = time.perf_counter() - start_time
    print(f"⏱️ Execution time: {duration:.2f} seconds")
    return result_df


import polars as pl
import time
from pathlib import Path


def compute_map_by_metric_memory_efficient(lf: pl.LazyFrame, limit_rows: None | int = None) -> pl.LazyFrame:
    """
    Memory-efficient computation of mean average precision (mAP) per METRIC.
    This version still uses too much memory for very large datasets - use chunked versions instead.
    """
    start_time = time.perf_counter()
    if limit_rows is not None:
        lf = lf.limit(limit_rows)
        print(f"⏳ Processing first {limit_rows:,} rows")

    # Filter out self-scores and add relevant flag in one step
    lf = lf.filter(pl.col("GLOSS_A_PATH") != pl.col("GLOSS_B_PATH")).with_columns(
        [(pl.col("GLOSS_A") == pl.col("GLOSS_B")).cast(pl.Int8).alias("relevant")]
    )

    # Simplified approach: just sort by score (no shuffling needed for mAP calculation)
    # The random shuffling was likely causing memory issues without adding value
    lf = lf.sort(["METRIC", "GLOSS_A_PATH", "SCORE"])

    # Use simpler ranking - just ordinal rank which is what we need for precision calculation
    lf = lf.with_columns(
        [pl.col("SCORE").rank("ordinal", descending=False).over(["METRIC", "GLOSS_A_PATH"]).alias("rank")]
    )

    # Calculate cumulative sum of relevant items (more memory efficient)
    lf = lf.with_columns([pl.col("relevant").cum_sum().over(["METRIC", "GLOSS_A_PATH"]).alias("relevant_count")])

    # Calculate precision only for relevant items
    lf = lf.with_columns(
        [
            pl.when(pl.col("relevant") == 1)
            .then(pl.col("relevant_count") / pl.col("rank"))
            .otherwise(None)
            .alias("precision_at_k")
        ]
    )

    # Use streaming engine throughout and collect results progressively
    result = (
        lf.filter(pl.col("precision_at_k").is_not_null())
        .group_by(["METRIC", "GLOSS_A_PATH"])
        .agg(pl.col("precision_at_k").mean().alias("average_precision"))
        .group_by("METRIC")
        .agg(pl.col("average_precision").mean().alias("mAP"))
        .collect(engine="streaming")
    )

    duration = time.perf_counter() - start_time
    print(f"⏱️ Execution time: {duration:.2f} seconds")
    return result


def compute_map_aggressive_chunking(
    dataset_path: str, metrics_to_process: list[str] = None, max_queries_per_chunk: int = 1000
) -> pl.DataFrame:
    """
    Aggressively chunk by both metric AND query paths to minimize memory usage.
    Processes small batches of queries at a time.
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")

    if metrics_to_process is None:
        # Get all available metrics
        metrics_to_process = [
            frag.partition_expression.operands[1].value
            for frag in dataset.get_fragments()
            if hasattr(frag.partition_expression.operands[1], "value")
        ]
        metrics_to_process = list(set(metrics_to_process))

    results = []

    for metric in metrics_to_process:
        print(f"🔄 Processing metric: {metric}")

        # Filter dataset to single metric
        filtered_dataset = dataset.filter(ds.field("METRIC") == metric)

        # Get unique query paths for this metric
        paths_lf = pl.scan_pyarrow_dataset(filtered_dataset).select("GLOSS_A_PATH").unique()
        unique_paths = paths_lf.collect(engine="streaming")["GLOSS_A_PATH"].to_list()

        print(f"  📊 Found {len(unique_paths):,} unique queries for {metric}")

        all_avg_precisions = []

        # Process queries in small chunks
        for i in range(0, len(unique_paths), max_queries_per_chunk):
            chunk_paths = unique_paths[i : i + max_queries_per_chunk]
            chunk_num = i // max_queries_per_chunk + 1
            total_chunks = (len(unique_paths) + max_queries_per_chunk - 1) // max_queries_per_chunk

            print(f"  📦 Processing chunk {chunk_num}/{total_chunks} ({len(chunk_paths)} queries)")

            try:
                # Process this small chunk
                chunk_avg_precisions = process_query_chunk_minimal_memory(filtered_dataset, chunk_paths)
                all_avg_precisions.extend(chunk_avg_precisions)

            except Exception as e:
                print(f"    ❌ Error in chunk {chunk_num}: {e}")
                continue

        # Calculate mAP for this metric
        if all_avg_precisions:
            map_value = sum(all_avg_precisions) / len(all_avg_precisions)
        else:
            map_value = 0.0

        results.append({"METRIC": metric, "mAP": map_value})
        print(f"  ✅ {metric}: mAP = {map_value:.4f} (from {len(all_avg_precisions):,} queries)")

    return pl.DataFrame(results)


def process_query_chunk_minimal_memory(dataset, query_paths: list[str]) -> list[float]:
    """
    Process a small chunk of queries with minimal memory usage.
    Returns list of average precision values.
    """
    avg_precisions = []

    for query_path in query_paths:
        try:
            # Load data for just this one query
            query_lf = (
                pl.scan_pyarrow_dataset(dataset)
                .filter(pl.col("GLOSS_A_PATH") == query_path)
                .filter(pl.col("GLOSS_A_PATH") != pl.col("GLOSS_B_PATH"))
                .select(["GLOSS_A", "GLOSS_B", "SCORE"])
                .sort("SCORE")
            )

            # Collect this small dataset
            query_data = query_lf.collect(engine="streaming")

            if len(query_data) == 0:
                continue

            # Calculate relevance and AP in pure Python (more memory efficient for small data)
            scores = query_data["SCORE"].to_list()
            gloss_a = query_data["GLOSS_A"].to_list()
            gloss_b = query_data["GLOSS_B"].to_list()

            # Calculate average precision
            relevant_count = 0
            precision_sum = 0.0

            for i, (a, b) in enumerate(zip(gloss_a, gloss_b)):
                if a == b:  # relevant
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    precision_sum += precision_at_i

            if relevant_count > 0:
                ap = precision_sum / relevant_count
                avg_precisions.append(ap)

        except Exception as e:
            print(f"    ⚠️ Error processing query {query_path}: {e}")
            continue

    return avg_precisions


def compute_map_chunked_by_metric(
    dataset_path: str, metrics_to_process: list[str] = None, chunk_size: int = 1000000
) -> pl.DataFrame:
    """
    Process each metric separately to minimize memory usage.
    This is the most memory-efficient approach for huge datasets.
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")

    if metrics_to_process is None:
        # Get all available metrics
        metrics_to_process = [
            frag.partition_expression.operands[1].value
            for frag in dataset.get_fragments()
            if hasattr(frag.partition_expression.operands[1], "value")
        ]
        metrics_to_process = list(set(metrics_to_process))

    results = []

    for metric in metrics_to_process:
        print(f"🔄 Processing metric: {metric}")

        # Filter dataset to single metric
        filtered_dataset = dataset.filter(ds.field("METRIC") == metric)

        # Convert to Polars LazyFrame
        lf = pl.scan_pyarrow_dataset(filtered_dataset)

        # Process in chunks if the metric is still too large
        try:
            metric_result = compute_map_single_metric_chunked(lf, metric, chunk_size)
            results.append(metric_result)
        except Exception as e:
            print(f"❌ Error processing {metric}: {e}")
            continue

    # Combine all results
    if results:
        return pl.concat(results)
    else:
        return pl.DataFrame({"METRIC": [], "mAP": []})


def compute_map_single_metric_chunked(lf: pl.LazyFrame, metric_name: str, chunk_size: int = 1000000) -> pl.DataFrame:
    """
    Process a single metric in chunks by GLOSS_A_PATH to minimize memory usage.
    """
    # Get unique GLOSS_A_PATH values to process in batches
    unique_paths = lf.select("GLOSS_A_PATH").unique().collect(engine="streaming")["GLOSS_A_PATH"].to_list()

    all_avg_precisions = []

    # Process paths in chunks
    for i in range(0, len(unique_paths), chunk_size):
        chunk_paths = unique_paths[i : i + chunk_size]
        print(
            f"  📦 Processing chunk {i//chunk_size + 1}/{(len(unique_paths) + chunk_size - 1)//chunk_size} "
            f"({len(chunk_paths)} paths)"
        )

        # Filter to current chunk of paths
        chunk_lf = lf.filter(pl.col("GLOSS_A_PATH").is_in(chunk_paths))

        # Process this chunk
        chunk_result = process_chunk_for_map(chunk_lf)
        all_avg_precisions.append(chunk_result)

    # Combine all chunks and calculate final mAP
    if all_avg_precisions:
        combined = pl.concat(all_avg_precisions)
        map_value = combined["average_precision"].mean()
        return pl.DataFrame({"METRIC": [metric_name], "mAP": [map_value]})
    else:
        return pl.DataFrame({"METRIC": [metric_name], "mAP": [0.0]})


def process_chunk_for_map(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Process a chunk of data to compute average precision per query.
    """
    # Filter out self-scores and add relevant flag
    lf = (
        lf.filter(pl.col("GLOSS_A_PATH") != pl.col("GLOSS_B_PATH"))
        .with_columns([(pl.col("GLOSS_A") == pl.col("GLOSS_B")).cast(pl.Int8).alias("relevant")])
        .sort(["GLOSS_A_PATH", "SCORE"])
    )

    # Add rank within each query
    lf = lf.with_columns([pl.col("SCORE").rank("ordinal", descending=False).over("GLOSS_A_PATH").alias("rank")])

    # Calculate cumulative relevant count
    lf = lf.with_columns([pl.col("relevant").cum_sum().over("GLOSS_A_PATH").alias("relevant_count")])

    # Calculate precision at each relevant position
    lf = lf.with_columns(
        [
            pl.when(pl.col("relevant") == 1)
            .then(pl.col("relevant_count") / pl.col("rank"))
            .otherwise(None)
            .alias("precision_at_k")
        ]
    )

    # Calculate average precision per query
    return (
        lf.filter(pl.col("precision_at_k").is_not_null())
        .group_by("GLOSS_A_PATH")
        .agg(pl.col("precision_at_k").mean().alias("average_precision"))
        .collect(engine="streaming")
    )


def compute_map_ultra_minimal_memory(dataset_path: str, metrics_to_process: list[str] = None) -> pl.DataFrame:
    """
    Ultra memory-efficient version that processes one metric and one query at a time.
    Slowest but uses minimal memory.
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")

    if metrics_to_process is None:
        metrics_to_process = [
            frag.partition_expression.operands[1].value
            for frag in dataset.get_fragments()
            if hasattr(frag.partition_expression.operands[1], "value")
        ]
        metrics_to_process = list(set(metrics_to_process))

    results = []

    for metric in metrics_to_process:
        print(f"🔄 Processing metric: {metric}")

        # Get all unique GLOSS_A_PATH values for this metric
        metric_dataset = dataset.filter(ds.field("METRIC") == metric)
        paths_df = pl.scan_pyarrow_dataset(metric_dataset).select("GLOSS_A_PATH").unique().collect(engine="streaming")
        unique_paths = paths_df["GLOSS_A_PATH"].to_list()

        avg_precisions = []

        # Process each query (GLOSS_A_PATH) individually
        for i, path in enumerate(unique_paths):
            if i % 1000 == 0:
                print(f"  📍 Processing query {i+1}/{len(unique_paths)}")

            # Get data for this specific query
            query_lf = (
                pl.scan_pyarrow_dataset(metric_dataset)
                .filter(pl.col("GLOSS_A_PATH") == path)
                .filter(pl.col("GLOSS_A_PATH") != pl.col("GLOSS_B_PATH"))
                .with_columns([(pl.col("GLOSS_A") == pl.col("GLOSS_B")).cast(pl.Int8).alias("relevant")])
                .sort("SCORE")
            )

            # Calculate AP for this query
            query_data = query_lf.collect(engine="streaming")

            if len(query_data) == 0:
                continue

            relevant_items = query_data["relevant"].to_list()

            if sum(relevant_items) == 0:  # No relevant items
                continue

            # Calculate average precision for this query
            ap = calculate_average_precision(relevant_items)
            avg_precisions.append(ap)

        # Calculate mAP for this metric
        if avg_precisions:
            map_value = sum(avg_precisions) / len(avg_precisions)
        else:
            map_value = 0.0

        results.append({"METRIC": metric, "mAP": map_value})
        print(f"  ✅ {metric}: mAP = {map_value:.4f}")

    return pl.DataFrame(results)


def calculate_average_precision(relevant_list: list[int]) -> float:
    """
    Calculate average precision for a single query given a list of relevance labels.
    """
    if not any(relevant_list):
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, is_relevant in enumerate(relevant_list):
        if is_relevant:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    return precision_sum / relevant_count if relevant_count > 0 else 0.0

    # Option 1: Simplified version (removes shuffling overhead)
    # lf = pl.scan_pyarrow_dataset(your_dataset)
    # result = compute_map_by_metric_memory_efficient(lf)

    # Option 2: Process by metric chunks (recommended for huge datasets)
    # result = compute_map_chunked_by_metric("path/to/your/dataset",
    #                                       metrics_to_process=["metric1", "metric2"],
    #                                       chunk_size=500000)

    # Option 3: Ultra minimal memory (slowest but safest)
    # result = compute_map_ultra_minimal_memory("path/to/your/dataset")

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean SCORE for each METRIC partition.")
    parser.add_argument("path", type=Path, help="Path to the METRIC=hive-partitioned Parquet dataset.")
    args = parser.parse_args()

    lf = pl.scan_parquet(args.path)
    show_head(lf)

    # run_experiments(args.path)

    # compute_score_by_metric(lf)

    result = compute_map_by_metric_memory_efficient(pl.scan_parquet(args.path), limit_rows=100_000_000)
    print(f"Optimized:\n{result}")
    for row in result.iter_rows():
        print(row)

    # result = compute_map_by_metric_safe_chunked(pl.scan_parquet(args.path), limit_rows=100_000_000)
    # print(f"Chunked:\n{result}")
    # for row in result.iter_rows():
    #     print(row)
    result = compute_map_chunked_by_metric(
        args.path, metrics_to_process=["metric1", "metric2"], chunk_size=500000
    )
    print(f"Chunked By Metric:\n{result}")
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
