import heapq
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import typer
from tqdm import tqdm

from pose_evaluation.evaluation.load_pyarrow_dataset import summarize_metric_partitions

app = typer.Typer()


def summarize_distance_matrix(
    dataset: ds.Dataset,
    query_path_col: str = "GLOSS_A_PATH",
    neighbor_path_col: str = "GLOSS_B_PATH",
    score_col: str = "SCORE",
    query_label_col: str = "GLOSS_A",
    neighbor_label_col: str = "GLOSS_B",
) -> dict[str, Any]:
    """
    Summarizes the sparsity and completeness of a KNN-style distance matrix
    in a memory-efficient way.
    """
    query_to_refcount = Counter()
    unique_queries = set()
    unique_references = set()
    unique_query_labels = set()
    unique_reference_labels = set()
    score_count = 0

    scan_cols = [query_path_col, neighbor_path_col, score_col, query_label_col, neighbor_label_col]
    scanner = dataset.scanner(columns=scan_cols)

    for batch in tqdm(scanner.to_batches(), desc="Scanning Batches"):
        q_paths = batch.column(query_path_col)
        r_paths = batch.column(neighbor_path_col)
        scores = batch.column(score_col)
        q_labels = batch.column(query_label_col)
        r_labels = batch.column(neighbor_label_col)

        for i in range(batch.num_rows):
            if not scores[i].is_valid:  # skip nulls
                continue

            q_path = q_paths[i].as_py()
            r_path = r_paths[i].as_py()
            q_label = q_labels[i].as_py()
            r_label = r_labels[i].as_py()

            query_to_refcount[q_path] += 1
            unique_queries.add(q_path)
            unique_references.add(r_path)
            unique_query_labels.add(q_label)
            unique_reference_labels.add(r_label)
            score_count += 1

    num_queries = len(unique_queries)
    num_references = len(unique_references)
    total_possible = num_queries * num_references

    full_rows = sum(1 for count in query_to_refcount.values() if count == num_references)
    avg_refs_per_query = sum(query_to_refcount.values()) / num_queries if num_queries else 0
    counts_distribution = Counter(query_to_refcount.values())

    summary = {
        "num_queries": num_queries,
        "num_references": num_references,
        "theoretical_filled_shape": (num_queries, num_references),
        "total_possible": total_possible,
        "recorded_scores": score_count,
        "fill_ratio": score_count / total_possible if total_possible else 0,
        "queries_with_full_reference_set": full_rows,
        "average_references_per_query": avg_refs_per_query,
        "reference_count_distribution": dict(counts_distribution),
        "unique_query_labels": len(unique_query_labels),
        "unique_reference_labels": len(unique_reference_labels),
    }

    print("\n--- Distance Matrix Summary ---")
    for key, value in summary.items():
        if key == "reference_count_distribution":
            print("\nReference count distribution (top 10):")
            for count, freq in counts_distribution.most_common(10):
                print(f"  {count:,} references → {freq:,} queries")
        else:
            print(f"{key.replace('_', ' ').capitalize()}: {value}")

    return summary


def compute_top_k_neighbors(
    dataset: ds.Dataset,
    k: int,
    score_col: str,
    query_path_col: str,
    neighbor_path_col: str,
    query_label_col: str,
    neighbor_label_col: str,
    min_references_per_query: int | None = None,
    max_queries: int | None = None,
) -> tuple[dict[tuple[str, str], list[tuple[float, str, str]]], dict[str, int]]:
    """
    Compute top-k nearest neighbors from a PyArrow dataset, assuming lower
    scores are better (e.g., distances). Only includes queries with at least
    `min_references_per_query` neighbor entries.

    Returns:
        - A dict mapping (query_path, query_label) to a sorted list of (score, neighbor_path, neighbor_label)
        - A stats dict with 'total', 'kept', 'skipped'

    """
    top_k = defaultdict(list)
    reference_counts = defaultdict(int)

    for batch in tqdm(dataset.to_batches(), desc="Processing batches"):
        table = batch.to_pydict()

        for q_path, n_path, score, q_label, n_label in zip(
            table[query_path_col],
            table[neighbor_path_col],
            table[score_col],
            table[query_label_col],
            table[neighbor_label_col],
            strict=False,
        ):
            key = (q_path, q_label)
            reference_counts[key] += 1
            heapq.heappush(top_k[key], (score, n_path, n_label))
            if len(top_k[key]) > k:
                # Pop *largest* (worst) score to keep only k smallest
                heapq._heapify_max(top_k[key])
                heapq.heappop(top_k[key])
                heapq.heapify(top_k[key])  # Restore min-heap property

    total = len(top_k)
    filtered_top_k = {}
    for key, heap in top_k.items():
        if min_references_per_query is None or reference_counts[key] >= min_references_per_query:
            filtered_top_k[key] = sorted(heap)

    if max_queries is not None:
        # Truncate to the first N entries (dicts preserve insertion order in Python 3.7+)
        filtered_top_k = dict(list(filtered_top_k.items())[:max_queries])

    stats = {
        "total": total,
        "kept": len(filtered_top_k),
        "skipped": total - len(filtered_top_k),
    }

    # Debug: Print a few sample neighbors for inspection
    print("\nSample top-k neighbors:")
    for i, ((q_path, q_label), neighbors) in enumerate(filtered_top_k.items()):
        print(f"Query: {q_path} ({q_label})")
        for score, n_path, n_label in neighbors:
            print(f"  -> {n_path} ({n_label}) with score {score:.4f}")
        print()
        if i >= 2:  # Limit to first 3 queries
            break

    return filtered_top_k, stats


def get_metric_filtered_datasets(
    dataset_path: Path, dataset: ds.Dataset, metric: str | None = None
) -> list[tuple[str | None, ds.Dataset]]:
    """
    Returns a list of (metric_name, filtered_dataset) tuples based on the
    user selection or provided metric string.
    """
    if "METRIC" not in dataset.partitioning.schema.names:
        typer.echo("No METRIC partition found in dataset.")
        return [(None, dataset)]

    metric_stats_dict = summarize_metric_partitions(dataset_path)
    metric_values = [str(m) for m in metric_stats_dict.keys()]

    if metric is None:
        typer.echo("Dataset has available METRIC values:")
        for i, val in enumerate(metric_values):
            typer.echo(f"{i + 1}. {val} ({metric_stats_dict[val]['num_rows']:,} rows)")
        typer.echo("a. All metrics")
        typer.echo("n. No filtering")
        typer.echo("x. Exit")

        choice = typer.prompt("Select a metric number (or 'a' for all, 'n' for none)")

        if choice.lower() == "a":
            selected_metrics = metric_values
        elif choice.lower() == "n":
            selected_metrics = [None]
        elif choice.lower() == "x":
            print("All right, fare well!")
            raise typer.Exit()
        else:
            try:
                selected_metrics = [metric_values[int(choice) - 1]]
            except (ValueError, IndexError) as e: # Catch the original exception as 'e' (or 'err')
                typer.echo("Invalid selection.")
                raise typer.Exit(code=1) from e # Chain the new exception to the original one
    else:
        if metric == "all":
            selected_metrics = metric_values
        elif metric == "none":
            selected_metrics = [None]
        elif metric in metric_values:
            selected_metrics = [metric]
        else:
            typer.echo(f"Metric '{metric}' not found in dataset.")
            raise typer.Exit(code=1)

    metric_datasets = []
    for m in selected_metrics:
        if m is None:
            filtered = dataset  # no filtering
        else:
            filtered = dataset.filter(ds.field("METRIC") == m)
        metric_datasets.append((m, filtered))

    return metric_datasets


def save_neighbors(
    top_k_results: dict,
    output_path: Path,
    query_path_col: str,
    query_label_col: str,
    neighbor_path_col: str,
    neighbor_label_col: str,
    score_col: str,
    overwrite: bool = False,
) -> None:
    """
    Save top-k results from neighbor dict to Parquet.

    Prompts before overwriting.
    """
    output_rows = []

    for (query_path, query_label), neighbors in top_k_results.items():
        for rank, (score, neighbor_path, neighbor_label) in enumerate(neighbors, start=1):
            output_rows.append(
                {
                    query_path_col: query_path,
                    query_label_col: query_label,
                    neighbor_path_col: neighbor_path,
                    neighbor_label_col: neighbor_label,
                    score_col: score,
                    "RANK": rank,
                }
            )

    if not output_rows:
        typer.echo("⚠️ No results to save.")
        return

    if output_path.exists():
        typer.echo(f"\nFile {output_path} already exists.")
        if overwrite:
            typer.echo(f"--overwrite give: Overwriting {output_path}")
            choice = "o"
        else:
            choice = typer.prompt("Choose an action: [o]verwrite / [n]ew name / [s]kip", default="s").lower()

        if choice == "o":
            pass
        elif choice == "n":
            while True:
                new_name = typer.prompt("Enter new file name")
                if not new_name.endswith(".parquet"):
                    new_name += ".parquet"
                new_path = output_path.parent / new_name
                if not new_path.exists():
                    output_path = new_path
                    break
                else:
                    typer.echo(f"File {new_path} already exists. Please choose a different name or Ctrl+C to abort.")
        else:
            typer.echo("Skipping save.")
            return

    typer.echo(f"Saving top-k neighbors to {output_path} ...")
    df = pd.DataFrame(output_rows)
    df.to_parquet(output_path, index=False)


def evaluate_top_k_results(
    top_k_results: dict,
    k: int,
    verbose: bool = False,
) -> float:
    """Evaluate top-k results and return accuracy only."""
    correct = 0
    total = 0

    for (query_path, query_label), neighbors in tqdm(top_k_results.items(), desc=f"Processing items, k={k}"):
        neighbor_labels = [label for _, _, label in neighbors[:k]]
        label_counts = Counter(neighbor_labels)
        predicted_label, predicted_label_count = label_counts.most_common(1)[0]
        is_correct = predicted_label == query_label
        correct += is_correct
        total += 1

        if verbose:
            typer.echo(
                f"{query_path}: {predicted_label} x {predicted_label_count} (true: {query_label}) "
                f"{'✓' if is_correct else '✗'} ({[f'{label} x {count}' for label, count in label_counts.items()]})"
            )
    typer.echo(f"Evaluated {total} queries, of which {correct} were correctly classified")

    return correct / total if total > 0 else 0.0


@app.command("analyze")
def analyze_neighbors_command(
    file_path: Path = typer.Argument(..., help="Path to a saved Parquet file of neighbors."),
    k: int = typer.Option(5, help="Value of k used in top-k neighbors."),
    query_path_col: str = typer.Option("GLOSS_A_PATH"),
    neighbor_path_col: str = typer.Option("GLOSS_B_PATH"),
    query_label_col: str = typer.Option("GLOSS_A"),
    neighbor_label_col: str = typer.Option("GLOSS_B"),
    score_col: str = typer.Option("SCORE"),
    verbose: bool = typer.Option(False, help="Print classification details."),
):
    """Analyze previously saved top-k neighbor results."""
    df = pd.read_parquet(file_path)
    grouped = df.groupby([query_path_col, query_label_col])

    top_k_results = {
        (query_path, query_label): list(
            zip(group[score_col], group[neighbor_path_col], group[neighbor_label_col], strict=False)
        )
        for (query_path, query_label), group in grouped
    }

    accuracy = evaluate_top_k_results(top_k_results, k=k, verbose=verbose)

    typer.echo(f"\nAccuracy from saved neighbors: {accuracy:.4f}")


@app.command("compute")
def do_knn(
    dataset_path: Path = typer.Argument(..., help="Path to Arrow/Parquet dataset."),
    k: int = typer.Option(1, help="Number of neighbors."),
    score_col: str = typer.Option("SCORE"),
    query_path_col: str = typer.Option("GLOSS_A_PATH"),
    neighbor_path_col: str = typer.Option("GLOSS_B_PATH"),
    query_label_col: str = typer.Option("GLOSS_A"),
    neighbor_label_col: str = typer.Option("GLOSS_B"),
    output_path: Path | None = typer.Option(None, help="Optional output path to save top-k results as Parquet."),
    verbose: bool = typer.Option(False, help="Print classification details."),
    metric: str | None = typer.Option(None, help="Metric partition value to evaluate."),
    max_queries: int | None = typer.Option(None, help="If given, will only process this many query files"),
    overwrite: bool = typer.Option(False, help="If given, will skip saving over existing files"),
):
    """
    Perform KNN classification using intra- or cross-split distances from a
    PyArrow dataset.
    """
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
    metric_datasets = get_metric_filtered_datasets(dataset_path, dataset, metric)

    for metric_name, dataset in metric_datasets:
        typer.echo("-" * 60)
        typer.echo(f"\nEvaluating metric: {metric_name or 'None'}")

        summary_stats = summarize_distance_matrix(dataset)

        if summary_stats["fill_ratio"] < 0.5:
            print(f"⚠️ Warning: distance matrix is sparse! Fill Ratio: {summary_stats['fill_ratio']}")

        top_k_results, filter_stats = compute_top_k_neighbors(
            dataset=dataset,
            k=k,
            score_col=score_col,
            query_path_col=query_path_col,
            neighbor_path_col=neighbor_path_col,
            query_label_col=query_label_col,
            neighbor_label_col=neighbor_label_col,
            min_references_per_query=summary_stats["num_references"],  # optional
            max_queries=max_queries,
        )

        typer.echo(
            f"Queries evaluated: {filter_stats['kept']} "
            f"(skipped {filter_stats['skipped']} under min_references={summary_stats['num_references']})"
        )

        proposed_name = f"{metric_name}_{summary_stats['num_queries']}_hyps_{summary_stats['num_references']}_refs_top_{k}_neighbors.parquet"
        if max_queries is not None:
            proposed_name = f"maxqueries_{max_queries}_{proposed_name}"

        if output_path is None:
            proposed_path = Path.cwd() / proposed_name
        else:
            output_path.mkdir(exist_ok=True, parents=True)
            proposed_path = output_path / proposed_name
        typer.echo(f"Results will be saved to {proposed_path}")

        save_neighbors(
            top_k_results,
            output_path=proposed_path,
            score_col=score_col,
            query_path_col=query_path_col,
            neighbor_path_col=neighbor_path_col,
            query_label_col=query_label_col,
            neighbor_label_col=neighbor_label_col,
            overwrite=overwrite,
        )

        accuracy = evaluate_top_k_results(
            top_k_results,
            k=k,
            verbose=verbose,
        )
        typer.echo(
            f"\nMetric: {metric_name}, k: {k}, Accuracy: {accuracy:.4f}, Queries: {filter_stats['kept']}, References: {summary_stats['num_references']}"
        )


if __name__ == "__main__":
    app()

# NOTE: GLOSS_A is the test set and GLOSS_B is the train set.
# I printed some samples from the table and got:
# Row 4:
#   SCORE                : 96.02032274735807
#   GLOSS_A              : ARMY
#   GLOSS_B              : MOST
#   SIGNATURE            : untrimmed_zspeed1.0_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked1...
#   GLOSS_A_PATH         : /opt/home/cleong/data/ASL_Citizen/poses/pose/04138050683379402-ARMY.pose
#   GLOSS_B_PATH         : /opt/home/cleong/data/ASL_Citizen/poses/pose/7040324327088798-MOST.pose
#   TIME                 : 0.02099267399898963
# 7040324327088798 is in ASL Citizen's train.csv
# 04138050683379402 is in ASL Citizen's test.csv
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && /opt/home/cleong/projects/pose-evaluation# python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/knn_eval.py metric_results_full_matrix/pyarrow_dataset/asl-citizen/testvstrain/METRIC\=untrimmed_zspeed1.0_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast/ --k 5 --verbose


# full matrix 300 scores
#
# cd /opt/home/cleong/projects/pose-evaluation/metric_results_full_matrix && bash copy_first_200.sh
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/load_parquets.py /opt/home/cleong/projects/pose-evaluation/metric_results_full_matrix/scores_200 --merge-dir /opt/home/cleong/projects/pose-evaluation/metric_results_full_matrix/pyarrow_dataset_200
# out_dir="knn_metric_analysis/5_27_84_metrics/" && kval=100 && mkdir -p "$out_dir" && python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/knn_eval.py compute /opt/home/cleong/projects/pose-evaluation/metric_results_full_matrix/pyarrow_dataset_200 --k 100 --verbose --output-path "$out_dir" 2>&1 |tee "$out_dir/knn_analysis_out_k$kval.txt"
