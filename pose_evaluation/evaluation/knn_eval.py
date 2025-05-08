import heapq
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
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
) -> Dict[str, Any]:
    """
    Summarizes the sparsity and completeness of a KNN-style distance matrix in a PyArrow dataset.

    Returns a dictionary with:
        - shape info
        - fill stats
        - reference count distribution
        - unique path/label counts
    """

    query_to_neighbors = defaultdict(set)
    all_queries = set()
    all_references = set()
    all_query_labels = set()
    all_reference_labels = set()
    score_count = 0

    for batch in tqdm(dataset.to_batches(), desc="Building Distance Matrix Summary"):
        query_paths = batch[query_path_col].to_pylist()
        neighbor_paths = batch[neighbor_path_col].to_pylist()
        scores = batch[score_col].to_pylist()
        query_labels = batch[query_label_col].to_pylist()
        neighbor_labels = batch[neighbor_label_col].to_pylist()

        for q_path, r_path, s, q_label, r_label in zip(
            query_paths, neighbor_paths, scores, query_labels, neighbor_labels
        ):
            if s is not None:
                query_to_neighbors[q_path].add(r_path)
                all_queries.add(q_path)
                all_references.add(r_path)
                all_query_labels.add(q_label)
                all_reference_labels.add(r_label)
                score_count += 1

    num_queries = len(all_queries)
    num_references = len(all_references)
    total_possible = num_queries * num_references

    neighbor_counts = {q: len(refs) for q, refs in query_to_neighbors.items()}
    full_rows = sum(1 for count in neighbor_counts.values() if count == num_references)
    avg_refs_per_query = sum(neighbor_counts.values()) / len(neighbor_counts) if neighbor_counts else 0
    counts_distribution = Counter(neighbor_counts.values())

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
        "unique_query_labels": len(all_query_labels),
        "unique_reference_labels": len(all_reference_labels),
    }

    # Optional: print in human-readable format
    print("\n--- Distance Matrix Summary ---")
    for key, value in summary.items():
        if key == "reference_count_distribution":
            print("\nReference count distribution (top 10):")
            for count, freq in counts_distribution.most_common(10):
                print(f"  {count} references → {freq} queries")
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
    # batch_size: int = 1024,
    min_references_per_query: Optional[int] = None,
) -> Tuple[Dict[Tuple[str, str], List[Tuple[float, str, str]]], Dict[str, int]]:
    """
    Compute top-k nearest neighbors from a PyArrow dataset.
    Only includes queries with at least `min_references_per_query` neighbor entries.

    Returns:
        - A dict mapping (query_path, query_label) to sorted list of (score, neighbor_path, neighbor_label)
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
        ):
            key = (q_path, q_label)
            reference_counts[key] += 1
            heapq.heappush(top_k[key], (-score, n_path, n_label))
            if len(top_k[key]) > k:
                heapq.heappop(top_k[key])

    total = len(top_k)
    filtered_top_k = {}
    for key, heap in top_k.items():
        if min_references_per_query is None or reference_counts[key] >= min_references_per_query:
            filtered_top_k[key] = sorted([(-s, n_path, n_label) for s, n_path, n_label in heap])

    kept = len(filtered_top_k)
    skipped = total - kept

    stats = {
        "total": total,
        "kept": kept,
        "skipped": skipped,
    }

    return filtered_top_k, stats





def get_metric_filtered_datasets(
    dataset_path: Path, dataset: ds.Dataset, metric: Optional[str] = None
) -> List[Tuple[Optional[str], ds.Dataset]]:
    """
    Returns a list of (metric_name, filtered_dataset) tuples
    based on the user selection or provided metric string.
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
            except (ValueError, IndexError):
                typer.echo("Invalid selection.")
                raise typer.Exit(code=1)
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


def evaluate_top_k_results(
    top_k_results: dict,
    k: int,
    query_path_col: str,
    query_label_col: str,
    neighbor_path_col: str,
    neighbor_label_col: str,
    score_col: str,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> float:
    """
    Evaluate top-k results and optionally save to a Parquet file.
    Returns the accuracy.
    """
    correct = 0
    total = 0
    output_rows = [] if output_path else None

    for (query_path, query_label), neighbors in tqdm(top_k_results.items(), desc="Processing items"):
        neighbor_labels = [label for _, _, label in neighbors]
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

        if output_path:
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

    accuracy = correct / total if total > 0 else 0.0

    if output_path and output_rows:
        # Check if file exists and confirm
        if output_path.exists():
            typer.echo(f"\nFile {output_path} already exists.")
            choice = typer.prompt("Choose an action: [o]verwrite / [n]ew name / [s]kip", default="s").lower()

            if choice == "o":
                pass  # proceed to write
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
                        typer.echo(
                            f"File {new_path} already exists. Please choose a different name or Ctrl+C to abort."
                        )
            else:
                typer.echo("Skipping save.")
                output_path = None

        if output_path:
            typer.echo(f"Saving top-{k} neighbors to {output_path} ...")
            df = pd.DataFrame(output_rows)
            df.to_parquet(output_path, index=False)

    return accuracy

@app.command()
def evaluate(
    dataset_path: Path = typer.Argument(..., help="Path to Arrow/Parquet dataset."),
    k: int = typer.Option(1, help="Number of neighbors."),
    score_col: str = typer.Option("SCORE"),
    query_path_col: str = typer.Option("GLOSS_A_PATH"),
    neighbor_path_col: str = typer.Option("GLOSS_B_PATH"),
    query_label_col: str = typer.Option("GLOSS_A"),
    neighbor_label_col: str = typer.Option("GLOSS_B"),
    output_path: Optional[Path] = typer.Option(None, help="Optional output path to save top-k results as Parquet."),
    verbose: bool = typer.Option(False, help="Print classification details."),
    metric: Optional[str] = typer.Option(None, help="Metric partition value to evaluate."),
):
    """
    Perform KNN classification using intra- or cross-split distances from a PyArrow dataset.
    """
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
    metric_datasets = get_metric_filtered_datasets(dataset_path, dataset, metric)

    for metric_name, dataset in metric_datasets:
        typer.echo(f"\nEvaluating metric: {metric_name or 'None'}")

        summary_stats = summarize_distance_matrix(dataset)

        if summary_stats["fill_ratio"] < 0.5:
            print(f"⚠️ Warning: distance matrix is sparse! Fill Ratio: {fill_ratio}")

        top_k_results, filter_stats = compute_top_k_neighbors(
            dataset=dataset,
            k=k,
            score_col=score_col,
            query_path_col=query_path_col,
            neighbor_path_col=neighbor_path_col,
            query_label_col=query_label_col,
            neighbor_label_col=neighbor_label_col,
            min_references_per_query=summary_stats["num_references"],  # optional
        )

        typer.echo(
            f"Queries evaluated: {filter_stats['kept']} "
            f"(skipped {filter_stats['skipped']} under min_references={summary_stats["num_references"]})"
        )

        if output_path is None:
            proposed_path = Path.cwd() / f"{metric_name}.parquet"
        else:            
            output_path.mkdir(exist_ok=True, parents=True)
            proposed_path = output_path / f"{metric_name}.parquet"
        typer.echo(f"Results will be saved to {proposed_path}")
            

        accuracy = evaluate_top_k_results(
            top_k_results=top_k_results,
            k=k,
            query_path_col=query_path_col,
            query_label_col=query_label_col,
            neighbor_path_col=neighbor_path_col,
            neighbor_label_col=neighbor_label_col,
            score_col=score_col,
            output_path=proposed_path,
            verbose=verbose,
        )

        typer.echo(f"\nTop-{k} Accuracy: {accuracy:.4f}")


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
