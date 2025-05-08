import heapq
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer
from tqdm import tqdm

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
    batch_size: int = 1024,
) -> Dict[Tuple[str, str], List[Tuple[float, str, str]]]:
    """
    Compute top-k nearest neighbors from a PyArrow dataset.

    Returns:
        Dict mapping (query_path, query_label) to sorted list of
        (score, neighbor_path, neighbor_label).
    """
    top_k = defaultdict(list)

    for batch in tqdm(dataset.to_batches(batch_size=batch_size), desc="Processing batches"):
        table = batch.to_pydict()

        for q_path, n_path, score, q_label, n_label in zip(
            table[query_path_col],
            table[neighbor_path_col],
            table[score_col],
            table[query_label_col],
            table[neighbor_label_col],
        ):
            key = (q_path, q_label)
            heapq.heappush(top_k[key], (-score, n_path, n_label))
            if len(top_k[key]) > k:
                heapq.heappop(top_k[key])

    return {key: sorted([(-s, n_path, n_label) for s, n_path, n_label in heap]) for key, heap in top_k.items()}


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
):
    """
    Perform KNN classification using intra- or cross-split distances from a PyArrow dataset.
    """
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")

    # if "METRIC" in dataset.partitioning.schema.names:
    #     print("METRIC in partition!")
    # else:
    #     print(dataset.partitioning)
    #     print("&&&&&&&&&&&&&& SCHEMA")
    #     print(dataset.partitioning.schema)
    #     print("&&&&&&&&&&&&&& DICTS")
    #     print(dataset.partitioning.dictionaries)
    # exit()
    summary_stats = summarize_distance_matrix(dataset)
    if summary_stats["fill_ratio"] < 0.5:
        print("⚠️ Warning: distance matrix is sparse.")

    # TODO: iterate over metrics

    top_k_results = compute_top_k_neighbors(
        dataset=dataset,
        k=k,
        score_col=score_col,
        query_path_col=query_path_col,
        neighbor_path_col=neighbor_path_col,
        query_label_col=query_label_col,
        neighbor_label_col=neighbor_label_col,
    )

    # Evaluation: compute top-1 accuracy
    correct = 0
    total = 0

    if output_path:
        output_rows = []

    for (query_path, query_label), neighbors in tqdm(top_k_results.items(), desc="Processing items"):
        neighbor_labels = [label for _, _, label in neighbors]
        label_counts = Counter(neighbor_labels)
        predicted_label = label_counts.most_common(1)[0][0]
        predicted_label_count = label_counts.most_common(1)[0][1]
        is_correct = predicted_label == query_label
        correct += is_correct
        total += 1

        if verbose:
            typer.echo(
                f"{query_path}: {predicted_label} x {predicted_label_count} (true: {query_label}) {'✓' if is_correct else '✗'}"
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
    typer.echo(f"\nTop-{k} Accuracy: {accuracy:.4f} ({correct}/{total})")

    if output_path:
        typer.echo(f"Saving top-{k} neighbors to {output_path} ...")
        table = pa.table(output_rows)
        pq.write_table(table, output_path)


if __name__ == "__main__":
    app()
# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && /opt/home/cleong/projects/pose-evaluation# python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/knn_eval.py metric_results_full_matrix/pyarrow_dataset/asl-citizen/testvstrain/METRIC\=untrimmed_zspeed1.0_normalizedbyshoulders_reduceholistic_defaultdist10.0_nointerp_dtw_fillmasked10.0_dtaiDTWAggregatedDistanceMetricFast/ --k 5 --verbose
