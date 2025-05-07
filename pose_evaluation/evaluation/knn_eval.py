import heapq
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer
from tqdm import tqdm

app = typer.Typer()


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
    dataset = ds.dataset(dataset_path, format="parquet")

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
        predicted_label = neighbors[0][2]
        is_correct = predicted_label == query_label
        correct += is_correct
        total += 1

        if verbose:
            typer.echo(f"{query_path}: {predicted_label} (true: {query_label}) {'✓' if is_correct else '✗'}")

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
