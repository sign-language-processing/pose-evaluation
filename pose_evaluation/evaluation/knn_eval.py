import typer
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import sys

app = typer.Typer()


@app.command()
def evaluate(
    dataset_path: Path = typer.Argument(..., help="Path to a Parquet/Arrow dataset."),
    k: int = typer.Option(1, help="Number of neighbors to use."),
    score_col: str = typer.Option("SCORE", help="Name of the distance/score column."),
    query_path_col: str = typer.Option("GLOSS_A_PATH", help="Column for query sample path."),
    neighbor_path_col: str = typer.Option("GLOSS_B_PATH", help="Column for neighbor sample path."),
    query_label_col: str = typer.Option("GLOSS_A", help="Column for query sample label."),
    neighbor_label_col: str = typer.Option("GLOSS_B", help="Column for neighbor sample label."),
    verbose: bool = typer.Option(False, help="Print classification report and confusion matrix."),
):
    """
    Perform KNN classification using precomputed intra-split distances from a PyArrow dataset.
    """
    if not dataset_path.exists():
        typer.echo(f"❌ Error: Dataset {dataset_path} not found.", err=True)
        raise typer.Exit(1)

    # Load dataset efficiently
    dataset = ds.dataset(dataset_path, format="parquet")
    typer.echo("📥 Scanning Arrow dataset...")
    table = dataset.to_table()
    df = table.to_pandas()

    # Ensure only distances between unique sample pairs (symmetric matrix assumed)
    unique_paths = sorted(df[query_path_col].unique())
    typer.echo(f"✅ Loaded {len(df)} distance rows between {len(unique_paths)} unique samples...")

    # Pivot to create distance matrix (dense)
    typer.echo("🔧 Creating full distance matrix...")
    dist_matrix = df.pivot(index=query_path_col, columns=neighbor_path_col, values=score_col)
    dist_matrix = dist_matrix.loc[unique_paths, unique_paths].values

    # Replace self-distances with inf
    np.fill_diagonal(dist_matrix, np.inf)

    # Extract labels
    label_map = df.drop_duplicates(query_path_col).set_index(query_path_col)[query_label_col]
    labels = label_map.loc[unique_paths].values

    # Fit and predict using KNN
    knn = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
    knn.fit(dist_matrix, labels)
    predictions = knn.predict(dist_matrix)

    acc = accuracy_score(labels, predictions)
    typer.echo(f"\n✅ KNN (k={k}) Accuracy: {acc:.4f}")

    if verbose:
        typer.echo("\n📋 Classification Report:")
        print(classification_report(labels, predictions, digits=3))

        typer.echo("🔀 Confusion Matrix:")
        print(confusion_matrix(labels, predictions))


if __name__ == "__main__":
    app()
