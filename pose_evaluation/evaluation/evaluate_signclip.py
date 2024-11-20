import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from pose_evaluation.metrics.signclip_distance_metric import SignCLIPEmbeddingDistanceMetric
from tqdm import tqdm

def load_embedding(file_path: Path) -> np.ndarray:
    """
    Load a SignCLIP embedding from a .npy file, ensuring it has the correct shape.
    
    Args:
        file_path (Path): Path to the .npy file.
    
    Returns:
        np.ndarray: The embedding with shape (768,).
    """
    embedding = np.load(file_path)
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding[0]  # Reduce shape from (1, 768) to (768,)
    return embedding

def match_embeddings_to_glosses(emb_dir: Path, split_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match .npy embeddings to the corresponding glosses based on the numerical ID.
    
    Args:
        emb_dir (Path): Directory containing the .npy files.
        split_df (pd.DataFrame): DataFrame containing the split file with the "Video file" column.
    
    Returns:
        pd.DataFrame: Updated DataFrame with an additional column for embeddings.
    """
    # Map video file IDs to embeddings
    embeddings_map = {}
    for npy_file in emb_dir.glob("*.npy"):
        numerical_id = npy_file.stem.split("-")[0]
        embeddings_map[numerical_id] = npy_file

    # Match embeddings to glosses
    embeddings = []
    for _, row in split_df.iterrows():
        video_file = row["Video file"]
        numerical_id = video_file.split("-")[0]
        npy_file = embeddings_map.get(numerical_id)

        if npy_file is not None:
            embeddings.append(load_embedding(npy_file))
        else:
            embeddings.append(None)  # Placeholder if no matching file

    split_df["embedding"] = embeddings
    return split_df

def evaluate_signclip(emb_dir: Path, split_file: Path, kind: str = "cosine"):
    """
    Evaluate SignCLIP embeddings using score_all.
    
    Args:
        emb_dir (Path): Directory containing .npy embeddings.
        split_file (Path): Path to the split CSV file.
        kind (str): Metric type ("cosine" or "l2"). Default is "cosine".
    """
    # Load split file
    split_df = pd.read_csv(split_file)
    
    # Match embeddings
    split_df = match_embeddings_to_glosses(emb_dir, split_df)
    
    # Filter out rows without embeddings
    valid_df = split_df.dropna(subset=["embedding"]).reset_index(drop=True)
    embeddings = valid_df["embedding"].tolist()

    # Initialize metric
    metric = SignCLIPEmbeddingDistanceMetric(kind=kind)

    # Compute all pairwise scores
    print(f"Computing {kind} distances for {len(embeddings)} embeddings...")
    
    scores = metric.score_all(embeddings, embeddings)

    # Save scores to a CSV file
    output_file = emb_dir / "signclip_scores.csv"
    results = []
    for i, hyp_row in valid_df.iterrows():
        for j, ref_row in valid_df.iterrows():
            results.append({
                "hyp": hyp_row["Video file"],
                "ref": ref_row["Video file"],
                "score": scores[i, j]
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Scores saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate SignCLIP embeddings with score_all.")
    parser.add_argument(
        "emb_dir", type=Path, help="Path to the directory containing SignCLIP .npy files"
    )
    parser.add_argument(
        "--split_file", type=Path, required=True, help="Path to the split CSV file (e.g., test.csv)"
    )
    parser.add_argument(
        "--kind", type=str, choices=["cosine", "l2"], default="cosine",
        help="Type of distance metric to use (default: cosine)"
    )
    args = parser.parse_args()

    evaluate_signclip(emb_dir=args.emb_dir, split_file=args.split, kind=args.kind)

if __name__ == "__main__":
    main()
    print(f"THIS SCRIPT NEEDS TESTING")
