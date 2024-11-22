import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from pose_evaluation.metrics.embedding_distance_metric import EmbeddingDistanceMetric
from tqdm import tqdm
import time

# python evaluation/evaluate_signclip.py /media/vlab/Aqsa-Deep-Storage/colin/ASL_Citizen/embeddings/sem-lex/ --split_file /media/vlab/Aqsa-Deep-Storage/colin/ASL_Citizen/splits/400_words_10_examples_each.csv
# (pose_evaluation) (base) vlab@vlab-desktop:~/projects/sign_language_processing/pose-evaluation/pose_evaluation$ python evaluation/evaluate_signclip.py /media/vlab/Aqsa-Deep-Storage/colin/ASL_Citizen/embeddings/sem-lex/ --split_file /media/vlab/Aqsa-Deep-Storage/colin/ASL_Citizen/splits/20x5_curated_sample.csv 
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
    metric = EmbeddingDistanceMetric(kind=kind, device="cpu")

    # Compute all pairwise scores
    print(f"Computing {kind} distances for {len(embeddings)} embeddings...")
    
    start_time = time.perf_counter()
    scores = metric.score_all(embeddings, embeddings)
    score_duration = time.perf_counter() - start_time
    print(f"Score_all took {score_duration:.3f} seconds")



    # Extract the "Video file" column
    files = valid_df["Video file"].tolist()

    # Create output file path
    output_file = Path("signclip_scores.csv")

    # Start timer
    start_time = time.perf_counter()

    # Create the Cartesian product of `files` with itself
    n = len(files)
    data = {
        "hyp": [files[i] for i in range(n) for j in range(n)],
        "ref": [files[j] for i in range(n) for j in range(n)],
        "score": scores.flatten()  # Flatten the 2D score matrix into a 1D array
    }


    # Construct the DataFrame
    results_df = pd.DataFrame(data)

    # Save to CSV
    results_df.to_csv(output_file, index=False)

    # End timer
    end_time = time.perf_counter()
    print(f"Saving DataFrame and writing to CSV took {end_time - start_time:.2f} seconds")



    # Save scores to a CSV file
    output_file = Path("signclip_scores.csv")
    results = []
    for i, hyp_row in tqdm(valid_df.iterrows(), total=valid_df.shape[0]):
        for j, ref_row in valid_df.iterrows():
            results.append({
                "hyp": hyp_row["Video file"],
                "ref": ref_row["Video file"],
                "score": scores[i, j].item()
            })

    df_start = time.perf_counter()
    results_df = pd.DataFrame(results)
    df_end = time.perf_counter()
    df_duration = df_end - df_start
    print(f"df took {df_duration}")



    


    csv_start = time.perf_counter()    
    results_df.to_csv(output_file, index=False)
    csv_end = time.perf_counter()
    csv_duration = csv_end - csv_start
    print(f"CSV took {csv_duration}")

    json_start = time.perf_counter()
    results_df.to_json(output_file.with_suffix(".json"), index=False)
    json_end = time.perf_counter()
    json_duration = json_end - json_start
    print(f"JSON took {json_duration}")
    
    np_start = time.perf_counter()
    np.save(output_file.with_suffix(".npy"), scores)
    np_end = time.perf_counter()
    np_duration = np_end-np_start
    print(f"np took {np_duration}")
    
    
    
    
    print(f"Scores of shape {scores.shape} saved to {output_file}")
    read_back_in = np.load(output_file.with_suffix(".npy"))    
    if np.allclose(read_back_in, scores):
        print("yay! All the same!")

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

    evaluate_signclip(emb_dir=args.emb_dir, split_file=args.split_file, kind=args.kind)

if __name__ == "__main__":
    main()
    print(f"THIS SCRIPT NEEDS TESTING")
