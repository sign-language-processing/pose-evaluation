import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from pose_evaluation.metrics.embedding_distance_metric import EmbeddingDistanceMetric
from tqdm import tqdm
import time
import torch
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
    import time

    # Step 1: Create a mapping of numerical IDs to .npy files
    map_start = time.perf_counter()
    embeddings_map = {
        npy_file.stem.split("-")[0]: npy_file
        for npy_file in emb_dir.glob("*.npy")
    }
    map_end = time.perf_counter()
    print(f"Creating embeddings map took {map_end - map_start:.4f} seconds")

    # Step 2: Vectorized matching of embeddings
    match_start = time.perf_counter()

    def get_embedding(video_file):
        numerical_id = video_file.split("-")[0]
        npy_file = embeddings_map.get(numerical_id)
        if npy_file is not None:
            return load_embedding(npy_file)
        return None

    split_df["embedding"] = split_df["Video file"].apply(get_embedding)
    match_end = time.perf_counter()
    print(f"Matching embeddings to glosses took {match_end - match_start:.4f} seconds")

    return split_df


def evaluate_signclip(emb_dir: Path, split_file: Path, kind: str = "cosine", out_path=None):
    """
    Evaluate SignCLIP embeddings using score_all.
    
    Args:
        emb_dir (Path): Directory containing .npy embeddings.
        split_file (Path): Path to the split CSV file.
        kind (str): Metric type ("cosine" or "l2"). Default is "cosine".
    """
    overall_start = time.perf_counter()  # Start overall benchmarking

    # Step 1: Load split file
    split_load_start = time.perf_counter()
    split_df = pd.read_csv(split_file)
    split_load_end = time.perf_counter()
    print(f"Loading split file took {split_load_end - split_load_start:.4f} seconds")
    # print(f"{split_df.info()}")

    # Step 2: Match embeddings to glosses
    match_start = time.perf_counter()
    split_df = match_embeddings_to_glosses(emb_dir, split_df)
    match_end = time.perf_counter()
    print(f"Matching embeddings to glosses took {match_end - match_start:.4f} seconds")
    # print(split_df.info())

    # Step 3: Filter out rows without embeddings
    filter_start = time.perf_counter()
    items_with_embeddings_df = split_df.dropna(subset=["embedding"]).reset_index(drop=True)
    embeddings = items_with_embeddings_df["embedding"].tolist()
    filter_end = time.perf_counter()
    print(f"Filtering embeddings took {filter_end - filter_start:.4f} seconds")
    print(items_with_embeddings_df.info())

    # Step 4: Initialize the distance metric
    metric_start = time.perf_counter()
    metric = EmbeddingDistanceMetric(kind=kind, device="cpu")
    metric_end = time.perf_counter()
    print(f"Initializing metric took {metric_end - metric_start:.4f} seconds")

    # Step 5: Compute all pairwise scores
    score_start = time.perf_counter()
    print(f"Computing {kind} distances for {len(embeddings)} embeddings...")
    scores = metric.score_all(embeddings, embeddings)
    score_end = time.perf_counter()
    print(f"Score_all took {score_end - score_start:.3f} seconds")

    # Step 6: Create output file path
    output_file = out_path
    if out_path is None:
        output_file = Path(f"signclip_scores_{split_file.name}").with_suffix(".npz")

    if not output_file.suffix == ".npz":
        output_file = Path(f"{output_file}.npz")
        

    print(f"Scores will be saved to {output_file}")

 

    # Step 7: Extract file list from DataFrame
    files_start = time.perf_counter()
    files = items_with_embeddings_df["Video file"].tolist()
    files_end = time.perf_counter()
    print(f"Extracting file list took {files_end - files_start:.4f} seconds")


    analysis_start = time.perf_counter()    
    index_to_check = 0
    number_to_check = 10
    print(f"The first {number_to_check} scores for {files[index_to_check]} to...")
    for ref, score in list(zip(files, scores[index_to_check]))[:number_to_check]:
        print("\t*------------->", f"{ref}".ljust(35), "\t", score.item())

    unique_glosses = items_with_embeddings_df['Gloss'].unique()
    print(f"We have a vocabulary of {len(unique_glosses)} glosses")
    gloss_indices = {}
    for gloss in items_with_embeddings_df['Gloss'].unique():
        gloss_indices[gloss] = items_with_embeddings_df.index[items_with_embeddings_df['Gloss'] == gloss].tolist()

    for gloss, indices in gloss_indices.items():
        print(f"Here are the {len(indices)} indices for {gloss}:{indices}")

    # Assuming 'scores' is your distance matrix and 'gloss_indices' is your dictionary of gloss indices
    find_class_distances_start = time.perf_counter()
    all_within_class_distances = np.array([])  # Initialize as empty NumPy array
    all_between_class_distances = np.array([])  # Initialize as empty NumPy array

    within_class_means_by_gloss = {}
    for gloss, indices in tqdm(gloss_indices.items(), desc="Finding mean values by gloss"):
        # Within-class distances
        within_class_distances = scores[np.ix_(indices, indices)]
        within_class_mean = torch.mean(within_class_distances)
        within_class_means_by_gloss[gloss] = within_class_mean
        within_class_distances = within_class_distances[np.triu_indices(len(indices), k=1)]
        all_within_class_distances = np.concatenate([all_within_class_distances, within_class_distances.ravel()])

        # Between-class distances
        other_indices = np.setdiff1d(np.arange(len(scores)), indices)
        between_class_distances = scores[np.ix_(indices, other_indices)]
        all_between_class_distances = np.concatenate([all_between_class_distances, between_class_distances.ravel()])
    find_class_distances_end = time.perf_counter()


    print(f"Finding within and without took {find_class_distances_end-find_class_distances_start}")

    for gloss, mean in within_class_means_by_gloss.items():
        print(f"Within {gloss}: {within_class_means_by_gloss[gloss]}")

    print(f"Mean within classes: {np.mean(all_within_class_distances)}")
    print(f"Mean between classes: {np.mean(all_between_class_distances)}")

    
    analysis_end = time.perf_counter()
    analysis_duration = analysis_end - analysis_start
    print(f"Analysis took {analysis_duration} seconds")
    
    

    # Step 8: Save the scores and files to a compressed file
    save_start = time.perf_counter()
    np.savez(output_file, scores=scores, files=files)
    save_end = time.perf_counter()
    print(f"Saving scores and files took {save_end - save_start:.4f} seconds")
    print(f"Scores of shape {scores.shape} with files list of length {len(files)} saved to {output_file}")

    # Step 9: Read back the saved scores
    read_start = time.perf_counter()
    read_back_in = np.load(f"{output_file}")
    read_end = time.perf_counter()
    print(f"Reading back the file took {read_end - read_start:.4f} seconds")

    # Step 10: Verify if the read data matches the original scores
    verify_start = time.perf_counter()
    if np.allclose(read_back_in["scores"], scores):
        print("Yay! All the same!")
    else:
        print("Mismatch found!")
    verify_end = time.perf_counter()
    print(f"Verification step took {verify_end - verify_start:.4f} seconds")

    # Overall time
    overall_end = time.perf_counter()
    print(f"Total script runtime: {overall_end - overall_start:.4f} seconds")



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

    parser.add_argument("--out_path", 
                        type=Path, 
                        help="Where to save output distance npz matrix+file list")

    args = parser.parse_args()

    evaluate_signclip(emb_dir=args.emb_dir, split_file=args.split_file, kind=args.kind, out_path=args.out_path)

if __name__ == "__main__":
    main()
