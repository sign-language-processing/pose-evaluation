import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pose_evaluation.metrics.embedding_distance_metric import EmbeddingDistanceMetric


def load_embedding(file_path: Path) -> np.ndarray:
    """
    Load a SignCLIP embedding from a .npy file, ensuring it has the correct
    shape.

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
    Match .npy embeddings to the corresponding glosses based on the
    numerical ID.

    Args:
        emb_dir (Path): Directory containing the .npy files.
        split_df (pd.DataFrame): DataFrame containing the split file with the "Video file" column.

    Returns:
        pd.DataFrame: Updated DataFrame with an additional column for embeddings.

    """
    # Step 1: Create a mapping of numerical IDs to .npy files
    map_start = time.perf_counter()
    embeddings_map = {npy_file.stem.split("-")[0]: npy_file for npy_file in emb_dir.glob("*.npy")}
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


def calculate_mean_distances(
    distance_matrix: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    exclude_self: bool = False,
) -> float:
    """
    Calculate the mean of distances between two sets of indices in a 2D
    distance matrix.

    Args:
        distance_matrix (torch.Tensor): A 2D tensor representing pairwise distances.
        indices_a (torch.Tensor): A tensor of row indices.
        indices_b (torch.Tensor): A tensor of column indices.
        exclude_self (bool): Whether to exclude distances where indices_a == indices_b.

    Returns:
        float: The mean distance between all pairs of (indices_a, indices_b).

    """
    # Create all pair combinations
    row_indices, col_indices = torch.meshgrid(indices_a, indices_b, indexing="ij")

    if exclude_self:
        # Apply a mask to remove self-distances
        mask = row_indices != col_indices
        row_indices = row_indices[mask]
        col_indices = col_indices[mask]

    # Gather distances
    selected_distances = distance_matrix[row_indices.flatten(), col_indices.flatten()]

    # Return the mean
    return selected_distances.mean().item()


def generate_synthetic_data(num_items, num_classes, num_items_per_class=4):
    torch.manual_seed(42)
    random.seed(42)
    # distance_matrix = torch.rand((num_items, num_items)) * 100
    distance_matrix = torch.full((num_items, num_items), 10.0)
    distance_matrix.fill_diagonal_(0)
    indices = list(range(num_items))
    random.shuffle(indices)

    classes = {
        f"CLASS_{i}": torch.tensor([indices.pop() for _ in range(num_items_per_class)]) for i in range(num_classes)
    }
    # Assign intra-class distances
    mean_values_by_class = {}
    for i, class_name in enumerate(classes.keys()):
        mean_value = i + 1
        mean_values_by_class[class_name] = mean_value
    for class_name, indices in classes.items():
        mean_value = mean_values_by_class[class_name]
        for i in indices:
            for j in indices:
                if i != j:  # Exclude self-distances
                    distance_matrix[i, j] = mean_value
    return classes, distance_matrix


def calculate_class_means(gloss_indices, scores):
    class_means_by_gloss = {}
    all_indices = torch.arange(scores.size(0), dtype=int)

    for gloss, indices in tqdm(gloss_indices.items(), desc="Finding mean values by gloss"):
        indices = torch.LongTensor(indices)
        class_means_by_gloss[gloss] = {}
        within_class_mean = calculate_mean_distances(scores, indices, indices, exclude_self=True)

        class_means_by_gloss[gloss]["in_class"] = within_class_mean

        complement_indices = all_indices[~torch.isin(all_indices, indices)]
        without_class_mean = calculate_mean_distances(scores, indices, complement_indices)
        class_means_by_gloss[gloss]["out_of_class"] = without_class_mean

    return class_means_by_gloss


# def calculate_class_means(gloss_indices, scores):
#    all_within_class_distances = np.array([])  # Initialize as empty NumPy array
#    all_between_class_distances = np.array([])  # Initialize as empty NumPy array
#    within_class_means_by_gloss = {}
#    for gloss, indices in tqdm(gloss_indices.items(), desc="Finding mean values by gloss"):
#        # Within-class distances
#        within_class_distances = scores[np.ix_(indices, indices)]
#        within_class_mean = torch.mean(within_class_distances)
#        within_class_means_by_gloss[gloss] = within_class_mean
#        within_class_distances = within_class_distances[np.triu_indices(len(indices), k=1)]
#        all_within_class_distances = np.concatenate([all_within_class_distances, within_class_distances.ravel()])
#
#        # Between-class distances
#        other_indices = np.setdiff1d(np.arange(len(scores)), indices)
#        between_class_distances = scores[np.ix_(indices, other_indices)]
#        all_between_class_distances = np.concatenate([all_between_class_distances, between_class_distances.ravel()])
#
#    for gloss, mean in within_class_means_by_gloss.items():
#        print(f"Within {gloss}: {within_class_means_by_gloss[gloss]}")
#
#    print(f"Mean within classes: {np.mean(all_within_class_distances)}")
#    print(f"Mean between classes: {np.mean(all_between_class_distances)}")
#    return within_class_means_by_gloss


def evaluate_signclip(emb_dir: Path, split_file: Path, out_path: Path, kind: str = "cosine"):  # pylint: disable=too-many-locals, too-many-statements
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
    # metric = EmbeddingDistanceMetric(kind=kind, device="cpu")
    metric = EmbeddingDistanceMetric(kind=kind)
    metric_end = time.perf_counter()
    print(f"Initializing metric took {metric_end - metric_start:.4f} seconds")

    # Step 5: Compute all pairwise scores
    score_start = time.perf_counter()
    print(f"Computing {kind} distances for {len(embeddings)} embeddings...")
    scores = metric.score_all(embeddings, embeddings)
    score_end = time.perf_counter()
    print(f"Score_all took {score_end - score_start:.3f} seconds")

    # Step 7: Extract file list from DataFrame
    files_start = time.perf_counter()
    files = items_with_embeddings_df["Video file"].tolist()
    files_end = time.perf_counter()
    print(f"Extracting file list took {files_end - files_start:.4f} seconds")

    analysis_start = time.perf_counter()
    index_to_check = 0
    number_to_check = 10
    print(f"The first {number_to_check} scores for {files[index_to_check]} to...")
    for ref, score in list(zip(files, scores[index_to_check], strict=False))[:number_to_check]:
        print("\t*------------->", f"{ref}".ljust(35), "\t", score.item())

    unique_glosses = items_with_embeddings_df["Gloss"].unique()
    print(f"We have a vocabulary of {len(unique_glosses)} glosses")
    gloss_indices = {}
    for gloss in items_with_embeddings_df["Gloss"].unique():
        gloss_indices[gloss] = items_with_embeddings_df.index[items_with_embeddings_df["Gloss"] == gloss].tolist()

    for gloss, indices in list(gloss_indices.items())[:10]:
        print(f"Here are the {len(indices)} indices for {gloss}:{indices}")

    find_class_distances_start = time.perf_counter()

    # synthetic_classes, synthetic_distances = generate_synthetic_data(30000, 2700, 8)
    # class_means = calculate_class_means(synthetic_classes, synthetic_distances)
    class_means = calculate_class_means(gloss_indices, scores)

    find_class_distances_end = time.perf_counter()

    print(f"Finding within and without took {find_class_distances_end - find_class_distances_start}")

    analysis_end = time.perf_counter()
    analysis_duration = analysis_end - analysis_start

    in_class_means = [mean_dict["in_class"] for mean_dict in class_means.values()]
    out_class_means = [mean_dict["out_of_class"] for mean_dict in class_means.values()]

    for gloss, means in list(class_means.items())[:10]:
        print(gloss, means)

    print(f"Mean of in-class means: {np.mean(in_class_means)}")
    print(f"Mean of out-of-class means: {np.mean(out_class_means)}")

    print(f"Analysis took {analysis_duration} seconds")

    # Step 8: Save the scores and files to a compressed file

    save_start = time.perf_counter()
    class_means_json = out_path.with_name(f"{out_path.stem}_class_means").with_suffix(".json")
    with open(class_means_json, "w", encoding="utf-8") as f:
        print(f"Writing class means to {f}")
        json.dump(class_means, f)
    np.savez(out_path, scores=scores, files=files)
    save_end = time.perf_counter()
    print(f"Saving scores and files took {save_end - save_start:.4f} seconds")
    print(f"Scores of shape {scores.shape} with files list of length {len(files)} saved to {out_path}")

    # Step 9: Read back the saved scores
    read_start = time.perf_counter()
    read_back_in = np.load(f"{out_path}")
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
        "emb_dir",
        type=Path,
        help="Path to the directory containing SignCLIP .npy files",
    )
    parser.add_argument(
        "--split_file",
        type=Path,
        required=True,
        help="Path to the split CSV file (e.g., test.csv)",
    )
    parser.add_argument(
        "--kind",
        type=str,
        choices=["cosine", "l2"],
        default="cosine",
        help="Type of distance metric to use (default: cosine)",
    )

    parser.add_argument(
        "--out_path",
        type=Path,
        help="Where to save output distance npz matrix+file list",
    )

    args = parser.parse_args()

    output_file = args.out_path
    if output_file is None:
        output_file = Path(f"signclip_scores_{args.split_file.name}").with_suffix(".npz")

    if output_file.suffix != ".npz":
        output_file = Path(f"{output_file}.npz")

    print(f"Scores will be saved to {output_file}")

    evaluate_signclip(
        emb_dir=args.emb_dir,
        split_file=args.split_file,
        out_path=output_file,
        kind=args.kind,
    )


if __name__ == "__main__":
    main()
