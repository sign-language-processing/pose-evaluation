import argparse
from pathlib import Path
import time
import json
import random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pose_evaluation.metrics.embedding_distance_metric import EmbeddingDistanceMetric


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

    # Step 1: Create a mapping of numerical IDs to .npy files
    map_start = time.perf_counter()
    embeddings_map = {
        npy_file.stem.split("-")[0]: npy_file for npy_file in emb_dir.glob("*.npy")
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


def calculate_mean_distances(
    distance_matrix: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    exclude_self: bool = False,
) -> float:
    """
    Calculate the mean of distances between two sets of indices in a 2D distance matrix.

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
        f"CLASS_{i}": torch.tensor([indices.pop() for _ in range(num_items_per_class)])
        for i in range(num_classes)
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

    for gloss, indices in tqdm(
        gloss_indices.items(), desc="Finding mean values by gloss"
    ):
        indices = torch.LongTensor(indices)
        class_means_by_gloss[gloss] = {}
        within_class_mean = calculate_mean_distances(
            scores, indices, indices, exclude_self=True
        )

        class_means_by_gloss[gloss]["in_class"] = within_class_mean

        complement_indices = all_indices[~torch.isin(all_indices, indices)]
        without_class_mean = calculate_mean_distances(
            scores, indices, complement_indices
        )
        class_means_by_gloss[gloss]["out_of_class"] = without_class_mean

    return class_means_by_gloss


def load_split_file(split_file: Path) -> pd.DataFrame:
    start = time.perf_counter()
    df = pd.read_csv(split_file)
    print(f"Loading split file took {time.perf_counter() - start:.4f} seconds")
    return df


def match_and_filter_embeddings(emb_dir: Path, split_df: pd.DataFrame):
    start = time.perf_counter()
    split_df = match_embeddings_to_glosses(emb_dir, split_df)
    print(
        f"Matching embeddings to glosses took {time.perf_counter() - start:.4f} seconds"
    )

    start = time.perf_counter()
    filtered_df = split_df.dropna(subset=["embedding"]).reset_index(drop=True)
    embeddings = filtered_df["embedding"].tolist()
    print(f"Filtering embeddings took {time.perf_counter() - start:.4f} seconds")
    print(filtered_df.info())
    return filtered_df, embeddings


def init_metric(kind: str):
    start = time.perf_counter()
    metric = EmbeddingDistanceMetric(kind=kind)
    print(f"Initializing metric took {time.perf_counter() - start:.4f} seconds")
    return metric


def compute_scores(metric, embeddings):
    start = time.perf_counter()
    print(f"Computing {metric.kind} distances for {len(embeddings)} embeddings...")
    scores = metric.score_all(embeddings, embeddings)
    print(f"Score_all took {time.perf_counter() - start:.3f} seconds")
    return scores


def extract_files(df: pd.DataFrame):
    start = time.perf_counter()
    files = df["Video file"].tolist()
    print(f"Extracting file list took {time.perf_counter() - start:.4f} seconds")
    return files


def analyze_results(df: pd.DataFrame, files, scores):
    start = time.perf_counter()
    print(f"The first 10 scores for {files[0]} to...")
    for ref, score in list(zip(files, scores[0]))[:10]:
        print("\t*------------->", f"{ref}".ljust(35), "\t", score.item())

    gloss_indices = {
        gloss: df.index[df["Gloss"] == gloss].tolist() for gloss in df["Gloss"].unique()
    }
    print(f"We have a vocabulary of {len(gloss_indices)} glosses")
    for gloss, indices in list(gloss_indices.items())[:10]:
        print(f"Here are the {len(indices)} indices for {gloss}:{indices}")

    cm_start = time.perf_counter()
    class_means = calculate_class_means(gloss_indices, scores)
    print(f"Finding within and without took {time.perf_counter() - cm_start}")

    for gloss, means in list(class_means.items())[:10]:
        print(gloss, means)

    print(
        f"Mean of in-class means: {np.mean([m['in_class'] for m in class_means.values()])}"
    )
    print(
        f"Mean of out-of-class means: {np.mean([m['out_of_class'] for m in class_means.values()])}"
    )
    print(f"Analysis took {time.perf_counter() - start} seconds")

    return class_means


def save_and_verify(scores, files, out_path: Path, class_means):
    start = time.perf_counter()
    class_means_json = out_path.with_name(f"{out_path.stem}_class_means").with_suffix(
        ".json"
    )
    with open(class_means_json, "w", encoding="utf-8") as f:
        print(f"Writing class means to {f}")
        json.dump(class_means, f)
    np.savez(out_path, scores=scores, files=files)
    print(f"Saving scores and files took {time.perf_counter() - start:.4f} seconds")
    print(
        f"Scores of shape {scores.shape} with files list of length {len(files)} saved to {out_path}"
    )

    start = time.perf_counter()
    read_back_in = np.load(f"{out_path}")
    print(f"Reading back the file took {time.perf_counter() - start:.4f} seconds")

    start = time.perf_counter()
    if np.allclose(read_back_in["scores"], scores):
        print("Yay! All the same!")
    else:
        print("Mismatch found!")
    print(f"Verification step took {time.perf_counter() - start:.4f} seconds")


def evaluate_signclip(
    emb_dir: Path, split_file: Path, out_path: Path, kind: str = "cosine"
):
    """
    Evaluate SignCLIP embeddings using score_all.

    Args:
        emb_dir (Path): Directory containing .npy embeddings.
        split_file (Path): Path to the split CSV file.
        kind (str): Metric type ("cosine" or "l2"). Default is "cosine".
    """
    overall_start = time.perf_counter()  # Start overall benchmarking

    split_df = load_split_file(split_file)
    split_df, embeddings = match_and_filter_embeddings(emb_dir, split_df)
    metric = init_metric(kind)
    scores = compute_scores(metric, embeddings)
    files = extract_files(split_df)
    class_means = analyze_results(split_df, files, scores)
    save_and_verify(scores, files, out_path, class_means)

    print(f"Total script runtime: {time.perf_counter() - overall_start:.4f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SignCLIP embeddings with score_all."
    )
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
        output_file = Path(f"signclip_scores_{args.split_file.name}").with_suffix(
            ".npz"
        )

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
