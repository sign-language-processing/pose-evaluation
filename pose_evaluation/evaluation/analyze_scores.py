from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px

tqdm.pandas()

# ,METRIC,SCORE,GLOSS_A,GLOSS_B,SIGNATURE,GLOSS_A_PATH,GLOSS_B_PATH,TIME
GLOSS_A_PATH = "GLOSS_A_PATH"
GLOSS_B_PATH = "GLOSS_B_PATH"
GLOSS_A = "GLOSS_A"
GLOSS_B = "GLOSS_B"
SCORE = "SCORE"
METRIC = "METRIC"
SIGNATURE = "SIGNATURE"
TIME = "TIME"


def precision_at_k(df: pd.DataFrame, k: int) -> float:
    precisions = []

    for gloss_a_path, group in df.groupby(GLOSS_A_PATH):
        # Exclude rows where 'Gloss B Path' matches 'Gloss A Path'
        filtered_group = group[group[GLOSS_B_PATH] != gloss_a_path]

        # Sort by score (lower is better)
        sorted_group = filtered_group.sort_values(by=SCORE, ascending=True)

        # Select top-k results
        top_k = sorted_group.head(k)

        # Compute precision@k as (# of correct matches in top-k) / k
        correct_matches = (top_k[GLOSS_A] == top_k[GLOSS_B]).sum()
        precisions.append(correct_matches / k)

    # Compute mean precision@k over all groups
    return sum(precisions) / len(precisions) if precisions else 0.0


def recall_at_k(df: pd.DataFrame, k: int) -> float:
    recalls = []

    for gloss_a_path, group in df.groupby(GLOSS_A_PATH):
        # Exclude trivial cases
        filtered_group = group[group[GLOSS_B_PATH] != gloss_a_path]

        # Sort by score (ascending)
        sorted_group = filtered_group.sort_values(by=SCORE, ascending=True)

        # Count total correct matches in the full set
        total_correct = (sorted_group[GLOSS_A] == sorted_group[GLOSS_B]).sum()
        if total_correct == 0:
            continue  # Skip if there are no true matches

        # Select top-k results
        top_k = sorted_group.head(k)

        # Compute recall@k
        correct_in_top_k = (top_k[GLOSS_A] == top_k[GLOSS_B]).sum()
        recalls.append(correct_in_top_k / total_correct)

    # Compute mean recall@k
    return sum(recalls) / len(recalls) if recalls else 0.0


def match_count_at_k(df: pd.DataFrame, k: int) -> int:
    match_count = 0

    for gloss_a_path, group in df.groupby(GLOSS_A_PATH):
        # Exclude trivial cases
        filtered_group = group[group[GLOSS_B_PATH] != gloss_a_path]

        # Sort by score (ascending)
        sorted_group = filtered_group.sort_values(by=SCORE, ascending=True)

        # Select top-k results
        top_k = sorted_group.head(k)

        # Count matches where 'Gloss A' == 'Gloss B'
        match_count += (top_k[GLOSS_A] == top_k[GLOSS_B]).sum()

    return match_count


def mean_average_precision(df: pd.DataFrame) -> float:
    average_precisions = []

    for gloss_a_path, group in df.groupby(GLOSS_A_PATH):
        # Exclude trivial cases
        filtered_group = group[group[GLOSS_B_PATH] != gloss_a_path]

        # Sort by score (ascending)
        sorted_group = filtered_group.sort_values(by=SCORE, ascending=True)

        # Identify relevant positions
        relevant_mask = sorted_group[GLOSS_A].values == sorted_group[GLOSS_B].values
        relevant_indices = np.where(relevant_mask)[0]  # Convert to NumPy array of indices

        if relevant_indices.size == 0:
            continue  # Skip if there are no correct matches

        # Compute precision at each relevant index
        ranks = np.arange(1, relevant_indices.size + 1)
        precision_at_ranks = (relevant_indices + 1) / ranks  # Using NumPy broadcasting

        # Average Precision for this query
        ap = np.mean(precision_at_ranks)
        average_precisions.append(ap)

    # Mean Average Precision
    return float(np.mean(average_precisions)) if average_precisions else 0.0


def mean_reciprocal_rank(df: pd.DataFrame) -> float:
    reciprocal_ranks = []

    for gloss_a_path, group in df.groupby(GLOSS_A_PATH):
        # Exclude rows where 'Gloss B Path' matches 'Gloss A Path'
        filtered_group = group[group[GLOSS_B_PATH] != gloss_a_path]

        # Sort by score in ascending order (lower score is better)
        sorted_group = filtered_group.sort_values(by=SCORE, ascending=True)

        # Find the first correct match
        relevant_mask = sorted_group[GLOSS_A].values == sorted_group[GLOSS_B].values
        relevant_indices = np.where(relevant_mask)[0]  # NumPy array of relevant indices

        if relevant_indices.size > 0:
            rank = relevant_indices[0] + 1  # Convert zero-based index to one-based rank
            reciprocal_ranks.append(1 / rank)

    # Compute Mean Reciprocal Rank
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def mean_match_count_at_k(df: pd.DataFrame, k: int) -> float:
    match_counts = []

    for gloss_a_path, group in df.groupby(GLOSS_A_PATH):
        # Exclude trivial cases
        filtered_group = group[group[GLOSS_B_PATH] != gloss_a_path]

        # Sort by score (ascending)
        sorted_group = filtered_group.sort_values(by=SCORE, ascending=True)

        # Select top-k results
        top_k = sorted_group.head(k)

        # Count matches where 'Gloss A' == 'Gloss B'
        match_count = (top_k[GLOSS_A] == top_k[GLOSS_B]).sum()
        match_counts.append(match_count)

    # Compute mean match count across all groups
    return sum(match_counts) / len(match_counts) if match_counts else 0.0


def standardize_path_order(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `Gloss A Path` always contains the original query path if it exists in the dataset."""
    df = df.copy()

    # Create a mask for rows where `Gloss B Path` appears in `Gloss A Path`
    mask = df[GLOSS_B_PATH].isin(df[GLOSS_A_PATH])

    # Swap columns where the mask is True
    df.loc[mask, [GLOSS_A_PATH, GLOSS_B_PATH]] = df.loc[mask, [GLOSS_B_PATH, GLOSS_A_PATH]].values
    df.loc[mask, [GLOSS_A, GLOSS_B]] = df.loc[mask, [GLOSS_B, GLOSS_A]].values

    return df


if __name__ == "__main__":

    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_analysis\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_analysis_with_times\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\embedding_analysis\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\combined_embedding_and_pose_stats\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\what_the_heck_why_pop\scores")
    stats_folder = Path(r"/opt/home/cleong/projects/pose-evaluation/metric_results/scores")

    # TODO: check if the number of CSVs has changed. If not, load deduplicated.

    analysis_folder = stats_folder.parent / "score_analysis"
    analysis_folder.mkdir(exist_ok=True)

    csv_stats_dfs = []
    for csv_file in tqdm(stats_folder.glob("*.csv"), desc="Loading stats csvs"):
        # print(f"Reading {csv_file}")
        # ,METRIC,SCORE,GLOSS_A,GLOSS_B,SIGNATURE,GLOSS_A_PATH,GLOSS_B_PATH,TIME
        csv_stats_df = pd.read_csv(
            csv_file,
            index_col=0,
            dtype={
                "METRIC": str,
                "GLOSS_A": str,
                "GLOSS_B": str,
                "SIGNATURE": str,
                "GLOSS_A_PATH": str,
                "GLOSS_B_PATH": str,
            },
            float_precision="high",
        )

        csv_stats_dfs.append(csv_stats_df)
        # print(f"{len(csv_stats_dfs[-1])} rows loaded")
    stats_df = pd.concat(csv_stats_dfs)

    print(f"{stats_df}")

    # Normalize path pairs by sorting them
    print("Creating path and gloss tuple pairs")
    stats_df["path_tuple"] = [tuple(x) for x in np.sort(stats_df[[GLOSS_A_PATH, GLOSS_B_PATH]].values, axis=1)]
    stats_df["gloss_tuple"] = [tuple(x) for x in np.sort(stats_df[[GLOSS_A, GLOSS_B]].values, axis=1)]
    metric_stats = defaultdict(list)
    metric_stats_at_k = defaultdict(list)

    metrics_to_analyze = stats_df[METRIC].unique()
    print(f"We have results for {len(metrics_to_analyze)}")
    # metrics_to_analyze = ["n-dtai-DTW-MJE (fast)", "MJE", "nMJE"]
    for metric_index, metric in enumerate(metrics_to_analyze):
        print(f"*" * 50)
        print(f"METRIC: {metric}")

        metric_df = stats_df[stats_df[METRIC] == metric]
        # print(metric_df.head())
        # print(metric_df.columns)
        metric_df[TIME].head()
        # print(metric_df[TIME].mean())
        # print(len(metric_df[metric_df[TIME].isna()]))

        signatures = metric_df[SIGNATURE].str.split("=").str[0].unique()
        nan_rows = metric_df[metric_df[SIGNATURE].isna()]
        if len(nan_rows) > 0:
            print("nan_rows:")
            print(nan_rows)

        empty_rows = metric_df[metric_df[SIGNATURE] == ""]
        if len(empty_rows) > 0:
            print("empty_rows")
            print(empty_rows)
            print(signatures)

        path_tuples = metric_df["path_tuple"].unique()
        gloss_tuples = metric_df["gloss_tuple"].unique()
        metric_glosses = metric_df[GLOSS_A].unique().tolist()
        metric_glosses.extend(metric_df[GLOSS_B].unique().tolist())
        metric_glosses = set(metric_glosses)

        # Group by 'gloss_tuple' and compute stats for 'score'
        not_self_score_df = metric_df[metric_df[GLOSS_A] != metric_df[GLOSS_B]]

        # print(f"{metric} has {len(not_self_score_df)} out-of-class-scores: ")
        out_of_class_gloss_stats = (
            not_self_score_df.groupby("gloss_tuple")[SCORE].agg(["count", "mean", "max", "min", "std"]).reset_index()
        )
        out_of_class_gloss_stats = out_of_class_gloss_stats.sort_values("count", ascending=False)
        out_of_class_gloss_stats[METRIC] = metric

        ################
        # Self-scores
        self_scores_df = metric_df[metric_df[GLOSS_A] == metric_df[GLOSS_B]]

        print(
            f"{metric} has \n*\t{len(metric_df)} distances, \n*\tcovering {len(metric_glosses)} glosses,\n*\twith {len(metric_df[GLOSS_A].unique())} unique query glosses,\n*\t{len(metric_df[GLOSS_B].unique())} unique ref glosses\n*\tin {len(gloss_tuples)} combinations \n*\twith {len(path_tuples)} file combinations. \n*\tThere are {len(not_self_score_df)} out-of-class-scores, \n*\tand {len(self_scores_df)} in-class scores"
        )

        gloss_stats_self = (
            self_scores_df.groupby("gloss_tuple")[SCORE].agg(["count", "mean", "max", "min", "std"]).reset_index()
        )
        # print(f"Gloss stats for self-scores for {metric}")
        # print(gloss_stats_self)

        ###############################
        # Add to metric stats
        # METRIC: [],
        # SIGNATURE: [],
        # "unique_gloss_pairs": [],
        # "total_count": [],
        # "self_scores_count": [],
        # "mean": [],
        # "max": [],
        # "std": [],
        # "std_of_gloss_std": [],
        # "std_of_of_gloss_mean": [],
        # "mean_of_gloss_mean": [],
        # metric_stats[METRIC].append(metric)

        ####
        # Rank by "mean" in ascending order (use ascending=False for descending)
        out_of_class_gloss_stats = out_of_class_gloss_stats.sort_values("mean", ascending=True)

        # Add a rank column based on the sorted order
        out_of_class_gloss_stats["rank"] = range(1, len(out_of_class_gloss_stats) + 1)

        assert len(signatures) == 1, signatures
        metric_stats[METRIC].append(metric)
        metric_stats[SIGNATURE].append(signatures[0])
        metric_stats["unique_gloss_pairs"].append(len(gloss_tuples))
        metric_stats["unique_glosses"].append(len(metric_glosses))
        metric_stats["total_count"].append(len(metric_df))

        metric_stats["self_scores_count"].append(len(self_scores_df))
        metric_stats["mean_self_score"].append(self_scores_df[SCORE].mean())
        metric_stats["std_self_score"].append(self_scores_df[SCORE].std())
        metric_stats["mean_of_gloss_self_score_means"].append(gloss_stats_self["mean"].mean())
        metric_stats["std_of_gloss_self_score_means"].append(gloss_stats_self["mean"].std())

        metric_stats["out_of_class_scores_count"].append(len(not_self_score_df))
        metric_stats["mean_out_of_class_score"].append(not_self_score_df[SCORE].mean())
        metric_stats["std_out_of_class_score"].append(not_self_score_df[SCORE].std())
        metric_stats["mean_of_out_of_class_score_means"].append(out_of_class_gloss_stats["mean"].mean())
        metric_stats["std_of_out_of_class_score_means"].append(out_of_class_gloss_stats["mean"].std())

        metric_stats["mean_score_time"].append(metric_df[TIME].mean())
        metric_stats["std_dev_of_score_time"].append(metric_df[TIME].std())

        metric_stats["mrr_of_same_gloss"].append(mean_reciprocal_rank(metric_df))
        metric_stats["map_of_same_gloss"].append(mean_average_precision(metric_df))

        print("#" * 30)

        ########################

        for k in tqdm(
            [1, 5, 10], desc=f"Metric #{metric_index}/{len(metrics_to_analyze)} Calculating retrieval metrics at k"
        ):
            metric_stats_at_k[METRIC].append(metric)
            metric_stats_at_k[SIGNATURE].append(signatures[0])
            metric_stats_at_k["k"].append(k)

            prec_at_k = precision_at_k(metric_df, k)
            rec_at_k = recall_at_k(metric_df, k)

            metric_stats_at_k["recall@k"].append(rec_at_k)
            metric_stats_at_k["precision@k"].append(prec_at_k)
            metric_stats_at_k["mean_match_count@k"].append(mean_match_count_at_k(metric_df, k))

            metric_stats[f"precision@{k}"].append(prec_at_k)
            metric_stats[f"recall@{k}"].append(rec_at_k)

        metric_stats["mean_out_mean_in_ratio"].append(not_self_score_df[SCORE].mean() / self_scores_df[SCORE].mean())

        print(f"Saving analysis outputs to {analysis_folder}")
        out_of_class_gloss_stats.to_csv(analysis_folder / f"{metric}_out_of_class_scores_by_gloss.csv", index=False)

    metric_stats = pd.DataFrame(metric_stats)
    print(metric_stats)
    metric_stats_out = analysis_folder / "stats_by_metric.csv"
    metric_stats.to_csv(metric_stats_out, index=False)

    metric_stats_at_k = pd.DataFrame(metric_stats_at_k)
    metric_stats_at_k_out = analysis_folder / "stats_by_metric_at_k.csv"
    metric_stats_at_k.to_csv(metric_stats_at_k_out, index=False)

    fig = px.box(
        stats_df,
        x=METRIC,
        y=TIME,
        points=None,
        title="Metric Pairwise Scoring Times (s)",
        color=METRIC,
    )
    # fig.update_layout(xaxis_type="category")
    # fig.update_layout(xaxis={"categoryorder": "category ascending"})
    # fig.update_layout(xaxis={"categoryorder": "trace"})
    # fig.update_layout(xaxis_tickangle=-45)  # Rotate x labels for readability
    fig.update_layout(xaxis=dict(categoryorder="trace", showticklabels=False, ticks=""))

    fig.write_html(analysis_folder / "metric_pairwise_scoring_time_distributions.html")
