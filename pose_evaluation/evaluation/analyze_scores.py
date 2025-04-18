from typing import Optional, List
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


def calculate_retrieval_stats(df: pd.DataFrame, ks: Optional[List[int]] = None) -> dict:
    if ks is None:
        ks = [1, 5, 10]
    results = {
        "mean_average_precision": 0.0,
        "mean_reciprocal_rank": 0.0,
    }
    per_k_stats = {k: {"precision": [], "recall": [], "match_count": 0} for k in ks}
    average_precisions = []
    reciprocal_ranks = []

    for gloss_a_path, group in df.groupby(GLOSS_A_PATH):
        filtered_group = group[group[GLOSS_B_PATH] != gloss_a_path]
        if filtered_group.empty:
            continue

        filtered_group = filtered_group.sample(frac=1, random_state=42)  # random_state for reproducibility
        sorted_group = filtered_group.sort_values(by=SCORE, ascending=True, kind="stable")
        correct_mask = sorted_group[GLOSS_A].values == sorted_group[GLOSS_B].values

        relevant_indices = np.where(correct_mask)[0]
        total_correct = correct_mask.sum()

        for k in ks:
            top_k_mask = correct_mask[:k]
            per_k_stats[k]["precision"].append(top_k_mask.sum() / k)
            if total_correct > 0:
                per_k_stats[k]["recall"].append(top_k_mask.sum() / total_correct)
            per_k_stats[k]["match_count"] += top_k_mask.sum()

        if relevant_indices.size > 0:
            ranks = np.arange(1, relevant_indices.size + 1)
            precision_at_ranks = (relevant_indices + 1) / ranks
            average_precisions.append(np.mean(precision_at_ranks))
            reciprocal_ranks.append(1 / (relevant_indices[0] + 1))

    # Global metrics
    results["mean_average_precision"] = float(np.mean(average_precisions)) if average_precisions else 0.0
    results["mean_reciprocal_rank"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    # Per-k metrics
    for k in ks:
        results[f"precision@{k}"] = float(np.mean(per_k_stats[k]["precision"])) if per_k_stats[k]["precision"] else 0.0
        results[f"recall@{k}"] = float(np.mean(per_k_stats[k]["recall"])) if per_k_stats[k]["recall"] else 0.0
        results[f"match_count@{k}"] = per_k_stats[k]["match_count"]

    return results


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
    metric_stats_out = analysis_folder / "stats_by_metric.csv"
    ks = [1, 5, 10]

    previous_stats_by_metric = None

    # TODO: check metric against previous stats, and skip calculating if possible
    if metric_stats_out.is_file():
        print(f"Loaded previous stats from {metric_stats_out}")
        previous_stats_by_metric = pd.read_csv(metric_stats_out)
        print(previous_stats_by_metric)
        print(previous_stats_by_metric.info())
        print(previous_stats_by_metric.describe())

    csv_stats_dfs = []
    for csv_file in tqdm(stats_folder.glob("*.csv"), desc="Loading scores csvs"):
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

        csv_stats_df["SIGNATURE"] = csv_stats_df["SIGNATURE"].apply(
            lambda x: x.split("=")[0].strip() if "=" in x else x.strip()
        )
        assert len(csv_stats_df[METRIC].unique()) == 1
        assert len(csv_stats_df[SIGNATURE].unique()) == 1, f"{csv_file}, {csv_stats_df[SIGNATURE].unique()}"

        csv_stats_dfs.append(csv_stats_df)        

    stats_df = pd.concat(csv_stats_dfs)

    print(f"{stats_df}")

    # Normalize path pairs by sorting them
    print("Creating gloss tuple pairs")
    # stats_df["path_tuple"] = [tuple(x) for x in np.sort(stats_df[[GLOSS_A_PATH, GLOSS_B_PATH]].values, axis=1)]
    stats_df["gloss_tuple"] = [tuple(x) for x in np.sort(stats_df[[GLOSS_A, GLOSS_B]].values, axis=1)]
    stats_by_metric = defaultdict(list)

    metrics_to_analyze = stats_df[METRIC].unique()
    print(f"We have results for {len(metrics_to_analyze)}")
    # metrics_to_analyze = ["n-dtai-DTW-MJE (fast)", "MJE", "nMJE"]
    for metric_index, metric in enumerate(metrics_to_analyze):
        print("*" * 50)
        print(f"METRIC #{metric_index}/{len(metrics_to_analyze)}: {metric}")

        metric_df = stats_df[stats_df[METRIC] == metric]
        # print(metric_df.head())
        # print(metric_df.columns)
        metric_df[TIME].head()
        # print(metric_df[TIME].mean())
        # print(len(metric_df[metric_df[TIME].isna()]))

        # Shuffle to prevent bias in case of score ties
        metric_df = metric_df.sample(frac=1, random_state=42)  # random_state for reproducibility

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

        # path_tuples = metric_df["path_tuple"].unique()
        gloss_tuples = metric_df["gloss_tuple"].unique()
        metric_glosses = metric_df[GLOSS_A].unique().tolist()
        metric_glosses.extend(metric_df[GLOSS_B].unique().tolist())
        metric_glosses = set(metric_glosses)

        not_self_score_df = metric_df[metric_df[GLOSS_A] != metric_df[GLOSS_B]]

        ################
        # Self-scores
        self_scores_df = metric_df[metric_df[GLOSS_A] == metric_df[GLOSS_B]]

        print(
            f"{metric} has \n"
            f"*\t{len(metric_df)} distances,\n"
            f"*\tcovering {len(metric_glosses)} glosses,\n"
            f"*\twith {len(metric_df[GLOSS_A].unique())} unique query glosses,\n"
            f"*\t{len(metric_df[GLOSS_B].unique())} unique ref glosses,\n"
            f"*\tin {len(gloss_tuples)} combinations,\n"
            # f"*\twith {len(path_tuples)} file combinations.\n"
            f"*\tThere are {len(not_self_score_df)} out-of-class-scores,\n"
            f"*\tand {len(self_scores_df)} in-class scores"
        )

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

        ############################################################################
        # Out of gloss
        # Rank by "mean" in ascending order (use ascending=False for descending)

        assert len(signatures) == 1, signatures
        stats_by_metric[METRIC].append(metric)
        stats_by_metric[SIGNATURE].append(signatures[0])
        stats_by_metric["unique_gloss_pairs"].append(len(gloss_tuples))
        stats_by_metric["unique_glosses"].append(len(metric_glosses))
        stats_by_metric["total_count"].append(len(metric_df))

        stats_by_metric["self_scores_count"].append(len(self_scores_df))
        stats_by_metric["mean_self_score"].append(self_scores_df[SCORE].mean())
        stats_by_metric["std_self_score"].append(self_scores_df[SCORE].std())

        stats_by_metric["out_of_class_scores_count"].append(len(not_self_score_df))
        stats_by_metric["mean_out_of_class_score"].append(not_self_score_df[SCORE].mean())
        stats_by_metric["std_out_of_class_score"].append(not_self_score_df[SCORE].std())

        stats_by_metric["mean_score_time"].append(metric_df[TIME].mean())
        stats_by_metric["std_dev_of_score_time"].append(metric_df[TIME].std())

        ##############################
        # Retrieval Stats
        # Calculate retrieval stats for each metric
        print("Calculating retrieval stats")

        retrieval_stats = calculate_retrieval_stats(metric_df, ks=ks)

        # Append global metrics
        stats_by_metric["mrr_of_same_gloss"].append(retrieval_stats["mean_reciprocal_rank"])
        stats_by_metric["map_of_same_gloss"].append(retrieval_stats["mean_average_precision"])

        # Append per-k metrics
        for k in ks:
            stats_by_metric[f"precision@{k}"].append(retrieval_stats[f"precision@{k}"])
            stats_by_metric[f"recall@{k}"].append(retrieval_stats[f"recall@{k}"])

        stats_by_metric["mean_out_mean_in_ratio"].append(not_self_score_df[SCORE].mean() / self_scores_df[SCORE].mean())
        stats_df = stats_df[stats_df[METRIC] != metric] # delete. 
        print(f"{len(metric_df)} metric rows removed, now stats_df is {len(stats_df)} long")
        print("#" * 50)

    stats_by_metric = pd.DataFrame(stats_by_metric)
    print(stats_by_metric)

    stats_by_metric.to_csv(metric_stats_out, index=False)

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


# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/analyze_scores.py
