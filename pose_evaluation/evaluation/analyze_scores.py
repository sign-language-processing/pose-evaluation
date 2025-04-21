from typing import Optional, List, Dict
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

    for gloss_a_path, group in tqdm(df.groupby(GLOSS_A_PATH), "Iterating over hyp paths", disable=len(df) < 1000):
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


def analyze_metric(metric_name: str, metric_df: pd.DataFrame, ks: List[int], out_folder=None) -> Dict[str, any]:
    result = {METRIC: metric_name}

    # Shuffle to prevent bias in case of score ties
    metric_df = metric_df.sample(frac=1, random_state=42)

    # Check signatures
    signatures = metric_df[SIGNATURE].str.split("=").str[0].unique()
    if metric_df[SIGNATURE].isna().any():
        print("nan_rows:")
        print(metric_df[metric_df[SIGNATURE].isna()])

    if (metric_df[SIGNATURE] == "").any():
        print("empty_rows:")
        print(metric_df[metric_df[SIGNATURE] == ""])
        print(signatures)

    assert len(signatures) == 1, signatures
    result[SIGNATURE] = signatures[0]

    # Gloss and gloss-pair stats
    gloss_tuples = metric_df["gloss_tuple"].unique()
    metric_glosses = set(metric_df[GLOSS_A].tolist() + metric_df[GLOSS_B].tolist())

    # Score subsets
    not_self_score_df = metric_df[metric_df[GLOSS_A] != metric_df[GLOSS_B]]
    self_scores_df = metric_df[metric_df[GLOSS_A] == metric_df[GLOSS_B]]

    result["unique_gloss_pairs"] = len(gloss_tuples)
    result["unique_glosses"] = len(metric_glosses)
    result["hyp_gloss_count"] = len(metric_df[GLOSS_A].unique())
    result["ref_gloss_count"] = len(metric_df[GLOSS_B].unique())
    result["total_count"] = len(metric_df)

    # Self-score stats
    result["self_scores_count"] = len(self_scores_df)
    result["mean_self_score"] = self_scores_df[SCORE].mean()
    result["std_self_score"] = self_scores_df[SCORE].std()

    gloss_stats_self = self_scores_df.groupby(GLOSS_A)[SCORE].agg(["mean", "std"])
    result["mean_of_gloss_self_score_means"] = gloss_stats_self["mean"].mean()
    result["std_of_gloss_self_score_means"] = gloss_stats_self["mean"].std()

    # Out-of-class stats
    result["out_of_class_scores_count"] = len(not_self_score_df)
    result["mean_out_of_class_score"] = not_self_score_df[SCORE].mean()
    result["std_out_of_class_score"] = not_self_score_df[SCORE].std()

    out_of_class_gloss_stats = not_self_score_df.groupby(GLOSS_A)[SCORE].agg(["mean", "std"])
    result["mean_of_out_of_class_score_means"] = out_of_class_gloss_stats["mean"].mean()
    result["std_of_out_of_class_score_means"] = out_of_class_gloss_stats["mean"].std()

    # Time stats
    result["mean_score_time"] = metric_df[TIME].mean()
    result["std_dev_of_score_time"] = metric_df[TIME].std()

    print(
        f"{metric_name} has \n"
        f"*\t{len(metric_df)} distances,\n"
        f"*\tcovering {len(metric_glosses)} glosses,\n"
        f"*\twith {len(metric_df[GLOSS_A].unique())} unique query glosses,\n"
        f"*\t{len(metric_df[GLOSS_B].unique())} unique ref glosses,\n"
        f"*\tin {len(gloss_tuples)} combinations,\n"
        f"*\t{len(not_self_score_df)} out-of-class scores,\n"
        f"*\t{len(self_scores_df)} in-class scores"
    )

    # Retrieval metrics
    retrieval_stats = calculate_retrieval_stats(metric_df, ks=ks)
    result.update(retrieval_stats)

    # mean in-gloss score
    # mean out-gloss score

    result["mean_out_gloss_score"] = not_self_score_df[SCORE].mean()
    result["mean_in_gloss_score"] = self_scores_df[SCORE].mean()
    result["mean_out_mean_in_ratio"] = result["mean_out_gloss_score"] / result["mean_in_gloss_score"]

    out_of_class_gloss_stats = (
        not_self_score_df.groupby("gloss_tuple")[SCORE].agg(["count", "mean", "max", "min", "std"]).reset_index()
    )
    out_of_class_gloss_stats = out_of_class_gloss_stats.sort_values("count", ascending=False)
    out_of_class_gloss_stats[METRIC] = metric_name
    out_of_class_gloss_stats.to_csv(out_folder / f"{metric}_out_of_class_scores_by_gloss.csv", index=False)

    return result


if __name__ == "__main__":

    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_analysis\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_analysis_with_times\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\embedding_analysis\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\combined_embedding_and_pose_stats\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\what_the_heck_why_pop\scores")
    stats_folder = Path(r"/opt/home/cleong/projects/pose-evaluation/metric_results/scores")

    # TODO: check if the number of CSVs has changed. If not, load deduplicated.

    analysis_folder = stats_folder.parent / "score_analysis_foo"
    analysis_folder.mkdir(exist_ok=True)
    metric_stats_out = analysis_folder / "stats_by_metric.csv"
    metric_by_gloss_stats_folder = analysis_folder / "metric_by_gloss_stats"
    metric_by_gloss_stats_folder.mkdir(exist_ok=True)
    ks = [1, 5, 10]

    previous_stats_by_metric = None

    # TODO: load from index json below
    if metric_stats_out.is_file():
        print(f"Loaded previous stats from {metric_stats_out}")
        previous_stats_by_metric = pd.read_csv(metric_stats_out)
        print(previous_stats_by_metric)
        print(previous_stats_by_metric.info())
        print(previous_stats_by_metric.describe())

    csv_stats_dfs = []
    csv_files = list(stats_folder.glob("*.csv"))
    for csv_file in tqdm(csv_files, desc="Loading scores csvs"):
        # print(f"Reading {csv_file}")
        # ,METRIC,SCORE,GLOSS_A,GLOSS_B,SIGNATURE,GLOSS_A_PATH,GLOSS_B_PATH,TIME
        scores_csv_df = pd.read_csv(
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

        scores_csv_df["SIGNATURE"] = scores_csv_df["SIGNATURE"].apply(
            lambda x: x.split("=")[0].strip() if "=" in x else x.strip()
        )
        assert len(scores_csv_df[METRIC].unique()) == 1
        assert len(scores_csv_df[SIGNATURE].unique()) == 1, f"{csv_file}, {scores_csv_df[SIGNATURE].unique()}"

        csv_stats_dfs.append(scores_csv_df)

    scores_df = pd.concat(csv_stats_dfs)

    print(f"{scores_df}")

    # Normalize gloss pairs by sorting them
    print("Creating gloss tuple pairs")
    # stats_df["path_tuple"] = [tuple(x) for x in np.sort(stats_df[[GLOSS_A_PATH, GLOSS_B_PATH]].values, axis=1)]
    scores_df["gloss_tuple"] = [tuple(x) for x in np.sort(scores_df[[GLOSS_A, GLOSS_B]].values, axis=1)]

    # TODO: This is where we should load from the index.
    metrics_to_analyze = scores_df[METRIC].unique()
    stats_by_metric = defaultdict(list)
    print(f"We have results for {len(metrics_to_analyze)}")
    # metrics_to_analyze = ["n-dtai-DTW-MJE (fast)", "MJE", "nMJE"]
    for metric_index, metric in enumerate(metrics_to_analyze):
        print("*" * 50)
        print(f"METRIC #{metric_index}/{len(metrics_to_analyze)}: {metric}")
        metric_df = scores_df[scores_df[METRIC] == metric]

        metric_stats = analyze_metric(metric, metric_df, ks, out_folder=metric_by_gloss_stats_folder)
        for k, v in metric_stats.items():
            stats_by_metric[k].append(v)

    stats_by_metric = pd.DataFrame(stats_by_metric)
    print(stats_by_metric)

    stats_by_metric.to_csv(metric_stats_out, index=False)


# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/analyze_scores.py
