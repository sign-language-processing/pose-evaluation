from typing import Optional, List, Dict
from collections import defaultdict
from pathlib import Path
import json

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalPrecision, RetrievalRecall

from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol, load_score_csv
from pose_evaluation.evaluation.index_score_files import ScoresIndexDFCol, index_scores


tqdm.pandas()


def calculate_retrieval_stats(df: pd.DataFrame, ks: Optional[List[int]] = None) -> dict:
    if ks is None:
        ks = [1, 5, 10]

    results = {
        "mean_average_precision": 0.0,
        "mean_reciprocal_rank": 0.0,
    }
    per_k_stats = {k: {"precision": [], "recall": [], "match_count": 0} for k in ks}

    all_preds = []
    all_targets = []
    all_indexes = []

    group_id = 0
    for gloss_a_path, group in tqdm(
        df.groupby(ScoreDFCol.GLOSS_A_PATH), "Iterating over hyp paths", disable=len(df) < 1000
    ):
        filtered_group = group[group[ScoreDFCol.GLOSS_B_PATH] != gloss_a_path]
        if filtered_group.empty:
            continue

        filtered_group = filtered_group.sample(frac=1, random_state=42)  # shuffle
        sorted_group = filtered_group.sort_values(
            by=ScoreDFCol.SCORE, ascending=True, kind="stable"
        )  # lower score = better
        correct_mask = sorted_group[ScoreDFCol.GLOSS_A].values == sorted_group[ScoreDFCol.GLOSS_B].values
        total_correct = correct_mask.sum()

        # Torchmetrics wants higher score = better, so we flip the sign
        scores = -torch.tensor(sorted_group[ScoreDFCol.SCORE].values, dtype=torch.float32)
        targets = torch.tensor(correct_mask.astype(np.int32), dtype=torch.int32)
        indexes = torch.full_like(targets, fill_value=group_id)

        all_preds.append(scores)
        all_targets.append(targets)
        all_indexes.append(indexes)

        # Manual per-k
        for k in ks:
            top_k = targets[:k]
            per_k_stats[k]["precision"].append(top_k.sum().item() / k)
            if total_correct > 0:
                per_k_stats[k]["recall"].append(top_k.sum().item() / total_correct)
            per_k_stats[k]["match_count"] += top_k.sum().item()

        group_id += 1

    if all_preds:
        print("Calculating torchmetrics")
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        indexes = torch.cat(all_indexes)

        preds = preds.to(dtype=torch.float)
        targets = targets.to(dtype=torch.int)
        indexes = indexes.to(dtype=torch.long)

        results["mean_average_precision"] = RetrievalMAP()(preds, targets, indexes).item()
        results["mean_reciprocal_rank"] = RetrievalMRR()(preds, targets, indexes).item()

    for k in ks:
        results[f"precision@{k}"] = float(np.mean(per_k_stats[k]["precision"])) if per_k_stats[k]["precision"] else 0.0
        results[f"recall@{k}"] = float(np.mean(per_k_stats[k]["recall"])) if per_k_stats[k]["recall"] else 0.0
        results[f"match_count@{k}"] = per_k_stats[k]["match_count"]

    return results


def standardize_path_order(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `Gloss A Path` always contains the original query path if it exists in the dataset."""
    df = df.copy()

    # Create a mask for rows where `Gloss B Path` appears in `Gloss A Path`
    mask = df[ScoreDFCol.GLOSS_B_PATH].isin(df[ScoreDFCol.GLOSS_A_PATH])

    # Swap columns where the mask is True
    df.loc[mask, [ScoreDFCol.GLOSS_A_PATH, ScoreDFCol.GLOSS_B_PATH]] = df.loc[
        mask, [ScoreDFCol.GLOSS_B_PATH, ScoreDFCol.GLOSS_A_PATH]
    ].values
    df.loc[mask, [ScoreDFCol.GLOSS_A, ScoreDFCol.GLOSS_B]] = df.loc[
        mask, [ScoreDFCol.GLOSS_B, ScoreDFCol.GLOSS_A]
    ].values

    return df


def analyze_metric(metric_name: str, metric_df: pd.DataFrame, ks: List[int], out_folder=None) -> Dict[str, any]:
    result = {ScoreDFCol.METRIC: metric_name}

    # Shuffle to prevent bias in case of score ties
    metric_df = metric_df.sample(frac=1, random_state=42)

    # Check signatures
    signatures = metric_df[ScoreDFCol.SIGNATURE].str.split("=").str[0].unique()
    if metric_df[ScoreDFCol.SIGNATURE].isna().any():
        print("nan_rows:")
        print(metric_df[metric_df[ScoreDFCol.SIGNATURE].isna()])

    if (metric_df[ScoreDFCol.SIGNATURE] == "").any():
        print("empty_rows:")
        print(metric_df[metric_df[ScoreDFCol.SIGNATURE] == ""])
        print(signatures)

    assert len(signatures) == 1, signatures
    result[ScoreDFCol.SIGNATURE] = signatures[0]

    # Gloss and gloss-pair stats
    gloss_tuples = metric_df["gloss_tuple"].unique()
    metric_glosses = set(metric_df[ScoreDFCol.GLOSS_A].tolist() + metric_df[ScoreDFCol.GLOSS_B].tolist())

    # Score subsets
    not_self_score_df = metric_df[metric_df[ScoreDFCol.GLOSS_A] != metric_df[ScoreDFCol.GLOSS_B]]
    self_scores_df = metric_df[metric_df[ScoreDFCol.GLOSS_A] == metric_df[ScoreDFCol.GLOSS_B]]

    result["unique_gloss_pairs"] = len(gloss_tuples)
    result["unique_glosses"] = len(metric_glosses)
    result["hyp_gloss_count"] = len(metric_df[ScoreDFCol.GLOSS_A].unique())
    result["ref_gloss_count"] = len(metric_df[ScoreDFCol.GLOSS_B].unique())
    result["total_count"] = len(metric_df)

    # Self-score stats
    result["self_scores_count"] = len(self_scores_df)
    result["mean_self_score"] = self_scores_df[ScoreDFCol.SCORE].mean()
    result["std_self_score"] = self_scores_df[ScoreDFCol.SCORE].std()

    gloss_stats_self = self_scores_df.groupby(ScoreDFCol.GLOSS_A)[ScoreDFCol.SCORE].agg(["mean", "std"])
    result["mean_of_gloss_self_score_means"] = gloss_stats_self["mean"].mean()
    result["std_of_gloss_self_score_means"] = gloss_stats_self["mean"].std()

    # Out-of-class stats
    result["out_of_class_scores_count"] = len(not_self_score_df)
    result["mean_out_of_class_score"] = not_self_score_df[ScoreDFCol.SCORE].mean()
    result["std_out_of_class_score"] = not_self_score_df[ScoreDFCol.SCORE].std()

    out_of_class_gloss_stats = not_self_score_df.groupby(ScoreDFCol.GLOSS_A)[ScoreDFCol.SCORE].agg(["mean", "std"])
    result["mean_of_out_of_class_score_means"] = out_of_class_gloss_stats["mean"].mean()
    result["std_of_out_of_class_score_means"] = out_of_class_gloss_stats["mean"].std()

    # Time stats
    result["mean_score_time"] = metric_df[ScoreDFCol.TIME].mean()
    result["std_dev_of_score_time"] = metric_df[ScoreDFCol.TIME].std()

    print(
        f"{metric_name} has \n"
        f"*\t{len(metric_df)} distances,\n"
        f"*\tcovering {len(metric_glosses)} glosses,\n"
        f"*\twith {len(metric_df[ScoreDFCol.GLOSS_A].unique())} unique query glosses,\n"
        f"*\t{len(metric_df[ScoreDFCol.GLOSS_B].unique())} unique ref glosses,\n"
        f"*\tin {len(gloss_tuples)} combinations,\n"
        f"*\t{len(not_self_score_df)} out-of-class scores,\n"
        f"*\t{len(self_scores_df)} in-class scores"
    )

    # Retrieval metrics
    retrieval_stats = calculate_retrieval_stats(metric_df, ks=ks)
    result.update(retrieval_stats)

    result["mean_out_mean_in_ratio"] = result["mean_out_of_class_score"] / result["mean_self_score"]

    out_of_class_gloss_stats = (
        not_self_score_df.groupby("gloss_tuple")[ScoreDFCol.SCORE]
        .agg(["count", "mean", "max", "min", "std"])
        .reset_index()
    )
    out_of_class_gloss_stats = out_of_class_gloss_stats.sort_values("count", ascending=False)
    out_of_class_gloss_stats[ScoreDFCol.METRIC] = metric_name
    out_of_class_gloss_stats.to_csv(out_folder / f"{metric_name}_out_of_class_scores_by_gloss.csv", index=False)

    return result


if __name__ == "__main__":
    # scores_folder = Path(r"/opt/home/cleong/projects/pose-evaluation/metric_results/scores")
    scores_folder = Path("/opt/home/cleong/projects/pose-evaluation/metric_results_round_2/scores")
    

    # TODO: check if the number of CSVs has changed. If not, load deduplicated.

    analysis_folder = scores_folder.parent / "score_analysis"
    analysis_folder.mkdir(exist_ok=True)


    score_files_index_path = analysis_folder / "score_files_index.json"
    

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

        print("Columns: ")
        for column in previous_stats_by_metric.columns:
            print("*\t", column)

    score_files_index = {}
    if score_files_index_path.is_file():
        print(f"Loading Score files index {score_files_index_path}")
        with score_files_index_path.open("r", encoding="utf-8") as f:
            score_files_index = json.load(f)
            print(score_files_index[ScoresIndexDFCol.SUMMARY])

    if score_files_index[ScoresIndexDFCol.SUMMARY]["total_scores"] != previous_stats_by_metric["total_count"].sum():
        print(f"Index and previous analysis score counts differ. Re-analysis needed")
        print("Index has")
        print(f"\t{score_files_index[ScoresIndexDFCol.SUMMARY]["unique_metrics"]:,} metrics")
        print(f"\t{score_files_index[ScoresIndexDFCol.SUMMARY]["unique_gloss_a"]:,} unique 'gloss a' (query/hyp) values")
        print(f"\t{score_files_index[ScoresIndexDFCol.SUMMARY]["unique_gloss_b"]:,} unique 'gloss b' (ref) values")
        print(f"\t{score_files_index[ScoresIndexDFCol.SUMMARY]["total_scores"]:,} scores")
        
        print()
        print("Previous analysis had")
        print(f"*\t{len(previous_stats_by_metric)} metrics")        
        print(f"*\t{previous_stats_by_metric["hyp_gloss_count"].max():,} 'gloss a' values (max)")
        print(f"*\t{previous_stats_by_metric["hyp_gloss_count"].min():,} 'gloss a' values (min)")
        print(f"*\t{previous_stats_by_metric["ref_gloss_count"].max():,} 'gloss b' values (max)")
        print(f"*\t{previous_stats_by_metric["ref_gloss_count"].min():,} 'gloss b' values (min)")
        print(f"*\t{previous_stats_by_metric["total_count"].sum():,} scores")
        # hyp_gloss_count
        
    else:
        print(f"Score count has not changed, no need to re-analyze. Quitting now.")
        exit()
    
    csv_stats_dfs = []
    csv_files = list(scores_folder.glob("*.csv"))
    # ['ACCENT', 'ADULT', 'AIRPLANE', 'APPEAR', 'BAG2', 'BANANA2', 'BEAK', 'BERRY', 'BIG', 'BINOCULARS', 'BLACK', 'BRAG', 'BRAINWASH', 'CAFETERIA', 'CALM', 'CANDY1', 'CASTLE2', 'CELERY', 'CHEW1', 'COLD', 'CONVINCE2', 'COUNSELOR', 'DART', 'DEAF2', 'DEAFSCHOOL', 'DECIDE2', 'DIP3', 'DOLPHIN2', 'DRINK2', 'DRIP', 'DRUG', 'EACH', 'EARN', 'EASTER', 'ERASE1', 'EVERYTHING', 'FINGERSPELL', 'FISHING2', 'FORK4', 'FULL', 'GOTHROUGH', 'GOVERNMENT', 'HIDE', 'HOME', 'HOUSE', 'HUNGRY', 'HURRY', 'KNITTING3', 'LEAF1', 'LEND', 'LIBRARY', 'LIVE2', 'MACHINE', 'MAIL1', 'MEETING', 'NECKLACE4', 'NEWSTOME', 'OPINION1', 'ORGANIZATION', 'PAIR', 'PEPSI', 'PERFUME1', 'PIG', 'PILL', 'PIPE2', 'PJS', 'REALSICK', 'RECORDING', 'REFRIGERATOR', 'REPLACE', 'RESTAURANT', 'ROCKINGCHAIR1', 'RUIN', 'RUSSIA', 'SCREWDRIVER3', 'SENATE', 'SHAME', 'SHARK2', 'SHAVE5', 'SICK', 'SNOWSUIT', 'SPECIALIST', 'STADIUM', 'SUMMER', 'TAKEOFF1', 'THANKSGIVING', 'THANKYOU', 'TIE1', 'TOP', 'TOSS', 'TURBAN', 'UNCLE', 'VAMPIRE', 'WASHDISHES', 'WEAR', 'WEATHER', 'WINTER', 'WORKSHOP', 'WORM', 'YESTERDAY']
    # glosses_to_load = ["ACCENT", "VAMPIRE", "RUSSIA", "BRAG", "WEATHER"]
    glosses_to_load = None

    # metrics_to_load = ["untrimmed_normalizedbyshoulders_hands_defaultdist0.0_nointerp_padwithfirstframe_fillmasked1.0_AggregatedPowerDistanceMetric","startendtrimmed_normalizedbyshoulders_youtubeaslkeypoints_defaultdist0.0_nointerp_padwithfirstframe_fillmasked0.0_AggregatedPowerDistanceMetric"]
    metrics_to_load = None

    for csv_file in tqdm(csv_files, desc="Loading scores csvs"):
        if glosses_to_load is not None:
            if not any(gloss_to_load in csv_file.name for gloss_to_load in glosses_to_load):
                continue
        if metrics_to_load is not None:
            if not any(metric_to_load in csv_file.name for metric_to_load in metrics_to_load):
                continue
        scores_csv_df = load_score_csv(csv_file=csv_file)
        csv_stats_dfs.append(scores_csv_df)

    scores_df = pd.concat(csv_stats_dfs)

    print(f"{scores_df}")

    # Normalize gloss pairs by sorting them
    print("Creating gloss tuple pairs")
    # TODO: This is where we should load from the index.
    scores_df["gloss_tuple"] = [
        tuple(x) for x in np.sort(scores_df[[ScoreDFCol.GLOSS_A, ScoreDFCol.GLOSS_B]].values, axis=1)
    ]

    metrics_to_analyze = scores_df[ScoreDFCol.METRIC].unique()
    stats_by_metric = defaultdict(list)
    print(f"We have results for {len(metrics_to_analyze)}")
    # metrics_to_analyze = ["n-dtai-DTW-MJE (fast)", "MJE", "nMJE"]
    for metric_index, metric in enumerate(metrics_to_analyze):
        print("*" * 50)
        print(f"METRIC #{metric_index}/{len(metrics_to_analyze)}: {metric}")
        metric_df = scores_df[scores_df[ScoreDFCol.METRIC] == metric]

        metric_stats = analyze_metric(metric, metric_df, ks, out_folder=metric_by_gloss_stats_folder)
        for k, v in metric_stats.items():
            stats_by_metric[k].append(v)

        scores_df = scores_df[scores_df[ScoreDFCol.METRIC] != metric]  # delete.
        print(f"{len(metric_df)} metric rows removed, now scores_df is {len(scores_df)} long")

    stats_by_metric = pd.DataFrame(stats_by_metric)
    print(stats_by_metric)
    stats_by_metric.to_csv(metric_stats_out, index=False)




# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/analyze_scores.py
