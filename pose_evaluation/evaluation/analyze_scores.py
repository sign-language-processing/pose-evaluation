import argparse
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from pathlib import Path
import json
import re 

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalPrecision, RetrievalRecall

from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol, load_score_csv
from pose_evaluation.evaluation.index_score_files import ScoresIndexDFCol, index_scores
from pose_evaluation.evaluation.load_pyarrow_dataset import load_dataset, load_metric_dfs

_SIGNATURE_RE = re.compile(r"default_distance:([\d.]+)")

tqdm.pandas()

def extract_signature_distance(signature: str) -> Optional[str]:
    """
    From a signature string, extract the float following 'default_distance:'.
    Returns None if not found.
    """
    m = _SIGNATURE_RE.search(signature)
    return float(m.group(1)) if m else None

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
    
    print(f"Calculating @k-metrics")
    for k in tqdm(ks, desc="metrics at k"):
        results[f"precision@{k}"] = float(np.mean(per_k_stats[k]["precision"])) if per_k_stats[k]["precision"] else 0.0
        results[f"recall@{k}"] = float(np.mean(per_k_stats[k]["recall"])) if per_k_stats[k]["recall"] else 0.0
        results[f"match_count@{k}"] = per_k_stats[k]["match_count"]

    # print(f"Calculating @k-metrics using TorchMetrics")
    # for k in tqdm(ks, desc="torchmetrics at k"):
    #     precision_at_k = RetrievalPrecision(top_k=k)
    #     recall_at_k = RetrievalRecall(top_k=k)

    #     results[f"precision@{k}(torch)"] = precision_at_k(preds, targets, indexes).item()
    #     results[f"recall@{k}(torch)"] = recall_at_k(preds, targets, indexes).item()
    
    # # Assertions for comparison
    # for k in ks:
    #     assert np.isclose(results[f"precision@{k}(torch)"], results[f"precision@{k}"], atol=1e-4), f"Precision@{k} mismatch: {results[f'precision@{k}(torch)']:.4f} vs {results[f'precision@{k}']:.4f}"
    #     assert np.isclose(results[f"recall@{k}(torch)"], results[f"recall@{k}"], atol=1e-4), f"Recall@{k} mismatch: {results[f'recall@{k}(torch)']:.4f} vs {results[f'recall@{k}']:.4f}"


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

    if len(signatures) > 1:
        print("Problematic rows:")
        for sig in signatures:
            print(f"Rows with signature '{sig}':")
            print(metric_df[metric_df[ScoreDFCol.SIGNATURE].str.split("=").str[0] == sig])
    assert len(signatures) == 1, f"{len(signatures)} Signatures: {signatures}, \nMetric: {metric_name}"
    result[ScoreDFCol.SIGNATURE] = signatures[0]

    # Gloss and gloss-pair stats
    metric_df["gloss_tuple"] = [
                tuple(x) for x in np.sort(metric_df[[ScoreDFCol.GLOSS_A, ScoreDFCol.GLOSS_B]].values, axis=1)
            ]
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
        f"*\t{len(metric_df):,} distances,\n"
        f"*\tcovering {len(metric_glosses):,} glosses,\n"
        f"*\twith {len(metric_df[ScoreDFCol.GLOSS_A].unique()):,} unique query glosses,\n"
        f"*\t{len(metric_df[ScoreDFCol.GLOSS_B].unique()):,} unique ref glosses,\n"
        f"*\tin {len(gloss_tuples):,} combinations,\n"
        f"*\t{len(not_self_score_df):,} out-of-class scores,\n"
        f"*\t{len(self_scores_df):,} in-class scores"
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

def load_score_parquet(parquet_file: Path) -> pd.DataFrame:
    """Loads a score Parquet file into a Pandas DataFrame."""
    return pd.read_parquet(parquet_file)

def load_metric_dfs_from_filenames(scores_folder: Path, file_format: str = "csv") -> Tuple[str, pd.DataFrame]:
    """
    Parses filenames to extract the metric name and loads/groups the corresponding data.

    Args:
        scores_folder: Path to the folder containing the score files.
        file_format: The format of the score files. Must be either "csv" or "parquet".
                     Defaults to "csv".

    Yields:
        A tuple containing the metric name (str) and the combined DataFrame (pd.DataFrame)
        for that metric.
    """
    if file_format not in ["csv", "parquet"]:
        raise ValueError(f"Invalid file_format: {file_format}. Must be 'csv' or 'parquet'.")

    file_extension = f"*.{file_format}"
    score_files = list(scores_folder.glob(file_extension))
    metric_files: Dict[str, list[Path]] = defaultdict(list)

    for score_file in tqdm(score_files, f"Parsing {file_format} filenames"):
        filename = score_file.stem  # Get filename without extension
        parts = filename.split("_")
        if len(parts) > 1 and "outgloss" in parts:
            gloss = parts[0]
            outgloss_index = parts.index("outgloss")
            metric_name_parts = parts[1:outgloss_index]
            metric_name = "_".join(metric_name_parts)
            metric_files[metric_name].append(score_file)
        else:
            print(f"Warning: Could not parse metric name from filename: {filename}")

    print(f"Found {len(metric_files.keys())} metrics")
    print(f"Found {len(metric_files.keys())} metrics: {list(metric_files.keys())[:10]}")

    for metric, files in metric_files.items():
        signatures_set = set()
        all_dfs = []
        processed_file_signatures = {}
        signature_files = defaultdict(list)
        try:
            for score_file in tqdm(files, desc=f"Loading {len(files)} {file_format.upper()} files for metric '{metric}'"):
                if file_format == "csv":
                    scores_df = load_score_csv(csv_file=score_file)
                elif file_format == "parquet":
                    scores_df = load_score_parquet(parquet_file=score_file)
                else:
                    raise ValueError("Unexpected file format")

                signatures = scores_df[ScoreDFCol.SIGNATURE].unique().tolist()
                assert len(signatures) == 1, f"{score_file} has more than one signature!: \n{signatures}"
                signatures_set.update(signatures)
                processed_file_signatures[str(score_file)] = signatures


                # Had a bug where the default distance wasn't being set correctly
                if "defaultdist" in metric_name:
                    default_distance = float(metric_name.split("defaultdist")[1].split("_")[0])
                    sig_distance = float(extract_signature_distance(signatures[0]))                    
                    assert default_distance == sig_distance, f"default_distance:{default_distance} does not match distance in signatures[0]: \nNAME:\n{metric_name}\nSIGNATURE:\n{signatures[0]}" 

                for sig in signatures:
                    signature_files[sig].append(str(score_file))

                all_dfs.append(scores_df)

            assert len(signatures_set) == 1, f"More than one signature found for {metric}, previously processed {len(processed_file_signatures.keys())}"
        except AssertionError as e:
            debug_path = Path.cwd()/"debug_jsons"
            debug_path.mkdir(exist_ok=True)
            processed_file_signatures_out_json = debug_path/f"{metric}_debugging_processed_{file_format}_signatures.json"
            signature_files_out_json = debug_path/f"{metric}_debugging_signature_{file_format}_files.json"
            print(processed_file_signatures_out_json.resolve())
            print(signature_files_out_json.resolve())
            with processed_file_signatures_out_json.open("w") as f:
                json.dump(processed_file_signatures, f, indent=4)

            with signature_files_out_json.open("w") as f:
                json.dump(signature_files, f, indent=4)
            raise e

        if all_dfs:

            combined_df = pd.concat(all_dfs, ignore_index=True)
            signatures = combined_df[ScoreDFCol.SIGNATURE].unique()

            assert len(signatures) == 1, f"More than one signature found for {metric}"
            # Normalize gloss pairs by sorting them
            combined_df["gloss_tuple"] = [
                tuple(x) for x in np.sort(combined_df[[ScoreDFCol.GLOSS_A, ScoreDFCol.GLOSS_B]].values, axis=1)
            ]
            yield metric, combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pose evaluation scores.")
    parser.add_argument(
        "scores_folder",
        type=Path,
        help="Path to the folder containing the score files, either a pyarrow parquet datset (default) or csv files",
    )
    parser.add_argument(
    "--file-format",
    type=str,
    choices=["pyarrow", "parquet", "csv"],
    default="pyarrow",
    help="Format of the score files to parse (pyarrow, parquets, or csvs). Defaults to pyarrow.",
)
    args = parser.parse_args()
    scores_folder = Path(args.scores_folder)

    analysis_folder = scores_folder.parent / "score_analysis"
    analysis_folder.mkdir(exist_ok=True)

    score_files_index_path = analysis_folder / "score_files_index.json"
    metric_stats_out = analysis_folder / "stats_by_metric.csv"
    metric_by_gloss_stats_folder = analysis_folder / "metric_by_gloss_stats"
    metric_by_gloss_stats_folder.mkdir(exist_ok=True)
    ks = [1, 5, 10]

    previous_stats_by_metric = None

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

    if previous_stats_by_metric is not None and score_files_index.get(ScoresIndexDFCol.SUMMARY, {}).get("total_scores") == previous_stats_by_metric["total_count"].sum():
        print(f"Score count has not changed, no need to re-analyze. Quitting now.")
        exit()
    else:
        print(f"Index and previous analysis score counts differ or no previous analysis found. Re-analysis needed")
        if score_files_index.get(ScoresIndexDFCol.SUMMARY):
            print("Index has")
            print(f"\t{score_files_index[ScoresIndexDFCol.SUMMARY]["unique_metrics"]:,} metrics")
            print(f"\t{score_files_index[ScoresIndexDFCol.SUMMARY]["unique_gloss_a"]:,} unique 'gloss a' (query/hyp) values")
            print(f"\t{score_files_index[ScoresIndexDFCol.SUMMARY]["unique_gloss_b"]:,} unique 'gloss b' (ref) values")
            print(f"\t{score_files_index[ScoresIndexDFCol.SUMMARY]["total_scores"]:,} scores")

        if previous_stats_by_metric is not None:
            print()
            print("Previous analysis had")
            print(f"*\t{len(previous_stats_by_metric)} metrics")
            print(f"*\t{previous_stats_by_metric["hyp_gloss_count"].max():,} 'gloss a' values (max)")
            print(f"*\t{previous_stats_by_metric["hyp_gloss_count"].min():,} 'gloss a' values (min)")
            print(f"*\t{previous_stats_by_metric["ref_gloss_count"].max():,} 'gloss b' values (max)")
            print(f"*\t{previous_stats_by_metric["ref_gloss_count"].min():,} 'gloss b' values (min)")
            print(f"*\t{previous_stats_by_metric["total_count"].sum():,} scores")

    stats_by_metric = defaultdict(list)
    metrics_analyzed = set()

    if args.file_format != "pyarrow":
        metric_generator  = load_metric_dfs_from_filenames(args.scores_folder, file_format=args.file_format)
    
    else:
        dataset= load_dataset(scores_folder)
        metric_generator  = load_metric_dfs(dataset)
    # i = 0
    # for metric, metric_df in tqdm(metric_generator, desc="Checking metric csvs"):
    #     print("*" * 50)
    #     print(f"Metric {i}: {metric} has {len(metric_df)} rows")
    #     i+=1


    # exit()
    for metric, metric_df in tqdm(metric_generator, desc="Analyzing metrics"):
        if metric in metrics_analyzed:
            print(f"Skipping already analyzed metric: {metric}")
            continue

        print("*" * 50)
        print(f"Analyzing metric: {metric}")
        metric_stats = analyze_metric(metric, metric_df, ks, out_folder=metric_by_gloss_stats_folder)
        for k, v in metric_stats.items():
            stats_by_metric[k].append(v)
        metrics_analyzed.add(metric)

    if stats_by_metric:
        stats_by_metric_df = pd.DataFrame(stats_by_metric)
        print(stats_by_metric_df)
        print(f"Saving to {metric_stats_out}")
        stats_by_metric_df.to_csv(metric_stats_out, index=False)
    else:
        print("No metrics were analyzed.")



# conda activate /opt/home/cleong/envs/pose_eval_src && cd /opt/home/cleong/projects/pose-evaluation && python pose_evaluation/evaluation/analyze_scores.py
