import ast
import gc
import hashlib
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_rel
from tqdm import tqdm

sns.set_theme()


def classify_relation(row):
    if row["known_lookalikes"] and row["semantically_related"]:
        return "Both"
    elif row["known_lookalikes"]:
        return "Lookalike"
    elif row["semantically_related"]:
        return "Semantic"
    else:
        return "Neither"


def plot_metric_boxplot(
    df: pd.DataFrame,
    col: str,
    metric: str,
    show: bool = False,
    out_path: Optional[Path] = None,
):
    df_filtered = df[df["METRIC"] == metric]

    if df_filtered.empty:
        print(f"No data found for metric: {metric}")
        return

    plt.figure(figsize=(7, 5))
    sns.boxplot(
        data=df_filtered,
        x="relation_type",
        y=col,
        palette={"Lookalike": "red", "Semantic": "blue", "Neither": "gray", "Both": "green"},
    )

    plt.title(f"{col} Distribution by Relation Type ({metric})", fontsize=14)
    plt.xlabel("Relation Type", fontsize=12)
    plt.ylabel(f"{col} Value", fontsize=12)

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    if show:
        plt.show()

    plt.close()


def plot_metric_histogram(
    df: pd.DataFrame,
    col: str,
    metric: str,
    bins: int = 10,
    kde: bool = True,
    show: bool = False,
    out_path: Optional[Path] = None,
    overlay_by_relation: bool = False,  # NEW
):
    """
    Plots a histogram of the specified column, filtering the dataframe by the specified 'metric'.
    Optionally overlays histograms by 'relation_type' if available.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        col (str): The column to plot.
        metric (str): The metric value to filter the dataframe.
        bins (int): Number of bins in the histogram.
        kde (bool): Whether to include the KDE curve.
        show (bool): Whether to show the plot immediately.
        out_path (Optional[Path]): Where to save the plot if specified.
        overlay_by_relation (bool): If True, overlay histograms by relation type.
    """
    # Filter dataframe by metric
    df_filtered = df[df["METRIC"] == metric]

    if df_filtered.empty:
        print(f"No data found for metric: {metric}")
        return

    plt.figure(figsize=(8, 5))

    if overlay_by_relation and "relation_type" in df_filtered.columns:
        colors = ({"Lookalike": "red", "Semantic": "blue", "Neither": "gray", "Both": "green"},)
        labels = ["Lookalike", "Semantic", "Neither", "Both"]

        for relation in labels:
            subset = df_filtered[df_filtered["relation_type"] == relation]
            if not subset.empty:
                sns.histplot(
                    subset[col],
                    bins=bins,
                    kde=kde,
                    label=relation,
                    color=colors.get(relation, "black"),
                    stat="density",
                    element="step",
                    fill=False,
                    linewidth=2,
                )
        plt.legend()
        plt.title(f"{col} Distribution by Relation Type ({metric})", fontsize=14)

    else:
        sns.histplot(
            df_filtered[col],
            bins=bins,
            kde=kde,
            color="royalblue",
            edgecolor="black",
        )
        plt.title(f"{col} Distribution ({metric})", fontsize=14)

    # Labels
    plt.xlabel(f"{col} Value", fontsize=12)
    plt.ylabel("Density" if overlay_by_relation else "Frequency", fontsize=12)

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    if show:
        plt.show()

    plt.close()


def plot_metric_scatter(
    df: pd.DataFrame,
    metric_x: str,
    metric_y: str,
    show: bool = False,
    png_path: Optional[Path] = None,
):
    # Filter for the two specified metrics
    df_x = df[df["METRIC"] == metric_x].rename(columns={"mean": "score_x"})
    df_y = df[df["METRIC"] == metric_y].rename(columns={"mean": "score_y"})

    # Merge on GLOSS_A and GLOSS_B
    merged_df = df_x.merge(df_y, on=["GLOSS_A", "GLOSS_B"], suffixes=("", "_y"))

    # Create labels
    merged_df["label"] = merged_df["GLOSS_A"] + " / " + merged_df["GLOSS_B"]

    # Define custom palette
    palette = {"Lookalike": "red", "Semantic": "blue", "Neither": "gray", "Both": "green"}

    # Generate shortened names and hash
    metric_x_short = metric_x[:15]
    metric_y_short = metric_y[:15]
    metric_x_hash = hashlib.md5(f"{metric_x}".encode()).hexdigest()[:8]
    metric_y_hash = hashlib.md5(f"{metric_y}".encode()).hexdigest()[:8]
    name_hash = hashlib.md5(f"{metric_x}|{metric_y}".encode()).hexdigest()[:8]

    if png_path:
        base_path = png_path.parent / f"{metric_x_hash}_{metric_y_hash}_{name_hash}"
        png_file = base_path.with_suffix(".png")
        txt_file = base_path.with_suffix(".txt")
    else:
        png_file = None
        txt_file = None

    # Create plot
    plt.figure(figsize=(8, 6))
    # Plot gray background dots first
    sns.scatterplot(
        data=merged_df[merged_df["relation_type"] == "Neither"],
        x="score_x",
        y="score_y",
        color="lightgray",
        alpha=0.5,
        s=40,
        edgecolor=None,
        label="Neither",
    )

    # Plot all other relation types on top
    sns.scatterplot(
        data=merged_df[merged_df["relation_type"] != "Neither"],
        x="score_x",
        y="score_y",
        hue="relation_type",
        palette=palette,
        alpha=0.8,
        s=60,
        edgecolor="black",
        linewidth=0.3,
    )

    plt.xlabel(f"{metric_x_hash}")
    plt.ylabel(f"{metric_y_hash}")
    plt.title(f"Mean Intergloss Scores:\n{metric_x_hash} vs\n {metric_y_hash}")
    plt.grid(True)

    if show:
        plt.show()
    if png_file:
        plt.tight_layout()
        plt.savefig(png_file)

    # Save metadata to a sidecar .txt file
    if txt_file:
        relation_counts = merged_df["relation_type"].value_counts().to_dict()
        with open(txt_file, "w") as f:
            f.write(f"Full X Metric: {metric_x}\n")
            f.write(f"Full Y Metric: {metric_y}\n")
            f.write(f"Hashed X Metric: {metric_x_hash}\n")
            f.write(f"Hashed Y Metric: {metric_y_hash}\n")
            f.write(f"Hashed Combination: {name_hash}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Number of points: {len(merged_df)}\n")
            f.write("Relation type counts:\n")
            for rel_type in ["Both", "Lookalike", "Semantic", "Neither"]:
                count = relation_counts.get(rel_type, 0)
                f.write(f"  {rel_type}: {count}\n")

    plt.close()


def plot_overlay_histograms(
    df_metric: pd.DataFrame, metric: str, save_plot_path: Optional[Path] = None, show_plot: bool = True
) -> None:
    plt.figure(figsize=(8, 5))

    colors = {"Lookalike": "blue", "Semantic": "green", "Neither": "gray"}
    labels = ["Lookalike", "Semantic", "Neither"]

    for relation in labels:
        scores = df_metric[df_metric["relation_type"] == relation]["score"]
        if not scores.empty:
            sns.histplot(
                scores,
                label=relation,
                color=colors[relation],
                stat="density",
                kde=True,
                bins=30,
                element="step",
                fill=False,
                linewidth=2,
            )

    plt.title(f"Score Distributions (Overlayed) for {metric}")
    plt.xlabel("Distance Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if save_plot_path:
        plt.savefig(save_plot_path)
    if show_plot:
        plt.show()

    plt.close()


def analyze_metric_relationships(
    df: pd.DataFrame,
    metric_x: str,
    metric_y: str,
    show_plot: bool = True,
    save_plot_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Analyze correlation and distance tendencies between two metrics.

    Args:
        df: DataFrame containing columns METRIC, GLOSS_A, GLOSS_B, mean, relation_type.
        metric_x: First metric name.
        metric_y: Second metric name.
        show_plot: Whether to display the plot.
        save_plot_path: Optional path to save the plot.

    Returns:
        summary_df: DataFrame summarizing correlations, means, and t-tests.
    """
    # Generate shortened names and hash
    metric_x_short = metric_x[:15]
    metric_y_short = metric_y[:15]
    metric_x_hash = hashlib.md5(f"{metric_x}".encode()).hexdigest()[:8]
    metric_y_hash = hashlib.md5(f"{metric_y}".encode()).hexdigest()[:8]
    name_hash = hashlib.md5(f"{metric_x}|{metric_y}".encode()).hexdigest()[:8]

    # Prepare merged dataframe
    df_x = df[df["METRIC"] == metric_x].rename(columns={"mean": "score_x"})
    df_y = df[df["METRIC"] == metric_y].rename(columns={"mean": "score_y"})
    merged_df = df_x.merge(df_y, on=["GLOSS_A", "GLOSS_B"], suffixes=("", "_y"))

    # Calculate correlations
    pearson_corr, _ = pearsonr(merged_df["score_x"], merged_df["score_y"])
    spearman_corr, _ = spearmanr(merged_df["score_x"], merged_df["score_y"])
    kendall_corr, _ = kendalltau(merged_df["score_x"], merged_df["score_y"])

    # Initialize summary dict
    summary = {
        "metric_x": metric_x,
        "metric_y": metric_y,
        "metric_x_hash": metric_x_hash,
        "metric_y_hash": metric_y_hash,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "kendall_corr": kendall_corr,
        "pearson_interpretation": interpret_correlation(pearson_corr),
        "spearman_interpretation": interpret_correlation(spearman_corr),
        "kendall_interpretation": interpret_correlation(kendall_corr),
        # other stuff...
    }

    # For Lookalike and Semantic
    for relation in ["Lookalike", "Semantic"]:
        sub_df = merged_df[merged_df["relation_type"] == relation]
        if len(sub_df) > 1:
            mean_x = sub_df["score_x"].mean()
            mean_y = sub_df["score_y"].mean()
            t_stat, p_val = ttest_rel(sub_df["score_x"], sub_df["score_y"])
        else:
            mean_x = mean_y = t_stat = p_val = float("nan")

        summary[f"{relation.lower()}_mean_x"] = mean_x
        summary[f"{relation.lower()}_mean_y"] = mean_y
        summary[f"{relation.lower()}_t_stat"] = t_stat
        summary[f"{relation.lower()}_p_val"] = p_val

    summary_df = pd.DataFrame([summary])

    # Plotting
    melted = pd.DataFrame(
        {
            "Relation": ["Lookalike", "Lookalike", "Semantic", "Semantic"],
            "Metric": [metric_x, metric_y, metric_x, metric_y],
            "Mean Distance": [
                summary["lookalike_mean_x"],
                summary["lookalike_mean_y"],
                summary["semantic_mean_x"],
                summary["semantic_mean_y"],
            ],
        }
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(data=melted, x="Relation", y="Mean Distance", hue="Metric")
    plt.title(f"Mean Distances for {metric_x_hash} vs {metric_y_hash}")
    plt.grid(True, axis="y")
    plt.tight_layout()

    if save_plot_path:
        base_path = save_plot_path.parent / f"{metric_x_hash}_{metric_y_hash}_{name_hash}"
        png_file = base_path.with_suffix(".png")
        save_plot_path = base_path / png_file

    if save_plot_path:
        plt.savefig(save_plot_path)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return summary_df


def create_gloss_tuple_aslkg(row):
    gloss_1 = row["subject"].split(":")[-1].upper()  # Extract and capitalize subject gloss
    gloss_2 = row["object"].split(":")[-1].upper()  # Extract and capitalize object gloss
    return tuple(sorted([gloss_1, gloss_2], reverse=False))  # Sort and create tuple


def create_gloss_tuple_known_lookalikes(row):
    gloss_1 = row["GLOSS_A"].upper()  # Extract and capitalize subject gloss
    gloss_2 = row["GLOSS_B"].upper()  # Extract and capitalize object gloss
    return tuple(sorted([gloss_1, gloss_2], reverse=False))  # Sort and create tuple


def normalize_gloss_tuple(val):
    if isinstance(val, tuple):
        return val
    try:
        # Try literal_eval first
        return ast.literal_eval(val)
    except:
        # Fall back to manually parsing something like (AIRPLANE, HOME)
        val = val.strip("() ")
        parts = [part.strip().strip("'\"") for part in val.split(",")]
        return tuple(parts)


def interpret_correlation(coef: float) -> str:
    if abs(coef) >= 0.9:
        return "very strong correlation"
    elif abs(coef) >= 0.7:
        return "strong correlation"
    elif abs(coef) >= 0.5:
        return "moderate correlation"
    elif abs(coef) >= 0.3:
        return "weak correlation"
    else:
        return "very weak or no correlation"


if __name__ == "__main__":

    # score_analysis_folder = Path("metric_results/4_22_2025_csvcount_17187_score_analysis_with_updated_MAP")
    # score_analysis_folder = Path(
    #     "/opt/home/cleong/projects/pose-evaluation/metric_results_round_2/4_23_2025_score_analysis_3300_trials"
    # )
    # score_analysis_folder = Path(
    #     "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4/5_12_2025_score_analysis_288_metrics_169_glosses"
    # )
    # score_analysis_folder = Path(
    #     "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4/5_12_2025_score_analysis_288_metrics_169_glosses"
    # )

    score_analysis_folder = Path("/data/petabyte/cleong/projects/pose-eval/Return4/score_analysis")
    aslkg_csv = Path("/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/ASLKG/edges_v2_noweights.tsv")
    known_lookalikes_csv = Path(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/SimilarSigns/deduped_sorted_similar_gloss_pairs.csv"
    )

    plots_folder = score_analysis_folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    # EmbeddingDistanceMetric_sem-lex_cosine_out_of_class_scores_by_gloss.csv
    scores_by_gloss_csvs = list(score_analysis_folder.rglob("*out_of_class_scores_by_gloss.csv"))
    print(f"Found {len(scores_by_gloss_csvs)} csvs containing scores by gloss")

    score_by_gloss_dfs_list = []

    for csv_file in tqdm(scores_by_gloss_csvs, desc="Loading CSVs"):
        csv_df = pd.read_csv(csv_file)
        score_by_gloss_dfs_list.append(csv_df)

    scores_by_gloss_df = pd.concat(score_by_gloss_dfs_list)
    scores_by_gloss_df[["GLOSS_A", "GLOSS_B"]] = scores_by_gloss_df["gloss_tuple"].str.extract(
        r"\('([^']*)', '([^']*)'\)"
    )
    print(scores_by_gloss_df.info())
    print(scores_by_gloss_df.head())

    ################################################################
    # Adding the ASL Knowledge Graph
    asl_knowledge_graph_df = pd.read_csv(aslkg_csv, delimiter="\t")
    # # get the "response" relation
    asl_knowledge_graph_df = asl_knowledge_graph_df[asl_knowledge_graph_df["relation"] == "response"]

    # # add gloss_tuple
    asl_knowledge_graph_df["gloss_tuple"] = asl_knowledge_graph_df.apply(create_gloss_tuple_aslkg, axis=1)
    print(asl_knowledge_graph_df.info())
    print(asl_knowledge_graph_df.head())

    print(f"ASL KNOWLEDGE GRAPH HEAD:")
    asl_knowledge_graph_df["gloss_tuple"] = asl_knowledge_graph_df["gloss_tuple"].apply(normalize_gloss_tuple)
    # print(asl_knowledge_graph_df["gloss_tuple"].head())
    print(asl_knowledge_graph_df.head())
    # print(asl_knowledge_graph_df.info())
    print()

    print(f"SCORES BY GLOSS HEAD")
    scores_by_gloss_df["gloss_tuple"] = scores_by_gloss_df["gloss_tuple"].apply(normalize_gloss_tuple)
    # print(scores_by_gloss_df["gloss_tuple"].head())
    print(scores_by_gloss_df.head())
    # print(scores_by_gloss_df.info())

    gloss_tuple_set = set(asl_knowledge_graph_df["gloss_tuple"])
    scores_by_gloss_df["semantically_related"] = scores_by_gloss_df["gloss_tuple"].isin(
        set(asl_knowledge_graph_df["gloss_tuple"])
    )
    print(set(asl_knowledge_graph_df["gloss_tuple"]).intersection(set(scores_by_gloss_df["gloss_tuple"])))

    print("*" * 50)
    print(scores_by_gloss_df[scores_by_gloss_df["semantically_related"] == True])

    #############################################
    # Lookalikes
    known_lookalikes_df = pd.read_csv(known_lookalikes_csv)
    print(known_lookalikes_df.head())

    # add gloss tuple
    known_lookalikes_df["gloss_tuple"] = known_lookalikes_df.apply(create_gloss_tuple_known_lookalikes, axis=1)
    scores_by_gloss_df["known_lookalikes"] = scores_by_gloss_df["gloss_tuple"].isin(
        set(known_lookalikes_df["gloss_tuple"])
    )

    print(f"KNOWN LOOKALIKES")
    print(scores_by_gloss_df[scores_by_gloss_df["known_lookalikes"] == True])
    print("*" * 50)

    scores_by_gloss_df["relation_type"] = scores_by_gloss_df.apply(classify_relation, axis=1)

    for rel_type in scores_by_gloss_df["relation_type"].unique():
        rel_type_df = scores_by_gloss_df[scores_by_gloss_df["relation_type"] == rel_type]
        rel_type_df = rel_type_df[["GLOSS_A", "GLOSS_B", "relation_type"]].drop_duplicates()
        rel_type_glosses = rel_type_df["GLOSS_A"].unique().tolist()
        rel_type_glosses.extend(rel_type_df["GLOSS_B"].unique().tolist())
        rel_type_glosses = list(set(rel_type_glosses))
        rel_type_out = score_analysis_folder / f"glosses_{rel_type}_relation.csv"
        print(f"Saving {len(rel_type_df)} {rel_type} tuples to {rel_type_out}")
        print(rel_type_glosses)
        rel_type_df.to_csv(rel_type_out)

    print("*" * 50)
    exit()

    # Example:EmbeddingDistanceMetric_sem-lex_cosine_out_of_class_scores_by_gloss.csv
    # gloss_tuple	count	mean	max	min	std	known_similar	metric	rank
    # ('DEER', 'MOOSE')	870	0.125122927317674	0.364014387130737	0.0213934779167175	0.0508914505643154	True	EmbeddingDistanceMetric_sem-lex_cosine	1
    # ('HUG', 'LOVE')	930	0.129592590242304	0.372491121292114	0.0322074890136718	0.0600298605817267	True	EmbeddingDistanceMetric_sem-lex_cosine	2
    # ('BUT', 'DIFFERENT')	930	0.139856820337234	0.28130042552948	0.023825466632843	0.0414892420832739	True	EmbeddingDistanceMetric_sem-lex_cosine	3
    # ('FAVORITE', 'GOOD')	32	0.140619456768036	0.271014928817749	0.0476789474487304	0.0497585180016914	False	EmbeddingDistanceMetric_sem-lex_cosine	4
    # ('ANIMAL', 'HAVE')	930	0.157980350461057	0.340065956115723	0.0243424773216247	0.051135208995542	True	EmbeddingDistanceMetric_sem-lex_cosine	5
    # ('CHALLENGE', 'GAME')	930	0.163187250334729	0.346950709819794	0.0585821270942688	0.050448722682526	True	EmbeddingDistanceMetric_sem-lex_cosine	6
    # ('SATURDAY', 'TUESDAY')	31	0.163267293284016	0.318226993083954	0.0510987639427185	0.0666946134674303	True	EmbeddingDistanceMetric_sem-lex_cosine	7
    # ('FAVORITE', 'SPICY')	32	0.169413153082132	0.287134885787964	0.104118287563324	0.0507829287998875	False	EmbeddingDistanceMetric_sem-lex_cosine	8
    # ('FAMILY', 'SANTA')	30	0.174653542041779	0.230535089969635	0.130753219127655	0.0242531823125804	False	EmbeddingDistanceMetric_sem-lex_cosine	9
    # ('FAVORITE', 'TASTE')	928	0.181375090739336	0.422960162162781	0.031768798828125	0.0689190830802117	True	EmbeddingDistanceMetric_sem-lex_cosine	10

    metrics = scores_by_gloss_df["METRIC"].unique().tolist()
    correlation_plots_folder = plots_folder / "metric_correlations"
    histogram_plots_folder = plots_folder / "metric_histograms"
    bar_plots_folder = plots_folder / "bar_plots"
    summaries_folder = plots_folder / "summaries"

    correlation_plots_folder.mkdir(exist_ok=True)
    histogram_plots_folder.mkdir(exist_ok=True)
    bar_plots_folder.mkdir(exist_ok=True)
    summaries_folder.mkdir(exist_ok=True)

    for metric in tqdm(metrics, desc="Generating histogram plots"):
        plot_metric_histogram(
            scores_by_gloss_df,
            metric=metric,
            col="mean",
            out_path=histogram_plots_folder / f"{metric}_intergloss_hist.png",
        )
        plot_metric_boxplot(
            scores_by_gloss_df,
            metric=metric,
            col="mean",
            out_path=histogram_plots_folder / f"{metric}_intergloss_boxplot.png",
        )
        gc.collect()
    combinations_count = math.comb(len(metrics), 2)
    print(f"We have intergloss scores for {len(metrics)} metrics, so there are {combinations_count} combinations")

    metric_summaries = []

    for metric1, metric2 in tqdm(
        combinations(metrics, 2), desc="generating correlation plots", total=combinations_count
    ):
        metric1_hash = hashlib.md5(f"{metric1}".encode()).hexdigest()[:8]
        metric2_hash = hashlib.md5(f"{metric2}".encode()).hexdigest()[:8]
        name_hash = hashlib.md5(f"{metric1}|{metric2}".encode()).hexdigest()[:8]
        summary = analyze_metric_relationships(
            scores_by_gloss_df,
            metric1,
            metric2,
            show_plot=False,
            save_plot_path=bar_plots_folder / f"{metric1}_versus_{metric2}_{name_hash}.png",
        )
        plot_metric_scatter(
            scores_by_gloss_df,
            metric1,
            metric2,
            show=False,
            png_path=correlation_plots_folder / f"{metric1}_versus_{metric2}.png",
        )

        # plot_metric_scatter_interactive(
        #     scores_by_gloss_df,
        #     metric1,
        #     metric2,
        #     show=False,
        #     html_path=correlation_plots_folder / f"{metric1}_versus_{metric2}.html",
        # )

        # print(summary)

        # if abs(summary["pearson_corr"].item()) > 0.7:
        #     print("\nHIGHLY CORRELATED")
        #     print(f"*metric1: {metric1}")
        #     print(f"*metric1: {metric1_hash}")
        #     print(f"*metric2: {metric2}")
        #     print(f"*metric2: {metric2_hash}")
        #     print(f"*hash: {name_hash}")
        #     print(summary["pearson_corr"])

        # summary.to_csv(summaries_folder / f"{name_hash}.csv")
        summary["hash"] = name_hash
        metric_summaries.append(summary)
        gc.collect()

    correlation_summary_df = pd.concat(metric_summaries)
    correlation_summary_df = correlation_summary_df.sort_values(by="pearson_corr", ascending=False)
    correlation_summary_df.to_csv(summaries_folder / "summary.csv")
