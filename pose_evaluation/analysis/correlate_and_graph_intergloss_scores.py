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


def plot_metric_histogram(
    df: pd.DataFrame,
    col: str,
    metric_to_plot: str,
    bins: int = 10,
    kde: bool = True,
    show=False,
    out_path: Optional[Path] = None,
):
    """
    Plots a histogram of the 'mean' column, filtering the dataframe by the specified 'metric'.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        metric (str): The metric value to filter the dataframe.
        bins (int): Number of bins in the histogram.
        kde (bool): Whether to include the KDE curve.
        color (str): Color of the histogram bars.
    """
    # Filter dataframe by metric
    df_filtered = df[df["METRIC"] == metric_to_plot]

    if df_filtered.empty:
        print(f"No data found for metric: {metric_to_plot}")
        return

    # Set seaborn style
    # sns.set_style("whitegrid")

    # Create the plot
    plt.figure(figsize=(7, 5))
    sns.histplot(
        df_filtered[col].tolist(),
        bins=bins,
        kde=kde,
        #  color=color, edgecolor="black"
    )

    # Labels and title
    plt.xlabel(f"{col} Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Mean Inter-gloss ({metric_to_plot})", fontsize=14)

    if show:
        plt.show()
    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.close()


def plot_metric_scatter_interactive(
    df: pd.DataFrame, metric_x: str, metric_y: str, show: bool = False, html_path: Optional[Path] = None
):
    # Filter for the two specified metrics
    df_x = df[df["METRIC"] == metric_x].rename(columns={"mean": "score_x"})
    df_y = df[df["METRIC"] == metric_y].rename(columns={"mean": "score_y"})

    # Merge on GLOSS_A and GLOSS_B
    merged_df = df_x.merge(df_y, on=["GLOSS_A", "GLOSS_B"], suffixes=("", "_y"))

    # Create labels
    merged_df["label"] = merged_df["GLOSS_A"] + " / " + merged_df["GLOSS_B"]

    custom_color_map = {"Both": "purple", "Lookalike": "blue", "Semantic": "green", "Neither": "lightgray"}

    # Create scatter plot without labels
    fig = px.scatter(
        merged_df,
        x="score_x",
        y="score_y",
        color="relation_type",
        title=f"{metric_x} vs {metric_y}",
        labels={"score_x": metric_x, "score_y": metric_y, "relation_type": "Relation Type"},
        color_discrete_map=custom_color_map,
    )

    # Add text labels as a separate trace with legend entry
    text_trace = go.Scatter(
        x=merged_df["score_x"],
        y=merged_df["score_y"],
        text=merged_df["label"],
        mode="text",
        textposition="top center",
        name="Labels",  # Separate legend entry
        showlegend=True,  # Allow toggling via legend
    )

    fig.add_trace(text_trace)

    # Improve layout
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        xaxis_title=metric_x,
        yaxis_title=metric_y,
        hovermode="closest",
    )

    if show:
        fig.show()
    if html_path:
        fig.write_html(html_path)

    del fig


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
    palette = {
        "Both": "purple",
        "Lookalike": "blue",
        "Semantic": "green",
        "Neither": "lightgray",
    }

    # Generate shortened names and hash
    metric_x_short = metric_x[:20]
    metric_y_short = metric_y[:20]
    name_hash = hashlib.md5(f"{metric_x}|{metric_y}".encode()).hexdigest()[:8]

    if png_path:
        base_path = png_path.parent / f"{metric_x_short}__{metric_y_short}__{name_hash}"
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

    plt.xlabel(f"{metric_x_short}")
    plt.ylabel(f"{metric_y_short}")
    plt.title(f"Mean Inter-gloss Scores:\n{metric_x_short} vs\n {metric_y_short}")
    plt.grid(True)

    if show:
        plt.show()
    if png_file:
        plt.tight_layout()
        plt.savefig(png_file)

    # Save metadata to a sidecar .txt file
    if txt_file:
        relation_counts = merged_df["relation_type"].value_counts().to_dict()
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"Full X Metric: {metric_x}\n")
            f.write(f"Full Y Metric: {metric_y}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Number of points: {len(merged_df)}\n")
            f.write("Relation type counts:\n")
            for rel_type in ["Both", "Lookalike", "Semantic", "Neither"]:
                count = relation_counts.get(rel_type, 0)
                f.write(f"  {rel_type}: {count}\n")

    plt.close()


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
    except (ValueError, SyntaxError, TypeError):
        # Fall back to manually parsing something like (AIRPLANE, HOME)
        val = val.strip("() ")
        parts = [part.strip().strip("'\"") for part in val.split(",")]
        return tuple(parts)


if __name__ == "__main__":

    score_analysis_folder = Path("metric_results/4_22_2025_csvcount_17187_score_analysis_with_updated_MAP")
    # score_analysis_folder = Path(
    #     "/opt/home/cleong/projects/pose-evaluation/metric_results_round_2/4_23_2025_score_analysis_3300_trials"
    # )
    aslkg_csv = Path("/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/ASLKG/edges_v2_noweights.tsv")
    known_lookalikes_csv = (
        Path("/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/SimilarSigns")
        / "deduped_sorted_similar_gloss_pairs.csv"
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
    # Adding the ASL Knowledge Graph: alas, none of these are in here.
    asl_knowledge_graph_df = pd.read_csv(aslkg_csv, delimiter="\t")
    # # get the "response" relation
    asl_knowledge_graph_df = asl_knowledge_graph_df[asl_knowledge_graph_df["relation"] == "response"]

    # # add gloss_tuple
    asl_knowledge_graph_df["gloss_tuple"] = asl_knowledge_graph_df.apply(create_gloss_tuple_aslkg, axis=1)
    print(asl_knowledge_graph_df.info())
    print(asl_knowledge_graph_df.head())

    print("ASL KNOWLEDGE GRAPH HEAD:")
    asl_knowledge_graph_df["gloss_tuple"] = asl_knowledge_graph_df["gloss_tuple"].apply(normalize_gloss_tuple)
    # print(asl_knowledge_graph_df["gloss_tuple"].head())
    print(asl_knowledge_graph_df.head())
    # print(asl_knowledge_graph_df.info())
    print()

    print("SCORES BY GLOSS HEAD")
    scores_by_gloss_df["gloss_tuple"] = scores_by_gloss_df["gloss_tuple"].apply(normalize_gloss_tuple)
    # print(scores_by_gloss_df["gloss_tuple"].head())
    print(scores_by_gloss_df.head())
    # print(scores_by_gloss_df.info())

    gloss_tuple_set = set(asl_knowledge_graph_df["gloss_tuple"])
    scores_by_gloss_df["semantically_related"] = scores_by_gloss_df["gloss_tuple"].isin(
        set(asl_knowledge_graph_df["gloss_tuple"])
    )
    print(set(asl_knowledge_graph_df["gloss_tuple"]).intersection(set(scores_by_gloss_df["gloss_tuple"])))

    print("*" * 20)
    semantically_related_items_df = scores_by_gloss_df[scores_by_gloss_df["semantically_related"]]
    print(semantically_related_items_df)

    #############################################
    # Lookalikes
    known_lookalikes_df = pd.read_csv(known_lookalikes_csv)
    print(known_lookalikes_df.head())

    # add gloss tuple
    known_lookalikes_df["gloss_tuple"] = known_lookalikes_df.apply(create_gloss_tuple_known_lookalikes, axis=1)
    scores_by_gloss_df["known_lookalikes"] = scores_by_gloss_df["gloss_tuple"].isin(
        set(known_lookalikes_df["gloss_tuple"])
    )
    scores_which_are_known_lookalikes = scores_by_gloss_df[scores_by_gloss_df["known_lookalikes"]]
    print(scores_which_are_known_lookalikes)

    scores_by_gloss_df["relation_type"] = scores_by_gloss_df.apply(classify_relation, axis=1)

    metrics = scores_by_gloss_df["METRIC"].unique().tolist()
    correlation_plots_folder = plots_folder / "metric_correlations"
    histogram_plots_folder = plots_folder / "metric_histograms"
    correlation_plots_folder.mkdir(exist_ok=True)
    histogram_plots_folder.mkdir(exist_ok=True)

    combinations_count = math.comb(len(metrics), 2)
    print(f"We have intergloss scores for {len(metrics)} metrics, so there are {combinations_count} combinations")

    for metric1, metric2 in tqdm(
        combinations(metrics, 2), desc="generating correlation plots", total=combinations_count
    ):
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
        gc.collect()

    for metric in tqdm(metrics, desc="Generating histogram plots"):
        plot_metric_histogram(
            scores_by_gloss_df,
            metric_to_plot=metric,
            col="mean",
            out_path=histogram_plots_folder / f"{metric}_intergloss_hist.png",
        )
        gc.collect()
