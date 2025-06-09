import io
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import re
import streamlit as st
import torch
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

# pio.templates.default = "plotly"

# https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/4
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# # or simply:
# torch.classes.__path__ = []

from pose_evaluation.evaluation.interpret_name import shorten_metric_name, descriptive_name, interpret_name

METRIC_COL = "METRIC"
SIGNATURE_COL = "SIGNATURE"
SHORT_COL = "SHORT"
DESCRIPTIVE_NAME_COL = "DESCRIPTIVE_NAME"


def plot_pareto_frontier(df: pd.DataFrame):
    st.subheader("Pareto Frontier Plot")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns to compute Pareto frontier.")
        return

    col1 = st.selectbox("Select X-axis column", numeric_cols, index=0)
    col2 = st.selectbox("Select Y-axis column", numeric_cols, index=1)
    normalize = st.checkbox("Normalize columns to [0, 1]?", value=False)

    if SHORT_COL not in df.columns:
        st.warning(f"Column '{SHORT_COL}' not found in the DataFrame.")
        return

    # Drop missing
    use_cols = [SHORT_COL, DESCRIPTIVE_NAME_COL, col1, col2]
    if "highlight" in df.columns:
        use_cols.append("highlight")
    if METRIC_COL in df.columns:
        use_cols.append(METRIC_COL)

    plot_df = df[use_cols].dropna().copy()

    # Normalize if needed
    if normalize:
        for col in [col1, col2]:
            plot_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min())

    # Compute Pareto frontier
    maximize_col1 = st.checkbox(f"Maximize {col1}?", value=True)
    maximize_col2 = st.checkbox(f"Maximize {col2}?", value=False)

    frontier = get_pareto_frontier(plot_df, col1, col2, maximize_col1, maximize_col2)

    fig = go.Figure()

    # Plot all points (grouped by highlight if available)
    if "highlight" in plot_df.columns:
        for label, group in plot_df.groupby("highlight"):
            fig.add_trace(
                go.Scatter(
                    x=group[col1],
                    y=group[col2],
                    mode="markers",
                    name=label,
                    marker=dict(size=6, opacity=0.7),
                    customdata=(
                        group[
                            [
                                DESCRIPTIVE_NAME_COL,
                                METRIC_COL,
                            ]
                        ]
                        if METRIC_COL in group
                        else group[[SHORT_COL]]
                    ),
                    hovertemplate=f"{DESCRIPTIVE_NAME_COL}: %{{customdata[0]}}<br>"
                    #   f"{'METRIC: %{{customdata[1]}}<br>' if METRIC_COL in group else ''}"
                                  f"{METRIC_COL}: %{{customdata[1]}}<br>" f"{col1}: %{{x:.3f}}<br>{col2}: %{{y:.3f}}<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_df[col1],
                y=plot_df[col2],
                mode="markers",
                name="All Points",
                marker=dict(size=6, opacity=0.5),
                customdata=plot_df[[DESCRIPTIVE_NAME_COL]],
                hovertemplate=f"{DESCRIPTIVE_NAME_COL}: %{{customdata[0]}}<br>{col1}: %{{x:.3f}}<br>{col2}: %{{y:.3f}}<extra></extra>",
            )
        )

    # Plot Pareto frontier
    fig.add_trace(
        go.Scatter(
            x=frontier[col1],
            y=frontier[col2],
            mode="lines+markers",
            name="Pareto Frontier",
            line=dict(color="red", width=1),
            marker=dict(
                size=12,
                color="rgba(0,0,0,0)",  # Transparent fill
                line=dict(
                    width=1,
                    # color='blue'  # Border color
                ),
            ),
            customdata=(
                frontier[[DESCRIPTIVE_NAME_COL, METRIC_COL]] if METRIC_COL in frontier else frontier[[SHORT_COL]]
            ),
            hovertemplate=f"{DESCRIPTIVE_NAME_COL}: %{{customdata[0]}}<br>"
                          f"{METRIC_COL}: %{{customdata[1]}}<br>"
                          f"{col1}: %{{x:.3f}}<br>{col2}: %{{y:.3f}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Pareto Frontier: {col1} vs {col2}",
        xaxis_title=col1,
        yaxis_title=col2,
        height=600,
        legend_title="Highlight Group" if "highlight" in plot_df.columns else "Legend",
    )

    st.plotly_chart(fig, use_container_width=True)


def get_pareto_frontier(
        df: pd.DataFrame, col1: str, col2: str, maximize_col1: bool, maximize_col2: bool
) -> pd.DataFrame:
    """Returns the Pareto frontier based on optimization directions."""
    data = df.copy()

    # Flip signs if we're maximizing a column
    if maximize_col1:
        data[col1] = -data[col1]
    if maximize_col2:
        data[col2] = -data[col2]

    # Sort by col1 then col2
    sorted_df = data.sort_values(by=[col1, col2], ascending=True)

    frontier = []
    best_col2 = float("inf")

    for _, row in sorted_df.iterrows():
        if row[col2] < best_col2:
            best_col2 = row[col2]
            frontier.append(row)

    frontier_df = pd.DataFrame(frontier)

    # Flip signs back for display
    if maximize_col1:
        frontier_df[col1] = -frontier_df[col1]
    if maximize_col2:
        frontier_df[col2] = -frontier_df[col2]

    # Return original columns
    return df.loc[frontier_df.index]


def find_single_token_diff_pairs(df: pd.DataFrame) -> List[Tuple[str, str, int]]:
    results = []
    metric_tokens = df["METRIC"].apply(lambda x: x.lower().split("_"))

    for (i1, tokens1), (i2, tokens2) in combinations(metric_tokens.items(), 2):
        if len(tokens1) != len(tokens2):
            continue  # Can't compare if lengths differ

        diffs = [i for i, (a, b) in enumerate(zip(tokens1, tokens2)) if a != b]

        if len(diffs) == 1:
            idx = diffs[0]
            results.append((df["METRIC"][i1], df["METRIC"][i2], idx))

    return results


def find_keyword_difference_pairs(df, keywords, metric_col="METRIC", verbose=True):
    """
    df: a pandas DataFrame containing metric strings
    keywords: list of keywords to identify differing positions
    metric_col: the name of the column in df containing metric strings
    verbose: whether to print debug info
    """
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in DataFrame.")

    metrics = df[metric_col].tolist()
    keywords = [kw.lower() for kw in keywords]
    results = []

    split_metrics = [(m, m.lower().split("_")) for m in metrics]

    for i, (m1, m1_parts) in enumerate(split_metrics):
        if verbose:
            print(f"\nExamining base metric[{i}]: {metrics[i]}")

        kw_indices = []
        for kw in keywords:
            for idx, token in enumerate(m1_parts):
                if kw in token:
                    kw_indices.append(idx)
                    break

        if verbose:
            print(f"  Found keyword indices: {kw_indices}")

        if len(kw_indices) != len(keywords):
            if verbose:
                print("  Skipping — not all keywords found in this metric.")
            continue

        for j, (m2, m2_parts) in enumerate(split_metrics):
            if i == j or len(m1_parts) != len(m2_parts):
                continue

            all_match = True
            for idx in range(len(m1_parts)):
                if idx in kw_indices:
                    if m1_parts[idx] == m2_parts[idx]:
                        if verbose:
                            print(f"    At index {idx}, expected difference but got match: {m1_parts[idx]}")
                        all_match = False
                        break
                else:
                    if m1_parts[idx] != m2_parts[idx]:
                        if verbose:
                            print(
                                f"    At index {idx}, expected match but got diff: {m1_parts[idx]} vs {m2_parts[idx]}"
                            )
                        all_match = False
                        break

            if all_match:
                if verbose:
                    print(f"  --> Match found: {metrics[i]} vs {metrics[j]}")
                results.append((metrics[i], metrics[j]))

    return results


def prettify_axis_label(label: str) -> str:
    return " ".join(word.capitalize() for word in label.split("_"))


def apply_minimal_layout(fig: go.Figure, size: int = 600) -> None:
    # fig.update_layout(
    #     title=None,
    #     xaxis=dict(visible=False),
    #     yaxis=dict(visible=False),
    #     margin=dict(l=0, r=0, t=0, b=0),
    #     showlegend=True,
    #     paper_bgcolor="white",
    #     plot_bgcolor="white",
    #     width=size,
    #     height=size,
    #     template="plotly_white",  # <- This is key
    # )

    # better, more visible, legend still tiny
    # fig.update_layout(
    #     title=None,
    #     xaxis=dict(visible=False),
    #     yaxis=dict(visible=False),
    #     margin=dict(l=0, r=0, t=0, b=0),
    #     showlegend=True,
    #     paper_bgcolor="white",
    #     plot_bgcolor="white",
    #     width=size,
    #     height=size,
    #     template="plotly_white",
    #     legend=dict(bgcolor="white", font=dict(color="black")),  # or another color if you want contrast
    # )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=40, color="black"),
        ),
        xaxis=dict(
            visible=True,  # set this True if you had it off earlier
            title=dict(
                # text="New X Label",
                font=dict(size=18)
            ),  # your desired label
            tickfont=dict(size=14),  # optional: tick label font size
        ),
        yaxis=dict(
            visible=True,  # set this True if you had it off earlier
            title=dict(
                # text="New X Label",
                font=dict(size=18)
            ),  # your desired label
            tickfont=dict(size=14),  # optional: tick label font size
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=40),  # leave space for legend if it's below
        template="plotly_white",
        width=size,
        height=size,
    )

    old_x_label = fig.layout.xaxis.title.text
    if old_x_label:
        fig.update_layout(xaxis_title_text=prettify_axis_label(old_x_label))

    old_y_label = fig.layout.yaxis.title.text
    if old_y_label:
        fig.update_layout(yaxis_title_text=prettify_axis_label(old_y_label))


def export_minimal_pdf(fig: go.Figure, output_path: Path, size: int = 1200) -> None:
    apply_minimal_layout(fig, size=size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_image(fig, str(output_path.with_suffix(".pdf")), format="pdf")
    print(f"Exported: {output_path.with_suffix('.pdf')}")


def plot_grouped_bar_chart(df: pd.DataFrame):
    st.subheader("Grouped Bar Chart For Two Measures")

    if SHORT_COL not in df.columns:
        st.warning(f"Column '{SHORT_COL}' not found in the DataFrame.")
        return

    # Let user select two numeric columns to compare
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns to plot grouped bar chart.")
        return

    col1 = st.selectbox("Select first column", numeric_cols, index=0)
    col2 = st.selectbox("Select second column", numeric_cols, index=1)

    # Normalize option
    normalize = st.checkbox("Normalize columns to [0, 1] scale?", value=False)

    # Sort dataframe by col1
    sorted_df = df[[SHORT_COL, col1, col2]].dropna().sort_values(by=col1, ascending=False).reset_index(drop=True)

    if normalize:
        sorted_df[col1] = (sorted_df[col1] - sorted_df[col1].min()) / (sorted_df[col1].max() - sorted_df[col1].min())
        sorted_df[col2] = (sorted_df[col2] - sorted_df[col2].min()) / (sorted_df[col2].max() - sorted_df[col2].min())

    # Create an x-label using row numbers and include SHORT_COL in hover
    x_labels = sorted_df.index.astype(str)

    fig = go.Figure(
        data=[
            go.Bar(
                name=col1,
                x=x_labels,
                y=sorted_df[col1],
                hovertemplate=f"{SHORT_COL}: %{{customdata[0]}}<br>{col1}: %{{y:.2f}}<extra></extra>",
                customdata=sorted_df[[SHORT_COL]],
            ),
            go.Bar(
                name=col2,
                x=x_labels,
                y=sorted_df[col2],
                hovertemplate=f"{SHORT_COL}: %{{customdata[0]}}<br>{col2}: %{{y:.2f}}<extra></extra>",
                customdata=sorted_df[[SHORT_COL]],
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Sorted Rows by First Column",
        yaxis_title="Value" if not normalize else "Normalized Value",
        title=f"Grouped Bar Chart: {col1} vs {col2} (Sorted by {col1})",
        legend_title="Columns",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


# --- Path input ---
# /opt/home/cleong/projects/pose-evaluation/metric_results_round_2/4_23_2025_score_analysis_3300_trials/stats_by_metric.csv,/opt/home/cleong/projects/pose-evaluation/metric_results/4_22_2025_csvcount_17187_score_analysis_with_updated_MAP/stats_by_metric.csv
# csv_paths_input = st.text_input(
#     "Enter paths to your CSV files (comma-separated)",
#     value="/opt/home/cleong/projects/pose-evaluation/metric_results_round_2/4_23_2025_score_analysis_3300_trials/stats_by_metric.csv,
# /opt/home/cleong/projects/pose-evaluation/metric_results/4_22_2025_csvcount_17187_score_analysis_with_updated_MAP/stats_by_metric.csv,/opt/home/cleong/projects/pose-evaluation/metric_results_z_offsets_combined/score_analysis/stats_by_metric.csv",
# )
csv_paths_default = [
    # "/opt/home/cleong/projects/pose-evaluation/metric_results/4_22_2025_csvcount_17187_score_analysis_with_updated_MAP/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results/score_analysis/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_round_2/4_23_2025_score_analysis_3300_trials/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_round_2/score_analysis/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_round_3/score_analysis/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_z_offsets_combined/score_analysis/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4/score_analysis/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_embeddings/score_analysis/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_1_2_z_combined_818_metrics/score_analysis/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_round_2/score_analysis/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4/score_analysis/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4/5_14_169_glosses_40680_metrics_score_analysis/stats_by_metric.csv",
    "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4/5_12_2025_score_analysis_288_metrics_169_glosses/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_embeddings/5_12_2025_embedding_score_analysis_169_glosses/stats_by_metric.csv",
    # "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4_pruned_to_match_embeddings/5_19_score_analysis_48_metrics_and_6_embedding_metrics_comparable/stats_by_metric.csv",
    # /opt/home/cleong/projects/pose-evaluation/metric_results_round_4_pruned_to_match_embeddings/5_19_score_analysis_48_and_2_and_6embedding_metrics_169_glosses
    "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4_pruned_to_match_embeddings/5_19_score_analysis_48_and_2_and_6embedding_metrics_169_glosses/stats_by_metric.csv",
    "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4_pruned_to_match_embeddings/5_21_score_analysis_1206metrics_169glosses/stats_by_metric.csv",
]
csv_paths_input = st.text_input(
    "Enter paths to your CSV files (comma-separated)",
    value=",".join(csv_paths_default),
)

csv_path_options = [p.strip() for p in csv_paths_input.split(",")]
csv_paths = st.multiselect(f"Which of these CSVs to load?", options=csv_path_options, default=csv_path_options)

if csv_paths_input:
    try:
        # Split input by comma and strip whitespace

        # Load and concatenate all CSVs
        df_list = []
        for path in csv_paths:
            df = pd.read_csv(path)
            df_list.append(df)
            st.write(f"Loaded {len(df)} from {path}")
        df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

    # Find duplicates in the METRIC_COL column
    df[SHORT_COL] = df[METRIC_COL].apply(shorten_metric_name)
    df[DESCRIPTIVE_NAME_COL] = df[METRIC_COL].apply(descriptive_name)
    df = df[[SHORT_COL] + [col for col in df.columns if col != SHORT_COL]]
    df = df[[DESCRIPTIVE_NAME_COL] + [col for col in df.columns if col != DESCRIPTIVE_NAME_COL]]
    duplicate_metrics = df[df.duplicated(SHORT_COL, keep=False)]
    st.write(
        f"Loaded {len(df)} metric stats, with {len(df[METRIC_COL].unique())} unique metric names and {len(df[SHORT_COL].unique())} unique short names"
    )

    # standardize names so that none of them contain others, e.g. "trimmed" within "untrimmed"
    # "trimmed" -> "startendtrimmed"
    df.loc[df[METRIC_COL].str.startswith("trimmed_"), METRIC_COL] = df[METRIC_COL].str.replace(
        "trimmed_", "startendtrimmed_", n=1
    )

    # "normalized" -> "normalizedbyshoulders"
    df[METRIC_COL] = df[METRIC_COL].str.replace("_normalized_", "_normalizedbyshoulders_", regex=False)

    if not duplicate_metrics.empty:
        show_dupes = st.checkbox(f"Show duplicate metrics: {len(duplicate_metrics)} rows")
        if show_dupes:
            duplicate_metrics = duplicate_metrics.sort_values(by=SHORT_COL)
            st.write(f"{len(duplicate_metrics)} Duplicate metrics found:")
            for short_dupe in duplicate_metrics[SHORT_COL].unique().tolist():
                st.write(f"{short_dupe}")
                st.write(duplicate_metrics[duplicate_metrics[SHORT_COL] == short_dupe])
    else:
        st.write("No duplicate metrics found.")

    # Drop duplicates by keeping the one with the highest total_count
    df_deduped = df.loc[df.groupby(SHORT_COL)["total_count"].idxmax()]

    st.write(f"Deduplicated metrics (kept the one with highest total_count): we now have {len(df_deduped)}")
    # st.write(df_deduped)
    df = df_deduped

    # stats
    metric_count = len(df[METRIC_COL].unique())
    total_distances_count = None
    trials_count = None

    # --- minimum query-gloss count
    min_hyp = st.number_input(
        "Filter out metrics with fewer than this many Query glosses? (0 to not filter)", min_value=0, value=0, step=1
    )
    if min_hyp > 0:
        df = df[df["hyp_gloss_count"] >= min_hyp]

    st.write(f"We have {len(df)} metrics left")

    # --- Column selection ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    sort_col = st.selectbox("Sort by column", numeric_cols)
    sort_col_capitalized = " ".join(s.capitalize() for s in sort_col.split("_"))
    sort_ascending = st.checkbox("Sort ascending?", value=False)

    # --- Keyword filtering ---
    exclude = st.text_input("Keywords to exclude? (comma-separated)", value="")
    include = st.text_input("Keywords to include? (comma-separated)", value="")

    metric_series = df[METRIC_COL].str.lower()

    match_all = st.checkbox("Require all keywords (AND)?", value=False)  # default is "any" (OR)

    if include:
        keywords = [kw.strip().lower() for kw in include.split(",") if kw.strip()]

        # Ensure your `metric_series` is lowercase for consistent matching
        metric_series = df["METRIC"].str.lower()

        if match_all:
            for kw in keywords:
                df = df[metric_series.str.contains(re.escape(kw), na=False)]
        else:
            pattern = "|".join(map(re.escape, keywords))
            df = df[metric_series.str.contains(pattern, na=False)]

    if exclude:
        keywords = [kw.strip().lower() for kw in exclude.split(",") if kw.strip()]
        pattern = "|".join(map(re.escape, keywords))
        df = df[~metric_series.str.contains(pattern, na=False)]

    # --- Multi-keyword matching ---
    keyword_input = st.text_input("Search / highlight by keyword(s) (comma-separated)", value="")

    multi_color = st.checkbox("Color bars by individual keyword?", value=True)

    # top N

    # hands,reduceholistic,removelegsandworld,return4 # hands gives best precision, removelegsandworld worst
    # zeropad,return4,interp15,interp120
    # return4,interp15,interp120,zeropad # shows clear badness

    # untrimmed,unnormalized
    # unnormalized # no strong pattern in precision, but interesting in mean in/out
    # removelegsandworld, return4
    # zeropad,dtw,return4 # same as AggregatedPowerDistance

    # AggregatedPowerDistance # all lower half of the precision list
    # AggregatedPowerDistance,return4,dtw

    # fillmasked0.0,fillmasked1.0,fillmasked10.0,return4 has no pattern I can see
    # defaultdist10.0,defaultdist1.0,defaultdist0.0,return4 has no pattern I can see

    # this is interesting for precision@10 and mean_score_time. Clear bands of performance
    # dtw,zeropad,hands,removelegsandworld,reduceholistic,return4,nointerp,interp15,interp120

    # dtw,zeropad,return4,nointerp,interp15,interp120 shows the mean_score_time effects well.

    # precision@5
    # dtw,hands,zeropad,return4 shows that dtw+hands is the clear winner
    # return4,dtw,zeropad,hands,reduceholistic,removelegsandworld,nointerp,interp15,interp120 precision@5: the top are all dtw hands

    # Keypoint Selection Strategy
    # hands,removelegsandworld,reduceholistic,youtubeaslkeypoints,return4

    # Sequence Alignment Strategy
    # dtw,zeropad,padwithfirstframe,return4

    # Interpolation
    # nointerp,interp15,interp120,return4

    df = df.copy()

    st.write(f"We have {metric_count} metrics originally, and after filters we have {len(df)}")
    if "hyp_gloss_count" in df.columns:
        trials_count = df["hyp_gloss_count"].sum()
        st.write(f"The total number of query-gloss+metric trials is {trials_count:,}")

    if "total_count" in df.columns:
        total_distances_count = df["total_count"].sum()
        st.write(f"The total number of pose distances is {total_distances_count:,}")

    if keyword_input.strip():
        keywords = [k.strip().lower() for k in keyword_input.split(",") if k.strip()]


        def match_keywords(text):
            matched = [kw for kw in keywords if kw in text.lower()]
            if matched:
                return " + ".join(sorted(set(matched))) if multi_color else f"Matched: {', '.join(keywords)}"
            return "Other"


        df["highlight"] = df.apply(
            lambda row: match_keywords(row[METRIC_COL]) if pd.notnull(row[METRIC_COL]) else "Other", axis=1
        )

        # Check SIGNATURE if METRIC didn't match any
        unmatched_mask = df["highlight"] == "Other"
        df.loc[unmatched_mask, "highlight"] = df.loc[unmatched_mask].apply(
            lambda row: match_keywords(row[SIGNATURE_COL]) if pd.notnull(row[SIGNATURE_COL]) else "Other", axis=1
        )
    else:
        df["highlight"] = "All"

    # --- Sort ---
    df = df.sort_values(by=sort_col, ascending=sort_ascending).reset_index(drop=True)
    df["RANK"] = df.index + 1  # Rank starts at 1

    top_n = st.number_input("Show top N rows (leave as 0 to show all)", min_value=0, value=0, step=1)
    if top_n > 0:
        df = df.head(top_n)

    # --- Plot ---
    title = f"{sort_col_capitalized} by Metric"

    top_or_bottom = ""
    if sort_ascending:
        top_or_bottom = "bottom"
    else:
        top_or_bottom = "top"

    legend_title = st.text_input("Legend Title", value="")
    if legend_title:
        title = title + f" and {legend_title}"

    if len(df) != metric_count:
        title = title + f" ({top_or_bottom} {len(df)} of {metric_count} metrics"
    else:
        title = title + f" ({metric_count} metrics"

    if len(df["hyp_gloss_count"].unique()) == 1:
        title = title + f" with {df['hyp_gloss_count'].iloc[0]} Query Glosses per Metric"
    elif trials_count is not None:
        title = title + f" over {trials_count:,} trials"

    if exclude:
        title = title + f", excl. '{exclude}'"

    if min_hyp > 0:
        title = title + f", excl. metrics w/ < {min_hyp:,} trials"

    # if total_distances_count is not None:
    #     title = title + f" with {total_distances_count:,} total distances"

    title = title + ")"
    title = st.text_input("Title for plot", value=title)
    show_data_labels = st.checkbox(f"Show data labels: {len(df)} rows")
    df["data_labels"] = df[sort_col].map(lambda x: f"{x:.5f}")
    text_arg = "data_labels" if show_data_labels else None

    fig = px.bar(
        df,
        x="RANK",
        y=sort_col,
        color="highlight",
        hover_data=["RANK", SHORT_COL, METRIC_COL, "hyp_gloss_count"],
        title=title,
        text=text_arg,
        category_orders={SHORT_COL: df[SHORT_COL].tolist()},
    )
    if legend_title:
        fig.update_layout(legend=dict(title=legend_title))

    if show_data_labels:
        fig.update_traces(textposition="auto", textfont_size=12)

    if st.button("Export PDF?"):
        # def export_minimal_pdf(fig: go.Figure, output_path: Path, is_3d: bool = True, size: int = 600) -> None:
        title_out = title.replace(" ", "") + ".pdf"
        fig_out = Path("/opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/plots") / f"{title_out}"
        st.write(f"Saved to {fig_out}")
        fig.write_image(fig_out.with_suffix(".png"))
        export_minimal_pdf(fig, output_path=fig_out)

    # fig.update_layout(xaxis_tickangle=45)
    # fig.update_layout(xaxis_ticktext=[], xaxis_tickvals=[])

    st.plotly_chart(fig, use_container_width=True)

    # --- Optional table ---
    if st.checkbox(f"Show data table: {len(df)} rows"):
        st.dataframe(df)
        st.write(df["METRIC"].tolist())
        df_csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Full Data Table CSV",
            data=df_csv_data,
            file_name=f"combined_metric_stats.csv",
            mime="text/csv",
        )
        markdown_output = ""
        #         r"""\begin{table}[ht]
        #     \centering
        #     \resizebox{\linewidth}{!}{%
        #     \begin{tabular}{lllccllrr}
        # \toprule
        # Base & Def &Fill & Trim & Norm & Pad & Keypoints & \textbf{mAP}$\uparrow$ & \textbf{P@10}$\uparrow$ \\
        # \midrule"""  # Initialize an empty string to store the Markdown
        metrics_to_markdown = df["METRIC"].tolist()
        metrics_to_markdown.reverse()
        for metric in metrics_to_markdown:
            metric_row = df[df["METRIC"] == metric].iloc[0]
            # markdown_lines = [
            #     f"\n% {metric} &\t{metric_row['mean_average_precision']:.2f} &\t{metric_row['precision@10']:.2f}&\t{metric_row['hyp_gloss_count']}\t\\\\",
            #     f"\n{descriptive_name(metric)} &\t{metric_row['mean_average_precision']:.2f} &\t{metric_row['precision@10']:.2f}&\t{metric_row['hyp_gloss_count']}\t\\\\",
            # ]
            choices = interpret_name(metric)
            if choices is not None:
                dd = choices["default"]
                fm = choices["fillmasked"]
                if choices["trim"]:
                    trim = r"\textcolor{green}{v}"
                else:
                    trim = r"\textcolor{red}{x}"

                if choices["normalize"]:
                    norm = r"\textcolor{green}{v}"
                else:
                    norm = r"\textcolor{red}{x}"

                seq_align = choices["seq_align"]
                if "first" in seq_align:
                    seq_align = "first"
                elif "zero" in seq_align:
                    seq_align = "zero"
                else:
                    seq_align = "/"

                kp = choices["keypoints"]
                if "removelegsandworld" == kp:
                    kp = "Upper Body"
                elif "youtube" in kp:
                    kp = "YT-ASL"
                elif "reduce" in kp:
                    kp = "Reduced"
                elif "hands" in kp:
                    kp = "Hands"

            else:
                dd = None
                fm = None
                trim = None
                norm = None
                seq_align = None
                kp = None

            markdown_lines = [
                # f"\n% {interpret_name(metric)}",
                f"\n% {descriptive_name(metric)}",
                f"\n% {metric} &\t{metric_row['mean_average_precision']:.2f} &\t{metric_row['precision@10']:.2f}\t\\\\",
                f"\n{descriptive_name(metric).split()[0]} & {dd} & {fm} & {trim} & {norm} & {seq_align} & {kp} &\t{metric_row['mean_average_precision'] * 100:.0f}\\% &\t{metric_row['precision@10'] * 100:.0f}\\%\t\\\\",
            ]

            for mdl in markdown_lines:
                if "Unknown" not in mdl:
                    markdown_output += mdl

                # st.code(mdl, language=None)
        #         markdown_output += r"""\bottomrule
        # \end{tabular}
        #     }
        #     \caption{Automatic meta-evaluation of reference-based metrics on retrieval, with 169 trials each. A representative sample, including the best-performing configurations, is shown in this table.}
        #     \label{tab:auto_eval}
        # \end{table}"""
        st.code(markdown_output, language=None)

    # --- Keyword Presence Effect Estimation (Group Comparison) ---
    st.markdown("## 🔍 Estimate Effect of a Keyword (Group Comparison)")
    effect_keyword = st.text_input(
        "Keyword to test effect of (case-insensitive, comma-separated)", key="group_effect_kw"
    )

    if not effect_keyword.strip():
        effect_keywords = [
            "dtw,dtaiDTWAggregatedDistanceMetricFast",
            "dtw",
            "zeropad",
            "hands",
            "reduceholistic",
            "removelegsandworld",
            "untrimmed",
            "startendtrimmed",
            "unnormalized",
            "normalizedbyshoulders",
            "nointerp",
            "return4",
            "padwithfirstframe",
            "youtubeaslkeypoints",
            "zspeed",
            "embedding",
            "pop_sign_finetune",
            "asl_citizen_finetune",
            "sem_lex_finetune",
            "asl_finetune",
            "asl_signs_finetune",
        ]
        for fps in [
            15,
            30,
            45,
            60,
            120,
        ]:
            effect_keywords.append(f"interp{fps}")

        for zspeed in [0.1, 1.0, 4.0, 100.0, 1000.0]:
            effect_keywords.append(f"zspeed{zspeed}")

        for dd in [
            0.0,
            1.0,
            10.0,
            100.0,
            1000.0,
        ]:
            effect_keywords.append(f"defaultdist{dd}")

        for fm in [
            0.0,
            1.0,
            10.0,
            100.0,
            1000.0,
        ]:
            effect_keywords.append(f"fillmasked{fm}")

    else:
        effect_keywords = [kw.strip() for kw in effect_keyword.split(",")]
        # effect_keywords = [effect_keyword]

    summary_data = []
    for effect_keyword in effect_keywords:
        kw = effect_keyword.strip().lower()
        df["_metric_lower"] = df[METRIC_COL].str.lower()

        has_kw = df[df["_metric_lower"].str.contains(kw)]
        no_kw = df[~df["_metric_lower"].str.contains(kw)]

        if len(has_kw) == 0:
            st.warning(f"No metrics contain keyword '{kw}'")
        elif len(no_kw) == 0:
            st.warning(f"All metrics contain keyword '{kw}'")
        else:
            avg_with = has_kw[sort_col].mean()
            max_with = has_kw[sort_col].max()
            min_with = has_kw[sort_col].min()
            avg_rank = has_kw["RANK"].mean()
            avg_without = no_kw[sort_col].mean()
            max_without = no_kw[sort_col].max()
            min_without = no_kw[sort_col].min()
            delta = avg_with - avg_without

            summary_data.append(
                {
                    "keyword": kw,
                    f"Δ {sort_col}": round(delta, 4),
                    f"count within {top_or_bottom} 100": (has_kw["RANK"] <= 100).sum(),
                    f"count within {top_or_bottom} 10": (has_kw["RANK"] <= 10).sum(),
                    f"count within {top_or_bottom} 5": (has_kw["RANK"] <= 5).sum(),
                    "mean (with kw)": round(avg_with, 4),
                    "mean (without kw)": round(avg_without, 4),
                    "n (with)": len(has_kw),
                    "n (without)": len(no_kw),
                    "mean metric rank": avg_rank,
                }
            )

            st.write(f"#### Effect of `{kw}`")
            st.write(f"Compared `{len(has_kw)}` metrics **with** '`{kw}`' vs `{len(no_kw)}` **without**.")
            st.write(f"**Average on '{sort_col}' with '{kw}':** `{avg_with:.4f}`")
            st.write(f"**Average on '{sort_col}' without '{kw}':** `{avg_without:.4f}`")
            st.write(f"**Estimated effect on '{sort_col}' of '{kw}':** `{delta:+.4f}`")
            st.write(f"{kw} count within {top_or_bottom} 100 by {sort_col}: {(has_kw['RANK'] <= 100).sum()}")
            st.write(f"{kw} count within {top_or_bottom} 10 by {sort_col}: {(has_kw['RANK'] <= 10).sum()}")
            st.write(f"{kw} count within {top_or_bottom} 5 by {sort_col}: {(has_kw['RANK'] <= 5).sum()}")

            if st.checkbox(f"Show distributions for {kw}?"):
                fig = go.Figure()

                fig.add_trace(
                    go.Histogram(
                        x=has_kw[sort_col],
                        name=f"Has '{kw}'",
                        marker_color="blue",
                        opacity=0.6,
                        histnorm="probability density",
                    )
                )

                fig.add_trace(
                    go.Histogram(
                        x=no_kw[sort_col],
                        name=f"No '{kw}'",
                        marker_color="orange",
                        opacity=0.6,
                        histnorm="probability density",
                    )
                )

                fig.update_layout(
                    barmode="overlay",
                    title=f"Distribution of '{sort_col}' by presence of '{kw}'",
                    xaxis_title=sort_col,
                    yaxis_title="Density",
                    legend=dict(x=0.7, y=0.95),
                )

                st.plotly_chart(fig, use_container_width=True)
    if st.checkbox(f"Pairwise comparisons for {effect_keywords}"):
        pairwise_effect_rows = []
        for effect_keywords_pairwise in effect_keywords:

            effect_keywords_pairwise = [k.strip() for k in effect_keywords_pairwise.split(",")]
            st.write(f"#### {effect_keywords_pairwise}")
            pairs = find_keyword_difference_pairs(df, effect_keywords_pairwise, metric_col="METRIC")
            # st.write(pairs)

            pairwise_keyword_effects = []
            missing_metric_pairs = []

            for metric_with_kw, metric_without_kw in pairs:
                row_with_kw = df[df["METRIC"] == metric_with_kw]
                row_without_kw = df[df["METRIC"] == metric_without_kw]

                if row_with_kw.empty or row_without_kw.empty:
                    missing_metric_pairs.append((metric_with_kw, metric_without_kw))
                    continue

                value_with_kw = row_with_kw[sort_col].values[0]
                value_without_kw = row_without_kw[sort_col].values[0]

                effect = value_with_kw - value_without_kw
                pairwise_keyword_effects.append(effect)

            # Compute and display the average effect
            if pairwise_keyword_effects:
                count_with = len(set(pair[0] for pair in pairs))
                count_without = len(set(pair[1] for pair in pairs))
                average_keyword_effect = sum(pairwise_keyword_effects) / len(pairwise_keyword_effects)
                st.write(f"Found {len(pairs)}, with {count_with} having and {count_without} not.")
                st.write(f"Average effect of {effect_keywords_pairwise} on '{sort_col}': {average_keyword_effect:.4f}")
                pairwise_effect_rows.append(
                    {
                        "Keywords": "+".join(effect_keywords_pairwise),
                        f"Average Pairwise Effect ({sort_col_capitalized})": average_keyword_effect,
                        "Pairs": len(pairs),
                        # "Metrics With": count_with,
                        # "Metrics Without": count_without,
                    }
                )

                # Build the table data
                effect_rows = []
                for (metric_with_kw, metric_without_kw), effect in zip(pairs, pairwise_keyword_effects):
                    effect_rows.append(
                        {
                            "Metric With Keyword(s)": metric_with_kw,
                            "Metric Without Keyword(s)": metric_without_kw,
                            f"{sort_col} Difference": round(effect, 4),
                        }
                    )

                # Convert to DataFrame and display

                if st.checkbox(f"Show individual result rows for {effect_keywords_pairwise}?"):
                    effects_df = pd.DataFrame(effect_rows)
                    st.dataframe(effects_df)

            else:
                st.warning(f"No effects computed for {effect_keywords_pairwise} — check for valid metric pairs.")

            # Optionally display missing matches
            if missing_metric_pairs:
                st.warning(f"Some metric pairs were not found in the DataFrame: {missing_metric_pairs}")

        st.write(f"### Pairwise Keyword Effects")
        pairwise_effect_df = pd.DataFrame(pairwise_effect_rows)
        pairwise_effect_df = pairwise_effect_df.sort_values(by=f"Average Pairwise Effect ({sort_col_capitalized})")
        st.dataframe(pairwise_effect_df)
        pairwise_effect_df_csv_data = pairwise_effect_df.to_csv(index=False)

        st.download_button(
            label="Download Pairwise Keyword Effects CSV",
            data=pairwise_effect_df_csv_data,
            file_name=f"pairwise_keyword_effects_on_{sort_col}.csv",
            mime="text/csv",
        )
        if st.checkbox(f"LaTeX version?"):
            latex_str = pairwise_effect_df.to_latex(
                index=False,
                caption=f"Average pairwise effect of various options on {sort_col_capitalized}",
                label=f"tab:avg_pairwise_effect_{sort_col}",
                float_format="%.3f".__mod__,
            )
            # wrapped = (
            #     "\\resizebox{\\textwidth}{!}{%\n"
            #     + latex_str +
            #     "}\n"
            # )
            st.code(latex_str)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by=f"Δ {sort_col}", ascending=sort_ascending)
        st.markdown("### 📋 Summary of Keyword Effects")
        st.dataframe(summary_df, use_container_width=True)
        csv_data = summary_df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"summary_of_keyword_effects_on_{sort_col}.csv",
            mime="text/csv",
        )

    # Plot two things against each other
    chart_type = st.selectbox("Choose chart type", ["Pareto Frontier", "Grouped Bar Chart"])
    if chart_type == "Grouped Bar Chart":
        plot_grouped_bar_chart(df)
    else:
        plot_pareto_frontier(df)

# conda activate /opt/home/cleong/envs/pose_eval_src && streamlit run pose_evaluation/evaluation/explore_metric_stats.py
