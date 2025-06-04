import re
import os
import torch
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pose_evaluation.evaluation.interpret_name import shorten_metric_name, descriptive_name, interpret_name

# https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/4
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]  # type: ignore

# # or simply:
# torch.classes.__path__ = []


METRIC_COL = "METRIC"
SIGNATURE_COL = "SIGNATURE"
SHORT_COL = "SHORT"


def plot_grouped_bar_chart(df_to_plot: pd.DataFrame):
    st.subheader("Grouped Bar Chart")

    if SHORT_COL not in df_to_plot.columns:
        st.warning(f"Column '{SHORT_COL}' not found in the DataFrame.")
        return

    # Let user select two numeric columns to compare
    numeric_cols_to_group = df_to_plot.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols_to_group) < 2:
        st.warning("Need at least two numeric columns to plot grouped bar chart.")
        return

    col1 = st.selectbox("Select first column", numeric_cols_to_group, index=0)
    col2 = st.selectbox("Select second column", numeric_cols_to_group, index=1)

    # Normalize option
    normalize = st.checkbox("Normalize columns to [0, 1] scale?", value=False)

    # Sort dataframe by col1
    sorted_df = (
        df_to_plot[[SHORT_COL, col1, col2]].dropna().sort_values(by=col1, ascending=False).reset_index(drop=True)
    )

    if normalize:
        sorted_df[col1] = (sorted_df[col1] - sorted_df[col1].min()) / (sorted_df[col1].max() - sorted_df[col1].min())
        sorted_df[col2] = (sorted_df[col2] - sorted_df[col2].min()) / (sorted_df[col2].max() - sorted_df[col2].min())

    # Create an x-label using row numbers and include SHORT_COL in hover
    x_labels = sorted_df.index.astype(str)

    grouped_bar_fig = go.Figure(
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
    grouped_bar_fig.update_layout(
        barmode="group",
        xaxis_title="Sorted Rows by First Column",
        yaxis_title="Value" if not normalize else "Normalized Value",
        title=f"Grouped Bar Chart: {col1} vs {col2} (Sorted by {col1})",
        legend_title="Columns",
        height=500,
    )

    st.plotly_chart(grouped_bar_fig, use_container_width=True)


# --- Path input ---
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
    "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4/5_14_169_glosses_40680_metrics_score_analysis/stats_by_metric.csv",
    "/opt/home/cleong/projects/pose-evaluation/metric_results_round_4/5_12_2025_score_analysis_288_metrics_169_glosses/stats_by_metric.csv",
    "/opt/home/cleong/projects/pose-evaluation/metric_results_embeddings/5_2_2025_embedding_score_analysis_169_glosses/stats_by_metric.csv",
]
csv_paths_input = st.text_input(
    "Enter paths to your CSV files (comma-separated)",
    value=",".join(csv_paths_default),
)

csv_path_options = [p.strip() for p in csv_paths_input.split(",")]
csv_paths = st.multiselect("Which of these CSVs to load?", options=csv_path_options, default=csv_path_options)

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
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        st.stop()

    # Find duplicates in the METRIC_COL column
    df[SHORT_COL] = df[METRIC_COL].apply(shorten_metric_name)
    df = df[[SHORT_COL] + [col for col in df.columns if col != SHORT_COL]]
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
    title = f"{sort_col.replace("_"," ").capitalize()} by Metric"

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

    if trials_count is not None:
        title = title + f" over {trials_count:,} trials"

    if exclude:
        title = title + f", excl. '{exclude}'"

    if min_hyp > 0:
        title = title + f", excl. metrics w/ < {min_hyp:,} trials"

    title = title + ")"
    title = st.text_input("Title for plot", value=title)
    show_data_labels = st.checkbox(f"Show data labels: {len(df)} rows")
    df["data_labels"] = df[sort_col].map(lambda x: f"{x:.5f}")
    text_arg = "data_labels" if show_data_labels else None

    distribution_fig = px.bar(
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
        distribution_fig.update_layout(legend=dict(title=legend_title))

    if show_data_labels:
        distribution_fig.update_traces(textposition="auto", textfont_size=12)

    # fig.update_layout(xaxis_tickangle=45)
    # fig.update_layout(xaxis_ticktext=[], xaxis_tickvals=[])

    st.plotly_chart(distribution_fig, use_container_width=True)

    # --- Optional table ---
    if st.checkbox(f"Show data table: {len(df)} rows"):
        st.dataframe(df)
        st.write(df["METRIC"].tolist())
        df_csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Full Data Table CSV",
            data=df_csv_data,
            file_name="combined_metric_stats.csv",
            mime="text/csv",
        )
        markdown_output = ""  # Initialize an empty string to store the Markdown
        metrics_to_markdown = df["METRIC"].tolist()
        metrics_to_markdown.reverse()
        for metric in metrics_to_markdown:
            metric_row = df[df["METRIC"] == metric].iloc[0]
            # markdown_lines = [
            #     f"\n% {metric} &\t{metric_row['mean_average_precision']:.2f} &\t{metric_row['precision@10']:.2f}&\t{metric_row['hyp_gloss_count']}\t\\\\",
            #     f"\n{descriptive_name(metric)} &\t{metric_row['mean_average_precision']:.2f} &\t{metric_row['precision@10']:.2f}&\t{metric_row['hyp_gloss_count']}\t\\\\",
            # ]
            markdown_lines = [
                f"\n% {interpret_name(metric)}",
                f"\n% {metric} &\t{metric_row['mean_average_precision']:.2f} &\t{metric_row['precision@10']:.2f}\t\\\\",
                f"\n{descriptive_name(metric)} &\t{metric_row['mean_average_precision']:.2f} &\t{metric_row['precision@10']:.2f}\t\\\\",
            ]

            for mdl in markdown_lines:
                if "Unknown" not in mdl:
                    markdown_output += mdl

                # st.code(mdl, language=None)
        st.code(markdown_output, language=None)

    # --- Keyword Presence Effect Estimation (Group Comparison) ---
    st.markdown("## 🔍 Estimate Effect of a Keyword (Group Comparison)")
    effect_keyword = st.text_input("Keyword to test effect of (case-insensitive)", key="group_effect_kw")
    effect_keywords = []
    if not effect_keyword.strip():
        effect_keywords = [
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

    if effect_keyword.strip():
        effect_keywords = [effect_keyword]

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
            st.write(f"{kw} count within {top_or_bottom} 10 by {sort_col}: {(has_kw['RANK']<= 10).sum()}")
            st.write(f"{kw} count within {top_or_bottom} 5 by {sort_col}: {(has_kw['RANK'] <= 5).sum()}")

            if st.checkbox(f"Show distributions for {kw}?"):

                distribution_fig = go.Figure()

                distribution_fig.add_trace(
                    go.Histogram(
                        x=has_kw[sort_col],
                        name=f"Has '{kw}'",
                        marker_color="blue",
                        opacity=0.6,
                        histnorm="probability density",
                    )
                )

                distribution_fig.add_trace(
                    go.Histogram(
                        x=no_kw[sort_col],
                        name=f"No '{kw}'",
                        marker_color="orange",
                        opacity=0.6,
                        histnorm="probability density",
                    )
                )

                distribution_fig.update_layout(
                    barmode="overlay",
                    title=f"Distribution of '{sort_col}' by presence of '{kw}'",
                    xaxis_title=sort_col,
                    yaxis_title="Density",
                    legend=dict(x=0.7, y=0.95),
                )

                st.plotly_chart(distribution_fig, use_container_width=True)

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
    plot_grouped_bar_chart(df)
