import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

METRIC_COL = "METRIC"
SIGNATURE_COL = "SIGNATURE"

# --- Path input ---
csv_path = st.text_input(
    "Enter path to your CSV file",
    value="/opt/home/cleong/projects/pose-evaluation/metric_results/score_analysis/stats_by_metric.csv",
)

if csv_path:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # stats
    metric_count = len(df[METRIC_COL].unique())
    total_distances_count = None
    trials_count = None

    # --- Column selection ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    sort_col = st.selectbox("Sort by column", numeric_cols)
    sort_ascending = st.checkbox("Sort ascending?", value=False)

    # --- Keyword filtering---
    exclude = st.text_input("Keywords to exclude? (comma-separated)", value="")
    if exclude:
        keywords = [kw.strip().lower() for kw in exclude.split(",") if kw.strip()]
        pattern = "|".join(map(re.escape, keywords))  # escape special characters
        df = df[~df[METRIC_COL].str.lower().str.contains(pattern, na=False)]

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

    df = df.copy()

    st.write(f"We have {metric_count} metrics")
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

    title = f"{sort_col} by Metric"

    if len(df) != metric_count:
        title = title + f" (top {len(df)}/{metric_count} metrics"
    else:
        title = title + f" ({metric_count} metrics"

    if exclude:
        title = title + f" excluding {exclude}"

    if trials_count is not None:
        title = title + f" with {trials_count:,} trials"

    if total_distances_count is not None:
        title = title + f" with {total_distances_count:,} total distances"

    title = title + ")"

    # --- Plot ---
    fig = px.bar(
        df,
        x="RANK",  # 👈 x-axis is now rank
        y=sort_col,
        color="highlight",
        hover_data=["RANK", METRIC_COL, SIGNATURE_COL],  # 👈 now shows rank on hover
        title=title,
        category_orders={METRIC_COL: df[METRIC_COL].tolist()},  # 🔥 preserves sorted order
    )

    # fig.update_layout(xaxis_tickangle=45)
    # fig.update_layout(xaxis_ticktext=[], xaxis_tickvals=[])

    st.plotly_chart(fig, use_container_width=True)

    # --- Optional table ---
    if st.checkbox(f"Show data table: {len(df)} rows"):
        st.dataframe(df)

    # --- Keyword Presence Effect Estimation (Group Comparison) ---
    st.markdown("## 🔍 Estimate Effect of a Keyword (Group Comparison)")
    effect_keyword = st.text_input("Keyword to test effect of (case-insensitive)", key="group_effect_kw")

    if effect_keyword.strip():
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
            avg_without = no_kw[sort_col].mean()
            delta = avg_with - avg_without

            st.write(f"Compared `{len(has_kw)}` metrics **with** keyword vs `{len(no_kw)}` **without**.")
            st.write(f"**Average with '{kw}':** `{avg_with:.4f}`")
            st.write(f"**Average without '{kw}':** `{avg_without:.4f}`")
            st.write(f"**Estimated effect of '{kw}':** `{delta:+.4f}`")

            if st.checkbox("Show distributions?"):

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
