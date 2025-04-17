import streamlit as st
import pandas as pd
import plotly.express as px

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

    # --- Column selection ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    sort_col = st.selectbox("Sort by column", numeric_cols)
    sort_ascending = st.checkbox("Sort ascending?", value=False)

    # --- Keyword filtering + color labeling ---
    # --- Multi-keyword matching ---
    keyword_input = st.text_input("Search / highlight by keyword(s) (comma-separated)", value="")
    multi_color = st.checkbox("Color bars by individual keyword?", value=True)
    # hands,return4,world,reduce
    # zeropad,return4,interp15,interp120

    df = df.copy()

    if keyword_input.strip():
        keywords = [k.strip().lower() for k in keyword_input.split(",") if k.strip()]

        def match_keywords(text):
            matched = [kw for kw in keywords if kw in text.lower()]
            if matched:
                return " + ".join(sorted(set(matched))) if multi_color else f"Matched: {', '.join(keywords)}"
            return "Other"

        df["highlight"] = df.apply(
            lambda row: match_keywords(row["METRIC"]) if pd.notnull(row["METRIC"]) else "Other", axis=1
        )

        # Check SIGNATURE if METRIC didn't match any
        unmatched_mask = df["highlight"] == "Other"
        df.loc[unmatched_mask, "highlight"] = df.loc[unmatched_mask].apply(
            lambda row: match_keywords(row["SIGNATURE"]) if pd.notnull(row["SIGNATURE"]) else "Other", axis=1
        )
    else:
        df["highlight"] = "All"

    # --- Sort ---
    df = df.sort_values(by=sort_col, ascending=sort_ascending)

    # --- Plot ---
    fig = px.bar(
        df,
        x="METRIC",
        y=sort_col,
        color="highlight",
        hover_data=["SIGNATURE"],
        title=f"{sort_col} by METRIC",
        category_orders={"METRIC": df["METRIC"].tolist()},  # 🔥 preserves sorted order
    )

    # fig.update_layout(xaxis_tickangle=45)
    fig.update_layout(xaxis_ticktext=[], xaxis_tickvals=[])

    st.plotly_chart(fig, use_container_width=True)

    # --- Optional table ---
    if st.checkbox("Show data table"):
        st.dataframe(df)
