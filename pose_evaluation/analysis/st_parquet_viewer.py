from pathlib import Path

import pandas as pd
import streamlit as st

st.title("🪵 Parquet Viewer")

# Text input for parquet file path
file_path = st.text_input("Enter path to Parquet file:", "")

# Try loading and displaying the parquet file
if file_path:
    path = Path(file_path)
    if not path.exists():
        st.error("File not found. Please check the path.")
    elif not path.suffix == ".parquet":
        st.warning("The file does not have a .parquet extension. Are you sure it's a Parquet file?")
    else:
        try:
            df = pd.read_parquet(path)
            st.success(f"Loaded file with shape: {df.shape}")

            # Viewer
            st.subheader("Data Viewer (read-only)")
            st.dataframe(df)

            # Optional: editable version
            st.subheader("Editable View (changes not saved)")
            edited_df = st.data_editor(df, use_container_width=True)
        except Exception as e:
            st.exception(f"Error loading file: {e}")

# conda activate /opt/home/cleong/envs/pose_eval_src && streamlit run pose_evaluation/analysis/st_parquet_viewer.py