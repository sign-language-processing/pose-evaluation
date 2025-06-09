import random
from pathlib import Path

import pandas as pd
import streamlit as st


def display_video_for_gloss(rows: pd.DataFrame, gloss_label: str, key_prefix: str):
    st.markdown(f"### {gloss_label}")
    video_paths = rows["VIDEO_FILE_PATH"].dropna().astype(str).tolist()

    if not video_paths:
        st.info("No video files available for this gloss.")
        return

    # Create mapping from filename to full path
    name_to_path = {Path(p).name: p for p in video_paths}
    filenames = list(name_to_path.keys())

    # Try to select a default with 'seed' in the name
    default_name = next((name for name in filenames if "seed" in name.lower()), None)
    default_idx = filenames.index(default_name) if default_name in filenames else 0

    selected_name = st.selectbox(
        f"Select a video for {gloss_label}",
        options=filenames,
        index=default_idx,
        key=f"{key_prefix}_selectbox",
    )

    selected_path = name_to_path[selected_name]
    start_time = random.uniform(0.1, 0.3)
    try:
        st.video(selected_path, autoplay=True, loop=True, start_time=start_time)
    except st.errors.StreamlitDuplicateElementId:
        st.video(selected_path, autoplay=False, loop=True, start_time=start_time)


@st.cache_data
def get_all_mp4s(videos_path: Path):
    return list(videos_path.rglob("*.mp4"))


@st.cache_data
def attach_video_paths(df: pd.DataFrame, file_paths: list) -> pd.DataFrame:
    df = df.copy()

    # Assuming file_paths is a list of Path objects
    # Extract VIDEO_ID from filenames and map to path
    video_id_to_path = {path.stem.split("-")[0]: str(path) for path in file_paths}
    # video_id_to_name = {path.stem.split("-")[0]: str(path.name) for path in file_paths}

    # Map VIDEO_ID in the dataframe to the corresponding file path
    df["VIDEO_FILE_PATH"] = df["VIDEO_ID"].map(video_id_to_path)
    # df["VIDEO_FILE_NAME"] = df["VIDEO_ID"].map(video_id_to_name)
    return df


# File paths
similar_pairs_path = Path(
    "/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/SimilarSigns/deduped_sorted_similar_gloss_pairs_with_manual_pairs.csv"
)
# dataset_path = Path("/opt/home/cleong/projects/pose-evaluation/dataset_dfs/asl-citizen.csv")
dataset_path = Path(
    "/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/SimilarSigns/asl_citizen_dataset_df_with_video_paths.csv"
)
videos_path = Path("/opt/home/cleong/projects/semantic_and_visual_similarity/nas_data/ASL_Citizen/ASL_Citizen/videos")

# Preprocess mp4s into string paths


# Load CSVs
if st.checkbox(f"Load from {similar_pairs_path}?"):
    similar_pairs_df = pd.read_csv(similar_pairs_path)
    st.info(f"{len(similar_pairs_df)} pairs loaded")
else:
    similar_pairs_df = pd.DataFrame(columns=["GLOSS_A", "GLOSS_B"])
dataset_df = pd.read_csv(dataset_path)

# load the videos
# mp4_paths_str = [str(p) for p in mp4s]
if "VIDEO_FILE_PATH" not in dataset_df.columns and dataset_df["VIDEO_FILE_PATHS"]:
    mp4s = get_all_mp4s(videos_path)
    st.write(f"Found {len(mp4s)} videos")
    dataset_df = attach_video_paths(dataset_df, mp4s)
    df_csv_data = dataset_df.to_csv(index=False)

    st.download_button(
        label="Download Full Data Table CSV",
        data=df_csv_data,
        file_name="dataset_df.csv",
        mime="text/csv",
    )
st.info(f"Loaded {len(dataset_df)} rows from {dataset_path}")
raw_input = st.text_area("Paste GLOSS pairs here (comma-separated, one per line):", height=200)

if raw_input:
    lines = raw_input.strip().splitlines()
    gloss_pairs = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        parts = sorted(parts)
        if len(parts) == 2:
            gloss_pairs.append({"GLOSS_A": parts[0], "GLOSS_B": parts[1]})
        else:
            st.warning(f"Skipping invalid line: {line}")

    input_pairs_df = pd.DataFrame(gloss_pairs)
    st.success(f"Parsed {len(input_pairs_df)} valid gloss pairs.")
    st.dataframe(input_pairs_df)

    # Combine with existing similar_pairs_df
    similar_pairs_df = pd.concat([input_pairs_df, similar_pairs_df], ignore_index=True)
    st.success(f"Prepended to existing pairs. New total: {len(similar_pairs_df)}")


# Get all unique glosses from GLOSS_A and GLOSS_B
unique_glosses = sorted(set(similar_pairs_df["GLOSS_A"]).union(similar_pairs_df["GLOSS_B"]))

# Multiselect filter
selected_glosses = st.multiselect("Filter pairs containing any of these glosses:", unique_glosses)
if selected_glosses:
    filtered_pairs_df = similar_pairs_df[
        similar_pairs_df["GLOSS_A"].isin(selected_glosses) | similar_pairs_df["GLOSS_B"].isin(selected_glosses)
    ]

else:
    filtered_pairs_df = similar_pairs_df

if st.checkbox("Filter manually verified similar?"):
    filtered_pairs_df = filtered_pairs_df[pd.isna(filtered_pairs_df["manual_verification"])]


if st.button("Shuffle?"):
    filtered_pairs_df = filtered_pairs_df.sample(frac=1).reset_index(drop=True)


top_n = st.number_input("Show top N rows (leave as 0 to show all)", min_value=0, value=0, step=1)


start_i = st.number_input("Starting Index?", min_value=0, max_value=len(filtered_pairs_df), value=0, step=max(top_n, 1))

if start_i > 0:
    filtered_pairs_df = filtered_pairs_df.iloc[start_i:]
if top_n > 0:
    filtered_pairs_df = filtered_pairs_df.head(top_n)


st.info(f"{len(filtered_pairs_df)} pairs")
if st.checkbox("Show filtered pairs?"):
    st.dataframe(filtered_pairs_df)
# Ensure the necessary columns exist
if "GLOSS" not in dataset_df.columns:
    st.error("Column 'GLOSS' not found in dataset_df")
elif not all(col in similar_pairs_df.columns for col in ["GLOSS_A", "GLOSS_B"]):
    st.error("Columns 'GLOSS_A' and/or 'GLOSS_B' not found in similar_pairs_df")
else:
    st.title("Similar Gloss Pair Viewer")

    for idx, row in filtered_pairs_df.iterrows():
        gloss_a = row["GLOSS_A"]
        gloss_b = row["GLOSS_B"]

        rows_a = dataset_df[dataset_df["GLOSS"] == gloss_a]
        rows_b = dataset_df[dataset_df["GLOSS"] == gloss_b]

        st.subheader(f"Pair {idx + 1}: {gloss_a} ↔ {gloss_b}")
        show_df = st.checkbox(f"Show dataframe for Pair {idx + 1}?", key=f"show_df_{idx}")
        show_videos = st.checkbox(f"Show video for Pair {idx + 1}?", key=f"show_vid_{idx}", value=True)
        manual_verification = row["manual_verification"]
        st.info(f"Manual Verification Status: {manual_verification}")
        if st.checkbox("similar?", key=f"similar_checkbox_{idx}", value=True):
            manual_verification_thoughts = "similar"
        else:
            manual_verification_thoughts = "not similar"
        manual_verification_thoughts = st.text_input(
            "Further thoughts on this similarity?", manual_verification_thoughts, key=f"manual_verify_text_{idx}"
        )
        if st.button(
            f"Update CSV ({gloss_a},{gloss_b}) with '{manual_verification_thoughts}'", key=f"update_csv_{idx}"
        ):
            # find similar_pairs_df row where "GLOSS_A" == gloss_a and "GLOSS_B" == gloss_b
            # overwrite "manual_verification" value with "not similar"
            # resave it to similar_pairs_path
            mask = (similar_pairs_df["GLOSS_A"] == gloss_a) & (similar_pairs_df["GLOSS_B"] == gloss_b)
            similar_pairs_df.loc[mask, "manual_verification"] = manual_verification_thoughts
            similar_pairs_df.to_csv(similar_pairs_path, index=False)
            st.success(f"Updated {similar_pairs_path}")

        col1, col2 = st.columns(2)
        if show_df:
            with col1:
                st.markdown(f"### GLOSS_A: {gloss_a}")
                st.dataframe(rows_a)

            with col2:
                st.markdown(f"### GLOSS_B: {gloss_b}")
                st.dataframe(rows_b)

        if show_videos:
            with col1:
                display_video_for_gloss(rows_a, f"GLOSS_A: {gloss_a}", f"vid_a_{idx}")

            with col2:
                display_video_for_gloss(rows_b, f"GLOSS_B: {gloss_b}", f"vid_b_{idx}")
