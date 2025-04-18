from collections import defaultdict
from typing import List
from pathlib import Path

import pandas as pd


STANDARDIZED_VIDEO_ID_COL_NAME = "VIDEO_ID"
STANDARDIZED_SPLIT_COL_NAME = "SPLIT"
STANDARDIZED_GLOSS_COL_NAME = "GLOSS"


def file_paths_list_to_df(
    file_paths: List[Path], prefix="", parse_metatadata_from_folder_structure=False
) -> pd.DataFrame:
    # Define the column names dynamically based on the prefix
    columns = {
        f"{prefix.upper()}_FILE_PATH" if prefix else "FILE_PATH": [str(f) for f in file_paths],
        # f"{prefix} FILE NAME" if prefix else "FILE NAME": [f.name for f in file_paths],
    }

    if parse_metatadata_from_folder_structure:
        columns.update(parse_split_and_gloss_from_file_paths(file_paths))

    # Create the DataFrame with the correct column names
    df_paths = pd.DataFrame(columns)

    return df_paths


def parse_split_and_gloss_from_file_paths(file_paths: List[Path], gloss_level=0, split_level=1):
    columns = defaultdict(list)

    for file_path in file_paths:
        parents = file_path.parents
        split_val = parents[split_level].name
        gloss_val = parents[gloss_level].name
        columns[STANDARDIZED_GLOSS_COL_NAME].append(gloss_val)
        columns[STANDARDIZED_SPLIT_COL_NAME].append(split_val)
    return columns


def df_to_standardized_df(
    df: pd.DataFrame,
    video_id_col="video_id",
    split_col="split",
    gloss_col="gloss",
    signer_id_col="signer_id",
):
    # Standardize to specific predictable names: "Video ID" or "video_id" for example,  becomes "VIDEO_ID"

    df = df.rename(
        columns={
            video_id_col: STANDARDIZED_VIDEO_ID_COL_NAME,
            split_col: STANDARDIZED_SPLIT_COL_NAME,
            gloss_col: STANDARDIZED_GLOSS_COL_NAME,
            signer_id_col: "PARTICIPANT_ID",
        }
    )

    # rename all columns to CAPITAL_UNDERSCORE format
    # Rename columns to uppercase with underscores

    df.columns = [col.replace(" ", "_").upper() for col in df.columns]

    # capitalize all glosses
    df[STANDARDIZED_GLOSS_COL_NAME] = df[STANDARDIZED_GLOSS_COL_NAME].str.upper()

    # lowercase all splits
    df[STANDARDIZED_SPLIT_COL_NAME] = df[STANDARDIZED_SPLIT_COL_NAME].str.lower()

    return df


def deduplicate_by_video_id(df, video_id_col="video_id", split_col="split", priority_order=None):
    if priority_order is None:
        priority_order = ["train", "val", "test"]
    # Sort by the priority of split, with 'train' first, then 'val', and 'test' last
    df["priority"] = df[split_col].apply(
        lambda x: priority_order.index(x) if x in priority_order else len(priority_order)
    )

    # Sort the DataFrame by video_id and priority, keeping the first occurrence per video_id with the highest priority
    df_sorted = df.sort_values(by=[video_id_col, "priority"], ascending=[True, True])

    # Drop duplicates, keeping the first occurrence of each video_id, which will be the one with the highest priority
    df_deduplicated = df_sorted.drop_duplicates(subset=[video_id_col], keep="first")

    # Drop the priority column now that we're done
    df_deduplicated = df_deduplicated.drop(columns=["priority"])

    return df_deduplicated


def find_duplicates(df: pd.DataFrame, column: str):
    """
    Finds and prints duplicate values in the specified column of a DataFrame.

    Parameters:
    - df: pd.DataFrame — the input DataFrame
    - column: str — the column to check for duplicates

    Returns:
    - duplicate_counts: pd.Series — counts of duplicated values
    - duplicate_rows: pd.DataFrame — rows with duplicated values
    """
    duplicate_rows = df[df.duplicated(subset=column, keep=False)].sort_values(by=column)
    duplicate_counts = df[column].value_counts()
    duplicate_counts = duplicate_counts[duplicate_counts > 1]

    print(f"Duplicate '{column}' counts:")
    print(duplicate_counts)

    print(f"\nRows with duplicate '{column}' values:")
    print(duplicate_rows)

    return duplicate_counts, duplicate_rows


def convert_eng_to_ase_gloss_translations(df, asl_knowledge_graph_df, translations_only=False):
    translation_df = asl_knowledge_graph_df[asl_knowledge_graph_df["relation"] == "has_translation"]
    # translation_df = asl_knowledge_graph_df[asl_knowledge_graph_df["source"] == "asllex"]
    translation_df.loc[:, "object"] = translation_df["object"].str.upper()
    # translation_df["object"] = translation_df["object"].str.upper()

    matching_translations = translation_df[translation_df["object"].isin(df["GLOSS"])]

    selected_translations = []
    for translated_word in matching_translations["object"].unique():
        translations = matching_translations[matching_translations["object"] == translated_word]
        translations_without_colon = []
        for translation in translations["subject"].tolist():
            translation = translation.split(":")[-1].upper()
            translations_without_colon.append(translation)

        if len(set(translations_without_colon)) == 1:
            translated_word_without_lang = translated_word.split(":")[1]
            translation = list(set(translations_without_colon))[0]

            if translated_word_without_lang == translation:
                selected_translations.append((translated_word, translation))

    mapping_dict = dict(selected_translations)

    if translations_only:
        # Filter rows where GLOSS is in the mapping keys
        df = df[df["GLOSS"].isin(mapping_dict)].copy()

    # Apply the mapping safely using .loc
    df.loc[:, "GLOSS"] = df["GLOSS"].map(mapping_dict).fillna(df["GLOSS"])
    return df
