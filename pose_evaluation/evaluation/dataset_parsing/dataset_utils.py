import pandas as pd
from pathlib import Path
from typing import List

STANDARDIZED_VIDEO_ID_COL_NAME = "VIDEO_ID"
STANDARDIZED_SPLIT_COL_NAME = "SPLIT"
STANDARDIZED_GLOSS_COL_NAME = "GLOSS"


def file_paths_list_to_df(file_paths: List[Path], prefix="") -> pd.DataFrame:
    # Define the column names dynamically based on the prefix
    columns = {
        f"{prefix.upper()}_FILE_PATH" if prefix else "FILE_PATH": [str(f) for f in file_paths],
        # f"{prefix} FILE NAME" if prefix else "FILE NAME": [f.name for f in file_paths],
    }

    # Create the DataFrame with the correct column names
    df_paths = pd.DataFrame(columns)

    return df_paths


def df_to_standardized_df(
    df: pd.DataFrame,
    video_id_col="video_id",
    split_col="split",
    gloss_col="gloss",
    signer_id_col="signer_id",
    keep_cols=None,
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
    df["GLOSS"] = df["GLOSS"].str.upper()

    if keep_cols:
        df = df[[col for col in keep_cols if col in df.columns]]

    return df


def deduplicate_by_video_id(df, video_id_col="video_id", split_col="split", priority_order=["train", "val", "test"]):
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
