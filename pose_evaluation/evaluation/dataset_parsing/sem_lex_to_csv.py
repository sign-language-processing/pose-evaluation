import pandas as pd
import typer
from pathlib import Path
from typing import Optional, List

from pose_evaluation.evaluation.dataset_parsing.collect_files import collect_files_main
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import (
    find_duplicates,
    file_paths_list_to_df,
    deduplicate_by_video_id,
    df_to_standardized_df,
    STANDARDIZED_VIDEO_ID_COL_NAME,
    STANDARDIZED_GLOSS_COL_NAME,
    STANDARDIZED_SPLIT_COL_NAME,
)

app = typer.Typer()


@app.command()
def collect_semlex(
    dataset_path: Path = typer.Argument(..., exists=True, file_okay=False),
    pose_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    metadata_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    video_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    embedding_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    out: Optional[Path] = typer.Option(None, exists=False, file_okay=False),
):
    """Read in Sem-Lex files and metadata, combine to one dataframe, and save out to csv"""

    result = collect_files_main(
        dataset_path=dataset_path,
        pose_files_path=pose_files_path,
        metadata_path=metadata_path,
        video_files_path=video_files_path,
        embedding_files_path=embedding_files_path,
        pose_patterns=["*.pose"],
        metadata_patterns=["*semlex_metadata.csv"],
        video_patterns=["*.webm"],
        embedding_patterns=["*.npy"],
    )

    for name, paths in result.items():
        typer.echo(f"🎯 Found {len(paths)} {name.replace('_', ' ')}. Samples:")
        for path in paths[:3]:
            typer.echo(f"* {path}")

    # metadata
    meta_dfs = []
    for meta_file in result["METADATA_FILES"]:

        # should just be the one file, semlex_metadata.csv
        df = pd.read_csv(meta_file, index_col=0, header=0)
        # should have 91148 rows, including the index row
        if len(df) == 91148:
            typer.echo(f"Found sem-lex metadata file: {meta_file}")

            print(f"Deduplicating by video ID and split")
            df = deduplicate_by_video_id(
                df, video_id_col="video_id", split_col="split", priority_order=["train", "val", "test"]
            )
            print(f"There are now {len(df)} rows")

            df = df_to_standardized_df(
                df,
                gloss_col="label",
                keep_cols=[STANDARDIZED_VIDEO_ID_COL_NAME, STANDARDIZED_GLOSS_COL_NAME, STANDARDIZED_SPLIT_COL_NAME],
            )

            meta_dfs.append(df)

    df = pd.concat(meta_dfs)
    assert len(df) == 88174  # should be exactly this long after deduplication

    for prefix in ["POSE", "VIDEO"]:

        files_df = file_paths_list_to_df(result[f"{prefix}_FILES"], prefix=prefix)
        files_df[STANDARDIZED_VIDEO_ID_COL_NAME] = files_df[f"{prefix}_FILE_PATH"].apply(lambda x: Path(x).stem)
        print(f"Merging {len(files_df)} {prefix} files into df")
        df = df.merge(files_df, on=STANDARDIZED_VIDEO_ID_COL_NAME, how="left")

    # pose_files_df = file_paths_list_to_df(result["pose_files"], prefix="POSE")
    # video_files_df = file_paths_list_to_df(result["video_files"], prefix="VIDEO")
    embedding_files_files_df = file_paths_list_to_df(result["EMBEDDING_FILES"], prefix="EMBEDDING")
    embedding_files_files_df["VIDEO_ID"] = embedding_files_files_df["EMBEDDING_FILE_PATH"].apply(
        lambda x: Path(x).stem.split("-")[0]
    )
    embedding_files_files_df["EMBEDDING_MODEL"] = embedding_files_files_df["EMBEDDING_FILE_PATH"].apply(
        lambda x: Path(x).stem.split("-using-model-")[-1]
    )
    print(embedding_files_files_df.head())

    df = df.merge(embedding_files_files_df, on=STANDARDIZED_VIDEO_ID_COL_NAME, how="left")
    print(df.info())
    print(df.head())

    # print(result.keys())
    if out is not None:
        if out.name.endswith(".csv"):
            df.to_csv(out, index=False)
        if out.name.endswith(".json"):
            df.to_json(out)


if __name__ == "__main__":
    app()
