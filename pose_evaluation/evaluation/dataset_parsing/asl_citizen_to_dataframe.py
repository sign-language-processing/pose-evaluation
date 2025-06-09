from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from pose_evaluation.evaluation.dataset_parsing.collect_files import collect_files_main
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import (
    DatasetDFCol,
    deduplicate_by_video_id,
    df_to_standardized_df,
    file_paths_list_to_df,
)

app = typer.Typer()


@app.command()
def collect(
    dataset_path: Path = typer.Argument(..., exists=True, file_okay=False),
    pose_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    metadata_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    video_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    out: Optional[Path] = typer.Option(None, exists=False, file_okay=True),
):
    """Read in Sem-Lex files and metadata, combine to one dataframe, and save out to csv"""
    # pylint: disable=duplicate-code
    result = collect_files_main(
        dataset_path=dataset_path,
        pose_files_path=pose_files_path,
        metadata_path=metadata_path,
        video_files_path=video_files_path,
        pose_patterns=["*.pose"],
        metadata_patterns=["train.csv", "val.csv", "test.csv"],
        video_patterns=["*.mp4"],
    )
    # pylint: enable=duplicate-code

    for name, paths in result.items():
        typer.echo(f"🎯 Found {len(paths)} {name.replace('_', ' ')}. Samples:")
        for path in paths[:3]:
            typer.echo(f"* {path}")

    # metadata
    meta_dfs = []
    for meta_file in result["METADATA_FILES"]:
        split_name = meta_file.stem

        df = pd.read_csv(meta_file, index_col=0, header=0)

        # 8336197103293617-CHAMP.mp4 becomes 8336197103293617
        df[DatasetDFCol.VIDEO_ID] = df["Video file"].apply(lambda x: Path(x).stem.split("-")[0])

        typer.echo(f"Found metadata file: {meta_file}")

        df["SPLIT"] = split_name
        df = df_to_standardized_df(
            df,
        )

        meta_dfs.append(df)

    df = pd.concat(meta_dfs)
    typer.echo(f"Deduplicating by video ID and split, currently there are {len(df)} rows")
    df = deduplicate_by_video_id(
        df, video_id_col="VIDEO_ID", split_col="SPLIT", priority_order=["train", "val", "test"]
    )
    typer.echo(f"There are now {len(df)} rows")

    for prefix in ["POSE", "VIDEO"]:

        files_df = file_paths_list_to_df(result[f"{prefix}_FILES"], prefix=prefix)
        files_df[DatasetDFCol.VIDEO_ID] = files_df[f"{prefix}_FILE_PATH"].apply(lambda x: Path(x).stem.split("-")[0])
        # typer.echo(files_df.head())
        typer.echo(f"Merging {len(files_df)} {prefix} files into df")
        df = df.merge(files_df, on=DatasetDFCol.VIDEO_ID, how="left")
    df = df_to_standardized_df(
        df,
        video_id_col=DatasetDFCol.VIDEO_ID,
        split_col=DatasetDFCol.SPLIT,
        gloss_col=DatasetDFCol.GLOSS,
    )
    typer.echo(df.info())
    typer.echo(df.head())
    if out is not None:
        if out.name.endswith(".csv"):
            df.to_csv(out, index=False)
        if out.name.endswith(".json"):
            df.to_json(out)


if __name__ == "__main__":
    app()
