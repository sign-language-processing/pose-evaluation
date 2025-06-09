import pandas as pd
import typer
from pathlib import Path
from typing import Optional

from pose_evaluation.evaluation.dataset_parsing.collect_files import collect_files_main
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import (
    file_paths_list_to_df,
    deduplicate_by_video_id,
    df_to_standardized_df,
    DatasetDFCol,
)

app = typer.Typer()


@app.command()
def collect_semlex(
    dataset_path: Path = typer.Argument(..., exists=True, file_okay=False),
    pose_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    metadata_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    video_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    out: Optional[Path] = typer.Option(None, exists=False, file_okay=True),
):
    """Read in files and metadata, combine to one dataframe, and save out to csv"""
    # pylint: disable=duplicate-code
    result = collect_files_main(
        dataset_path=dataset_path,
        pose_files_path=pose_files_path,
        metadata_path=metadata_path,
        video_files_path=video_files_path,
        pose_patterns=["*.pose"],
        metadata_patterns=["*semlex_metadata.csv"],
        video_patterns=["*.webm"],
    )
    # pylint: enable=duplicate-code

    # metadata
    meta_dfs = []
    for meta_file in result["METADATA_FILES"]:

        # should just be the one file, semlex_metadata.csv
        df = pd.read_csv(meta_file, index_col=0, header=0)
        # should have 91148 rows, including the index row
        if len(df) == 91148:
            typer.echo(f"Found sem-lex metadata file: {meta_file}")

            typer.echo("Deduplicating by video ID and split")

            typer.echo(f"There are now {len(df)} rows")

            df = df_to_standardized_df(
                df,
                gloss_col="label",
            )

            meta_dfs.append(df)

    df = pd.concat(meta_dfs)

    for prefix in ["POSE", "VIDEO"]:
        files_df = file_paths_list_to_df(result[f"{prefix}_FILES"], prefix=prefix)
        files_df[DatasetDFCol.VIDEO_ID] = files_df[f"{prefix}_FILE_PATH"].apply(lambda x: Path(x).stem)
        typer.echo(f"Merging {len(files_df)} {prefix} files into df")
        df = df.merge(files_df, on=DatasetDFCol.VIDEO_ID, how="left")

    df = df_to_standardized_df(
        df,
        video_id_col=DatasetDFCol.VIDEO_ID,
        split_col=DatasetDFCol.SPLIT,
        gloss_col=DatasetDFCol.GLOSS,
    )

    df = df[
        [
            col
            for col in [
                DatasetDFCol.GLOSS,
                DatasetDFCol.SPLIT,
                DatasetDFCol.VIDEO_ID,
                "POSE_FILE_PATH",
                # "VIDEO_FILE_PATH",
            ]
            if col in df.columns
        ]
    ]
    df = deduplicate_by_video_id(
        df, video_id_col="VIDEO_ID", split_col="SPLIT", priority_order=["train", "val", "test"]
    )

    assert len(df["VIDEO_ID"].unique()) == len(df["VIDEO_ID"])
    assert len(df["VIDEO_ID"].unique()) == len(df["POSE_FILE_PATH"])
    assert len(df) == 88174
    typer.echo(df.info())
    typer.echo(df.head())
    typer.echo(df.describe())

    if out is not None:
        if out.name.endswith(".csv"):
            df.to_csv(out, index=False)
        if out.name.endswith(".json"):
            df.to_json(out)


if __name__ == "__main__":
    app()
