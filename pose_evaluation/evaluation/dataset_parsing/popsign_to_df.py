from pathlib import Path
from typing import Optional


import typer


from pose_evaluation.evaluation.dataset_parsing.collect_files import collect_files_main
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import (
    file_paths_list_to_df,
    df_to_standardized_df,
    STANDARDIZED_VIDEO_ID_COL_NAME,
    STANDARDIZED_GLOSS_COL_NAME,
    STANDARDIZED_SPLIT_COL_NAME,
)

app = typer.Typer()


@app.command()
def collect(
    dataset_path: Path = typer.Argument(..., exists=True, file_okay=False),
    pose_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    video_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    out: Optional[Path] = typer.Option(None, exists=False, file_okay=True),
):
    # pylint: disable=duplicate-code
    result = collect_files_main(
        dataset_path=dataset_path,
        pose_files_path=pose_files_path,
        metadata_path=None,
        video_files_path=video_files_path,
        pose_patterns=["*.pose"],
        metadata_patterns=None,
        video_patterns=["*.mp4"],
    )
    # pylint: enable=duplicate-code

    files_dfs = []
    for prefix in ["POSE", "VIDEO"]:

        files_df = file_paths_list_to_df(
            result[f"{prefix}_FILES"], prefix=prefix, parse_metatadata_from_folder_structure=True
        )
        files_df[STANDARDIZED_VIDEO_ID_COL_NAME] = files_df[f"{prefix}_FILE_PATH"].apply(lambda x: Path(x).stem)

        files_dfs.append(files_df)

    df = files_dfs[0]
    for files_df in files_dfs[1:]:
        typer.echo(f"Merging {len(files_df)} {prefix} files into df")

        # Merge temporarily to check for mismatches
        merged_check = df.merge(files_df, on="VIDEO_ID", suffixes=("_x", "_y"))

        # Assert that SPLIT and GLOSS columns match
        assert (
            merged_check[f"{STANDARDIZED_SPLIT_COL_NAME}_x"] == merged_check[f"{STANDARDIZED_SPLIT_COL_NAME}_y"]
        ).all(), f"{STANDARDIZED_SPLIT_COL_NAME} values do not match"
        assert (
            merged_check[f"{STANDARDIZED_GLOSS_COL_NAME}_x"] == merged_check[f"{STANDARDIZED_GLOSS_COL_NAME}_y"]
        ).all(), f"{STANDARDIZED_GLOSS_COL_NAME} values do not match"

        files_df = files_df.drop(columns=["GLOSS", "SPLIT"])
        df = df.merge(files_df, on=STANDARDIZED_VIDEO_ID_COL_NAME, how="left")

    print(f"There are {len(df[STANDARDIZED_VIDEO_ID_COL_NAME].unique())} unique video ids")

    df = df_to_standardized_df(
        df,
        video_id_col=STANDARDIZED_VIDEO_ID_COL_NAME,
        split_col=STANDARDIZED_SPLIT_COL_NAME,
        gloss_col=STANDARDIZED_GLOSS_COL_NAME,
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
