import pandas as pd
import typer
from pathlib import Path

from pose_evaluation.evaluation.dataset_parsing.collect_files import parse_id_and_model_name_from_embedding_file
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import file_paths_list_to_df, DatasetDFCol

app = typer.Typer()


def get_embeddings_df(embeddings_folder: Path, split_id_on_dash=False):
    embedding_files = list(embeddings_folder.rglob("*using-model*.npy"))
    prefix = "EMBEDDING"
    files_df = file_paths_list_to_df(embedding_files, prefix=prefix)
    # id, model_name = parse_id_and_model_name_from_embedding_file(path)
    # files_df[DatasetDFCol.VIDEO_ID]
    # files_df[DatasetDFCol.EMBEDDING_MODEL]
    files_df[[DatasetDFCol.VIDEO_ID, DatasetDFCol.EMBEDDING_MODEL]] = files_df[f"{prefix}_FILE_PATH"].apply(
        lambda path: pd.Series(parse_id_and_model_name_from_embedding_file(path))
    )

    return files_df


@app.command()
def process(
    input_csv: Path = typer.Argument(..., exists=True, help="Path to input CSV file"),
    embeddings_folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Path to folder with embeddings"),
    output_csv: Path = typer.Option(Path("output.csv"), help="Path to output CSV file"),
):
    typer.echo(f"Reading input CSV: {input_csv}")
    typer.echo(f"Using embeddings from: {embeddings_folder}")
    typer.echo(f"Will write output to: {output_csv}")

    dataset_df = pd.read_csv(input_csv)
    typer.echo("**** Dataset DF: ****")
    typer.echo(dataset_df.head())
    typer.echo(dataset_df.info())
    typer.echo()

    typer.echo("**** Embedding DF: ****")
    embeddings_df = get_embeddings_df(embeddings_folder)
    typer.echo(embeddings_df.head())
    typer.echo(embeddings_df.info())
    typer.echo()

    typer.echo("**** Merged DF: ****")
    merged_df = dataset_df.merge(embeddings_df, on=DatasetDFCol.VIDEO_ID, how="left")
    if len(merged_df) == len(dataset_df):
        # ASL Citizen has slightly different filenames...
        embeddings_df[DatasetDFCol.VIDEO_ID] = embeddings_df[DatasetDFCol.VIDEO_ID].astype(str).str.split("-").str[0]
        merged_df = dataset_df.merge(embeddings_df, on=DatasetDFCol.VIDEO_ID, how="left")
    typer.echo(merged_df.head())
    typer.echo(merged_df.info())
    typer.echo()

    merged_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    app()
# python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/dataset_parsing/add_embeddings_to_dataset_dataframe.py dataset_dfs/semlex.csv /opt/home/cleong/data/Sem-Lex/embeddings/ --output-csv /opt/home/cleong/projects/pose-evaluation/dataset_dfs_with_embed/semlex.csv
# python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/dataset_parsing/add_embeddings_to_dataset_dataframe.py dataset_dfs/asl-citizen.csv /opt/home/cleong/data/ASL_Citizen/re-embed/ --output-csv /opt/home/cleong/projects/pose-evaluation/dataset_dfs_with_embed/asl-citizen.csv
# python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/dataset_parsing/add_embeddings_to_dataset_dataframe.py dataset_dfs/popsign_asl.csv /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/PopSignASL/embeddings --output-csv /opt/home/cleong/projects/pose-evaluation/dataset_dfs_with_embed/popsign_asl.csv
