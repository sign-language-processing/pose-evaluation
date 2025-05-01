import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
import shutil
import numpy as np
from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol, load_score_csv


def load_score_parquet(parquet_file: Path) -> pd.DataFrame:
    """Loads a score Parquet file into a Pandas DataFrame."""
    return pd.read_parquet(parquet_file)


def save_score_parquet(df: pd.DataFrame, parquet_file: Path):
    """Saves a score DataFrame to a Parquet file."""
    df.to_parquet(parquet_file, index=False)


def save_score_csv(df: pd.DataFrame, csv_file: Path):
    """Saves a score DataFrame to a CSV file."""
    df.to_csv(csv_file, index=False)


def fix_filenames_deduplicate_content_to_out(scores_folder: Path, out_folder: Path, file_format: str = "csv"):
    """
    Identifies and renames score files. Copies all processed files to the out_folder.
    If a filename collision occurs, it deduplicates the content and saves the result in the out_folder.

    Args:
        scores_folder: Path to the folder containing the score files.
        out_folder: Path to the folder where processed files will be copied.
        file_format: The format of the score files. Must be either "csv" or "parquet".
                     Defaults to "csv".
    """
    if file_format not in ["csv", "parquet"]:
        raise ValueError(f"Invalid file_format: {file_format}. Must be 'csv' or 'parquet'.")

    out_folder.mkdir(parents=True, exist_ok=True)
    file_extension = f"*.{file_format}"
    score_files = list(scores_folder.glob(file_extension))
    processed_files = set()

    for score_file in tqdm(score_files, f"Processing {file_format} files"):
        try:
            if file_format == "csv":
                df = load_score_csv(score_file)
            else:
                df = load_score_parquet(score_file)

            signatures = df[ScoreDFCol.SIGNATURE].unique().tolist()
            new_filepath = None

            if len(signatures) == 1:
                signature = signatures[0]
                match = re.search(r"default_distance:([\d.]+)", signature)
                if match:
                    signature_distance = float(match.group(1))
                    filename = score_file.stem
                    filename_parts = filename.split("_")
                    if len(filename_parts) > 1 and "outgloss" in filename_parts:
                        outgloss_index = filename_parts.index("outgloss")
                        metric_name_parts = filename_parts[1:outgloss_index]
                        current_metric_name = "_".join(metric_name_parts)
                        file_distance_match = re.search(r"defaultdist([\d.]+)", current_metric_name)
                        if file_distance_match:
                            file_distance = float(file_distance_match.group(1))
                            if abs(signature_distance - file_distance) > 1e-6:  # Check for significant difference
                                new_metric_name_parts = []
                                for part in metric_name_parts:
                                    if "defaultdist" in part:
                                        new_metric_name_parts.append(f"defaultdist{signature_distance}")
                                    else:
                                        new_metric_name_parts.append(part)
                                new_metric_name = "_".join(new_metric_name_parts)
                                new_filename_base = f"{filename_parts[0]}_{new_metric_name}_outgloss_{'_'.join(filename_parts[outgloss_index+1:])}"
                                new_filepath = out_folder / f"{new_filename_base}.{file_format}"

            if new_filepath:
                if new_filepath in processed_files:
                    print(
                        f"Collision detected in output folder: {new_filepath.name} (from {score_file.name} and another file) - attempting to deduplicate content."
                    )
                    try:
                        if file_format == "csv":
                            existing_df = load_score_csv(new_filepath)
                        else:
                            existing_df = load_score_parquet(new_filepath)
                        combined_df = pd.concat([df, existing_df]).drop_duplicates().reset_index(drop=True)
                        if file_format == "csv":
                            save_score_csv(combined_df, new_filepath)
                        else:
                            save_score_parquet(combined_df, new_filepath)
                        print(f"Successfully deduplicated and saved to {new_filepath.name}")
                        processed_files.add(new_filepath)
                    except Exception as e_dedup:
                        print(f"Error during deduplication for {new_filepath.name}: {e_dedup}")
                else:
                    if file_format == "csv":
                        save_score_csv(df, new_filepath)
                    else:
                        save_score_parquet(df, new_filepath)
                    print(f"Renamed and saved to: {new_filepath.name} (from {score_file.name})")
                    processed_files.add(new_filepath)
            else:
                # No renaming needed, just copy to the output folder
                output_path = out_folder / score_file.name
                shutil.copy2(score_file, output_path)
                processed_files.add(output_path)
                print(f"Copied: {score_file.name} -> {output_path.name}")

        except Exception as e:
            print(f"Error processing {score_file.name}: {e}")

    print(f"Finished processing. {len(processed_files)} files in the output folder: {out_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix misnamed score files, deduplicate content on collision, and output to a specified folder."
    )
    parser.add_argument(
        "scores_folder",
        type=Path,
        help="Path to the folder containing the score files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to the output folder where processed files will be saved.",
    )
    parser.add_argument(
        "--file-format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Format of the score files to process (csv or parquet). Defaults to csv.",
    )
    args = parser.parse_args()
    fix_filenames_deduplicate_content_to_out(args.scores_folder, args.out, args.file_format)
