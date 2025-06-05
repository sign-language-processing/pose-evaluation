#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd

from tqdm import tqdm

from pose_evaluation.evaluation.score_dataframe_format import load_score_csv


def convert_csvs_to_parquet(
    folder: Path, out_dir: Path | None = None, remove_original: bool = False, score_csv_format=False
):
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder}")
        return

    if score_csv_format:
        print(f"Using score csv format")

    out_dir = out_dir or folder
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in tqdm(csv_files, desc="Converting CSV files to Parquet"):

        if score_csv_format:
            df = load_score_csv(csv_path)
        else:
            df = pd.read_csv(csv_path)
        parquet_path = out_dir / csv_path.with_suffix(".parquet").name
        df.to_parquet(parquet_path, index=False)
        # print(f"Converted: {csv_path.name} → {parquet_path}")
        if remove_original:
            csv_path.unlink()
            print(f"Deleted original: {csv_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV files in a folder to Parquet.")
    parser.add_argument("folder", type=Path, help="Path to folder containing CSV files")
    parser.add_argument("-o", "--out", type=Path, help="Optional output folder for Parquet files")
    parser.add_argument("--remove", action="store_true", help="Remove original CSV files after conversion")
    parser.add_argument(
        "--not-score-files",
        action="store_true",
        help="By default, will load with specific score CSV columns, datatypes. If this is given, will just use pd.read_csv.",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Error: {args.folder} is not a directory.")
        return
    use_score_files_format = not args.not_score_files
    convert_csvs_to_parquet(
        folder=args.folder, out_dir=args.out, remove_original=args.remove, score_csv_format=use_score_files_format
    )


if __name__ == "__main__":
    main()
