"""
Script was intended to let me edit CSV files to e.g. fix metric names.

I forgot that some other columns might also need fixing, e.g. the
SIGNATURE
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from pose_evaluation.evaluation.score_dataframe_format import load_score_csv


def main():
    parser = argparse.ArgumentParser(description="Replace strings in a specific column of score CSVs.")
    parser.add_argument("--folder", type=Path, required=True, help="Folder containing score CSVs.")
    parser.add_argument("--column", type=str, required=True, help="Column name to modify.")
    parser.add_argument("--old", type=str, required=True, help="Substring to replace.")
    parser.add_argument("--new", type=str, required=True, help="New substring.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be changed, don't save anything.")

    args = parser.parse_args()

    for csv_path in tqdm(sorted(args.folder.glob("*.csv"))):
        # print(f"\n🔍 Checking: {csv_path}")
        df = load_score_csv(csv_path)

        # Show before/after if anything will be changed
        column_values = df[args.column]
        changed = column_values.str.contains(args.old, regex=False)
        if not changed.any():
            # print("  ✅ No occurrences found.")
            continue

        updated = column_values[changed].str.replace(args.old, args.new, regex=False)

        if args.dry_run:
            print(f"\n❌ {len(updated)} in {csv_path}, here are a few")
            for i, (old_val, new_val) in enumerate(zip(column_values[changed], updated, strict=False)):
                if i <= 5:
                    print(f"  📝 Would update row {changed.index[i]}: \n{old_val} \n→ \n{new_val}")
        else:
            df[args.column] = column_values.str.replace(args.old, args.new, regex=False)
            df.to_csv(csv_path)
            print(f"  💾 Updated {len(updated)} rows and saved: {csv_path.name}")


if __name__ == "__main__":
    main()
