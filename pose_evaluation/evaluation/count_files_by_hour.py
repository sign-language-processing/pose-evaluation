#!/usr/bin/env python3

import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Group files by modification hour and count them.")
    parser.add_argument("--path", default="metric_results/scores", type=Path, help="Path to glob for files, e.g., 'metric_results/scores/*'")
    args = parser.parse_args()

    path = Path(args.path).glob("*")

    timestamps = []
    for file in path:
        if file.is_file():
            stat = file.stat()
            dt = datetime.fromtimestamp(stat.st_mtime)
            hour_key = dt.strftime("%Y-%m-%d %H")  # e.g., '2025-04-17 15'
            timestamps.append(hour_key)

    counter = Counter(timestamps)
    for hour, count in sorted(counter.items()):
        print(f"{count:4} {hour}")

    if counter:
        total_files = sum(counter.values())
        total_hours = len(counter)
        avg_per_hour = total_files / total_hours
        print(f"\nAverage files/hour: {avg_per_hour:.2f}")
        print(f"\nTotal Files: {total_files}")
    else:
        print("No files found.")


if __name__ == "__main__":
    main()
