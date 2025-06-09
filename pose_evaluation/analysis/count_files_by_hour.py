#!/usr/bin/env python3

import argparse
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Group files by modification time and count them.")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to glob for files, e.g., 'metric_results/scores/*'",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=3,
        help="Only include files modified within the past N hours (default: 3).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="Target number of files. If set, estimates how many hours needed to reach it at current rate.",
    )
    args = parser.parse_args()

    now = datetime.now()
    cutoff = now - timedelta(hours=args.hours)

    paths = list(Path(args.path).rglob("*"))

    hour_counts = Counter()
    minute_counts = Counter()
    total_files = 0

    for file in tqdm(paths, desc="Processing files", disable=True):
        if file.is_file():
            stat = file.stat()
            dt = datetime.fromtimestamp(stat.st_mtime)
            if dt >= cutoff:
                hour_key = dt.strftime("%Y-%m-%d %H")
                minute_key = dt.strftime("%Y-%m-%d %H:%M")
                hour_counts[hour_key] += 1
                minute_counts[minute_key] += 1
                total_files += 1

    average_strs = []
    avg_per_minute = None

    if hour_counts:
        print("\nFiles per hour:")
        for hour, count in sorted(hour_counts.items()):
            print(f"{count:4} {hour}")

        avg_per_hour = total_files / len(hour_counts)
        average_strs.append(f"\nAverage files/hour: {avg_per_hour:.2f}")

    if minute_counts:
        # print("\nFiles per minute:")
        # for minute, count in sorted(minute_counts.items()):
        #     print(f"{count:4} {minute}")

        earliest = min(datetime.strptime(k, "%Y-%m-%d %H:%M") for k in minute_counts)
        duration_minutes = max(1, int((now - earliest).total_seconds() // 60))
        avg_per_minute = total_files / duration_minutes
        average_strs.append(f"Average files/minute: {avg_per_minute:.2f} (over {duration_minutes} minute(s))")

    for avg_str in average_strs:
        print(avg_str)

    print(f"\nTotal files: {len(paths):,}")
    print(f"Total files in last {args.hours} hour(s): {total_files:,}")

    # Target projection
    if args.target_count is not None and avg_per_minute:
        files_remaining = args.target_count - len(paths)
        if files_remaining <= 0:
            print(f"\nTarget of {args.target_count:,} files already reached or exceeded.")
        else:
            est_minutes_needed = files_remaining / avg_per_minute
            est_hours_needed = est_minutes_needed / 60
            # est_hours_needed = files_remaining / avg_per_hour
            # {percent:.2f}%
            percent = len(paths) / args.target_count * 100
            print(
                f"\n{percent:.2f}% done. Remaining: {files_remaining:,}/{args.target_count:,}. Estimated time to reach target: {est_hours_needed:.2f} hour(s)"
            )


if __name__ == "__main__":
    main()