#!/usr/bin/env python3
import argparse
import logging
import re
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm
from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol, load_score_csv

# --- Constants & Regexes ------------------------------------------------
# Signature: default_distance:<float>
_SIGNATURE_RE = re.compile(r"default_distance:([\d.]+)")
# Filename metric: defaultdist<float>
_DEFAULTDIST_RE = re.compile(r"defaultdist[\d.]+")

# --- Helper Functions ---------------------------------------------------


def extract_metric_name_from_filename(stem: str) -> Optional[str]:
    """
    Extract everything between the first underscore and '_outgloss_' in the file stem.
    e.g., 'GLOSS_trimmed_normalized_defaultdist10.0_extra_outgloss_4x_score_results'
    returns 'trimmed_normalized_defaultdist10.0_extra'
    """
    if "_outgloss_" not in stem or "_" not in stem:
        return None
    _, rest = stem.split("_", 1)
    metric, _ = rest.split("_outgloss_", 1)
    return metric


def extract_signature_distance(signature: str) -> Optional[str]:
    """
    From a signature string, extract the float following 'default_distance:'.
    Returns None if not found.
    """
    m = _SIGNATURE_RE.search(signature)
    return m.group(1) if m else None


def build_new_metric_name(old_metric: str, signature_dist: str) -> str:
    """
    Normalize prefixes and substrings, then align 'defaultdist' to signature:
      1) If starts with 'trimmed', replace that prefix with 'startendtrimmed'.
         (leave 'untrimmed' unchanged)
      2) Replace any '_normalized_' with '_normalizedbyshoulders_'
         (exact match; '_unnormalized_' is unaffected)
      3) Replace existing 'defaultdist<old>' with 'defaultdist<signature_dist>'
         or append '_defaultdist<signature_dist>' if missing.
    """
    metric = old_metric
    # 1) trimmed -> startendtrimmed
    if metric.startswith("trimmed"):
        metric = "startendtrimmed" + metric[len("trimmed") :]
    # 2) exact replace normalized
    metric = metric.replace("_normalized_", "_normalizedbyshoulders_")
    # 3) defaultdist replacement or append
    if _DEFAULTDIST_RE.search(metric):
        metric = _DEFAULTDIST_RE.sub(f"defaultdist{signature_dist}", metric)
    else:
        metric = f"{metric}_defaultdist{signature_dist}"
    return metric


def build_new_filename(old_stem: str, new_metric: str) -> str:
    """
    Replace the metric segment of old_stem with new_metric, preserving everything before the first '_'
    and after '_outgloss_'.
    """
    prefix, _, rest = old_stem.partition("_")
    # rest = old_metric + '_outgloss_' + suffix
    _, _, suffix = rest.partition("_outgloss_")
    return f"{prefix}_{new_metric}_outgloss_{suffix}"


def dedupe_and_append(out_path: Path, df_new: pd.DataFrame, ext: str) -> None:
    """
    If out_path exists, read it, concatenate new df, drop duplicates, and overwrite.
    Uses load_score_csv for CSVs to preserve dtypes.
    """
    if ext == "csv":
        df_existing = load_score_csv(out_path)
    else:
        df_existing = pd.read_parquet(out_path)

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates().reset_index(drop=True)

    if ext == "csv":
        df_combined.to_csv(out_path, index=False)
    else:
        df_combined.to_parquet(out_path, index=False)


# --- Core Processing ----------------------------------------------------


def process_scores(scores_dir: Path, out_dir: Path, ext: str = "csv") -> Dict[str, int]:
    """
    Process all *.{ext} files in scores_dir, enforcing that
    METRIC and filename defaultdist match the signature default_distance.

    Special case: if old_metric does not contain 'defaultdist' or
    signature lacks 'default_distance:', simply copy unchanged.
    Skip files with multiple distinct signatures.
    """
    stats = {
        "total_inputs": 0,
        "copied_unchanged": 0,
        "renamed": 0,
        "skipped_multi_signature": 0,
        "skipped_copy_special": 0,
        "deduplicated": 0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    for infile in tqdm(sorted(scores_dir.glob(f"*.{ext}"))):
        stats["total_inputs"] += 1

        # Load DataFrame with correct dtypes
        if ext == "csv":
            df = load_score_csv(infile)
        else:
            df = pd.read_parquet(infile)

        sigs = df[ScoreDFCol.SIGNATURE].unique()
        if len(sigs) != 1:
            logging.warning(
                "%s has %d distinct signatures; skipping.",
                infile.name,
                len(sigs),
            )
            stats["skipped_multi_signature"] += 1
            continue

        signature = sigs[0]
        sig_dist = extract_signature_distance(signature)

        stem = infile.stem
        old_metric = extract_metric_name_from_filename(stem) or ""

        # Special case: no defaultdist in metric OR no default_distance: in signature
        if "defaultdist" not in old_metric or sig_dist is None:
            # copy file as-is
            out_path = out_dir / infile.name
            # ensure METRIC dtype preserved
            if ext == "csv":
                df.to_csv(out_path, index=False)
            else:
                df.to_parquet(out_path, index=False)
            stats["copied_unchanged"] += 1
            stats["skipped_copy_special"] += 1
            logging.info("Copied without processing: %s", infile.name)
            continue

        # Build corrected metric name & filename
        new_metric = build_new_metric_name(old_metric, sig_dist)
        new_stem = build_new_filename(stem, new_metric)
        out_path = out_dir / f"{new_stem}.{ext}"

        # Update METRIC column
        df = df.copy()
        df[ScoreDFCol.METRIC] = new_metric

        # Save or dedupe-as-needed
        if out_path.exists():
            dedupe_and_append(out_path, df, ext)
            stats["deduplicated"] += 1
            logging.info("Deduplicated into: %s", out_path.name)
        else:
            if ext == "csv":
                df.to_csv(out_path, index=False)
            else:
                df.to_parquet(out_path, index=False)
            stats["renamed"] += 1
            logging.info("Written: %s", out_path.name)

    return stats


# --- CLI ---------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fix defaultdist mismatches between filename and SIGNATURE in score files"
    )
    parser.add_argument("scores_folder", type=Path, help="Directory containing GLOSS_*_outgloss_*.{csv,parquet} files")
    parser.add_argument("--out", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--ext", choices=["csv", "parquet"], default="csv", help="File extension to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)

    stats = process_scores(args.scores_folder, args.out, args.ext)

    # Summary
    summary = (
        f"Processed: {stats['total_inputs']} inputs\n"
        f"  • Copied unchanged:       {stats['copied_unchanged']}\n"
        f"  • Renamed/corrected:      {stats['renamed']}\n"
        f"  • Deduplicated:           {stats['deduplicated']}\n"
        f"  • Skipped multi-signature:{stats['skipped_multi_signature']}\n"
        f"  • Skipped special-copy:   {stats['skipped_copy_special']}"
    )
    print("\n=== Summary ===")
    print(summary)

    # Write summary.txt
    out_summary = args.out / "summary.txt"
    out_summary.write_text(summary)
    logging.info("Summary written to %s", out_summary)


if __name__ == "__main__":
    main()
