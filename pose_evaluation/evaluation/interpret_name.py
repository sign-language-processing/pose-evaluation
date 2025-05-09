from pathlib import Path
import hashlib
import re

import warnings

import pandas as pd
import numpy as np


from pose_evaluation.evaluation.create_metrics import get_metrics


def extract_with_prefix(s: str, prefix: str, cast_fn=float):
    """Extract value following a prefix ending with an underscore and followed by an underscore."""
    match = re.search(rf"_{re.escape(prefix)}([0-9.]+)_", s)
    return cast_fn(match.group(1)) if match else None


def extract_flagged_choice(s: str, options: list[str]):
    """Return the first matching option found as a substring surrounded by underscores."""
    for opt in options:
        if f"_{opt}_" in s:
            return opt
    return None


def interpret_name(metric_name: str):
    choices = {}

    # Measure type
    if "dtaiDTWAggregatedDistanceMetricFast" in metric_name:
        choices["measure"] = "dtaiDTWAggregatedDistanceFast"
    elif "AggregatedPowerDistanceMetric" in metric_name:
        choices["measure"] = "AggregatedPowerDistance"
    else:
        return None

    # Trim type
    choices["trim"] = metric_name.startswith(("startendtrimmed_", "trimmed_"))

    # Default distance and fillmasked
    for key in ["defaultdist", "fillmasked"]:
        val = extract_with_prefix(metric_name, key)
        if val is not None:
            choices["default" if key == "defaultdist" else "fillmasked"] = val

    # Interpolation
    if "_nointerp_" in metric_name:
        choices["interp"] = None
    elif "_interp" in metric_name:
        interp_val = extract_with_prefix(metric_name, "interp")
        choices["interp"] = interp_val
    else:
        raise ValueError(f"Invalid interp in: {metric_name}")

    # Normalization
    if "_normalized_" in metric_name or "_normalizedbyshoulders_" in metric_name:
        choices["normalize"] = True
    elif "_unnormalized_" in metric_name:
        choices["normalize"] = False

    # Keypoints
    keypoints_opts = [
        "removelegsandworld",
        "reduceholistic",
        "hands",
        "youtubeaslkeypoints",
    ]
    keypoints_choice = extract_flagged_choice(metric_name, keypoints_opts)
    if keypoints_choice:
        choices["keypoints"] = keypoints_choice

    # Z-speed
    zspeed = extract_with_prefix(metric_name, "zspeed")
    choices["zspeed"] = zspeed

    # Sequence alignment
    seq_align = extract_flagged_choice(metric_name, ["dtw", "padwithfirstframe", "zeropad"])
    if seq_align:
        choices["seq_align"] = seq_align

    return choices


def shorten_metric_name(metric_name: str):

    abbrev = {
        "normalize": "n",
        "trim": "t",
        "default": "dd",
        "fillmasked": "fm",
        "interp": "int",
        "keypoints": "kp",
        "zspeed": "zs",
        "measure": "",
        "original": "original",
        "seq_align": "sa",
    }

    value_abbrev = {
        "removelegsandworld": "rmlegwld",
        "reduceholistic": "reduce",
        "hands": "hand",
        "youtubeaslkeypoints": "yt",
        "zeropad": "zp",
        "padwithfirstframe": "p1f",
    }

    choices_short = []
    choices = interpret_name(metric_name)
    if choices is None:
        # e.g. "Return4Metric"
        return metric_name
    for choice_name, choice in choices.items():

        if isinstance(choice, bool):
            if choice == True:
                choice = "y"
            else:
                choice = "n"
        # print(choice_name, abbrev[choice_name], choice)

        if choice is None:
            continue

        if choice_name in abbrev:
            choice_name = abbrev[choice_name]

        if choice in value_abbrev:
            choice = value_abbrev[choice]

        if choice_name == "":
            choices_short.append(f"{choice}")
        else:
            choices_short.append(f"{choice_name}:{choice}")
    short = "_".join(choices_short)
    # print(metric_name)
    # print(short)
    return short


if __name__ == "__main__":
    METRIC_FILES = [
        "/opt/home/cleong/projects/pose-evaluation/metric_results/4_22_2025_csvcount_17187_score_analysis_with_updated_MAP/stats_by_metric.csv",
        "/opt/home/cleong/projects/pose-evaluation/metric_results_round_2/4_23_2025_score_analysis_3300_trials/stats_by_metric.csv",
    ]
    dfs = [pd.read_csv(f) for f in METRIC_FILES]

    df = pd.concat(dfs)

    choices = []
    shorts = []
    historical_metrics = df["METRIC"].unique().tolist()
    print(f"Loaded {len(historical_metrics):,} metrics we have previously used.")
    constructed_metrics = [m.name for m in get_metrics()]
    print(f"Loaded {len(constructed_metrics):,} newly constructed metrics")
    metrics = historical_metrics + constructed_metrics
    print(f"The total is {len(metrics):,}.")
    metrics = list(set(metrics))
    print(f"After deduplication, the total is {len(metrics)}.")

    for metric in metrics:
        metric_choices = interpret_name(metric)

        if metric_choices is not None:
            short = shorten_metric_name(metric)
            shorts.append(short)
            # print(short)
            metric_choices["original"] = metric
            metric_choices["short"] = short
            metric_choices["hash"] = hashlib.md5(f"{metric}".encode()).hexdigest()[:8]
            choices.append(metric_choices)

    interpretation = pd.DataFrame(choices)
    interpretation.sort_values(by="short")

    # print(interpretation)
    counts = []

    # Find duplicates in the "short" column
    duplicates = interpretation[interpretation.duplicated(subset="short", keep=False)]
    duplicates = duplicates.sort_values(by="short")

    # Display the original and short columns for these duplicates
    # print(duplicates[["original", "short"]])

    interpretation_csv = Path("metric_name_interpretation.csv")
    dupe_short_names_csv = Path("metric_name_short_dupes.csv")
    interpretation.to_csv(interpretation_csv)
    duplicates.to_csv(dupe_short_names_csv)

    for col in interpretation.columns:
        uniques = interpretation[col].unique()
        if col not in ["original", "short", "hash"]:
            print("\\item", col, len(uniques), uniques.tolist())
        else:
            print(f"Total {col}:{len(interpretation[col]):,}, Unique:{len(uniques):,}")
        counts.append(len(uniques))
    print(f"{len(interpretation):,} interpretations written to {interpretation_csv.resolve()}")
    print(f"{len(duplicates):,} dupes written to {dupe_short_names_csv.resolve()}")
    print("duplicate short names:", len(duplicates))
