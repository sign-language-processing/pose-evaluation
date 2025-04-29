import pandas as pd
import numpy as np


def interpret_name(metric_name: str):

    choices = {}
    # choices["original"] = metric_name

    if "dtaiDTWAggregatedDistanceMetricFast" in metric_name:
        choices["measure"] = "dtaiDTWAggregatedDistanceFast"
    elif "AggregatedPowerDistanceMetric" in metric_name:
        choices["measure"] = "AggregatedPowerDistance"
    else:
        return None

    if metric_name.startswith("untrimmed_"):
        choices["trim"] = False
    if metric_name.startswith("trimmed_"):
        choices["trim"] = True
    elif metric_name.startswith("startendtrimmed_"):
        choices["trim"] = True
    else:
        choices["trim"] = False

    for masked_fill in [0.0, 1.0, 10.0]:
        if f"_defaultdist{masked_fill}_" in metric_name:
            choices["default"] = masked_fill

    for masked_fill in [0.0, 1.0, 10.0]:
        if f"_fillmasked{masked_fill}_" in metric_name:
            choices["fillmasked"] = masked_fill

    # for interp in ["nointerp", "interp15", "interp120"]:
    if "_nointerp_" in metric_name:
        choices["interp"] = None
    elif "_interp" in metric_name:

        fps = float(metric_name.split("interp")[1].split("_")[0])

        if np.isnan(fps):
            print(metric_name)
            exit()

        choices["interp"] = fps
        # for fps in [15, 120]:
        # if f"_interp{fps}_" in metric_name:
        #     choices["interp"] = fps
    else:
        print(metric_name)
        exit()

    if "_normalized_" in metric_name or "_normalizedbyshoulders_" in metric_name:
        choices["normalize"] = True
    elif "_unnormalized_" in metric_name:
        choices["normalize"] = False

    for keypoints in [
        "removelegsandworld",
        "reduceholistic",
        "hands",
        "youtubeaslkeypoints",
    ]:
        if f"_{keypoints}_" in metric_name:
            choices["keypoints"] = keypoints

    if "_zspeed" in metric_name:
        zspeed = float(metric_name.split("zspeed")[1].split("_")[0])
        choices["zspeed"] = zspeed
    else:
        choices["zspeed"] = None

    for seq in ["dtw", "padwithfirstframe", "zeropad"]:
        if f"_{seq}_" in metric_name:
            choices["seq_align"] = seq

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
    for choice_name, choice in choices.items():

        if isinstance(choice, bool):
            if choice == True:
                choice = "y"
            else:
                choice = "n"
        print(choice_name, abbrev[choice_name], choice)

        if choice is None:
            continue
        choice_name = abbrev[choice_name]

        if choice in value_abbrev:
            choice = value_abbrev[choice]

        if choice_name == "":
            choices_short.append(f"{choice}")
        else:
            choices_short.append(f"{choice_name}:{choice}")
    short = "_".join(choices_short)
    print(metric_name)
    print(short)
    return short


if __name__ == "__main__":
    dfs = [
        pd.read_csv(
            "/opt/home/cleong/projects/pose-evaluation/metric_results/4_22_2025_csvcount_17187_score_analysis_with_updated_MAP/stats_by_metric.csv"
        ),
        pd.read_csv(
            "/opt/home/cleong/projects/pose-evaluation/metric_results_round_2/4_23_2025_score_analysis_3300_trials/stats_by_metric.csv"
        ),
    ]
    df = pd.concat(dfs)

    choices = []
    shorts = []
    metrics = df["METRIC"].unique().tolist()
    for metric in metrics:
        metric_choices = interpret_name(metric)

        if metric_choices is not None:
            short = shorten_metric_name(metric)
            shorts.append(short)
            print(short)
            metric_choices["original"] = metric
            metric_choices["short"] = short
            choices.append(metric_choices)

    interpretation = pd.DataFrame(choices)

    print(interpretation)
    counts = []
    for col in interpretation.columns:
        uniques = interpretation[col].unique()
        print("\\item", col, len(uniques), uniques.tolist())
        counts.append(len(uniques))

    print(np.prod(counts))

    # Find duplicates in the "short" column
    duplicates = interpretation[interpretation.duplicated(subset="short", keep=False)]
    duplicates = duplicates.sort_values(by="short")

    # Display the original and short columns for these duplicates
    print(duplicates[["original", "short"]])
    interpretation.to_csv("metric_name_interpretation.csv")
    duplicates.to_csv("metric_name_short_dupes.csv")
