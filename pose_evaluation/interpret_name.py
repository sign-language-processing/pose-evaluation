import pandas as pd
import numpy as np

def interpret_name(metric_name:str):

    choices = {}

    if "dtaiDTWAggregatedDistanceMetricFast" in metric_name:
        choices["measure"] = "dtaiDTWAggregatedDistanceMetricFast"
    elif "AggregatedPowerDistanceMetric" in metric_name:
        choices["measure"] = "AggregatedPowerDistanceMetric"
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
            choices["fillmasked"]= masked_fill

    for interp in ["nointerp", "interp15", "interp120"]:
        if f"_{interp}_" in metric_name:
            choices["interp"] = interp

    if "_normalized_" in metric_name or "_normalizedbyshoulders_" in metric_name:
        choices["normalize"] = True
    elif "_unnormalized_" in metric_name:
        choices["normalize"] = False
            

    for keypoints in ["removelegsandworld","reduceholistic","hands","youtubeaslkeypoints",]:
        if f"_{keypoints}_" in metric_name:
            choices["keypoints"] = keypoints
    
    return choices
        
    

if __name__ == "__main__":
    # df = pd.read_csv("/opt/home/cleong/projects/pose-evaluation/metric_results/4_22_2025_csvcount_17187_score_analysis_with_updated_MAP/stats_by_metric.csv")
    df = pd.read_csv("/opt/home/cleong/projects/pose-evaluation/metric_results_round_2/4_23_2025_score_analysis_3300_trials/stats_by_metric.csv")

    choices = []
    for metric in df["METRIC"].tolist():
        metric_choices = interpret_name(metric)
        if metric_choices is not None:
            choices.append(metric_choices)

    interpretation = pd.DataFrame(choices)
    
    print(interpretation)
    counts = []
    for col in interpretation.columns:
        uniques = interpretation[col].unique()
        print("\\item", col, len(uniques), uniques.tolist())
        counts.append(len(uniques))

    print(np.prod(counts))