import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol

if __name__ == "__main__":
    scores_folder = Path.cwd() / "metric_results_round_2/scores"
    analysis_folder = scores_folder.parent / "score_analysis"
    score_files_index_path = analysis_folder / "score_files_index.json"

    with score_files_index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
        for key, value in index.items():
            print(f"{key} has {len(value)} values")
            if key == ScoreDFCol.GLOSS_A:
                glosses = [str(g) for g in value.keys()]
                print("glosses:", glosses)
                for gloss in glosses:
                    print(f"* {gloss}: {len(index[ScoreDFCol.GLOSS_A][gloss])} csvs with results")
            if key == "SUMMARY":
                print(value)

            if key == ScoreDFCol.METRIC:
                for metric_name, csv_list in value.items():
                    print(f"Metric {metric_name} has {len(csv_list)} files")
