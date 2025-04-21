import argparse
from pathlib import Path
from collections import defaultdict
import json

import pandas as pd
from tqdm import tqdm

GLOSS_A_PATH = "GLOSS_A_PATH"
GLOSS_B_PATH = "GLOSS_B_PATH"
GLOSS_A = "GLOSS_A"
GLOSS_B = "GLOSS_B"
SCORE = "SCORE"
METRIC = "METRIC"
SIGNATURE = "SIGNATURE"
TIME = "TIME"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scores_folder",
        type=Path,
        nargs="?",
        default=Path.cwd() / "metric_results/scores",  # run from the root of the repo by default
        help="Path to folder containing score CSVs",
    )
    parser.add_argument("--metric", type=str, help="Filter by metric name")
    parser.add_argument("--gloss-a", type=str, help="Filter by gloss A")

    args = parser.parse_args()
    scores_folder = args.scores_folder
    # scores_folder = Path(r"/opt/home/cleong/projects/pose-evaluation/metric_results/scores")
    analysis_folder = scores_folder.parent / "score_analysis"
    score_files_index_out = analysis_folder / "score_files_index.json"

    score_files = list(scores_folder.glob("*.csv"))
    score_file_names = list(set([f.name for f in score_files]))
    assert len(score_file_names) == len(score_files)
    print(f"We have {len(score_files)} score files")
    total_count = 0
    index_dict = {
        "PATHS": {},
        METRIC: defaultdict(list),
        GLOSS_A: defaultdict(list),
        GLOSS_B: defaultdict(list),
    }
    for i, score_file in enumerate(tqdm(score_files, desc="Reading score files")):
        score_file_name = str(score_file.name)

        scores_csv_df = pd.read_csv(
            score_file,
            index_col=0,
            dtype={
                "METRIC": str,
                "GLOSS_A": str,
                "GLOSS_B": str,
                "SIGNATURE": str,
                "GLOSS_A_PATH": str,
                "GLOSS_B_PATH": str,
            },
            float_precision="high",
        )
        scores_csv_df["SIGNATURE"] = scores_csv_df["SIGNATURE"].apply(
            lambda x: x.split("=")[0].strip() if "=" in x else x.strip()
        )

        assert len(scores_csv_df[SIGNATURE].unique()) == 1
        assert len(scores_csv_df[METRIC].unique()) == 1
        assert len(scores_csv_df[GLOSS_A].unique()) == 1
        metric = scores_csv_df[METRIC][0]
        signature = scores_csv_df[SIGNATURE][0]
        gloss_a = scores_csv_df[GLOSS_A][0]

        if args.metric and metric != args.metric:
            continue
        if args.gloss_a and gloss_a != args.gloss_a:
            continue

        index_dict["PATHS"][score_file_name] = str(score_file)
        index_dict[METRIC][metric].append(score_file_name)
        index_dict[GLOSS_A][gloss_a].append(score_file_name)
        total_count += len(scores_csv_df)

        if i % 100 == 0:
            print(
                f"Read {i} score files so far:"
                f"\n  • {len(index_dict[METRIC])} metrics"
                f"\n  • {len(index_dict[GLOSS_A])} hypothesis glosses"
                f"\n  • {len(index_dict[GLOSS_B])} reference glosses"
                f"\n  • {total_count:,} total distance scores"
            )

        for gloss_b in scores_csv_df[GLOSS_B].unique().tolist():
            index_dict[GLOSS_B][gloss_b].append(score_file_name)
    # exit()
    # print(json.dumps(index_dict, indent=4))
    print("*" * 40)
    print("Done Indexing Score files!")
    print(
        f"Read {len(score_files)} score files total:"
        f"\n  • {len(index_dict[METRIC])} metrics"
        f"\n  • {len(index_dict[GLOSS_A])} hypothesis glosses"
        f"\n  • {len(index_dict[GLOSS_B])} reference glosses"
        f"\n  • {total_count:,} total distance scores"
    )

    index_dict["SUMMARY"] = {
        "total_score_files": len(score_files),
        "total_scores": total_count,
        "unique_metrics": len(index_dict[METRIC]),
        "unique_gloss_a": len(index_dict[GLOSS_A]),
        "unique_gloss_b": len(index_dict[GLOSS_B]),
    }
    with score_files_index_out.open("w", encoding="utf-8") as f:
        json.dump(index_dict, f, indent=2, sort_keys=True)
