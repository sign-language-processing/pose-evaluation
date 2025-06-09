import argparse
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol, load_score_csv


class ScoresIndexDFCol:
    PATHS = "PATHS"
    SUMMARY = "SUMMARY"


def index_scores(
    scores_folder: Path,
    filter_metric: Optional[str] = None,
    filter_gloss_a: Optional[str] = None,
):
    analysis_folder = scores_folder.parent / "score_analysis"
    score_files_index_out = analysis_folder / "score_files_index.json"

    score_files = list(scores_folder.glob("*.csv"))
    score_file_names = list(set([f.name for f in score_files]))
    assert len(score_file_names) == len(score_files)
    print(f"We have {len(score_files)} score files")
    total_count = 0
    index_dict = {
        ScoresIndexDFCol.PATHS: {},
        ScoreDFCol.METRIC: defaultdict(list),
        ScoreDFCol.GLOSS_A: defaultdict(list),
        ScoreDFCol.GLOSS_B: defaultdict(list),
    }
    for i, score_file in enumerate(tqdm(score_files, desc="Reading score files")):
        score_file_name = str(score_file.name)

        scores_csv_df = load_score_csv(csv_file=score_file)
        try:
            scores_csv_df[ScoreDFCol.SIGNATURE] = scores_csv_df[ScoreDFCol.SIGNATURE].apply(
                lambda x: x.split("=")[0].strip() if "=" in x else x.strip()
            )
        except TypeError as e:
            print(score_file)
            raise e

        assert len(scores_csv_df[ScoreDFCol.SIGNATURE].unique()) == 1
        assert len(scores_csv_df[ScoreDFCol.METRIC].unique()) == 1
        assert len(scores_csv_df[ScoreDFCol.GLOSS_A].unique()) == 1
        metric = scores_csv_df[ScoreDFCol.METRIC][0]
        # signature = scores_csv_df[ScoreDFCol.SIGNATURE][0]
        gloss_a = scores_csv_df[ScoreDFCol.GLOSS_A][0]

        if filter_metric and metric != filter_metric:
            continue
        if filter_gloss_a and gloss_a != filter_gloss_a:
            continue

        index_dict[ScoresIndexDFCol.PATHS][score_file_name] = str(score_file)
        index_dict[ScoreDFCol.METRIC][metric].append(score_file_name)
        index_dict[ScoreDFCol.GLOSS_A][gloss_a].append(score_file_name)
        total_count += len(scores_csv_df)

        if i % 100 == 0:
            print(
                f"Read {i} score files so far:"
                f"\n  • {len(index_dict[ScoreDFCol.METRIC])} metrics"
                f"\n  • {len(index_dict[ScoreDFCol.GLOSS_A])} hypothesis glosses"
                f"\n  • {len(index_dict[ScoreDFCol.GLOSS_B])} reference glosses"
                f"\n  • {total_count:,} total distance scores"
            )

        for gloss_b in scores_csv_df[ScoreDFCol.GLOSS_B].unique().tolist():
            index_dict[ScoreDFCol.GLOSS_B][gloss_b].append(score_file_name)
    # exit()
    # print(json.dumps(index_dict, indent=4))
    print("*" * 40)
    print("Done Indexing Score files!")
    print(
        f"Read {len(score_files)} score files total:"
        f"\n  • {len(index_dict[ScoreDFCol.METRIC])} metrics"
        f"\n  • {len(index_dict[ScoreDFCol.GLOSS_A])} hypothesis glosses"
        f"\n  • {len(index_dict[ScoreDFCol.GLOSS_B])} reference glosses"
        f"\n  • {total_count:,} total distance scores"
    )

    index_dict[ScoresIndexDFCol.SUMMARY] = {
        "total_score_files": len(score_files),
        "total_scores": total_count,
        "unique_metrics": len(index_dict[ScoreDFCol.METRIC]),
        "unique_gloss_a": len(index_dict[ScoreDFCol.GLOSS_A]),
        "unique_gloss_b": len(index_dict[ScoreDFCol.GLOSS_B]),
    }
    with score_files_index_out.open("w", encoding="utf-8") as f:
        json.dump(index_dict, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scores_folder",
        type=Path,
        nargs="?",
        default=Path.cwd() / "metric_results_round_2/scores",  # run from the root of the repo by default
        help="Path to folder containing score CSVs",
    )
    parser.add_argument("--metric", type=str, help="Filter by metric name")
    parser.add_argument("--gloss-a", type=str, help="Filter by gloss A")

    args = parser.parse_args()
    scores_folder = args.scores_folder

    index_scores(scores_folder=scores_folder, filter_metric=args.metric, filter_gloss_a=args.gloss_a)
