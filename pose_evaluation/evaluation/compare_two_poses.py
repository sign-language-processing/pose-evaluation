from typing import List
import typer
import pandas as pd
from pose_format import Pose
from pathlib import Path
from pose_evaluation.evaluation.create_metrics import get_metrics
from pose_evaluation.evaluation.load_splits_and_run_metrics import get_filtered_metrics

app = typer.Typer()


def extract_key(path_str):
    path_str = path_str.split("\\")[-1]
    return Path(Path(path_str).name.replace(".pose.zst", "")).stem


def update_csv_paths(paths_csv):
    dataset_df_csv = Path("/opt/home/cleong/projects/pose-evaluation/dataset_dfs/asl-citizen.csv")

    paths_df = pd.read_csv(paths_csv)
    paths_df["Gloss A Key"] = paths_df["Gloss A Path"].apply(extract_key)
    paths_df["Gloss B Key"] = paths_df["Gloss B Path"].apply(extract_key)

    print(paths_df)  # Gloss A Path,	Gloss B Path are in this
    print(paths_df["Gloss B Key"])  # Gloss A Path,	Gloss B Path are in this

    dataset_df = pd.read_csv(dataset_df_csv)  # POSE_FILE_PATH is in this
    dataset_df["Key"] = dataset_df["POSE_FILE_PATH"].apply(lambda p: Path(p).stem.replace(".pose", ""))
    print(dataset_df)

    # Create lookup dict from key to full POSE_FILE_PATH
    pose_path_lookup = dict(zip(dataset_df["Key"], dataset_df["POSE_FILE_PATH"]))
    paths_df["Gloss A Path"] = paths_df["Gloss A Key"].map(pose_path_lookup)
    paths_df["Gloss B Path"] = paths_df["Gloss B Key"].map(pose_path_lookup)

    # Drop helper columns if desired
    paths_df.drop(columns=["Gloss A Key", "Gloss B Key"], inplace=True)
    print(paths_df.head())

    output_dir = Path("/opt/home/cleong/projects/pose-evaluation/debug_zspeed/prelim_results_updated_paths")
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    output_path = output_dir / paths_csv.name  # Preserve original filename
    paths_df.to_csv(output_path, index=False)

    print(f"Saved updated CSV to: {output_path}")


@app.command()
def compare(
    query_pose: Path = typer.Argument(..., exists=True, help="Path to the query .pose file"),
    ref_pose: Path = typer.Argument(..., exists=True, help="Path to the reference .pose file"),
    specific_metrics: List[str] = typer.Option(
        None, help="If specified, will add these metrics to the list of filtered metrics"
    ),
    specific_metrics_csv: Path = typer.Option(
        None, help="If specified, will read the metrics from this CSV and add those"
    ),
    include_keywords: List[str] = typer.Option(
        None, help="Will filter metrics to only those that include any of these"
    ),
    exclude_keywords: List[str] = typer.Option(
        None, help="Will filter metrics to only those that include none of these"
    ),
):

    # paths_csvs = Path("/opt/home/cleong/projects/pose-evaluation/debug_zspeed/zspeed_results_from_preliminary/").glob(
    #     "*.csv"
    # )
    # for csv in paths_csvs:
    #     update_csv_paths(csv)

    # old_results_df = pd.read_csv(
    #     "/opt/home/cleong/projects/pose-evaluation/debug_zspeed/zspeed_results_from_preliminary/ALASKA_FACE_n-dtai-DTW-dtw_mje_dtai_fast_hands_z_offsets_score_results.csv"
    # )

    # for row in old_results_df.iterrows():
    #     print(row[["Gloss A Path", "Gloss B Path", "score"]])

    # typer.echo(f"Comparing poses:")
    # typer.echo(f"  Query: {query_pose}")
    # typer.echo(f"  Reference: {ref_pose}")

    metrics = get_metrics(include_masked=True)
    metrics = get_filtered_metrics(
        metrics,
        top10=False,
        top10_nohands_nointerp=False,
        top10_nohands_nodtw_nointerp=False,
        top50_nointerp=False,
        return4=False,
        specific_metrics=specific_metrics,
        csv_path=specific_metrics_csv,
        include_keywords=include_keywords,
        exclude_keywords=exclude_keywords,
    )
    for i, metric in enumerate(metrics):
        print(f"\n  Metric #{i}: {metric.name}")
        print(f"  Metric Signature: {metric}")
        hyp, ref = Pose.read(Path(query_pose).read_bytes()), Pose.read(Path(ref_pose).read_bytes())
        score = metric.score_with_signature(hyp, ref)
        typer.echo(f"  Score: {score.score}")


if __name__ == "__main__":
    app()

# Preliminary PhD Proposal results had this:
# score	Gloss A	Gloss B	Gloss A Path	Gloss B Path
# 1.0841141015	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\5892937353167862-FACE.pose.zst
# 1.5390595976	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\5976044526254161-FACE.pose.zst
# 3.4919609517	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\06117592760150714-FACE.pose.zst
# 3.289287599	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\9714074937680455-FACE.pose.zst
# 4.2774428154	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\5062351660044471-FACE.pose.zst
# 5.139735343	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\9692568743211929-FACE.pose.zst
# 8.1432730096	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\889629849987267-FACE.pose.zst
# 8.5872208648	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\03504395991070819-FACE.pose.zst
# 9.0912335072	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\45357856584990053-FACE.pose.zst
# 13.4655002536	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\8640277473918241-FACE.pose.zst
# 10.1791526978	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\39480858175448996-FACE.pose.zst
# 13.3209474395	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\8823645538409584-FACE.pose.zst
# 15.3920693698	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\9246436366130377-FACE.pose.zst
# 15.6334309204	ALASKA	FACE	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst	C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\7532215729692444-FACE.pose.zst


# both of these give Score: 1.084113992805835
# conda activate /opt/home/cleong/envs/pose_eval_src && python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/compare_two_poses.py /opt/home/cleong/data/ASL_Citizen/poses/pose/19151595519059494-ALASKA.pose /opt/home/cleong/data/ASL_Citizen/poses/pose/5892937353167862-FACE.pose --specific-metrics "untrimmed_zspeed1.0_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast"
# python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/compare_two_poses.py /opt/home/cleong/data/ASL_Citizen/poses/pose/19151595519059494-ALASKA.pose /opt/home/cleong/data/ASL_Citizen/poses/pose/5892937353167862-FACE.pose --specific-metrics "untrimmed_zspeed1.0_normalizedbyshoulders_hands_defaultdist0.0_nointerp_dtw_maskinvalidvals_dtaiDTWAggregatedDistanceMetricFast"


# /opt/home/cleong/projects/pose-evaluation/debug_zspeed/Preliminary Results on ASL Known Lookalikes/scores/ALASKA_FACE_dtw_mje_dtai_fast_hands_xy0_score_results.csv
