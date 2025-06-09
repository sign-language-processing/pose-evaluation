from pathlib import Path

import pandas as pd
from pose_format import Pose
from tqdm import tqdm

from pose_evaluation.evaluation.create_metrics import get_metrics
from pose_evaluation.evaluation.load_splits_and_run_metrics import get_filtered_metrics


def save_results(entries, output_csv_path):
    df = pd.DataFrame.from_dict(entries)
    df.to_csv(output_csv_path, index=False)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    # point this to the folders with 0.pose, 1.pose, etc.
    # Zifan says:
    # for reference, always use the raw pose for full keypoints
    # for sign_mt/sign_mt_v2, same
    # for sockeye, the MT system does not produce the face, so we have <id>.pose (direct system output) and <id>.imputed.pose (the face imputed with an average face). I'd use the imputed one for completeness

    # TODO: to support Embedding Metrics:
    # 0. Embed all the .pose files and save as .npy
    # 1. Instantiate Embedding Metric
    # EmbeddingDistanceMetric(model="asl_citizen_finetune_checkpoint_best")
    # 2. load in embeddings:
    # embedding = np.load(file_path)
    # if embedding.ndim == 2 and embedding.shape[0] == 1:
    #     embedding = embedding[0]  # Reduce shape from (1, 768) to (768,)
    # return embedding
    # 3. Replace the paths below with the .npy files that match the .poses

    sockeye_folder = Path("/opt/home/cleong/data/zifan_signsuisse/system_outputs/signsuisse_test/sockeye/")
    sign_mt_v2_folder = Path("/opt/home/cleong/data_munging/local_data/zifan_signsuisse/signsuisse_test/sign_mt_v2")
    sign_mt_folder = Path(
        "/opt/home/cleong/data_munging/local_data/zifan_signsuisse/system_outputs/signsuisse_test/sign_mt"
    )
    ref_folder = Path("/opt/home/cleong/data/zifan_signsuisse/system_outputs/signsuisse_test/ref/")

    stats_by_metric_csv = Path(
        "/opt/home/cleong/projects/pose-evaluation/stats_by_metric_sorted_by_mean_average_precision by Metric (top 822 of 18558 metrics over 44,254 trials, excl. 'embedding', excl. metrics w less than 5 trials).csv"
    )
    stats_by_metric_df = pd.read_csv(stats_by_metric_csv)

    stats_by_metric_df = stats_by_metric_df.sort_values(by="mean_average_precision", ascending=False, kind="stable")
    print(f"Loaded {len(stats_by_metric_df)} rows")
    print(stats_by_metric_df[["SHORT", "mean_average_precision"]].head())

    item_count = 1000

    # use raw for reference
    ref_paths = [ref_folder / f"{i}.raw.pose" for i in range(item_count)]

    # use raw for sign_mt_v2
    sign_mt_v2_paths = [sign_mt_v2_folder / f"{i}.raw.pose" for i in range(item_count)]

    sign_mt_paths = [sign_mt_folder / f"{i}.raw.pose" for i in range(item_count)]

    # use imputed for sockeye
    sockeye_paths = [sockeye_folder / f"{i}.imputed.pose" for i in range(item_count)]

    system_paths = {
        "sockeye": sockeye_paths,
        "sign_mt_v2": sign_mt_v2_paths,
        "sign_mt": sign_mt_paths,
    }

    system = "pose-eval"
    entries = []
    metrics = get_metrics()

    # # out of 10k+ metric combinations, filter to just the top 10
    # FILTER_METRICS = True
    # if FILTER_METRICS:
    #     metrics = get_filtered_metrics(
    #         metrics,
    #         top10=True,
    #         top10_nohands_nointerp=False,
    #         top10_nohands_nodtw_nointerp=False,
    #         top50_nointerp=False,
    #         return4=False,
    #     )

    metric_names_to_use = set(stats_by_metric_df["METRIC"].head(50).tolist())
    print(metric_names_to_use)
    metrics = [m for m in metrics if m.name in metric_names_to_use]

    print(f"USING THE FOLLOWING METRICS")
    for i, m in enumerate(metrics):
        print(i, m.name)

    # exit()

    # Split metrics into chunks of 10, if you want to do many metrics
    chunk_size = 10

    for batch_id, metric_chunk in enumerate(chunks(metrics, chunk_size)):
        chunk_start = batch_id * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(metrics))

        for system, hyp_paths in system_paths.items():

            output_csv_path = f"/opt/home/cleong/data/zifan_signsuisse/zspeed_metrics/metrics.{system}-{len(metrics)}metrics-metric{chunk_start}-metric{chunk_end}.csv"

            print(f"Running metrics for {system}, {chunk_start} to {chunk_end} and saving to {output_csv_path}")
            entries = []

            for i in tqdm(
                range(item_count), desc=f"Running metrics {chunk_start}-{chunk_end} on pose pairs for system {system}"
            ):
                hyp_file = hyp_paths[i]
                ref_file = ref_paths[i]
                hyp = Pose.read(Path(hyp_file).read_bytes())
                ref = Pose.read(Path(ref_file).read_bytes())

                entry = {
                    "data": "signsuisse_test",
                    "system": system,
                    "example_id": i,
                    # these two can be copied from existing metrics.{system}.csv files
                    # 'source': language_map[language]['source'],
                    # 'target': language_map[language]['target'],
                }

                for metric in tqdm(metric_chunk, disable=len(metric_chunk) < 20, desc="Running Metrics"):
                    try:
                        score = metric.score_with_signature(hyp, ref)
                        entry[f"{metric.name}"] = score.score
                    except KeyError as e:
                        pass
                    except ValueError as e:
                        print(f"Error on {metric.name}: {e}")

                entries.append(entry)

            save_results(entries, output_csv_path)
