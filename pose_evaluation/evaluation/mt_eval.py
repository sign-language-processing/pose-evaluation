from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pose_format import Pose

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
    hyp_folder = Path("/opt/home/cleong/data/zifan_signsuisse/system_outputs/signsuisse_test/sockeye/")
    ref_folder = Path("/opt/home/cleong/data/zifan_signsuisse/system_outputs/signsuisse_test/ref/")

    system = "pose-eval"
    entries = []
    metrics = get_metrics()

    # out of 10k+ metric combinations, filter to just the top 10
    FILTER_METRICS = true
    if FILTER_METRICS:
        metrics = get_filtered_metrics(
            metrics,
            top10=True,
            top10_nohands_nointerp=False,
            top10_nohands_nodtw_nointerp=False,
            top50_nointerp=False,
            return4=False,
        )

    # Split metrics into chunks of 10, if you want to do many metrics
    chunk_size = 10

    for batch_id, metric_chunk in enumerate(chunks(metrics, chunk_size)):
        chunk_start = batch_id * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(metrics))
        output_csv_path = f"/opt/home/cleong/data/zifan_signsuisse/metrics/metrics.{system}-{len(metrics)}metrics-metric{chunk_start}-metric{chunk_end}.csv"

        print(f"Running metrics {chunk_start} to {chunk_end} and saving to {output_csv_path}")
        entries = []

        for i in tqdm(range(1000), desc=f"Running metrics {chunk_start}-{chunk_end} on pose pairs"):
            hyp_file = hyp_folder / f"{i}.pose"
            ref_file = ref_folder / f"{i}.pose"
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
