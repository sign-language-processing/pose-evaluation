from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pose_format import Pose

from pose_evaluation.evaluation.create_metrics import get_metrics
from pose_evaluation.evaluation.load_splits_and_run_metrics import get_filtered_metrics


def save_results(entries, output_csv_path):
    df = pd.DataFrame.from_dict(entries)
    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    hyp_folder = Path("/opt/home/cleong/data/zifan_signsuisse/system_outputs/signsuisse_test/sockeye/")
    ref_folder = Path("/opt/home/cleong/data/zifan_signsuisse/system_outputs/signsuisse_test/ref/")

    system = "pose-eval"
    output_csv_path = f"/opt/home/cleong/data/zifan_signsuisse/metrics/metrics.{system}.csv"
    entries = []
    metrics = get_metrics()
    # metrics = get_filtered_metrics(
    #     metrics,
    #     # top10=True,
    #     top10_nohands_nointerp=False,
    #     top10_nohands_nodtw_nointerp=False,
    #     top50_nointerp=False,
    #     return4=True,
    # )

    for i in tqdm(range(1000), desc=f"Running {len(metrics)} metrics on pose pairs"):
        hyp_file = hyp_folder / f"{i}.pose"
        ref_file = ref_folder / f"{i}.pose"
        hyp = Pose.read(Path(hyp_file).read_bytes())
        ref = Pose.read(Path(ref_file).read_bytes())
        # https://github.com/J22Melody/iict-eval-private/blob/text2pose/metrics/metrics.py#L90C17-L96C18
        entry = {
            "data": "signsuisse_test",
            "system": system,
            "example_id": i,
        }

        for metric in tqdm(metrics, disable=len(metrics) < 20, desc="Running Metrics"):
            # print(metric.name)
            try:
                score = metric.score_with_signature(hyp, ref)
                entry[f"{metric.name}"] = score.score
            except KeyError as e:
                # print(f"Had an error on {metric.name}: {e}")
                pass
            except ValueError as e:
                print(f"Had an error on {metric.name}: {e}")

        entries.append(entry)

        if i % 10 == 0:
            save_results(entries, output_csv_path)
    save_results(entries, output_csv_path)
