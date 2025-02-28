from pathlib import Path
from pose_format import Pose
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.base import BaseMetric
from pose_evaluation.metrics.test_distance_metric import get_poses
from pose_evaluation.utils.pose_utils import zero_pad_shorter_poses

if __name__ == "__main__":
    # Define file paths for test pose data
    reference_file = (
        Path("pose_evaluation") / "utils" / "test" / "test_data" / "colin-1-HOUSE.pose"
    )
    hypothesis_file = (
        Path("pose_evaluation") / "utils" / "test" / "test_data" / "colin-2-HOUSE.pose"
    )

    # Choose whether to load real files or generate test poses
    # They have different lengths, and so some metrics will crash!
    # Change to False to generate fake poses with known distances, e.g. all 0 and all 1
    use_real_files = True

    if use_real_files:
        poses = [
            Pose.read(hypothesis_file.read_bytes()),
            Pose.read(reference_file.read_bytes()),
        ]
        # TODO: add PosePreprocessors to PoseDistanceMetrics, with their own signatures
        poses = zero_pad_shorter_poses(poses)

    else:
        hypothesis, reference = get_poses(2, 2, conf1=1, conf2=1)
        poses = [hypothesis, reference]

    # Define distance metrics
    mean_l1_metric = DistanceMetric(
        "mean_l1_metric", distance_measure=AggregatedPowerDistance(1, 17)
    )
    metrics = [
        BaseMetric("base"),
        DistanceMetric("PowerDistanceMetric", AggregatedPowerDistance(2, 1)),
        DistanceMetric("AnotherPowerDistanceMetric", AggregatedPowerDistance(1, 10)),
        mean_l1_metric,
        DistanceMetric(
            "max_l1_metric",
            AggregatedPowerDistance(
                order=1, aggregation_strategy="max", default_distance=0
            ),
        ),
        DistanceMetric(
            "MeanL2Score",
            AggregatedPowerDistance(
                order=2, aggregation_strategy="mean", default_distance=0
            ),
        ),
    ]

    # Evaluate each metric on the test poses
    for metric in metrics:
        print("*" * 10)
        print(metric.get_signature().format())
        print(metric.get_signature().format(short=True))

        try:
            score = metric.score(poses[0], poses[1])
            print(f"SCORE: {score}")
            print("SCORE With Signature:")
            score_with_sig = metric.score_with_signature(poses[0], poses[1])
            print(score_with_sig)
            print(repr(score_with_sig))
            print(f"{type(score_with_sig)}")

            # Verify that score behaves like a float
            doubled = score_with_sig * 2
            print(f"score * 2 = {doubled}")
            print(type(doubled))

        except NotImplementedError:
            print(f"{metric} score not implemented")
        print("*" * 10)
