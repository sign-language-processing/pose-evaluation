from pathlib import Path
from pose_format import Pose
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.base import BaseMetric
from pose_evaluation.metrics.dtw_metric import (
    DTWAggregatedPowerDistanceMeasure,
    DTWAggregatedScipyDistanceMeasure,
)
from pose_evaluation.metrics.test_distance_metric import get_poses
from pose_evaluation.metrics.pose_processors import (
    ZeroPadShorterPosesProcessor,
    get_standard_pose_processors,
)

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
    USE_REAL_FILES = True

    if USE_REAL_FILES:
        poses = [
            Pose.read(hypothesis_file.read_bytes()),
            Pose.read(reference_file.read_bytes()),
        ]
        # TODO: add PosePreprocessors to PoseDistanceMetrics, with their own signatures
        # poses = zero_pad_shorter_poses(poses)

    else:
        hypothesis, reference = get_poses(2, 2, conf1=1, conf2=1)
        poses = [hypothesis, reference]

    hypotheses = [pose.copy() for pose in poses]
    references = [pose.copy() for pose in poses]

    # Define distance metrics
    mean_l1_metric = DistanceMetric(
        "mean_l1_metric",
        distance_measure=AggregatedPowerDistance(order=1, default_distance=17),
    )
    metrics = [
        BaseMetric("base"),
        DistanceMetric(
            "PowerDistanceMetric", AggregatedPowerDistance(order=2, default_distance=1)
        ),
        DistanceMetric(
            "AnotherPowerDistanceMetric",
            AggregatedPowerDistance("A custom name", order=1, default_distance=10),
        ),
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
            pose_preprocessors=[ZeroPadShorterPosesProcessor()],
        ),
        DistanceMetric(
            "MeanL2Score",
            AggregatedPowerDistance(
                order=2, aggregation_strategy="mean", default_distance=0
            ),
        ),
        PoseMetric(),
        DistanceMetric(
            "DTWPowerDistance",
            DTWAggregatedPowerDistanceMeasure(
                aggregation_strategy="mean", default_distance=0.0, order=2
            ),
            pose_preprocessors=get_standard_pose_processors(
                zero_pad_shorter=False, reduce_holistic_to_face_and_upper_body=True
            ),
        ),
        DistanceMetric(
            "DTWScipyDistance",
            DTWAggregatedScipyDistanceMeasure(
                aggregation_strategy="mean", default_distance=0.0, metric="sqeuclidean"
            ),
            pose_preprocessors=get_standard_pose_processors(
                zero_pad_shorter=False, reduce_holistic_to_face_and_upper_body=True
            ),
        ),
    ]

    # Evaluate each metric on the test poses
    for metric in metrics:
        print("*" * 10)
        print(metric.name)
        print("\nSIGNATURE: ")
        print(metric.get_signature().format())

        print("\nSIGNATURE (short): ")
        print(metric.get_signature().format(short=True))

        try:
            #
            print("\nSCORE ALL with Signature (short):")
            print(
                metric.score_all_with_signature(
                    hypotheses, references, short=True, progress_bar=True
                )
            )

            score = metric.score(poses[0], poses[1])
            print(f"\nSCORE: {score}")

            print("\nSCORE With Signature:")
            print(metric.score_with_signature(poses[0], poses[1]))

            print("\nSCORE with Signature (short):")
            print(metric.score_with_signature(poses[0], poses[1], short=True))

        except NotImplementedError:
            print(f"{metric} score not implemented")
        print("*" * 10)
