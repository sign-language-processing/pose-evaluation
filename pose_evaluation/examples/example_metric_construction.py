from pathlib import Path

from pose_format import Pose

from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import (
    DTWAggregatedPowerDistanceMeasure,
    DTWAggregatedScipyDistanceMeasure,
    DTWDTAIImplementationDistanceMeasure,
)
from pose_evaluation.metrics.embedding_distance_metric import EmbeddingDistanceMetric
from pose_evaluation.metrics.ham2pose import Ham2Pose_nMSE
from pose_evaluation.metrics.pose_processors import (
    HideLegsPosesProcessor,
    NormalizePosesProcessor,
    ReduceHolisticPoseProcessor,
    ZeroPadShorterPosesProcessor,
    get_standard_pose_processors,
)
from pose_evaluation.metrics.test_distance_metric import get_poses

if __name__ == "__main__":
    # Define file paths for test pose data
    # /opt/home/cleong/projects/pose-evaluation/pose_evaluation/utils/test/test_data/mediapipe/standard_landmarks/colin-1-HOUSE.pose
    test_data_path = (
        Path("pose_evaluation").resolve() / "utils" / "test" / "test_data" / "mediapipe" / "standard_landmarks"
    )
    reference_file = test_data_path / "colin-1-HOUSE.pose"
    hypothesis_file = test_data_path / "colin-2-HOUSE.pose"

    # Choose whether to load real files or generate test poses
    # They have different lengths, and so some metrics will crash!
    # Metrics with ZeroPadShorterPosesProcessor, DTWMetrics are fine.
    # Change to False to generate fake poses with known distances, e.g. all 0 and all 1\
    USE_REAL_FILES = True

    if USE_REAL_FILES:
        poses = [
            Pose.read(hypothesis_file.read_bytes()),
            Pose.read(reference_file.read_bytes()),
        ]

    else:
        hypothesis, reference = get_poses(2, 2, conf1=1, conf2=1)
        poses = [hypothesis, reference]

    hypotheses = [pose.copy() for pose in poses]
    references = [pose.copy() for pose in poses]

    #############################
    # Abstract classes:

    # BaseMetric does not actually have score() function
    # base_metric = BaseMetric("base")

    # PoseMetric calls preprocessors before scoring,
    # It is also an abstract class
    # PoseMetric("pose base"),

    # Segments first, also abstract.
    # SegmentedPoseMetric("SegmentedMetric")

    # Define distance metrics
    metrics = [
        # a DistanceMetric uses a DistanceMeasure to calculate distances between two Poses
        # This one is effectively (normalized) Average Position Error (APE)
        # as it by default will run zero-padding of the shorter pose, and normalization,
        # and AggregatedPowerDistance does mean absolute (euclidean) distances by default.
        DistanceMetric(
            "NormalizedAveragePositionError",
            AggregatedPowerDistance(),  #
        ),
        # Customizing Distances
        # Distance Measures have signatures as well.
        # You can set options on the DistanceMeasure and they will be reflected in the signature.
        # This one would be distance_measure:{power_distance|pow:1.0|dflt:1.0|agg:max}
        DistanceMetric(
            "MaxL1DistanceMetric",
            AggregatedPowerDistance(order=1, default_distance=1, aggregation_strategy="max"),  #
        ),
        # Customizing Preprocessing
        # A DistanceMetric is a PoseMetric, and so it will call PosePreprocessors before scoring
        # get_standard_pose_processors gives you some default options,
        # for example you could decide not to remove the legs
        DistanceMetric(
            "CustomizedPosePreprocessorsWithLegsMetric",
            distance_measure=AggregatedPowerDistance("A custom name", order=1, default_distance=10),
            pose_preprocessors=get_standard_pose_processors(
                remove_legs=False,  # If you want the legs
            ),
        ),
        # Recreating Existing Metrics: Average Position Error/ Mean Joint Error
        # As defined in Ham2Pose,
        # APE is "the average L2 distance between the predicted and the GT pose keypoints
        # across all frames and data samples. Since it compares absolute positions,
        # it is sensitive to different body shapes and slight changes
        # in timing or position of the performed movement"
        # So we:
        # - Select AggregatedPowerDistance measure
        # - set the order to 2 (Euclidean distance)
        # - set the aggregation strategy to mean
        # - recreate the set of preprocessors from https://github.com/rotem-shalev/Ham2Pose/blob/main/metrics.py#L32-L62
        # (adapting to MediaPipe Holistic keypoints format instead of OpenPose)
        DistanceMetric(
            "AveragePositionError",
            AggregatedPowerDistance(order=2, aggregation_strategy="mean", default_distance=0),
            pose_preprocessors=[
                NormalizePosesProcessor(),
                HideLegsPosesProcessor(),
                ZeroPadShorterPosesProcessor(),
                ReduceHolisticPoseProcessor(),
            ],
        ),
        # Recreating Dynamic Time Warping - Mean Joint Error
        # As before, only now we use the Dynamic Time Warping version!
        # DistanceMetric(
        #     "DTWPowerDistance",
        #     DTWAggregatedPowerDistanceMeasure(aggregation_strategy="mean", default_distance=0.0, order=2),
        #     pose_preprocessors=get_standard_pose_processors(
        #         zero_pad_shorter=False, reduce_holistic_to_face_and_upper_body=True
        #     ),
        # ),
        # We can also implement a version that uses scipy distances "cdist"
        # This lets us experiment with e.g. jaccard
        # Options are listed at the documentation for scipy:
        # https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.spatial.distance.cdist.html
        # DistanceMetric(
        #     "DTWScipyDistance",
        #     DTWAggregatedScipyDistanceMeasure(aggregation_strategy="mean", default_distance=0.0, metric="jaccard"),
        #     pose_preprocessors=get_standard_pose_processors(
        #         zero_pad_shorter=False, reduce_holistic_to_face_and_upper_body=True
        #     ),
        # ),
        DistanceMetric(
            "n-dtai-DTW-MJE (fast)",
            distance_measure=DTWDTAIImplementationDistanceMeasure(use_fast=True),
            pose_preprocessors=get_standard_pose_processors(
                reduce_holistic_to_face_and_upper_body=True, zero_pad_shorter=False
            ),
        ),
    ]

    # metrics = [EmbeddingDistanceMetric(model="ModelName")]

    # Ham2Pose_nMSE = DistanceMetric(
    #     "nMSE",
    #     distance_measure=
    #     pose_processors = [RemoveWorldLandmarksProcessor(), ReduceHolisticProcessor(), NormalizePosesProcessor()]
    # )

    # Evaluate each metric on the test poses
    for metric in metrics:
        print("*" * 10)
        print(metric.name)

        print("\nMETRIC __str__: ")
        print(str(metric))

        print("\nMETRIC to repr: ")
        print(repr(metric))

        print("\nSIGNATURE: ")
        print(metric.get_signature().format())

        print("\nSIGNATURE (short): ")
        print(metric.get_signature().format(short=True))

        try:
            if isinstance(metric, EmbeddingDistanceMetric):
                print(
                    "Sorry, this is an embedding metric, it can't handle poses! You need to load the .npy files to arrays of shape (768,) first!"
                )
                continue
            #
            print("\nSCORE ALL with Signature (short):")
            print(metric.score_all_with_signature(hypotheses, references, short=True, progress_bar=True))

            score = metric.score(poses[0], poses[1])
            print(f"\nSCORE: {score}")

            print("\nSCORE With Signature:")
            print(metric.score_with_signature(poses[0], poses[1]))

            print("\nSCORE with Signature (short):")
            print(metric.score_with_signature(poses[0], poses[1], short=True))

        except NotImplementedError:
            print(f"{metric} score not implemented")
        print("*" * 10)
