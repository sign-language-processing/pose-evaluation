from pathlib import Path

from pose_format import Pose

from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.base import BaseMetric
from pose_evaluation.metrics.test_distance_metric import get_poses
# from pose_evaluation.metrics.pose_processors import get_standard_pose_processors
# from pose_evaluation.metrics.dynamic_time_warping_metric import DTWMetric

if __name__ == "__main__":

    reference_file = Path(r"pose_evaluation\utils\test\test_data\colin-1-HOUSE.pose")
    hypothesis_file = Path(r"pose_evaluation\utils\test\test_data\colin-2-HOUSE.pose")

    # poses = [Pose.read(hypothesis_file.read_bytes()),Pose.read(reference_file.read_bytes())] # not the same shape!
    # hypothesis, reference = get_poses(2, 2, conf1=0, conf2=0)
    hypothesis, reference = get_poses(2, 2, conf1=1, conf2=1)
    poses = [hypothesis, reference]
    # signature = DistanceMetric().get_signature()
    # print(signature)

    # MeanJointError = DistanceMetric(
    #     distance_measure=PowerDistance(2),
    #     pose_preprocessors=get_standard_pose_processors(),
    # )
    distance_measure = AggregatedPowerDistance(1, 17)
    mean_l1_metric = DistanceMetric("mean_l1_metric", distance_measure=distance_measure)

    metrics = [
        BaseMetric("base"),
        DistanceMetric("PowerDistanceMetric", AggregatedPowerDistance(2, 1)),
        DistanceMetric("AnotherPowerDistanceMetric", distance_measure=AggregatedPowerDistance(1, 10)),
        mean_l1_metric,
        DistanceMetric("max_l1_metric", distance_measure=AggregatedPowerDistance(order=1, aggregation_strategy="max", default_distance=0))
        # DTWMetric(),
        # MeanJointError,
    ]

    for m in metrics:

        print("*" * 10)
        print(m.get_signature().format())
        print(m.get_signature().format(short=True))
        
        try:
            score = m.score(poses[0], poses[1])
            print(f"SCORE: {score}")
            print(f"SCORE With Signature: ")
            score = m.score_with_signature(poses[0], poses[1])
            print(f"{score}")
            print(repr(score))
            print(f"{type(score)}")
            
            # still behaves like a float
            doubled = score*2
            print(f"score * 2 = {doubled}")
            print(type(doubled))

        except NotImplementedError:
            print(f"{m} score not implemented")
        print("*" * 10)


    # hypothesis, reference = get_poses(2, 2, conf1=0, conf2=0)
    # print(reference.body.data.mask)

    

    # score = mean_l1_metric.score()

    # print(f"SCORE: {mean_l1_metric.score(poses[0], poses[1])}")

    # print(distance_measure._calculate_distances(hyp_data=hypothesis.body.data, ref_data=reference.body.data))
