from pose_evaluation.metrics.distance_measure import (
    PowerDistance,
)
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.ham2pose_distances import (
    Ham2PoseMSEDistance,
    Ham2PoseMaskedEuclideanDistance,
    Ham2PoseAPEDistance,
)
from pose_evaluation.metrics.mje_metric import MeanJointErrorMetric
from pose_evaluation.metrics.dynamic_time_warping_metric import DTWMetric
from pose_evaluation.metrics.pose_processors import (
    get_standard_pose_processors,
)

if __name__ == "__main__":

    metrics = []
    MJEMetric = (
        MeanJointErrorMetric()
    )  # automatically sets distance measure, zero-padding.
    metrics.append(MJEMetric)

    Ham2Pose_DTW_MJE_Metric = DTWMetric(
        name="DTW_MJE",
        distance_measure=PowerDistance(2, 0),
        pose_preprocessors=get_standard_pose_processors(zero_pad_shorter=False),
    )

    Ham2Pose_nDTW_MJE_Metric = DTWMetric(
        name="nDTW_MJE",
        distance_measure=Ham2PoseMaskedEuclideanDistance(),
        pose_preprocessors=get_standard_pose_processors(zero_pad_shorter=False),
    )

    metrics.append(Ham2Pose_DTW_MJE_Metric)
    metrics.append(Ham2Pose_nDTW_MJE_Metric)

    # Ham2Pose APE is a PowerDistance. But with a few preprocessors.
    # 1. standard preprocessors
    # 2. then these: basically this is "zero_pad_shorter", and also setting masked values to zero.
    # if len(trajectory1) < len(trajectory2):
    #     diff = len(trajectory2) - len(trajectory1)
    #     trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 3))))
    # elif len(trajectory2) < len(trajectory1):
    #     trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 3))))
    # pose1_mask = np.ma.getmask(trajectory1)
    # pose2_mask = np.ma.getmask(trajectory2)
    # trajectory1[pose1_mask] = 0
    # trajectory1[pose2_mask] = 0
    # trajectory2[pose1_mask] = 0
    # trajectory2[pose2_mask] = 0
    # 3. pointwise aggregate by SUM
    # sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    # 4. trajectorywise aggregate by MEAN
    # np.sqrt(sq_error).mean()
    Ham2PoseAPEMetric = DistanceMetric(
        name="Ham2PoseAPEMetric",
        distance_measure=Ham2PoseAPEDistance(),
        pose_preprocessors=get_standard_pose_processors(
            zero_pad_shorter=True, set_masked_values_to_origin=True
        ),
    )
    metrics.append(Ham2PoseAPEMetric)

    # MSE from Ham2Pose is zero-padding, plus set to origin, and then squared error.
    # if len(trajectory1) < len(trajectory2):
    #     diff = len(trajectory2) - len(trajectory1)
    #     trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 3))))
    # elif len(trajectory2) < len(trajectory1):
    #     trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 3))))
    # pose1_mask = np.ma.getmask(trajectory1)
    # pose2_mask = np.ma.getmask(trajectory2)
    # trajectory1[pose1_mask] = 0
    # trajectory1[pose2_mask] = 0
    # trajectory2[pose1_mask] = 0
    # trajectory2[pose2_mask] = 0
    # sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    # return sq_error.mean()
    Ham2PoseMSEMetric = DistanceMetric(
        name="Ham2PoseMSEMetric",
        distance_measure=Ham2PoseMSEDistance(),
        pose_preprocessors=get_standard_pose_processors(
            zero_pad_shorter=True, set_masked_values_to_origin=True
        ),
    )
    metrics.append(Ham2PoseMSEMetric)

    for metric in metrics:
        print("*" * 30)
        print(f"METRIC: {metric}")
        print(metric.get_signature())
        print(metric.get_signature().format(short=True))
        print()
