from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.distance_metric import DistanceMetric, PowerDistance
from pose_evaluation.metrics.pose_processors import NormalizePosesProcessor, get_standard_pose_processors
from pose_evaluation.metrics.dynamic_time_warping_metric import DTWMetric


def test_pose_metric_signature_has_preprocessor_information():
    metric = PoseMetric("PoseMetric", pose_preprocessors=[NormalizePosesProcessor()])
    
    assert "pose_preprocessers" in metric.get_signature().format()
    assert "pre" in metric.get_signature().format(short=True)

    metric = PoseMetric("PoseMetric")
    assert "pose_preprocessers" not in metric.get_signature().format()
    assert "pre" not in metric.get_signature().format(short=True)



if __name__ == "__main__":

    MeanJointError= DistanceMetric(distance_measure=PowerDistance(2), 
                                   pose_preprocessors=get_standard_pose_processors(),
                                   )

    metrics = [DistanceMetric(), 
               DistanceMetric(distance_measure=PowerDistance(2, 1)),
               DTWMetric(),
               MeanJointError
               ]

    for metric in metrics:
        
        print("*"*10)
        print(metric.get_signature().format())
        print(metric.get_signature().format(short=True))
        # print(metric.distance_measure)

