from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.pose_processors import NormalizePosesProcessor
from pose_evaluation.metrics.aggregate_distances_strategy import DistancesAggregator


def test_pose_metric_signature_has_preprocessor_information():
    metric = PoseMetric("PoseMetric", pose_preprocessors=[NormalizePosesProcessor()])
    
    assert "pose_preprocessers" in metric.get_signature().format()
    assert "pre" in metric.get_signature().format(short=True)

    metric = PoseMetric("PoseMetric")
    assert "pose_preprocessers" not in metric.get_signature().format()
    assert "pre" not in metric.get_signature().format(short=True)



if __name__ == "__main__":

    metrics = [DistanceMetric(), DistanceMetric(distances_aggregator=DistancesAggregator("mean")), DistanceMetric(alignment_strategy="zero_pad_shorter", distances_aggregator=DistancesAggregator("max"))]

    for metric in metrics:
        
        print("*"*10)
        print(metric.get_signature().format())
        print(metric.get_signature().format(short=True))

