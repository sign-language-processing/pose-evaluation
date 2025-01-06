from typing import Literal, List

from pose_format import Pose

from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.utils.pose_utils import pose_hide_low_conf, get_preprocessed_pose

class DynamicTimeWarpingMeanJointError(DistanceMetric):
    def __init__(self, kind: Literal["manhattan", "euclidean"] = "euclidean", normalize_missing:bool=True):
        super().__init__(kind)

    def score_all(self, hypotheses:List[Pose], references:List[Pose], progress_bar=True):
        # TODO: 
        return super().score_all(hypotheses, references, progress_bar)
    
    
    def score(self, hypothesis:Pose, reference:Pose):
        # TODO
        return super().score(hypothesis, reference)