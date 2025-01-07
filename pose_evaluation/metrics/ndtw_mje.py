from typing import Literal, List

from pose_format import Pose

from pose_evaluation.metrics.distance_metric import DistanceMetric, ValidDistanceKinds
from pose_evaluation.utils.pose_utils import pose_hide_low_conf, preprocess_pose

class DynamicTimeWarpingMeanJointError(DistanceMetric):
    def __init__(self, kind: ValidDistanceKinds = "euclidean", 
                 normalize_poses:bool=True,
                 reduce_poses:bool=False,
                 remove_legs:bool=True,
                 remove_world_landmarks:bool=False,
                 conf_threshold_to_drop_points:None|int=None,
                 ):
        super().__init__(kind)

        self.normalize_poses = normalize_poses
        self.reduce_reference_poses = reduce_poses
        self.remove_legs = remove_legs
        self.remove_world_landmarks = remove_world_landmarks
        self.conf_threshold_to_drop_points = conf_threshold_to_drop_points

    def score_all(self, hypotheses:List[Pose], references:List[Pose], progress_bar=True):
        # TODO: 
        return super().score_all(hypotheses, references, progress_bar)
    
    
    def score(self, hypothesis:Pose, reference:Pose):
        # TODO
        return super().score(hypothesis, reference)