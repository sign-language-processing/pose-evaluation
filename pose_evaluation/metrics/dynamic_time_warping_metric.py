from typing import Literal, List, TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from pose_format import Pose

from pose_evaluation.metrics.distance_measure import DistanceMeasure, PowerDistance
from pose_evaluation.metrics.base_pose_metric import PoseMetric, PoseMetricSignature
from pose_evaluation.metrics.distance_metric import DistanceMetric, ValidPointDistanceKinds
from pose_evaluation.metrics.pose_processors import PoseProcessor
from pose_evaluation.metrics.pose_processors import get_standard_pose_processors



class DTWSignature(PoseMetricSignature):
    def __init__(self, args: dict):
        super().__init__(args)

        self._abbreviated.update({
           "radius":"rad",
            "distance_measure": "dist",
            "trajectory":"traj",
            }
        )

        self.signature_info.update(
            {
               "radius": args.get("radius", None),
                "distance_measure": args.get("distance_measure", None),
                "trajectory": args.get("trajectory", None)
            }
        )

class DTWMetric(PoseMetric):
  _SIGNATURE_TYPE = DTWSignature
  def __init__(self, 
               name: str = "DTWMetric", 
               radius:int = 1,
               distance_measure: Optional[DistanceMeasure] = None,
               trajectory:Literal['keypoints', "frames"] = 'keypoints',
               higher_is_better: bool = False, 
               pose_preprocessors: None | List[PoseProcessor] = None):
     super().__init__(name, higher_is_better, pose_preprocessors)

     self.radius = radius

     if distance_measure is None:
        self.distance_measure = PowerDistance()
     else:
        self.distance_measure = distance_measure
     self.trajectory = trajectory

  def score(self, hypothesis: Pose, reference: Pose):
    hypothesis, reference = self.process_poses([hypothesis, reference])
    if self.trajectory == "keypoints":
        keypoint_trajectory_distances = []
        tensor1 = hypothesis.body.points_perspective()
        tensor2 = reference.body.points_perspective()
        for keypoint_trajectory1, keypoint_trajectory2 in zip(tensor1, tensor2): 
            keypoint_trajectory_distances.append(fastdtw(keypoint_trajectory1, keypoint_trajectory2, radius=self.radius, dist=self.distance_measure))
            return float(np.mean(keypoint_trajectory_distances))
    

    tensor1 = hypothesis.body.data
    tensor2 = reference.body.data
    return fastdtw(tensor1, tensor2, radius=self.radius, dist=self.distance_measure)

    
    