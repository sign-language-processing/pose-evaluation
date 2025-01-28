from typing import Literal, Iterable, Tuple, List, cast, get_args, Callable, Union, TYPE_CHECKING, Optional
import numpy as np
from numpy import ma
import scipy.spatial.distance as scipy_distances
from fastdtw import fastdtw
if TYPE_CHECKING:
    from scipy.spatial.distance import _MetricKind
from pose_format import Pose, PoseBody
from pose_evaluation.utils.pose_utils import zero_pad_shorter_poses
from pose_evaluation.metrics.base import SignatureMixin
from pose_evaluation.metrics.base_pose_metric import PoseMetric, PoseMetricSignature
from pose_evaluation.metrics.distance_measure import DistanceMeasure, PowerDistance
from pose_evaluation.metrics.pose_processors import PoseProcessor, ZeroPadShorterPosesProcessor, SetMaskedValuesToOriginPositionProcessor

BuildTrajectoryStrategy = Literal['keypoint', 'frame']
TrajectoryAlignmentStrategy = Literal["zero_pad_shorter", "truncate_longer", "by_reference"]
MaskedKeypointPositionStrategy = Literal["skip_masked", "return_zero", "masked_to_origin", "ref_return_zero_hyp_to_origin", "undefined"]

KeypointPositionType = Union[Tuple[float, float, float], Tuple[float, float]] # XYZ or XY
ValidPointDistanceKinds = Literal["euclidean", "manhattan"] 




class DistanceMetricSignature(PoseMetricSignature):
    def __init__(self, args: dict):
        super().__init__(args)
        self.update_signature_and_abbr("distance_measure", "dist", args)
        self.update_signature_and_abbr("trajectory", "trj", args)


class DistanceMetric(PoseMetric):
    """Metrics that compute some sort of distance"""
    _SIGNATURE_TYPE = DistanceMetricSignature

    def __init__(self, 
                 name = "DistanceMetric",
                 distance_measure: Optional[DistanceMeasure] = None,
                 pose_preprocessors: None | List[PoseProcessor] = None, 
                 trajectory:BuildTrajectoryStrategy="keypoint",
                 ):
        super().__init__(name=name, higher_is_better=False, pose_preprocessors=pose_preprocessors)

        if distance_measure is None:
            self.distance_measure = PowerDistance()

        else:
            self.distance_measure = distance_measure
        

        self.trajectory = trajectory
        
    

    def score(self, hypothesis: Pose, reference: Pose) -> float:
        hypothesis, reference = self.process_poses([hypothesis, reference])
        if self.trajectory == "keypoint":
            return self.distance_measure(hypothesis.body.points_perspective(), reference.body.points_perspective())
        
        return self.distance_measure(hypothesis.body.data, reference.body.data)

    # def coord_pair_distance_function(self, hyp_coordinate:KeypointPositionType, ref_coordinate:KeypointPositionType) -> float:
    #     # if self
    #     raise NotImplementedError
    #     if self.__point_pair_metric_function is None:
    #         try: 
    #             metric_kind = self.distance_calculation_kind
    #             if TYPE_CHECKING:
    #                  metric_kind=cast(_MetricKind, self.distance_calculation_kind)

    #             return scipy_distances.pdist([hyp_coordinate, ref_coordinate],metric=metric_kind).item() 
    #         except ValueError as e:
    #             raise NotImplementedError(f"{self.distance_calculation_kind} distance function not implemented, and not in scipy: {e}")

    #     return self.__point_pair_metric_function(hyp_coordinate, ref_coordinate)
    
    
    # def align_trajectories(self, hyp_points, ref_point):
    #     if self.alignment_strategy == "zero_pad":
    #         raise NotImplementedError
    #     if self.alignment_strategy == "truncate":
    #         raise NotImplementedError


    # def trajectory_pair_distance_function(self, hyp_trajectory:List[KeypointPositionType], ref_trajectory:List[KeypointPositionType]) ->float:
    #     distances = []
    #     if self.alignment_strategy is None:
    #         if len(hyp_trajectory) != len(ref_trajectory):
    #             raise ValueError(f"Cannot calculate distances between trajectories with different lengths: {len(hyp_trajectory)} vs {len(ref_trajectory)}. Perhaps preprocess?")
        

    #     for i, coords in enumerate(zip(hyp_trajectory, ref_trajectory)):
    #         hyp_coord, ref_coord = coords

    #         dist = self.coord_pair_distance_function(hyp_coordinate=hyp_coord, ref_coordinate=ref_coord)
    #         distances.append(dist)
    #         # assert dist >= 0, f"{i}: {dist}, {hyp_coord}, {ref_coord}"
    #     return self.aggregate_point_distances(distances)
    
        

    # def score_along_keypoint_trajectories(self, hypothesis: Pose, reference: Pose)->float:
    #     hyp_points = hypothesis.body.points_perspective() # 560, 1, 93, 3 for example. joint-points, frames, xyz
    #     ref_points = reference.body.points_perspective()



    #     # hyp_points, ref_points = self.align_trajectories(hyp_points, ref_points) 



    #     if hyp_points.shape[0] != ref_points.shape[0] or hyp_points.shape[-1] != ref_points.shape[-1]:
    #         raise ValueError(
    #             f"Shapes of hyp ({hyp_points.shape}) and ref ({ref_points.shape}) unequal. Not supported by {self.name}"
    #             )
        
    #     point_errors = []

    #     for hyp_point_data, ref_point_data in zip(hyp_points, ref_points):
    #         # shape is people, frames, xyz
    #         # NOTE: assumes only one person! # TODO: pytest test checking this.
    #         assert hyp_point_data.shape[0] == 1, f"{self} metric expects only one person. Hyp shape given: {hyp_point_data.shape}"
    #         assert ref_point_data.shape[0] == 1, f"{self} metric expects only one person. Reference shape given: {ref_point_data.shape}"
    #         hyp_point_trajectory = hyp_point_data[0]
    #         ref_point_trajectory = ref_point_data[0]
    #         # point_errors.append(self.trajectory_pair_distance_function(hyp_point_trajectory, ref_point_trajectory))
    #         point_errors.append(self.distance_measure(hyp_point_trajectory, ref_point_trajectory))

    #     return self.aggregate_point_distances(point_errors)
    
    

