from typing import Literal, Tuple, List, cast, get_args, Callable, Union, TYPE_CHECKING
import numpy as np
from numpy import ma
import scipy.spatial.distance as scipy_distances
if TYPE_CHECKING:
    from scipy.spatial.distance import _MetricKind
from pose_format import Pose
from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.utils.pose_utils import PoseProcessor, NormalizePosesProcessor, RemoveLegsPosesProcessor, HideLowConfProcessor, ZeroPadShorterPosesProcessor, ReducePosesToCommonComponentsProcessor, RemoveWorldLandmarksProcessor, RemoveComponentsProcessor
ValidDistanceKinds = Literal["euclidean", "manhattan"]
UnequalSequenceLengthStrategy = Literal["zero_pad_shorter", "dynamic_time_warping"]
MissingKeypointStrategy = Literal["interpolate"]
MismatchedComponentsStrategy = Literal["reduce"]
PosesTransformerFunctionType = Callable[[List[Pose]], List[Pose]]
KeypointPositionType = Union[Tuple[float, float, float], Tuple[float, float]] # XYZ or XY

class DistanceMetric(PoseMetric):
    def __init__(self, 
                 spatial_distance_function_kind: ValidDistanceKinds = "euclidean",
                 preprocessors: Union[None, List[PoseProcessor]] = None,
                 normalize_poses:bool=True,
                 reduce_poses_to_common_points:bool=True,
                 zero_pad_shorter_sequence:bool=True,
                 remove_legs:bool=True,
                 remove_world_landmarks:bool=True,
                 conf_threshold_to_drop_points:None|float=None,
                 ):
        super().__init__(f"DistanceMetric {spatial_distance_function_kind}", higher_is_better=False)
        if preprocessors is None:
            self.preprocessers = [             
            ]
        if normalize_poses:
            self.preprocessers.append(NormalizePosesProcessor())
        if reduce_poses_to_common_points:
            self.preprocessers.append(ReducePosesToCommonComponentsProcessor())
        if zero_pad_shorter_sequence:
            self.preprocessers.append(ZeroPadShorterPosesProcessor())
        if remove_legs:
            self.preprocessers.append(RemoveLegsPosesProcessor())
        if remove_world_landmarks:
            self.preprocessers.append(RemoveWorldLandmarksProcessor())
        if conf_threshold_to_drop_points:
            self.preprocessers.append(HideLowConfProcessor(conf_threshold=conf_threshold_to_drop_points))
        

        if spatial_distance_function_kind == "euclidean":
            self.__point_pair_metric_function = scipy_distances.euclidean
        elif spatial_distance_function_kind == "manhattan":
            self.__point_pair_metric_function = scipy_distances.cityblock
            # return scipy_distances.cityblock(hyp_coordinate, ref_coordinate)
        else:

            # lambda
            self.__point_pair_metric_function = None
        # except ValueError as e:
        #     raise NotImplementedError(f"{self.spatial_distance_function_kind} distance function not implemented, and not in scipy: {e}")

        self.spatial_distance_function_kind = spatial_distance_function_kind


    def point_pair_distance_function(self, hyp_coordinate, ref_coordinate) -> float:
        if self.__point_pair_metric_function is None:
            try: 
                metric_kind = self.spatial_distance_function_kind
                if TYPE_CHECKING:
                     metric_kind=cast(_MetricKind, self.spatial_distance_function_kind)

                return scipy_distances.pdist([hyp_coordinate, ref_coordinate],metric=metric_kind).item() 
            except ValueError as e:
                raise NotImplementedError(f"{self.spatial_distance_function_kind} distance function not implemented, and not in scipy: {e}")

        return self.__point_pair_metric_function(hyp_coordinate, ref_coordinate)
    
    def preprocess_poses(self, poses:List[Pose])->List[Pose]:
        for preprocessor in self.preprocessers:
            preprocessor = cast(PoseProcessor, preprocessor)
            poses = preprocessor.process_poses(poses)
        return poses


        # return preprocess_poses(poses=poses, 
        #                         normalize_poses=self.normalize_poses,
        #                         conf_threshold_to_drop_points=self.conf_threshold_to_drop_points,
        #                         reduce_poses_to_common_points=self.reduce_poses_to_common_points,
        #                         remove_legs=self.remove_legs,
        #                         remove_world_landmarks=self.remove_world_landmarks,
        #                         zero_pad_shorter_pose=self.zero_pad_shorter_sequence,                                
        #                         )

        
    

    def trajectory_pair_distance_function(self, hyp_trajectory, ref_trajectory) ->float:
        arrays = [hyp_trajectory, ref_trajectory]
        hyp_trajectory = arrays[0]
        ref_trajectory = arrays[1]
        spatial_point_distances = []
        if len(hyp_trajectory) != len(ref_trajectory):
            raise ValueError(f"Cannot calculate distances between trajectories with different lengths: {len(hyp_trajectory)} vs {len(ref_trajectory)}. Perhaps preprocess?")

        for i, coords in enumerate(zip(hyp_trajectory, ref_trajectory)):
            hyp_coord, ref_coord = coords
            dist = self.point_pair_distance_function(hyp_coordinate=hyp_coord, ref_coordinate=ref_coord)
            spatial_point_distances.append(dist)
            assert dist >= 0, f"{i}: {dist}, {hyp_coord}, {ref_coord}"
        return np.mean(spatial_point_distances)        
        

    def score_along_keypoint_trajectories(self, hypothesis: Pose, reference: Pose)->float:
        assert np.count_nonzero(np.isnan(hypothesis.body.data)) == 0
        assert np.count_nonzero(np.isnan(reference.body.data)) == 0
        hyp_points = hypothesis.body.points_perspective() # 560, 1, 93, 3 for example. joint-points, frames, xyz
        ref_points = reference.body.points_perspective()

        if hyp_points.shape[0] != ref_points.shape[0] or hyp_points.shape[-1] != ref_points.shape[-1]:
            raise ValueError(
                f"Shapes of hyp ({hyp_points.shape}) and ref ({ref_points.shape}) unequal. Not supported by {self.name}"
                )
        
        point_errors = []
        for hyp_point_data, ref_point_data in zip(hyp_points, ref_points):
            # shape is people, frames, xyz
            # NOTE: assumes only one person! # TODO: pytest test checking this.
            assert hyp_point_data.shape[0] == 1, f"{self} metric expects only one person. Hyp shape given: {hyp_point_data.shape}"
            assert ref_point_data.shape[0] == 1, f"{self} metric expects only one person. Reference shape given: {ref_point_data.shape}"
            hyp_point_trajectory = hyp_point_data[0]
            ref_point_trajectory = ref_point_data[0]
            point_errors.append(self.trajectory_pair_distance_function(hyp_point_trajectory, ref_point_trajectory))

        return float(np.mean(point_errors))
    
    # def preprocess_and_score(self, hypothesis: Pose, reference: Pose) -> float:
    #     poses = [hypothesis, reference]
    #     poses = self.preprocess_poses(poses)
    #     return self.score(*poses)


    def score(self, hypothesis: Pose, reference: Pose) -> float:
        hypothesis, reference = self.preprocess_poses([hypothesis, reference])
        return self.score_along_keypoint_trajectories(hypothesis, reference)



        # arrays = [poses[0].body.data, poses[1].body.data]


        

        # # Calculate the error
        # error = arrays[0] - arrays[1]

        # # for l2/euclidean, we need to calculate the error for each point
        # if self.kind == "euclidean":
        #     # the last dimension is the 3D coordinates
        #     error = ma.power(error, 2)
        #     error = error.sum(axis=-1)
        #     error = ma.sqrt(error)
        # else:
        #     error = ma.abs(error)

        # error = error.filled(0)
        # return error.sum()
