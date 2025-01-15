from typing import Literal, Tuple, List

from numpy import ma
import scipy.spatial.distance as scipy_distances
from pose_format import Pose

from pose_evaluation.metrics.base_pose_metric import PoseMetric
ValidDistanceKinds = Literal["euclidean", "manhattan"]

class DistanceMetric(PoseMetric):
    def __init__(self, kind: ValidDistanceKinds = "euclidean"):
        super().__init__(f"DistanceMetric {kind}", higher_is_better=False)
        self.kind = kind

        

    def __pad_shorter_arrays(self, arrays:List):
        max_length = max(len(array) for array in arrays)
        # Pad the shorter array with zeros
        for i, array in enumerate(arrays):
            if len(array) < max_length:
                shape = list(array.shape)
                shape[0] = max_length - len(array)
                padding_tensor = ma.zeros(shape)
                arrays[i] = ma.concatenate([array, padding_tensor], axis=0)
        return arrays
        


    def element_score(self, hyp_coordinate, ref_coordinate) -> float:
        if self.kind == "euclidean":
            return scipy_distances.euclidean(hyp_coordinate, ref_coordinate)
        if self.kind == "manhattan":
            return scipy_distances.cityblock(hyp_coordinate, ref_coordinate)
        raise NotImplementedError(f"{self.kind} element_score not implemented")


    def trajectory_score(self, hyp_trajectory, ref_trajectory) ->float:
        arrays = [hyp_trajectory, ref_trajectory]
        arrays = self.__pad_shorter_arrays(arrays)
        hyp_trajectory = arrays[0]
        ref_trajectory = arrays[1]
        element_wise_errors = []

        for hyp_coord, ref_coord in zip(hyp_trajectory, ref_trajectory):
            element_wise_errors.append(self.element_score(hyp_coordinate=hyp_coord, ref_coordinate=ref_coord))

        return sum(element_wise_errors)/len(element_wise_errors)
        

        

    def score_along_joint_trajectories(self, hypothesis: Pose, reference: Pose):
        hyp_points = hypothesis.body.points_perspective() # 560, 1, 93, 3 for example. joint-points, frames, xyz
        ref_points = reference.body.points_perspective()


        if hyp_points.shape[0] != ref_points.shape[0] or hyp_points.shape[-1] != ref_points.shape[-1]:
            raise ValueError(
                f"Shapes of hyp ({hyp_points.shape}) and ref ({ref_points.shape}) unequal. Not supported by {self.name}"
                )
        
        point_count = hyp_points.shape[0]
        total_error = 0.0  
        for hyp_point_data, ref_point_data in zip(hyp_points, ref_points):
            # shape is people, frames, xyz
            # NOTE: assumes only one person! # TODO: pytest test checking this.
            assert hyp_point_data.shape[0] == 1, f"{self} metric expects only one person. Hyp shape given: {hyp_point_data.shape}"
            assert ref_point_data.shape[0] == 1, f"{self} metric expects only one person. Reference shape given: {ref_point_data.shape}"
            hyp_point_trajectory = hyp_point_data[0]
            ref_point_trajectory = ref_point_data[0]
            total_error += self.trajectory_score(hyp_point_trajectory, ref_point_trajectory)

        return total_error/point_count

    def score(self, hypothesis: Pose, reference: Pose) -> float:
        arrays = [hypothesis.body.data, reference.body.data]
        max_length = max(len(array) for array in arrays)
        # Pad the shorter array with zeros
        for i, array in enumerate(arrays):
            if len(array) < max_length:
                shape = list(array.shape)
                shape[0] = max_length - len(array)
                padding_tensor = ma.zeros(shape)
                arrays[i] = ma.concatenate([array, padding_tensor], axis=0)



        # Calculate the error
        error = arrays[0] - arrays[1]

        # for l2/euclidean, we need to calculate the error for each point
        if self.kind == "euclidean":
            # the last dimension is the 3D coordinates
            error = ma.power(error, 2)
            error = error.sum(axis=-1)
            error = ma.sqrt(error)
        else:
            error = ma.abs(error)

        error = error.filled(0)
        return error.sum()
