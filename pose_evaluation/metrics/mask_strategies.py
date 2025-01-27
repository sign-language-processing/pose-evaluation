import numpy as np
from pose_format import Pose
# from pose_evaluation.metrics.pose_processors import PoseProcessor
from pose_evaluation.metrics.distance_metric import MaskedKeypointPositionStrategy
# from pose_evaluation.utils.pose_utils import copy_pose, set



# def masked_euclidean(hyp_point, ref_point):
#     if np.ma.is_masked(ref_point):  # reference label keypoint is missing
#         return 0
#     elif np.ma.is_masked(hyp_point):  # reference label keypoint is not missing, other label keypoint is missing
#         return euclidean((0, 0, 0), ref_point)/2
#     d = euclidean(hyp_point, ref_point)
# #     return d

# class MasksToOriginProcessor(PoseProcessor):
#     def __init__(self, name="masked_to_origin") -> None:
#         super().__init__(name)

#     def process_pose(self, pose: Pose) -> Pose:
#         pose = copy_pose(pose)
#         # frames, person, keypoint, xyz
#         for frame_index, frame in enumerate(pose.body.data):
#             for person_index, person in enumerate(frame):
#                 for keypoint_index, keypoint_value in enumerate(person):
#                     if np.ma.is_masked(keypoint_value):
#                         pose.body.data[frame_index][person_index][keypoint_index] = np.zeros_like(keypoint_value)

#         return pose



        



    