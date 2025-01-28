from pathlib import Path
from typing import List, Tuple, Dict, Union, Iterable
import numpy as np
from pose_format import Pose
from pose_format.utils.openpose import OpenPose_Components
from pose_format.utils.openpose_135 import OpenPose_Components as OpenPose135_Components
# from pose_format.utils.holistic import holistic_components # creates an error: ImportError: Please install mediapipe with: pip install mediapipe
from collections import defaultdict
from pose_format.utils.generic import pose_normalization_info, pose_hide_legs, fake_pose
    

def pose_remove_world_landmarks(pose: Pose)->Pose:
    return remove_components(pose, ["POSE_WORLD_LANDMARKS"])

# TODO: remove, and use the one implemented in the latest pose_format
def detect_format(pose: Pose) -> str:
    component_names = [c.name for c in pose.header.components]
    mediapipe_components = [
        "POSE_LANDMARKS",
        "FACE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS",
        "POSE_WORLD_LANDMARKS",
    ]
    
    openpose_components = [c.name for c in OpenPose_Components]
    openpose_135_components = [c.name for c in OpenPose135_Components]
    for component_name in component_names:
        if component_name in mediapipe_components:
            return "mediapipe"
        if component_name in openpose_components:
            return "openpose"
        if component_name in openpose_135_components:
            return "openpose_135"

    raise ValueError(
        f"Unknown pose header schema with component names: {component_names}"
    )

def get_component_names_and_points_dict(pose:Pose)->Tuple[List[str], Dict[str, List[str]]]:
    component_names = []
    points_dict = defaultdict(list)
    for component in pose.header.components:
            component_names.append(component.name)
            
            for point in component.points:
                points_dict[component.name].append(point)

    return component_names, points_dict

def remove_components(
    pose: Pose, components_to_remove: List[str]|str, points_to_remove: List[str]|str|None=None
):
    if points_to_remove is None:
        points_to_remove = []
    if isinstance(components_to_remove, str):
        components_to_remove = [components_to_remove]
    if isinstance(points_to_remove, str):
        points_to_remove = [points_to_remove]
    components_to_keep = []
    points_dict = {}

    for component in pose.header.components:
        if component.name not in components_to_remove:
            components_to_keep.append(component.name)
            points_dict[component.name] = []
            for point in component.points:
                if point not in points_to_remove:
                    points_dict[component.name].append(point)

    return pose.get_components(components_to_keep, points_dict)




def pose_remove_legs(pose: Pose) -> Pose:
    detected_format = detect_format(pose)
    if detected_format == "mediapipe":
        mediapipe_point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        mediapipe_sides = ["LEFT", "RIGHT"]
        point_names_to_remove = [
            side + "_" + name
            for name in mediapipe_point_names
            for side in mediapipe_sides
        ]
    else:
        raise NotImplementedError(
            f"Remove legs not implemented yet for pose header schema {detected_format}"
        )

    pose = remove_components(pose, [], point_names_to_remove)
    return pose



def copy_pose(pose: Pose) -> Pose:
    return pose.get_components([component.name for component in pose.header.components])



def get_face_and_hands_from_pose(pose: Pose) -> Pose:
    # based on MediaPipe Holistic format.
    components_to_keep = [
        "FACE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS",
    ]
    return pose.get_components(components_to_keep)

def load_pose_file(pose_path: Path) -> Pose:
    pose_path = Path(pose_path).resolve()
    with pose_path.open("rb") as f:
        pose = Pose.read(f.read())
    return pose


def reduce_pose_components_and_points_to_intersection(poses: Iterable[Pose]) -> List[Pose]:
    poses = [copy_pose(pose) for pose in poses]
    component_names_for_each_pose = []
    point_dict_for_each_pose = []
    for pose in poses:
        names, points_dict = get_component_names_and_points_dict(pose)        
        component_names_for_each_pose.append(set(names))
        point_dict_for_each_pose.append(points_dict)

    set_of_common_components = list(set.intersection(*component_names_for_each_pose))

    common_points = {}
    for component_name in set_of_common_components:
        max_length = 0
        min_length = np.inf
        points_for_each_pose = []
        for point_dict in point_dict_for_each_pose:
            points_list = point_dict.get(component_name)
            if points_list is None:
                min_length =0 
            max_length = max(max_length, len(points_list))
            min_length = min(min_length, len(points_list))
            points_for_each_pose.append(set(points_list))
        set_of_common_points = list(set.intersection(*points_for_each_pose))

        if min_length < max_length and min_length>0:
            common_points[component_name] = set_of_common_points



    poses = [pose.get_components(set_of_common_components, common_points) for pose in poses]
    return poses

def zero_pad_shorter_poses(poses:Iterable[Pose]) -> List[Pose]:
    poses = [copy_pose(pose) for pose in poses]
    # arrays = [pose.body.data for pose in poses]


    # first dimension is frames. Then People, joint-points, XYZ or XY
    max_frame_count = max(len(pose.body.data) for pose in poses) 
    # Pad the shorter array with zeros
    for pose in poses:
        if len(pose.body.data) < max_frame_count:
            desired_shape = list(pose.body.data.shape)
            desired_shape[0] = max_frame_count - len(pose.body.data)
            padding_tensor = np.ma.zeros(desired_shape)
            padding_tensor_conf = np.ones(desired_shape[:-1])
            pose.body.data = np.ma.concatenate([pose.body.data, padding_tensor], axis=0)
            pose.body.confidence = np.concatenate([pose.body.confidence, padding_tensor_conf])
    return poses    



# def preprocess_poses(
#     poses: List[Pose],
#     normalize_poses: bool = True,
#     reduce_poses_to_common_points: bool = False,
#     remove_legs: bool = True,
#     remove_world_landmarks: bool = False,
#     conf_threshold_to_drop_points: None | float = None,
#     zero_pad_shorter_pose = True, 
# ) -> List[Pose]:
#     for pose in poses:
#         assert np.count_nonzero(np.isnan(pose.body.data)) == 0
#     # NOTE: this is a lot of arguments. Perhaps a list may be better?
#     if reduce_poses_to_common_points:
        
#         poses = reduce_pose_components_and_points_to_intersection(poses)

#     poses = [
#         preprocess_pose(
#             pose,
#             normalize_poses=normalize_poses,
#             remove_legs=remove_legs,
#             remove_world_landmarks=remove_world_landmarks,
#             conf_threshold_to_drop_points=conf_threshold_to_drop_points,
#         )
#         for pose in poses
#     ]
#     for pose in poses:
#         assert np.count_nonzero(np.isnan(pose.body.data)) == 0

#     if zero_pad_shorter_pose:
#         poses = zero_pad_shorter_poses(poses)
#     return poses



def set_masked_to_origin_position(pose:Pose)->Pose:
    pose = copy_pose(pose)
    # frames, person, keypoint, xyz
    
    pose.body.data = np.ma.array(pose.body.data.filled(0), mask=False)

    return pose


# def pre_align_with_dtw(hyp: Pose, ref:Pose):

#     x = hyp_trajectory
#     y = ref_trajectory
#     _, path = fastdtw(x.data, y.data)  # Use the raw data for DTW computation

#     # Initialize lists for aligned data and masks
#     aligned_x_data = []
#     aligned_y_data = []
    
#     aligned_x_mask = []
#     aligned_y_mask = []
    
#     # Loop through the DTW path
#     for xi, yi in path:
#         # Append aligned data
#         aligned_x_data.append(x.data[xi])
#         aligned_y_data.append(y.data[yi])
        
#         # Append aligned masks (directly use .mask)
#         aligned_x_mask.append(x.mask[xi])
#         aligned_y_mask.append(y.mask[yi])
    
#     # Create aligned masked arrays
#     aligned_x = np.ma.array(aligned_x_data, mask=aligned_x_mask)
#     aligned_y = np.ma.array(aligned_y_data, mask=aligned_y_mask)
#     return aligned_x, aligned_y

# def preprocess_pose(
#     pose: Pose,
#     normalize_poses: bool = True,
#     remove_legs: bool = True,
#     remove_world_landmarks: bool = False,
#     conf_threshold_to_drop_points: None | float = None,
# ) -> Pose:
#     assert np.count_nonzero(np.isnan(pose.body.data)) == 0
#     pose = copy_pose(pose)
#     if normalize_poses:
#         # note: latest version (not yet released) does it automatically
#         pose = pose.normalize(pose_normalization_info(pose.header))
#         # TODO: https://github.com/sign-language-processing/pose/issues/146

#     # Drop legs
#     if remove_legs:
#         try: 
#             pose = pose_remove_legs(pose)
#         except NotImplementedError as e:
#             print(f"Could not remove legs: {e}")
#             # raise Warning(f"Could not remove legs: {e}")

#     # not used, typically.
#     if remove_world_landmarks:
#         pose = pose_remove_world_landmarks(pose)
#         assert np.count_nonzero(np.isnan(pose.body.data)) == 0

#     # hide low conf
#     if conf_threshold_to_drop_points is not None:
#         pose_hide_low_conf(pose, confidence_threshold=conf_threshold_to_drop_points)
#         assert np.count_nonzero(np.isnan(pose.body.data)) == 0

#     return pose


def pose_hide_low_conf(pose: Pose, confidence_threshold: float = 0.2) -> None:
    mask = pose.body.confidence <= confidence_threshold
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask, mask], axis=3)
    masked_data = np.ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data