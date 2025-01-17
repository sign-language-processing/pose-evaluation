import argparse
from pathlib import Path
from pose_format import Pose

from pose_evaluation.metrics import ape_metric, distance_metric, ndtw_mje_metric
from pose_evaluation.utils.pose_utils import load_pose_file, preprocess_pose, get_component_names_and_points_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load files and run score_all")
    parser.add_argument("pose_dir", type=Path, help="Path to the directory containing SignCLIP .npy files")

    args = parser.parse_args()

    pose_files = args.pose_dir.glob("*.pose")
    poses = [load_pose_file(pose_file) for pose_file in pose_files]
    print(f"Loaded {len(poses)} poses from {args.pose_dir}")
    original_component_names, original_points_dict = get_component_names_and_points_dict(poses[0])

    preprocessed_poses = [preprocess_pose(pose) for pose in poses]
    preprocessed_component_names, preprocessed_points_dict = get_component_names_and_points_dict(preprocessed_poses[0])

    print(len(original_component_names))
    print(len(preprocessed_component_names))
    preprocessed_poses_not_normalized = [preprocess_pose(pose, normalize_poses=False) for pose in poses]
    
    
    
    only_reduced_poses = [pose.get_components(preprocessed_component_names, preprocessed_points_dict) for pose in poses]




    # metric = ndtw_mje.DynamicTimeWarpingMeanJointErrorMetric()
    print(f"Reduced poses to {len(only_reduced_poses[0].header.components)} components and {only_reduced_poses[0].header.total_points()} points")
    print(only_reduced_poses[0].body.data.shape) # (93, 1, 560, 3) for example
    print(only_reduced_poses[0].body.points_perspective().shape) # (560, 1, 93, 3)

    for metric_class in distance_metric.DistanceMetric.__subclasses__():
        metric = metric_class()
        print(metric)

        # metric = ape_metric.AveragePositionErrorMetric()

        
        # print(metric.score_all(poses, preprocessed_poses))
        # print(metric.score_all(only_reduced_poses, preprocessed_poses))
        print(metric.score_all(preprocessed_poses, preprocessed_poses))
        # print(metric.score_all(only_reduced_poses, only_reduced_poses))



        