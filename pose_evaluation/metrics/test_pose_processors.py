from pathlib import Path
from pose_format import Pose
from typing import List

from pose_evaluation.metrics.pose_processors import TrimMeaninglessFramesPoseProcessor


def test_trim_pose(real_refined_landmark_pose_file_paths: List[Path]):
    for pose_path in real_refined_landmark_pose_file_paths:
        if "needs-trim" in pose_path.name:
            pose = Pose.read(pose_path.read_bytes())

            original_frame_count = pose.body.data.shape[0]

            processor = TrimMeaninglessFramesPoseProcessor(start=True, end=True)

            processed_pose = processor.process_pose(pose)

            # not expecting it to edit the original
            assert (
                    pose.body.data.shape[0] == original_frame_count
            ), f"Original data changed! Frames before: {original_frame_count}. Now: {pose.body.data.shape[0]}"

            # should have fewer frames
            assert (
                    processed_pose.body.data.shape[0] < pose.body.data.shape[0]
            ), f"{pose_path}, {pose.body}, {processed_pose.body}"
