from abc import ABC
from importlib import resources

import numpy as np
from pose_format import Pose
from scipy.optimize import linear_sum_assignment
from sign_language_segmentation.bin import load_model, predict
from sign_language_segmentation.src.utils.probs_to_segments import probs_to_segments

from pose_evaluation.metrics.base_pose_metric import PoseMetric


class SegmentedPoseMetric(PoseMetric, ABC):
    def __init__(self, isolated_metric: PoseMetric):
        super().__init__("SegmentedPoseMetric", higher_is_better=isolated_metric.higher_is_better)

        self.isolated_metric = isolated_metric

        model_path = resources.path("sign_language_segmentation", "dist/model_E1s-1.pth")
        self.segmentation_model = load_model(model_path)

    # pylint: disable=too-many-locals
    def score(self, hypothesis: Pose, reference: Pose) -> float:
        # Process input files
        processed_hypothesis, processed_reference = self.process_poses([hypothesis, reference])
        # Predict segments BIO
        hypothesis_probs = predict(self.segmentation_model, processed_hypothesis)["sign"]
        reference_probs = predict(self.segmentation_model, processed_reference)["sign"]
        # Convert to discrete segments
        hypothesis_signs = probs_to_segments(hypothesis_probs, 60, 50)
        reference_signs = probs_to_segments(reference_probs, 60, 50)

        print(hypothesis_signs)  # TODO convert segments to Pose objects

        # Fallback to isolated metric if no segments are found
        if len(hypothesis_signs) == 0 or len(reference_signs) == 0:
            return self.isolated_metric.score(hypothesis, reference)

        # Pad with empty segment to make sure the number of signs is the same
        if len(hypothesis_signs) != len(reference_signs):
            max_length = max(len(hypothesis_signs), len(reference_signs))
            hypothesis_signs += [(0, 0)] * (max_length - len(hypothesis_signs))
            reference_signs += [(0, 0)] * (max_length - len(reference_signs))

        # Match each hypothesis sign with each reference sign
        cost_matrix = self.isolated_metric.score_all(hypothesis_signs, reference_signs, progress_bar=False)
        cost_tensor = np.array(cost_matrix)
        if not self.isolated_metric.higher_is_better:
            cost_tensor = 1 - cost_tensor

        row_ind, col_ind = linear_sum_assignment(cost_tensor)
        pairs = list(zip(row_ind, col_ind, strict=False))
        values = [cost_matrix[row][col] for row, col in pairs]
        return sum(values) / len(values)
