import numpy as np
from fastdtw import fastdtw
import numpy.ma as ma
from tqdm import tqdm

from pose_evaluation.metrics.distance_measure import (
    AggregatedPowerDistance,
    AggregationStrategy,
)


class DTWAggregatedPowerDistance(AggregatedPowerDistance):
    def __init__(
        self,
        order: int = 2,
        default_distance: float = 0,
        aggregation_strategy: AggregationStrategy = "mean",
    ) -> None:
        super().__init__(order, default_distance, aggregation_strategy)

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> float:
        keypoint_count = hyp_data.shape[
            2
        ]  # Assuming shape: (frames, person, keypoints, xyz)
        traj_distances = np.empty(keypoint_count)  # Preallocate a NumPy array

        for i, (hyp_traj, ref_traj) in tqdm(
            enumerate(self._get_keypoint_trajectories(hyp_data, ref_data)),
            desc="getting dtw distances for trajectories",
            total=keypoint_count,
        ):
            distance, _ = fastdtw(hyp_traj, ref_traj, dist=self._calculate_distances)
            traj_distances[i] = distance  # Store distance in the preallocated array

        return self._aggregate(traj_distances)
