from fastdtw import fastdtw
import numpy as np
from scipy.spatial.distance import cdist
import numpy.ma as ma  # pylint: disable=consider-using-from-import
from tqdm import tqdm

from pose_evaluation.metrics.distance_measure import (
    AggregatedDistanceMeasure,
    AggregationStrategy,
    masked_array_power_distance,
)


class DTWAggregatedDistanceMeasure(AggregatedDistanceMeasure):
    def __init__(
        self,
        name="DTWAggregatedDistanceMeasure",
        default_distance: float = 0,
        aggregation_strategy: AggregationStrategy = "mean",
    ) -> None:
        super().__init__(
            name=name,
            default_distance=default_distance,
            aggregation_strategy=aggregation_strategy,
        )

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray, progress=False) -> float:
        keypoint_count = hyp_data.shape[2]  # Assuming shape: (frames, person, keypoints, xyz)
        traj_distances = ma.empty(keypoint_count)  # Preallocate a NumPy array

        for i, (hyp_traj, ref_traj) in tqdm(
            enumerate(self._get_keypoint_trajectories(hyp_data, ref_data)),
            desc="getting dtw distances for trajectories",
            total=keypoint_count,
            disable=not progress,
        ):
            distance, _ = fastdtw(hyp_traj, ref_traj, dist=self._calculate_pointwise_distances)
            traj_distances[i] = distance  # Store distance in the preallocated array
        traj_distances = ma.array(traj_distances)
        return self._aggregate(traj_distances)

    def _calculate_pointwise_distances(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> ma.MaskedArray:
        raise NotImplementedError("_calculate_pointwise_distances must be a callable that can be passed to fastdtw")


class DTWAggregatedPowerDistanceMeasure(DTWAggregatedDistanceMeasure):
    def __init__(
        self,
        name="DTWAggregatedDistanceMeasure",
        default_distance: float = 0,
        aggregation_strategy: AggregationStrategy = "mean",
        order=2,
    ) -> None:
        super().__init__(
            name=name,
            default_distance=default_distance,
            aggregation_strategy=aggregation_strategy,
        )
        self.power = order

    def _calculate_pointwise_distances(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> ma.MaskedArray:
        return masked_array_power_distance(
            hyp_data=hyp_data, ref_data=ref_data, power=self.power, default_distance=self.default_distance
        )


class DTWAggregatedScipyDistanceMeasure(DTWAggregatedDistanceMeasure):
    def __init__(
        self,
        name="DTWAggregatedDistanceMeasure",
        default_distance: float = 0,
        aggregation_strategy: AggregationStrategy = "mean",
        metric: str = "euclidean",
    ) -> None:
        super().__init__(
            name=name,
            default_distance=default_distance,
            aggregation_strategy=aggregation_strategy,
        )
        self.metric = metric

    def _calculate_pointwise_distances(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> ma.MaskedArray:
        hyp_data = hyp_data.reshape(1, -1)  # Adds a new leading dimension
        ref_data = ref_data.reshape(1, -1)

        return cdist(hyp_data, ref_data, metric=self.metric)  # type: ignore "no overloads match" but it works fine


class DTWAggregatedPowerDistanceMeasure(DTWAggregatedDistanceMeasure):
    def __init__(
        self,
        name="DTWAggregatedDistanceMeasure",
        default_distance: float = 0,
        aggregation_strategy: AggregationStrategy = "mean",
        order=2,
    ) -> None:
        super().__init__(
            name=name,
            default_distance=default_distance,
            aggregation_strategy=aggregation_strategy,
        )
        self.power = order

    def _calculate_pointwise_distances(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> ma.MaskedArray:
        return masked_array_power_distance(
            hyp_data=hyp_data, ref_data=ref_data, power=self.power, default_distance=self.default_distance
        )


class DTWOptimizedDistanceMeasure(DTWAggregatedDistanceMeasure):
    """Optimized according to https://github.com/slaypni/fastdtw/blob/master/fastdtw/_fastdtw.pyx#L71-L76
    This function runs fastest if the following conditions are satisfied:
            1) x and y are either 1 or 2d numpy arrays whose dtype is a
               subtype of np.float
            2) The dist input is a positive integer or None
    """

    def __init__(
        self,
        name="DTWOptimizedDistanceMeasure",
        default_distance: float = 0,
        aggregation_strategy: AggregationStrategy = "mean",
        power=2,
        masked_fill_value=0,
    ) -> None:
        super().__init__(
            name=name,
            default_distance=default_distance,
            aggregation_strategy=aggregation_strategy,
        )
        self.power = power
        self.masked_fill_value = masked_fill_value

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray, progress=False) -> float:
        # https://github.com/slaypni/fastdtw/blob/master/fastdtw/_fastdtw.pyx#L71-L76
        # fastdtw goes more quickly if...
        # 1) x and y are either 1 or 2d numpy arrays whose dtype is a
        #    subtype of np.float
        # 2) The dist input is a positive integer or None
        #
        # So we convert to ndarray by filling in masked values, and we ensure the datatype.
        hyp_data = hyp_data.filled(self.masked_fill_value).astype(float)
        ref_data = ref_data.filled(self.masked_fill_value).astype(float)
        keypoint_count = hyp_data.shape[2]  # Assuming shape: (frames, person, keypoints, xyz)
        traj_distances = ma.empty(keypoint_count)  # Preallocate a NumPy array

        for i, (hyp_traj, ref_traj) in tqdm(
            enumerate(self._get_keypoint_trajectories(hyp_data, ref_data)),
            desc="getting dtw distances for trajectories",
            total=keypoint_count,
            disable=not progress,
        ):
            distance, _ = fastdtw(hyp_traj, ref_traj, self.power)
            traj_distances[i] = distance  # Store distance in the preallocated array
        traj_distances = ma.array(traj_distances)
        return self._aggregate(traj_distances)
