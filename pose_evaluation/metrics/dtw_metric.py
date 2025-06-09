import warnings

import numpy as np
import numpy.ma as ma  # pylint: disable=consider-using-from-import
from dtaidistance import dtw_ndim
from fastdtw import fastdtw  # type: ignore
from scipy.spatial.distance import cdist
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
        trajectory_distances = ma.empty(keypoint_count)  # Preallocate a NumPy array

        for i, (hyp_trajectory, ref_trajectory) in tqdm(
            enumerate(self._get_keypoint_trajectories(hyp_data, ref_data)),
            desc="getting dtw distances for trajectories",
            total=keypoint_count,
            disable=not progress,
        ):
            distance, _ = fastdtw(hyp_trajectory, ref_trajectory, dist=self._calculate_pointwise_distances)
            # distance is an ndarray of shape (1,)
            trajectory_distances[i] = distance.item()  # Store distance in the preallocated array
        trajectory_distances = ma.array(trajectory_distances)
        return self._aggregate(trajectory_distances)

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
        name="DTWAggregatedScipyDistanceMeasure",
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


class DTWOptimizedDistanceMeasure(DTWAggregatedDistanceMeasure):
    """Optimized according to https://github.com/slaypni/fastdtw/blob/master/fastdtw/_fastdtw.pyx#L71-L76
    This function runs fastest if the following conditions are satisfied:
            1) x and y are either 1 or 2d numpy arrays whose dtype is a
               subtype of np.float
            2) The dist input is a positive integer or None
    """

    def __init__(  # pylint: disable=too-many-arguments
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
        trajectory_distances = ma.empty(keypoint_count)  # Preallocate a NumPy array

        for i, (hyp_trajectory, ref_trajectory) in tqdm(
            enumerate(self._get_keypoint_trajectories(hyp_data, ref_data)),
            desc="getting dtw distances for trajectories",
            total=keypoint_count,
            disable=not progress,
        ):
            distance, _ = fastdtw(hyp_trajectory, ref_trajectory, self.power)
            trajectory_distances[i] = distance  # Store distance in the preallocated array
        trajectory_distances = ma.array(trajectory_distances)
        return self._aggregate(trajectory_distances)


# https://forecastegy.com/posts/dynamic-time-warping-dtw-libraries-python-examples/
class DTWDTAIImplementationDistanceMeasure(AggregatedDistanceMeasure):

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        name="dtaiDTWAggregatedDistanceMeasure",
        default_distance: float = 0,
        aggregation_strategy: AggregationStrategy = "mean",
        use_fast=True,
    ) -> None:
        super().__init__(
            name=name,
            default_distance=default_distance,
            aggregation_strategy=aggregation_strategy,
        )
        self.use_fast = use_fast

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray, progress=False) -> float:
        keypoint_count = hyp_data.shape[2]  # Assuming shape: (frames, person, keypoints, xyz)
        trajectory_distances = ma.masked_array(  # preallocate for speed
            data=np.empty(keypoint_count, dtype=float),
            mask=np.ones(keypoint_count, dtype=bool),  # all masked by default
        )

        for i, (hyp_trajectory, ref_trajectory) in tqdm(
            enumerate(self._get_keypoint_trajectories(hyp_data, ref_data)),
            desc="getting dtw distances for trajectories",
            total=keypoint_count,
            disable=not progress,
        ):
            hyp_trajectory = np.asarray(hyp_trajectory, dtype=np.float64)
            ref_trajectory = np.asarray(ref_trajectory, dtype=np.float64)
            if self.use_fast:
                distance = dtw_ndim.distance_fast(hyp_trajectory, ref_trajectory)
            else:
                distance = dtw_ndim.distance(hyp_trajectory, ref_trajectory)
            trajectory_distances[i] = distance
            trajectory_distances.mask[i] = False

            #     trajectory_distances[i] = distance
            #     # trajectory_distances.mask[i] = False  # explicitly unmask this value
            # else:
            #     # it is masked still
            #     # TODO: option to just skip nan trajectory distances
            #     trajectory_distances[i] = self.default_distance
            #     # trajectory_distances.mask[i] = False

        distance = self._aggregate(trajectory_distances)
        if distance is None or np.isnan(distance) or np.isinf(distance):
            warnings.warn(
                f"Invalid distance calculated, setting to default value {self.default_distance}, hyp shape: {hyp_data.shape} with {np.isnan(hyp_data).sum()} nans and {ma.count_masked(hyp_data)} masked, ref shape: {ref_data.shape} with {np.isnan(ref_data).sum()} nans and {ma.count_masked(ref_data)} masked",
                category=RuntimeWarning,
            )
            distance = self.default_distance

        return float(distance)
