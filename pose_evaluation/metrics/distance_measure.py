from typing import Iterable, Literal, Callable, Optional
from numpy.ma.core import MaskedArray as MaskedArray
from pose_evaluation.metrics.aggregate_distances import DistanceAggregator
from pose_format import Pose
from pose_evaluation.metrics.base import Signature, SignatureMixin
import numpy as np


PointwiseDistanceFunction = Callable[[MaskedArray, MaskedArray], float]




class DistanceMeasureSignature(Signature):
     def __init__(self, args: dict):
        super().__init__(args)
        self.update_signature_and_abbr("name", "n", args)


class DistanceMeasure(SignatureMixin):
    __SIGNATURE_TYPE = DistanceMeasureSignature
    def get_distance(self, hyp_data: np.ma.MaskedArray, ref_data: np.ma.MaskedArray)->float:
            raise NotImplementedError    
    
    def __call__(self, hyp_data: np.ma.MaskedArray, ref_data: np.ma.MaskedArray)->float:
         return self.get_distance(hyp_data, ref_data)

class PowerDistanceSignature(Signature):
     def __init__(self, args: dict):
          super().__init__(args)
          self.update_signature_and_abbr("power", "pow", args)
          self.update_signature_and_abbr("default_distance", "def_d", args)


class PowerDistance(DistanceMeasure):
    _SIGNATURE_TYPE = PowerDistanceSignature
    def __init__(self, power: int = 2, default_distance=0):
        self.name= "power_distance"
        self.power = power
        self.default_distance = default_distance

    def get_distance(self, hyp_data: np.ma.MaskedArray, ref_data: np.ma.MaskedArray)->float:
        return (hyp_data - ref_data).pow(self.power).abs().filled(self.default_distance).mean()
    

class AggregatePointWiseThenAggregateTrajectorywiseDistanceSignature(DistanceMeasureSignature):
     
     def __init__(self, args: dict):
        super().__init__(args)
        self.update_signature_and_abbr("pointwise_distance_function", "pwd", args)
        self.update_signature_and_abbr("pointwise_aggregator", "pt_agg", args)
        self.update_signature_and_abbr("trajectorywise_aggregator", "tw_agg", args)
        

          

class AggregatePointWiseThenAggregateTrajectorywiseDistance(DistanceMeasure):
     _SIGNATURE_TYPE = AggregatePointWiseThenAggregateTrajectorywiseDistanceSignature
     
     def __init__(self, 
                  pointwise_distance_function:Optional[PointwiseDistanceFunction],
                  pointwise_aggregator:Optional[DistanceAggregator], 
                  trajectorywise_aggregator:DistanceAggregator) -> None:
          super().__init__()
          self.pointwise_distance_function = pointwise_distance_function
          self.pointwise_aggregator = pointwise_aggregator
          self.trajectorywise_aggregator = trajectorywise_aggregator
     
     def pointwise_distance(self, hyp_point:MaskedArray, ref_point:MaskedArray)->float:
          if self.pointwise_distance_function is not None:
            return self.pointwise_distance_function(hyp_point, ref_point)
          raise NotImplementedError(f"Undefined pointwise distance function for {self}")
     
     def get_pointwise_distances(self, hyp_traj:MaskedArray, ref_traj:MaskedArray)->Iterable[float]:
        pointwise_distances = []
        for hyp_point, ref_point in zip(hyp_traj, ref_traj):
               pointwise_distances.append(self.pointwise_distance(hyp_point, ref_point))
        return pointwise_distances

     def get_trajectory_pair_distance(self, hyp_traj:MaskedArray, ref_traj:MaskedArray)->float:
          pointwise_distances = self.get_pointwise_distances(hyp_traj, ref_traj)
          return self.aggregate_pointwise_distances(pointwise_distances)
     
     def get_trajectory_pair_distances(self, hyp_trajectories:MaskedArray, ref_trajectories:MaskedArray)->Iterable[float]:
          return [self.get_trajectory_pair_distance(hyp_traj, ref_traj) for hyp_traj, ref_traj in zip(hyp_trajectories, ref_trajectories)]
     
     def aggregate_pointwise_distances(self, pointwise_distances:Iterable[float])->float:
          if self.pointwise_aggregator is not None:
            return self.pointwise_aggregator(pointwise_distances)
          raise NotImplementedError(f"No pointwise aggregator for {self}")
     
     def aggregate_trajectory_distances(self, trajectory_distances:Iterable[float]) -> float:
          return self.trajectorywise_aggregator(trajectory_distances)
     
     def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
          keypoint_trajectory_distances = self.get_trajectory_pair_distances(hyp_data, ref_data)
          return self.aggregate_trajectory_distances(keypoint_trajectory_distances)

