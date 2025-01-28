from typing import Callable, Iterable, Literal
from numpy.ma import MaskedArray
from pose_evaluation.metrics.base import Signature, SignatureMixin


AggregationStrategy = Literal["sum", "mean", "max"]
DistanceAggregatorFunction = Callable[[Iterable[float]], float]

class DistanceAggregatorSignature(Signature):
     def __init__(self, args: dict):
          super().__init__(args)
          self.update_signature_and_abbr("aggregation_strategy", "s", args)
          

class DistanceAggregator(SignatureMixin):     
     _SIGNATURE_TYPE = DistanceAggregatorSignature
     
     def __init__(self, aggregation_strategy:AggregationStrategy) -> None:
          self.aggregator_function = get_aggregator_function(strategy=aggregation_strategy)
          self.aggregation_strategy = aggregation_strategy
          
     def __call__(self, distances:Iterable[float] ) -> float:
          return self.aggregator_function(distances)


def create_maskedarrray_and_cast_result_to_float(callable:Callable) -> DistanceAggregatorFunction:
     return lambda a: float(callable(MaskedArray(a)))

def get_aggregator_function(strategy:AggregationStrategy)->DistanceAggregatorFunction:
     
     if strategy == "max":
          return create_maskedarrray_and_cast_result_to_float(MaskedArray.max)
     
     if strategy == "mean":
          return create_maskedarrray_and_cast_result_to_float(MaskedArray.mean)
     
     if strategy == "sum":
          return create_maskedarrray_and_cast_result_to_float(MaskedArray.sum)