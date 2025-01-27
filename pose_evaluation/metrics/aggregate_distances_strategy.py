from typing import Literal
from pose_evaluation.metrics.base import SignatureMixin
# from numpy.typing import ArrayLike
import numpy as np

AggregationStrategy = Literal["mean", "sum", "max"]

class DistancesAggregator(SignatureMixin):

    # aggregation_strategy: AggregationStrategy

    def __init__(self, aggregation_strategy:AggregationStrategy) -> None:
        self.aggregation_strategy = f"{aggregation_strategy}"

    def aggregate(self, distances):
        if self.aggregation_strategy == "sum":
            return np.sum(distances)
        if self.aggregation_strategy == "mean":
            return np.mean(distances)
        if self.aggregation_strategy == "max":
            return np.max(distances)
        
    def __str__(self):
        return f"{self.aggregation_strategy}"
    
    def __repr__(self):
        return f"{self.aggregation_strategy}"
        
    def get_signature(self) -> str:
        return f"{self}"


