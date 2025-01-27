from abc import ABC
from pose_evaluation.metrics.base import SignatureMixin
import numpy as np


class Distance(SignatureMixin):
    def get_signature(self) -> str:
        return "|".join([f"{key}:{value}" for key, value in self.__dict__.items()])
    
    def get_distance(self, hyp_data: np.ma.MaskedArray, ref_data: np.ma.MaskedArray)->float:
            raise NotImplementedError
    
    def __call__(self, hyp_data: np.ma.MaskedArray, ref_data: np.ma.MaskedArray)->float:
         return self.get_distance(hyp_data, ref_data)
         
         


class PowerDistance(Distance):
    def __init__(self, power: int = 2, default_distance=0):
        self.name= "power_distance"
        self.power = power
        self.default_distance = default_distance

    def get_distance(self, hyp_data: np.ma.MaskedArray, ref_data: np.ma.MaskedArray)->float:
        return (hyp_data - ref_data).pow(self.power).abs().filled(self.default_distance).mean()
    

# def masked_euclidean(point1, point2):
#     if np.ma.is_masked(point2):  # reference label keypoint is missing
#         return 0
#     elif np.ma.is_masked(point1):  # reference label keypoint is not missing, other label keypoint is missing
#         return euclidean((0, 0, 0), point2)/2
#     d = euclidean(point1, point2)
#     return d