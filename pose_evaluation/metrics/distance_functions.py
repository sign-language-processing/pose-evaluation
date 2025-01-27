# from typing import Tuple, Literal
# import numpy as np
# from scipy.spatial.distance import euclidean, cityblock

# ValidDistanceKinds = Literal["euclidean", "manhattan", "masked_euclidean"]


# class PointDistanceCalculator():

#     def __init__(self, point_distance_calculation_method_name: ValidDistanceKinds = "euclidean", ):
#         if point_distance_calculation_method_name == "manhattan":
#             self.__point_distance_calcuation_function = euclidean
#         elif point_distance_calculation_method_name == "manhattan":
#             self.__point_distance_calcuation_function = cityblock
#         elif point_distance_calculation_method_name == "masked_euclidean":
#             self.__point_distance_calcuation_function = masked_euclidean
#         else:
#             raise ValueError(f"unknown distance function")
#         self.name = f"dist:{point_distance_calculation_method_name}"

#     def calculate_distance(self, hyp_point: Tuple[float], ref_point:Tuple[float]) -> float:
#         return self.__point_distance_calcuation_function(hyp_point, ref_point)
    
#     def __str__(self):
#         return f"dist:{self.name}"

        



def mse(trajectory1, trajectory2):
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 3))))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 3))))
    pose1_mask = np.ma.getmask(trajectory1)
    pose2_mask = np.ma.getmask(trajectory2)
    trajectory1[pose1_mask] = 0
    trajectory1[pose2_mask] = 0
    trajectory2[pose1_mask] = 0
    trajectory2[pose2_mask] = 0
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return sq_error.mean()