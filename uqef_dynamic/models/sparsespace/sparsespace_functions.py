import math
import json
import numpy as np
import pandas as pd
import  pathlib
import pickle
import scipy
import scipy.integrate as integrate
import time
from typing import List, Optional, Dict, Any, Union
 
# from uqef_dynamic.models.time_dependent_baseclass.time_dependent_model import TimeDependentModel

from sparseSpACE.Function import *

from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqef_dynamic_utils

from uqef_dynamic.models.ishigami import IshigamiModel
# from uqef_dynamic.models.ishigami import IshigamiStatistics
# from uqef_dynamic.models.hbv_sask import HBVSASKModelUQ
# from uqef_dynamic.models.hbv_sask import HBVSASKStatistics

# class IshigamiFunction(Function, IshigamiModel.IshigamiModel):
#     def __init__(self, configurationObject=None, inputModelDir=None, workingDir=None, *args, **kwargs):
#         super().__init__(configurationObject=configurationObject, inputModelDir=inputModelDir, 
#         workingDir=workingDir, *args, **kwargs)

#     def __call__(self, coordinates: Union[Tuple[float, ...], Sequence[Tuple[float]]]) -> Sequence[float]:
#         # return self.eval_vectorized(coordinates)
#         return super().__call__(coordinates)

#     def __call__(self, i_s: Optional[List[int]] = [0, ], parameters: Optional[Union[Dict[str, Any], List[Any]]] = None, 
#     raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs):
#         # return self.run(i_s=i_s, parameters=parameters, raise_exception_on_model_break=raise_exception_on_model_break, *args, **kwargs)
#         return super().__call__(i_s=i_s, parameters=parameters, raise_exception_on_model_break=raise_exception_on_model_break, *args, **kwargs)

#     def eval(self, coordinates: Tuple[float, ...]) -> float:
#         return super().eval(coordinates)

#     def eval_vectorized(self, coordinates: Sequence[Tuple[float]]) -> Sequence[float]:
#         return super().eval_vectorized(coordinates)


#     def get_analytical_sobol_indices(self, **kwargs):
#         return super().get_analytical_sobol_indices()

#     def get_expectation(self):
#         return super().get_expectation()
    
#     def get_variance(self):
#         return super().get_variance()

#     def get_first_order_sobol_indices(self):
#         return super().get_first_order_sobol_indices()

#     def getAnalyticSolutionIntegral(self, start, end):
#         return super().getAnalyticSolutionIntegral(start, end)

#     def getAnalyticSolutionIntegral(self, start: Tuple[float, ...], end: Tuple[float, ...]) -> float:
#         return super().getAnalyticSolutionIntegral(start, end)



class IshigamiFunction(Function):
    def __init__(self):
        super().__init__()
        a = 7
        b = 0.1
        self.ishigamiModelObject = IshigamiModel.IshigamiModel(
            configurationObject=None, a=a, b=b)
        self.global_eval_counter = 0
        self.dim = 3

    def output_length(self) -> int:
        return 1

    def eval(self, coordinates):
        # assert (len(coordinates) == self.dim), len(coordinates)
        self.global_eval_counter += 1
        results = self.ishigamiModelObject(
            i_s=[self.global_eval_counter, ],
            parameters=[coordinates, ],
            raise_exception_on_model_break=True,
        )
        # alternatively
        # value_of_interest = results[0][0]['result_time_series'][self.ishigamiModelObject.qoi_column].values[0]
        # return value_of_interest  # np.array(value_of_interest)
        df_simulation_result = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe_simple(
            results, extract_only_qoi_columns=True, 
        )
        result = np.array(df_simulation_result[self.ishigamiModelObject.qoi_column].values)
        return result[0]

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        # f_values = np.empty((*np.shape(coordinates)[:-1], self.output_length()))

        # print(f"DEBUG: coordinates-{coordinates}")
        # print(f"DEBUG: coordinates.shape-{coordinates.shape}")
        # print(f"DEBUG: type(coordinates)-{type(coordinates)}")
        # print(f"DEBUG: coordinates[0].shape-{coordinates[0].shape}")
        # print(f"DEBUG: type(coordinates[0])-{type(coordinates[0])}")

        ndim = coordinates.ndim
        if ndim > 2:
            # for i, coordinate in enumerate(coordinates):
            return self.eval_vectorized(coordinates.reshape(-1, self.dim)).reshape(coordinates.shape[:-1] + (self.output_length(),))
        else:
            results = self.ishigamiModelObject(
                i_s=range(coordinates.shape[0]), 
                parameters=coordinates,
                raise_exception_on_model_break=True,
            ) 
            # df_simulation_result, _, _, _, _, _ =  uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe(
            #     results, extract_only_qoi_columns=True, 
            # )
            # print(f"Original full df_simulation_result-{df_simulation_result}")
            df_simulation_result = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe_simple(
                results, extract_only_qoi_columns=True, 
            )
            result = np.array(df_simulation_result[self.ishigamiModelObject.qoi_column].values)
            return result

    # def getAnalyticSolutionIntegral(self, start, end): assert "Not implemented"

    # @staticmethod
    def get_analytical_sobol_indices(self, **kwargs):
        sobol_m_analytical, sobol_t_analytical = self.ishigamiModelObject.get_analytical_sobol_indices()
        return sobol_m_analytical, sobol_t_analytical

#################
# Set of functions from SparseSpACE
#################
def generate_and_scale_coeff_and_weights(dim, b, w_norm=1, anisotropic=False):
    coeffs = cp.Uniform(0, 1).sample(dim)  # TODO Think of using some quasiMC method
    l1 = np.linalg.norm(coeffs, 1)
    coeffs = coeffs * b / l1
    if anisotropic:
        coeffs = np.array([coeff*np.exp(i/dim) for coeff, i in zip(coeffs, range(1,dim+1))])  # less isotropic
    weights = cp.Uniform(0,1).sample(dim)
    l1 = np.linalg.norm(weights, 1)
    weights = weights * w_norm / l1
    return coeffs, weights

# g-function of Sobol: https://www.sfu.ca/~ssurjano/gfunc.html
class GFunction(Function):
    def __init__(self, dim=2, **kwargs):
        super().__init__()
        self.dim = dim
        # self.a = 0.5 * np.array(range(dim))
        # self.a = np.array([0.5 * (i - 1) for i in range(dim)])
        self.a = np.array([0.5 * i for i in range(dim)])

    def eval(self, coordinates):
        assert len(coordinates) == self.dim
        return np.prod([(abs(4.0 * coordinates[d] - 2.0) + self.a[d]) / (1.0 + self.a[d]) for d in range(self.dim)])

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        value_of_interest = np.zeros(np.shape(coordinates)[:-1])  #TODO ?
        i = 0
        for coordinate in coordinates:
            value_of_interest[i] = np.prod([(abs(4.0 * coordinate[d] - 2.0) + self.a[d]) / (1.0 + self.a[d]) for d in range(self.dim)])
            i += 1
        return value_of_interest

    # Uniform distributions in [0, 1] are required for this Function.
    def get_expectation(self):
        return 1.0

    def get_variance(self):
        mom2 = np.prod([1.0 + 1.0 / (3.0 * (1.0 + a_d) ** 2) for a_d in self.a])
        return mom2 - 1.0

    # ~ def get_first_order_sobol_indices(self):
        # This seems to be wrong
        # ~ fac = 1.0 / np.prod([1.0 / (3.0 * (1.0 + a_d) ** 2) for a_d in self.a])
        # ~ return [fac * 1.0 / (3.0 * (1.0 + self.a[d]) ** 2) for d in range(self.dim)]

        # ~ ivar = 1.0 / self.get_variance()
        # ~ return [ivar * (1.0 + 1.0 / (3 * (1.0 + self.a[i]) ** 2)) for i in range(self.dim)]

    def getAnalyticSolutionIntegral(self, start, end):
        assert all([v == 0.0 for v in start])
        assert all([v == 1.0 for v in end])
        return self.get_expectation()


class GenzGaussian(Function):
    def __init__(self, midpoint, coefficients):
        super().__init__()
        self.midpoint = midpoint
        self.coefficients = coefficients

    def eval(self, coordinates):
        dim = len(coordinates)
        assert (dim == len(self.coefficients))
        summation = 0.0
        for d in range(dim):
            summation -= (self.coefficients[d] ** 2) * (coordinates[d] - self.midpoint[d]) ** 2
        return np.exp(summation)

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        result = np.exp(-1 * np.inner((coordinates - self.midpoint) ** 2, (self.coefficients ** 2)))
        self.check_vectorization(coordinates, result)
        return result

    # def getAnalyticSolutionIntegral(self, start, end):
    #     dim = len(start)
    #     # print lowerBounds,upperBounds,coefficients, midpoints
    #     result = 1.0
    #     sqPiHalve = np.sqrt(np.pi) * 0.5
    #     for d in range(dim):
    #         result = result * (
    #                 sqPiHalve * scipy.special.erf(np.sqrt(self.coefficients[d]) * (end[d] - self.midpoint[d])) -
    #                 sqPiHalve * scipy.special.erf(
    #             np.sqrt(self.coefficients[d]) * (start[d] - self.midpoint[d]))) / np.sqrt(self.coefficients[d])
    #     return result


# corrected discontinuous based on Conrad and Marzouk
class GenzDiscontinious(Function):
    def __init__(self, coeffs, border):
        super().__init__()
        self.coeffs = coeffs
        self.border = border
        self.dim = len(coeffs)

    def eval(self, coordinates):
        result = 0
        for d in range(self.dim):
            if coordinates[d] >= self.border[d]:
                return 0.0
            result = self.coeffs[d] * coordinates[d]
        return np.exp(result)

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        result = np.zeros(np.shape(coordinates)[:-1])
        filter = np.all(coordinates < self.border, axis=-1)
        result[filter] = np.exp(np.inner(coordinates[filter], self.coeffs))
        self.check_vectorization(coordinates, result)
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1
        end = list(end)
        for d in range(self.dim):
            if start[d] >= self.border[d]:
                return 0.0
            else:
                end[d] = min(end[d], self.border[d])
                result *= (np.exp(self.coeffs[d] * end[d]) - np.exp(self.coeffs[d] * start[d])) / self.coeffs[d]
        return result