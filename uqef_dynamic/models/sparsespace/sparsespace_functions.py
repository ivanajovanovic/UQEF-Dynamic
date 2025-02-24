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
from uqef_dynamic.models.hbv_sask import HBVSASKModelUQ
# from uqef_dynamic.models.hbv_sask import HBVSASKStatistics

# TODO Write a wrapper for combiObject to extend from UQEF/UQEF-Dynamic Model

# ============================
# HBV
# ============================
class HBVSASKFunctionUQEF(HBVSASKModelUQ.HBVSASKModelUQ, Function):
    def __init__(self, configurationObject, inputModelDir, workingDir, single_qoi: str=None, **kwargs):
        # super().__init__(configurationObject=configurationObject, inputModelDir=inputModelDir, workingDir=workingDir, **kwargs)
        HBVSASKModelUQ.HBVSASKModelUQ.__init__(self, configurationObject=configurationObject, \
            inputModelDir=inputModelDir, workingDir=workingDir, **kwargs)
        Function.__init__(self)
        self.global_eval_counter = 0
        self.single_qoi = single_qoi
        if self.single_qoi is None:
            self.single_qoi = self.list_qoi_column[0]

        self.vector_output = True

        if isinstance(self.single_qoi, dict):
            single_qoi = self.single_qoi['qoi']
            single_gof = self.single_qoi['gof']
            self.single_qoi = single_qoi
            self.single_gof = single_gof
        else:
            self.single_gof = None

        assert self.single_qoi in self.list_qoi_column
        if self.single_gof is not None:
            # This indicates that the single_qoi is some goodness of fit measure
            # Problem might be if GoF is computed for multiple qois (i.e., list_qoi_column); then we need to know which one to use!
            # print(f"DEBUGGIN self.list_objective_function_names_qoi-{self.list_objective_function_names_qoi}")
            # print(f"DEBUGGIN self.objective_function-{self.objective_function}")
            if self.qoi == utility.GOF:
                if not self.list_objective_function_names_qoi or self.list_objective_function_names_qoi is None:
                    raise ValueError(f"Problem")
                if self.single_gof not in self.list_objective_function_names_qoi:
                    raise ValueError(f"Problem")
            elif not any(self.list_calculate_GoF):
                raise ValueError(f"Problem")
            elif not self.objective_function or self.objective_function is None:
                raise ValueError(f"Problem")
            elif self.single_gof not in self.objective_function:
                raise ValueError(f"Problem")

    def __call__(self, *args, **kwargs):
        # Example: call both __call__ methods conditionally or sequentially
        if len(args) == 1:
            return Function.__call__(self, *args)
        elif 'coordinates' in kwargs:
            return Function.__call__(self, coordinates=kwargs['coordinates'])
        elif 'i_s' in kwargs and 'parameters' in kwargs:
            return HBVSASKModelUQ.HBVSASKModelUQ.__call__(self, \
                i_s=kwargs.pop('i_s'), parameters=kwargs.pop('parameters'), **kwargs)
        else:
            raise TypeError("Invalid number of arguments")

    def output_length(self) -> int:
        if self.single_gof is not None:
            self.vector_output = False
            output_length = 1
        else:
            output_length=len(self.get_simulation_range())
        return output_length
    
    def getAnalyticSolutionIntegral(self, start, end): 
        raise NotImplementedError("Not implemented")
        # assert "Not implemented"

    def eval(self, coordinates):
        coordinates = np.array(coordinates)
        if coordinates.ndim == 1:
            assert coordinates.shape[0] == self.dim
            parameters = [coordinates,]
        elif coordinates.ndim == 2:
            assert coordinates.shape[1] == self.dim
            parameters = coordinates
        self.global_eval_counter += 1
        results = self.run(
            i_s=[self.global_eval_counter, ],
            parameters=parameters,
            raise_exception_on_model_break=True,
        )
        if self.single_gof is not None:
            _, _, df_index_parameter_gof_values, _, _ ,_  = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe(
                results_array=results, extract_only_qoi_columns=False, 
                time_column_name=self.time_column_name, 
            )
            # print(f"DEBUGGIN df_index_parameter_gof_values-{df_index_parameter_gof_values}")
            # print(f"DEBUGGIN df_index_parameter_gof_values.columns-{df_index_parameter_gof_values.columns}")
            results = df_index_parameter_gof_values.loc[df_index_parameter_gof_values[utility.QOI_ENTRY] == self.single_qoi][self.single_gof].values[0]
            # TODO Process results further to minimize...
        else:
            df_simulation_result = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe_simple(
                results_array=results, extract_only_qoi_columns=False, 
                time_column_name=self.time_column_name, 
                qoi_columns=self.qoi_column, 
                index_column_name=self.index_column_name,
            )
            df_simulation_result.sort_values(
                by=[self.index_column_name, self.time_column_name], ascending=[True, True], 
                inplace=True, kind='quicksort', na_position='last'
            )
            results = np.array(df_simulation_result[self.single_qoi].values)
            assert results.shape[0] == self.output_length()
        return results

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        ndim = np.array(coordinates).ndim
        if ndim > 2:
            return self.eval_vectorized(coordinates.reshape(-1, self.dim)).reshape(coordinates.shape[:-1] + (self.output_length(),))
        else:
            try:
                results = self.run(
                    i_s=range(coordinates.shape[0]), 
                    parameters=coordinates,
                    raise_exception_on_model_break=True,
                ) 
            except:
                raise Exception(f"Sorry but model broke and we can not proceede")
            
            if self.single_gof is not None:
                _, _, df_index_parameter_gof_values, _, _ ,_  = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe(
                    results_array=results, extract_only_qoi_columns=False, 
                    time_column_name=self.time_column_name, 
                )
                df_index_parameter_gof_values.sort_values(
                    by=self.index_column_name, ascending=True, 
                    inplace=True, kind='quicksort', na_position='last'
                )
                results = df_index_parameter_gof_values.loc[df_index_parameter_gof_values[utility.QOI_ENTRY] == self.single_qoi][self.single_gof].values
                # TODO Process results further to minimize...
            else:
                df_simulation_result = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe_simple(
                    results_array=results, extract_only_qoi_columns=False, 
                    time_column_name=self.time_column_name, 
                )
                df_simulation_result.sort_values(
                    by=[self.index_column_name, self.time_column_name], ascending=[True, True], 
                    inplace=True, kind='quicksort', na_position='last'
                )
                grouped = df_simulation_result.groupby(self.index_column_name)
                results = np.empty(shape=(coordinates.shape[0], self.output_length()), dtype=float)
                for group_key, group_df in grouped:
                    qoi_values = group_df[self.single_qoi].values
                    results[group_key,:] = np.array(qoi_values)
                assert results.shape[1] == self.output_length()
            return results 

# ============================


class HBVFunction(Function):
    def __init__(self, configurationObject, inputModelDir, workingDir, single_qoi: str=None, **kwargs):
        super().__init__()
        self.modelObject = HBVSASKModelUQ.HBVSASKModelUQ(
            configurationObject=configurationObject, inputModelDir=inputModelDir, workingDir=workingDir, **kwargs)
        self.global_eval_counter = 0
        self.dim = self.modelObject.dim
        self.single_qoi = single_qoi
        if self.single_qoi is None:
            self.single_qoi = self.modelObject.list_qoi_column[0]
        assert self.single_qoi in self.modelObject.list_qoi_column

    def output_length(self) -> int:
        return len(self.modelObject.get_simulation_range())

    def getAnalyticSolutionIntegral(self, start, end): 
        raise NotImplementedError("Not implemented")
        # assert "Not implemented"

    def eval(self, coordinates):
        coordinates = np.array(coordinates)
        if coordinates.ndim == 1:
            assert coordinates.shape[0] == self.dim
            parameters = [coordinates,]
        elif coordinates.ndim == 2:
            assert coordinates.shape[1] == self.dim
            parameters = coordinates
        self.global_eval_counter += 1
        print(f"DEBUGGING coordinates.shape - {coordinates.shape}")
        results = self.modelObject(
            i_s=[self.global_eval_counter, ],
            parameters=parameters,
            raise_exception_on_model_break=True,
        )
        df_simulation_result = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe_simple(
            results_array=results, extract_only_qoi_columns=False, 
            time_column_name=self.modelObject.time_column_name, 
            qoi_columns=self.modelObject.qoi_column, 
            index_column_name=self.modelObject.index_column_name,
        )
        df_simulation_result.sort_values(
            by=[self.modelObject.index_column_name, self.modelObject.time_column_name], ascending=[True, True], 
            inplace=True, kind='quicksort', na_position='last'
        )
        results = np.array(df_simulation_result[self.single_qoi].values)
        print(f"DEBUGGING results.shape - {results.shape}")
        print(f"DEBUGGING self.output_length() - {self.output_length()}")
        assert results.shape[0] == self.output_length()
        return results

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        ndim = np.array(coordinates).ndim
        if ndim > 2:
            print(f"DEBUGGING coordinates.shape - {coordinates.shape}")
            return self.eval_vectorized(coordinates.reshape(-1, self.dim)).reshape(coordinates.shape[:-1] + (self.output_length(),))
        else:
            try:
                results = self.modelObject(
                    i_s=range(coordinates.shape[0]), 
                    parameters=coordinates,
                    raise_exception_on_model_break=True,
                ) 
            except:
                raise Exception(f"Sorry but model broke and we can not proceede")
            df_simulation_result = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe_simple(
                results_array=results, extract_only_qoi_columns=False, 
                time_column_name=self.modelObject.time_column_name, 
            )
            df_simulation_result.sort_values(
                by=[self.modelObject.index_column_name, self.modelObject.time_column_name], ascending=[True, True], 
                inplace=True, kind='quicksort', na_position='last'
            )
            grouped = df_simulation_result.groupby(self.modelObject.index_column_name)
            print(f"DEBUGGING coordinates.shape - {coordinates.shape}")
            results = np.empty(shape=(coordinates.shape[0], self.output_length()), dtype=float)
            for group_key, group_df in grouped:
                qoi_values = group_df[self.single_qoi].values
                results[group_key,:] = np.array(qoi_values)
            print(f"DEBUGGING results.shape - {results.shape}")
            print(f"DEBUGGING self.output_length() - {self.output_length()}")
            assert results.shape[1] == self.output_length()
            return results 


# ============================
# Ishigami
# ============================
class IshigamiFunctionUQEF(IshigamiModel.IshigamiModel, Function):
    def __init__(self, configurationObject, inputModelDir=None, workingDir=None, single_qoi: str=None, **kwargs):
        # super().__init__(configurationObject=configurationObject, inputModelDir=inputModelDir, workingDir=workingDir, **kwargs)
        a = 7
        b = 0.1
        IshigamiModel.IshigamiModel.__init__(self, 
            configurationObject=configurationObject, a=a, b=b)
        Function.__init__(self)
        self.global_eval_counter = 0
        self.single_qoi = single_qoi
        if self.single_qoi is None:
            self.single_qoi = self.list_qoi_column[0]

        self.dim = 3

        self.vector_output = True

        if isinstance(self.single_qoi, dict):
            single_qoi = self.single_qoi['qoi']
            single_gof = self.single_qoi['gof']
            self.single_qoi = single_qoi
            self.single_gof = single_gof
        else:
            self.single_gof = None

        assert self.single_qoi in self.list_qoi_column
        if self.single_gof is not None:
            # This indicates that the single_qoi is some goodness of fit measure
            # Problem might be if GoF is computed for multiple qois (i.e., list_qoi_column); then we need to know which one to use!
            # print(f"DEBUGGIN self.list_objective_function_names_qoi-{self.list_objective_function_names_qoi}")
            # print(f"DEBUGGIN self.objective_function-{self.objective_function}")
            if self.qoi == utility.GOF:
                if not self.list_objective_function_names_qoi or self.list_objective_function_names_qoi is None:
                    raise ValueError(f"Problem")
                if self.single_gof not in self.list_objective_function_names_qoi:
                    raise ValueError(f"Problem")
            elif not any(self.list_calculate_GoF):
                raise ValueError(f"Problem")
            elif not self.objective_function or self.objective_function is None:
                raise ValueError(f"Problem")
            elif self.single_gof not in self.objective_function:
                raise ValueError(f"Problem")

    def __call__(self, *args, **kwargs):
        # Example: call both __call__ methods conditionally or sequentially
        if len(args) == 1:
            return Function.__call__(self, *args)
        elif 'coordinates' in kwargs:
            return Function.__call__(self, coordinates=kwargs['coordinates'])
        elif 'i_s' in kwargs and 'parameters' in kwargs:
            return IshigamiModel.IshigamiModel.__call__(self, \
                i_s=kwargs.pop('i_s'), parameters=kwargs.pop('parameters'), **kwargs)
        else:
            raise TypeError("Invalid number of arguments")

    def output_length(self) -> int:
        # return 1
        if self.single_gof is not None:
            self.vector_output = False
            output_length = 1
        else:
            output_length=len(self.get_simulation_range())
        return output_length
    
    # def getAnalyticSolutionIntegral(self, start, end): 
    #     raise NotImplementedError("Not implemented")
    #     # assert "Not implemented"

    def eval(self, coordinates):
        coordinates = np.array(coordinates)
        if coordinates.ndim == 1:
            assert coordinates.shape[0] == self.dim
            parameters = [coordinates,]
        elif coordinates.ndim == 2:
            assert coordinates.shape[1] == self.dim
            parameters = coordinates
        self.global_eval_counter += 1
        results = self.run(
            i_s=[self.global_eval_counter, ],
            parameters=parameters,
            raise_exception_on_model_break=True,
        )
        if self.single_gof is not None:
            _, _, df_index_parameter_gof_values, _, _ ,_  = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe(
                results_array=results, extract_only_qoi_columns=False, 
                time_column_name=self.time_column_name, 
            )
            # print(f"DEBUGGIN df_index_parameter_gof_values-{df_index_parameter_gof_values}")
            # print(f"DEBUGGIN df_index_parameter_gof_values.columns-{df_index_parameter_gof_values.columns}")
            results = df_index_parameter_gof_values.loc[df_index_parameter_gof_values[utility.QOI_ENTRY] == self.single_qoi][self.single_gof].values[0]
        else:
            df_simulation_result = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe_simple(
                results_array=results, extract_only_qoi_columns=False, 
                time_column_name=self.time_column_name, 
                qoi_columns=self.qoi_column, 
                index_column_name=self.index_column_name,
            )
            df_simulation_result.sort_values(
                by=[self.index_column_name, self.time_column_name], ascending=[True, True], 
                inplace=True, kind='quicksort', na_position='last'
            )
            results = np.array(df_simulation_result[self.single_qoi].values)
            assert results.shape[0] == self.output_length()
        return results

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        ndim = np.array(coordinates).ndim
        if ndim > 2:
            return self.eval_vectorized(coordinates.reshape(-1, self.dim)).reshape(coordinates.shape[:-1] + (self.output_length(),))
        else:
            try:
                results = self.run(
                    i_s=range(coordinates.shape[0]), 
                    parameters=coordinates,
                    raise_exception_on_model_break=True,
                ) 
            except:
                raise Exception(f"Sorry but model broke and we can not proceede")
            
            if self.single_gof is not None:
                _, _, df_index_parameter_gof_values, _, _ ,_  = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe(
                    results_array=results, extract_only_qoi_columns=False, 
                    time_column_name=self.time_column_name, 
                )
                df_index_parameter_gof_values.sort_values(
                    by=self.index_column_name, ascending=True, 
                    inplace=True, kind='quicksort', na_position='last'
                )
                results = df_index_parameter_gof_values.loc[df_index_parameter_gof_values[utility.QOI_ENTRY] == self.single_qoi][self.single_gof].values
            else:
                df_simulation_result = uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe_simple(
                    results_array=results, extract_only_qoi_columns=False, 
                    time_column_name=self.time_column_name, 
                )
                df_simulation_result.sort_values(
                    by=[self.index_column_name, self.time_column_name], ascending=[True, True], 
                    inplace=True, kind='quicksort', na_position='last'
                )
                grouped = df_simulation_result.groupby(self.index_column_name)
                results = np.empty(shape=(coordinates.shape[0], self.output_length()), dtype=float)
                for group_key, group_df in grouped:
                    qoi_values = group_df[self.single_qoi].values
                    results[group_key,:] = np.array(qoi_values)
                assert results.shape[1] == self.output_length()
            return results 

# ============================
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
        coordinates = np.array(coordinates)
        if coordinates.ndim == 1:
            assert coordinates.shape[0] == self.dim
            parameters = [coordinates,]
        elif coordinates.ndim == 2:
            assert coordinates.shape[1] == self.dim
            parameters = coordinates
        results = self.ishigamiModelObject(
            i_s=[self.global_eval_counter, ],
            parameters=parameters,
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

# ============================
# G-Functions
# ============================


# g-function of Sobol: https://www.sfu.ca/~ssurjano/gfunc.html
class GFunction(Function):
    def __init__(self, dim=2, **kwargs):
        super().__init__( **kwargs)
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

# ============================
# Genz Functions
# ============================

LIST_OF_GENZ_FUNCTIONS = ["corner_peak", "product_peak", "oscillatory", "gaussian", "discontinuous"]

# utility for Genz Family of Functions
b_1 = 1.5  # 9.0
b_2 = 7.2
b_3 = 1.85
b_4 = 7.03
b_5 = 20.4
b_6 = 4.3
#
GENZ_DICT = {
    "oscillatory": b_1, "product_peak": b_2, "corner_peak": b_3,
    "gaussian": b_4, "continous": b_5, "discontinuous": b_6
}


def generate_and_scale_coeff_and_weights_genz(dim, b, w_norm=1, anisotropic=False):
    coeffs = cp.Uniform(0, 1).sample(dim)  # TODO Think of using some quasiMC method
    l1 = np.linalg.norm(coeffs, 1)
    coeffs = coeffs * b / l1
    if anisotropic:
        coeffs = np.array([coeff*np.exp(i/dim) for coeff, i in zip(coeffs, range(1,dim+1))])  # less isotropic
    weights = cp.Uniform(0,1).sample(dim)
    l1 = np.linalg.norm(weights, 1)
    weights = weights * w_norm / l1
    return coeffs, weights


class CornerPeakFunction(GenzCornerPeak):
    def __init__(self, coeffs, **kwargs):
        super().__init__(coeffs)


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