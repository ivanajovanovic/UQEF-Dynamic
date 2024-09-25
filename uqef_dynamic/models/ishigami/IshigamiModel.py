import math
import json
import numpy as np
import pandas as pd
import time
from typing import List, Optional, Dict, Any, Union
 
from uqef_dynamic.models.time_dependent_baseclass.time_dependent_model import TimeDependentModel

def model(p, x):
    a, b = p
    x1, x2, x3 = x
    f = np.sin(x1) + a * math.pow(np.sin(x2), 2) + b * math.pow(x3, 4) * np.sin(x1)
    return f


def model_2d(p, x):
    a, b = p
    x1, x3 = x
    x2 = 0
    # f = np.sin(q[:, 0]) + self.a * (np.sin(q[:, 1]) ** 2) + (self.b * np.sin(q[:, 0]) * (q[:, 2] ** 4))
    f = np.sin(x1) + a * math.pow(np.sin(x2), 2) + b * math.pow(x3, 4) * np.sin(x1)
    return f

def ishigami_func(coordinates, a_model_param=7, b_model_param=0.1):
    x1, x2, x3 = coordinates
    return math.sin(x1) + a_model_param * (math.sin(x2)) ** 2 + b_model_param * x3 ** 4 * math.sin(x1)


class IshigamiModelSetUp():
    def __init__(self, configurationObject):
        pass


class IshigamiModel(TimeDependentModel):
    def __init__(self, configurationObject, inputModelDir=None, workingDir=None, *args, **kwargs):
        # Model.__init__(self)
        super().__init__(configurationObject, inputModelDir, workingDir, *args, **kwargs)

        # if isinstance(configurationObject, dict) or configurationObject is None:
        #     self.configurationObject = configurationObject
        # else:
        #     with open(configurationObject) as f:
        #         self.configurationObject = json.load(f)

    def _setup_model_related(self, **kwargs):
        if "a" in kwargs:
            self.a = kwargs['a']
        else:
            try:
                self.a = self.configurationObject["other_model_parameters"]["a"]
            except KeyError:
                self.a = 7

        if "b" in kwargs:
            self.b = kwargs['b']
        else:
            try:
                self.b = self.configurationObject["other_model_parameters"]["b"]
            except KeyError:
                self.b = 0.1

    def _timespan_setup(self, **kwargs):
        self.t = self.t_sol = [0.0, ]
        self.t_starting = self.t_final = self.t_interest = 0.0

    # def timesteps(self):
    #     return self.t

    # def prepare(self, *args, **kwargs):
    #     pass

    # def assertParameter(self, parameter):
    #     pass

    # def normaliseParameter(self, parameter):
    #     return parameter

    def run(
        self, i_s: Optional[List[int]] = [0, ], 
        parameters: Optional[Union[Dict[str, Any], List[Any]]] = None,
        raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs
        ):
        results_array = super().run(i_s=i_s, parameters=parameters, raise_exception_on_model_break=raise_exception_on_model_break, *args, **kwargs)
        return results_array

    # def run(self, i_s, parameters, *args, **kwargs):

    #     # print(f"[Ishigami Model] {i_s}: paramater: {parameters}")

    #     results = []

    #     for ip in range(0, len(i_s)):
    #         start = time.time()
    #         i = i_s[ip]
    #         parameter = parameters[ip]

    #         args = self.a, self.b
    #         x = parameter[0], parameter[1], parameter[2]
    #         f_result = model(args, x)

    #         end = time.time()
    #         runtime = end - start

    #         results.append((f_result, runtime))

    #     return results

    def _parameters_configuration(self, parameters, take_direct_value, *args, **kwargs):
        """
        This function should return a dictionary of parameters to be used in the model.
        This is the first argument of the model_run function.

        Note: it should contain only uncertain parameters.
        """
        parameters_dict = {}
        parameters_dict["x1"] = parameters[0]
        parameters_dict["x2"] = parameters[1]
        parameters_dict["x3"] = parameters[2]
        return parameters_dict

    def _model_run(self, parameters_dict, *args, **kwargs):
        temp_results = math.sin(parameters_dict["x1"]) + \
            self.a * math.pow(math.sin(parameters_dict["x2"]), 2) + \
                self.b * math.pow(parameters_dict["x3"], 4) * math.sin(parameters_dict["x1"])
        return temp_results
        # pass

    def _process_model_output(self, model_output, unique_run_index, *args, **kwargs):
        result_dict_inner = {self.time_column_name: self.t_sol, self.index_column_name: unique_run_index, self.qoi_column: model_output} 
        result_df = pd.DataFrame(result_dict_inner)
        # print(f"DEBUGGIN result_df = {result_df}")
        return result_df
    
    def _transform_model_output(self, model_output_df, *args, **kwargs):
        pass

    def get_analytical_sobol_indices(self, type="both"):
        """
        Returns the analytical Sobol indices for the Ishigami function.
        :param type: str, optional
            The type of Sobol indices to return. Options are "both", "main", "total".
        :return: np.array or tuple of two np.arrays where the first entry are main indices and the second entry are
        total Sobol indices.
        """
        vm1 = 0.5*(1+(self.b*np.pi**4)/5)**2
        vm1 = (self.b*np.pi**4)/5 + ((self.b**2)*np.pi**8)/50 + 0.5  # Sudret!
        vm2 = self.a**2/8
        vm3 = 0.0
        vm12 = 0.0
        vm23 = 0.0
        vm13 = 8 * self.b**2 * np.pi ** 8 / 225
        # vm13 = 19 * self.b**2 * np.pi ** 8 / 450  # Ravi!
        vm123 = 0.0

        v = self.a**2/8 + (self.b*np.pi**4)/5 + (self.b**2*np.pi**8)/18 + 0.5
        v = vm1 + vm2 + vm13
        assert np.abs(v - (vm1 + vm2 + vm13)) < 0.001

        if type=="both":
            sm1 = vm1/v
            sm2 = vm2/v
            sm3 = 0.0  # vm3/v

            st1 = (vm1 + vm13)/v
            st2 = vm2/v
            st3 = vm13/v
            # Sobol_m_analytical = np.array([0.3138/0.3139, 0.4424/0.4424, 0.0/0.0000], dtype=np.float64)
            sobol_m_analytical = np.array([sm1, sm2, sm3], dtype=np.float64)

            # Sobol_t_analytical = np.array([0.5574/0.5576, 0.4424/0.4424, 0.2436/0.2437], dtype=np.float64)
            sobol_t_analytical = np.array([st1, st2, st3], dtype=np.float64)
            return sobol_m_analytical, sobol_t_analytical
        elif type=="main" or type=="m":
            sm1 = vm1/v
            sm2 = vm2/v
            sm3 = 0.0
            return np.array([sm1, sm2, sm3], dtype=np.float64)
        elif type=="total" or type=="t":
            st1 = (vm1 + vm13)/v
            st2 = vm2/v
            st3 = vm13/v
            return np.array([st1, st2, st3], dtype=np.float64)
        else:
            raise ValueError(f"Unknown type {type}.")


# Ravi's code
# class Ishigami():
#
#     def __init__(self, a, b, lower=-np.pi, upper=np.pi):
#         self.lower, self.upper, self.dim = lower, upper, 3
#         self.a, self.b = a, b
#         self.num_eval_lf, self.num_eval_hf = 0, 0
#
#     def transform_coordinates(self, x):
#         return x * (self.upper - self.lower) + self.lower
#
#     def hf(self, x): # this is the main Ishigami function
#         temp = np.atleast_2d(x)
#         if temp.shape[1] != self.dim:
#             temp = temp.T
#         q = self.transform_coordinates(temp)
#         f = np.sin(q[:, 0]) + self.a * (np.sin(q[:, 1])**2) + (self.b * np.sin(q[:, 0]) * (q[:, 2]**4))
#         self.num_eval_hf += len(temp)
#         return f
#
#     def lf(self, x):
#     	# This is just a dummy function to test multi-fidelity Ishigami toy problems :D
#         temp = np.atleast_2d(x)
#         if temp.shape[1] != self.dim:
#             temp = temp.T
#         q = self.transform_coordinates(temp)
#         a, b = self.a +0.1, self.b
#         f =(np.sin(q[:, 0]) + a * (np.sin(q[:, 1])**2) + (b * np.sin(q[:, 0]) * (q[:, 2]**4)) ) + 0.02
#         self.num_eval_lf += len(temp)
#         return f
#
#     def calculate_statistics(self):
#         mean = self.a * 0.5
#         D1 = (self.b * np.pi**4 / 5) + (self.b**2 * np.pi**8 / 50) + 0.5
#         D2 = self.a**2 / 8
#         D13 = 19 * self.b**2 * np.pi**8 / 450
#         D = D1 + D2 + D13
#         assert np.abs(D - (D1 + D2 + D13)) < 0.001
#         local_sobol = [D1/D, D2/D, 0.]
#         global_sobol = [(D1+D13)/D, D2/D, D13/D]
#         return mean, D, local_sobol, global_sobol


def ishigami_func_vec(coordinates, a_model_param=7, b_model_param=0.1):
    x1, x2, x3 = coordinates
    x1_is_array = isinstance(x1, list) or isinstance(x1, np.ndarray)
    x2_is_array = isinstance(x2, list) or isinstance(x2, np.ndarray)
    x3_is_array = isinstance(x3, list) or isinstance(x3, np.ndarray)

    if x1_is_array and x2_is_array and x3_is_array:
        x1_array = np.array(x1)
        x2_array = np.array(x2)
        x3_array = np.array(x3)
        result_vec = np.empty_like(x1_array)
        i=0
        for x1_loc, x2_loc, x3_loc in zip(x1_array, x2_array, x3_array):
            result_vec[i] = ishigami_func((x1_loc, x2_loc, x3_loc), a_model_param=a_model_param, b_model_param=b_model_param)
            i+=1
    elif x1_is_array and x2_is_array and isinstance(x3, (int, float)):
        x1_array = np.array(x1)
        x2_array = np.array(x2)
        result_vec = np.empty_like(x1_array)
        i=0
        for x1_loc, x2_loc in zip(x1_array, x2_array):
            result_vec[i] = ishigami_func((x1_loc, x2_loc, x3), a_model_param=a_model_param, b_model_param=b_model_param)
            i+=1
    elif x1_is_array and isinstance(x2, (int, float)) and x3_array:
        x1_array = np.array(x1)
        x3_array = np.array(x3)
        result_vec = np.empty_like(x1_array)
        i=0
        for x1_loc, x3_loc in zip(x1_array, x3_array):
            result_vec[i] = ishigami_func((x1_loc, x2, x3_loc), a_model_param=a_model_param, b_model_param=b_model_param)
            i+=1
    elif isinstance(x1, (int, float)) and x2_is_array and x3_is_array:
        x2_array = np.array(x2)
        x3_array = np.array(x3)
        result_vec = np.empty_like(x2_array)
        i=0
        for x2_loc, x3_loc in zip(x2_array, x3_array):
            result_vec[i] = ishigami_func((x1, x2_loc, x3_loc), a_model_param=a_model_param, b_model_param=b_model_param)
            i+=1
    elif isinstance(x1, (int, float)) and x2_is_array and isinstance(x3, (int, float)):
        x2_array = np.array(x2)
        result_vec = np.empty_like(x2_array)
        i=0
        for x2_loc in x2_array:
            result_vec[i] = ishigami_func((x1, x2_loc, x3), a_model_param=a_model_param, b_model_param=b_model_param)
            i+=1
    elif isinstance(x1, (int, float)) and isinstance(x2, (int, float)) and x3_is_array:
        x3_array = np.array(x3)
        result_vec = np.empty_like(x3_array)
        i=0
        for x3_loc in x3_array:
            result_vec[i] = ishigami_func((x1, x2, x3_loc), a_model_param=a_model_param, b_model_param=b_model_param)
            i+=1
    elif isinstance(x1, (int, float)) and isinstance(x2, (int, float)) and isinstance(x3, (int, float)):
        result_vec = ishigami_func((x1, x2, x3), a_model_param=a_model_param, b_model_param=b_model_param)
    return result_vec