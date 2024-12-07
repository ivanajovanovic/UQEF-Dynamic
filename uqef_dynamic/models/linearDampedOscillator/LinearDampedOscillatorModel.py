from functools import reduce
import pandas as pd
from pathlib import Path
import numpy as np
import time
from typing import List, Optional, Dict, Any, Union
from scipy.integrate import odeint

from uqef_dynamic.models.time_dependent_baseclass.time_dependent_model import TimeDependentModel, TimeDependentModelConfig
from uqef_dynamic.utils import utility


def model(w, t, p):
    x1, x2 		= w
    c, k, f, w 	= p
    f = [x2, f*np.cos(w*t) - k*x1 - c*x2]
    return f


def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t):
    sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)
    return sol


class LinearDampedOscillatorModelSetUp(TimeDependentModelConfig):
    def __init__(self, configurationObject: dict, deep_copy=False, *args: Any, **kwargs: Any):
        super().__init__(configurationObject, deep_copy, *args, **kwargs)


class LinearDampedOscillatorModel(TimeDependentModel):
    def __init__(self, configurationObject, inputModelDir=None, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, inputModelDir, workingDir, *args, **kwargs)
        # self.modelConfig = LinearDampedOscillatorModelSetUp(configurationObject, deep_copy=False, *args, **kwargs)
    
    def _setup(self, **kwargs):
        super()._setup(**kwargs)

    def _setup_model_related(self, **kwargs):
        """
        This function should be used to setup the (general) model related parameters, e.g., those that are static 
        or that are not dependent on the specific model run.

        It is called in the _setup function. kwargs are the same as in the _setup function and constructor.
        """
        self.model = model
        self.atol = kwargs.get("atol", 1e-10)
        self.rtol = kwargs.get("rtol", 1e-10) 
        self.default_par_info_dict = {}
        # TODO - maybe read the default values from the configuration file  
        self.default_par_info_dict["c"] = {"lower": 0.08, "upper": 0.12, "default": 0.1}
        self.default_par_info_dict["k"] = {"lower": 0.03, "upper": 0.04, "default": 0.035}
        self.default_par_info_dict["f"] = {"lower": 0.08, "upper": 0.12, "default": 0.1}
        self.default_par_info_dict["w"] = {"lower": 0.0, "upper": 1.0, "default": 1.0}
        self.default_par_info_dict["y0"] = {"lower": 0.45, "upper": 0.55, "default": 0.5}
        self.default_par_info_dict["y1"] = {"lower": -0.05, "upper": 0.05, "default": 0.0}
        self.c_default = self.default_par_info_dict["c"]["default"]
        self.k_default = self.default_par_info_dict["k"]["default"]
        self.f_default = self.default_par_info_dict["f"]["default"]
        self.w_default = self.default_par_info_dict["w"]["default"]
        self.y0_default = self.default_par_info_dict["y0"]["default"]
        self.y1_default = self.default_par_info_dict["y1"]["default"]
        self.init_cond =  self.y0_default, self.y1_default

    def _timespan_setup(self, **kwargs):
        t_sol = kwargs.get('t_sol', self.dict_config_time_settings.get('t_sol', None))
        self.t_interest = kwargs.get('t_interest', self.dict_config_time_settings.get('t_interest', None))
        if self.t_interest == "None":
            self.t_interest = None

        if t_sol is None:
            self.t_start = kwargs.get('t_start', self.dict_config_time_settings.get('t_start', 0))
            self.t_end = kwargs.get('t_end', self.dict_config_time_settings.get('t_end', 20))
            self.dt = kwargs.get('t_end', self.dict_config_time_settings.get('dt', 0.01))
            grid_size = int(self.t_end / self.dt) + 1
            self.t_sol = self.t = [i * self.dt for i in range(grid_size)]
            if self.t_interest is not None and not self.t_interest in self.t_sol:
                print(f"WARNING: t_interest = {self.t_interest} is not in the t_sol = {self.t_sol}. Instead a middle point is selected.")
                self.t_interest = self.t_sol[int(len(self.t_sol) // 2)]
        else:
            self.t_sol = self.t = t_sol
            self.t_start = t_sol[0]
            self.t_end = t_sol[-1]
            self.dt = (self.t_end - self.t_start) / (len(self.t_sol))
        if self.t_interest is not None and self.t_interest > len(self.t_sol):
            self.t_interest = self.t_sol[int(len(self.t_sol) // 2)]

    # def _parameters_configuration(self, parameters, take_direct_value, *args, **kwargs):
    #     """
    #     This function should return a dictionary of parameters to be used in the specific (i.e., single) 
    #     model run. This is the first argument of the model_run function.

    #     Note: it should contain only uncertain parameters.
    #     """
    #     parameters_dict = utility.configuring_parameter_values(
    #         parameters=parameters,
    #         configurationObject=self.configurationObject,
    #         default_par_info_dict=self.default_par_info_dict,
    #         take_direct_value=take_direct_value
    #         )
    #     return parameters_dict

    def _model_run(self, parameters_dict, *args, **kwargs):
        c = parameters_dict.get("c", self.c_default)
        k = parameters_dict.get("k", self.k_default)
        f = parameters_dict.get("f", self.f_default)
        w = parameters_dict.get("w", self.w_default)
        y0 = parameters_dict.get("y0", self.y0_default)
        y1 = parameters_dict.get("y1", self.y1_default)
        args = c, k, f, w
        init_cond = y0, y1
        value_of_interest = discretize_oscillator_odeint(
            self.model, self.atol, self.rtol, init_cond, args, self.t)
        return value_of_interest

    def _process_model_output(self, model_output, unique_run_index, *args, **kwargs):
        t_sol = self.t_sol

        assert len(model_output) == len(t_sol), f"model_output = {model_output}, t_sol = {t_sol}"

        list_single_result_df = []
        for single_qoi_column in self.list_qoi_column:
            if single_qoi_column=="displacement":
                result_dict_inner = {self.time_column_name: t_sol, self.index_column_name: unique_run_index, single_qoi_column: model_output[:,0]} 
            elif single_qoi_column=="velocity":
                result_dict_inner = {self.time_column_name: t_sol, self.index_column_name: unique_run_index, single_qoi_column: model_output[:,1]} 
            single_result_df = pd.DataFrame(result_dict_inner)
            if self.t_interest is not None:
                single_result_df = single_result_df[single_result_df[self.time_column_name]==self.t_interest]
            list_single_result_df.append(single_result_df)
        if len(list_single_result_df)>1:
            result_df = reduce(lambda left, right: pd.merge(left, right, on=[self.time_column_name, self.index_column_name ], how='inner'), list_single_result_df)
        elif len(list_single_result_df)==1:
            result_df = list_single_result_df[0]
        else:
            result_df = None
        # print(f"DEBUGGIN result_df = {result_df}")
        return result_df
