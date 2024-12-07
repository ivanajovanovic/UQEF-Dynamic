import pandas as pd
from pathlib import Path
import time
from typing import List, Optional, Dict, Any, Union
import numpy as np

from uqef_dynamic.models.time_dependent_baseclass.time_dependent_model import TimeDependentModel

class simpleOscillatorUQ(TimeDependentModel):

    def __init__(self, configurationObject, inputModelDir=None, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, inputModelDir, workingDir, *args, **kwargs)
    
    def _setup(self, **kwargs):
        super()._setup(**kwargs)

    def _timespan_setup(self, **kwargs):
        t_sol = kwargs.get('t_sol', self.dict_config_time_settings.get('t_sol', None))
        self.t_interest = kwargs.get('t_interest', self.dict_config_time_settings.get('t_interest', None))
        if self.t_interest == "None":
            self.t_interest = None
        t_start = kwargs.get('t_start', self.dict_config_time_settings.get('t_start', 0))
        t_end = kwargs.get('t_end', self.dict_config_time_settings.get('t_end', 10))
        N = kwargs.get('N', self.dict_config_time_settings.get('N', 200))
        if t_sol is None:
            self.t_sol = np.linspace(t_start, t_end, N)
            self.t_start = t_start
            self.t_end = t_end
            self.N = N
            if self.t_interest is not None and not self.t_interest in self.t_sol:
                print(f"WARNING: t_interest = {self.t_interest} is not in the t_sol = {self.t_sol}. Instead a middle point is selected.")
                self.t_interest = self.t_sol[int(len(self.t_sol) // 2)]
        else:
            self.t_sol = t_sol
            self.t_start = t_sol[0]
            self.t_end = t_sol[-1]
            self.N = len(t_sol)
        if self.t_interest is not None and self.t_interest > len(self.t_sol):
            self.t_interest = self.t_sol[int(len(self.t_sol) // 2)]
        # print(f"DEBUGGING: t_sol = {self.t_sol}")

    # def _setup_model_related(self, **kwargs):
    #     pass

    def _parameters_configuration(self, parameters, take_direct_value, *args, **kwargs):
        """
        This function should return a dictionary of parameters to be used in the model.
        This is the first argument of the model_run function.

        Note: it should contain only uncertain parameters.
        """
        parameters_dict = {}
        parameters_dict["alpha"] = parameters[0]
        parameters_dict["beta"] = parameters[1]
        parameters_dict["l"] = parameters[2]
        return parameters_dict

    def _model_run(self, parameters_dict, *args, **kwargs):
        temp_results = parameters_dict["l"]*np.exp(-parameters_dict["alpha"]*self.t_sol)*(np.cos(parameters_dict["beta"]*self.t_sol)+parameters_dict["alpha"]/parameters_dict["beta"]*np.sin(parameters_dict["beta"]*self.t_sol))
        return temp_results

    def _process_model_output(self, model_output, unique_run_index, *args, **kwargs):
        assert len(model_output) == len(self.t_sol), f"model_output = {model_output}, t_sol = {self.t_sol}"
        result_dict_inner = {self.time_column_name: self.t_sol, self.index_column_name: unique_run_index, self.qoi_column: model_output} 
        result_df = pd.DataFrame(result_dict_inner)
        if self.t_interest is not None:
            result_df = result_df[result_df[self.time_column_name]==self.t_interest]
        # print(f"DEBUGGIN result_df = {result_df}")
        return result_df
        