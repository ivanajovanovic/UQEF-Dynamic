import pandas as pd
from pathlib import Path
import time
from typing import List, Optional, Dict, Any, Union
import numpy as np

from uqef_dynamic.models.time_dependent_baseclass.time_dependent_model import TimeDependentModel

class simpleOscilatorUQ(TimeDependentModel):

    def __init__(self, configurationObject, inputModelDir, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, inputModelDir, workingDir, *args, **kwargs)
    
    def _setup(self, **kwargs):
        super()._setup(**kwargs)

    def _timespan_setup(self, **kwargs):
        t_sol = kwargs.get('t_sol', self.dict_config_time_settings.get('t_sol', None))
        t_start = kwargs.get('t_start', self.dict_config_time_settings.get('t_start', 0))
        t_end = kwargs.get('t_end', self.dict_config_time_settings.get('t_end', 10))
        N = kwargs.get('N', self.dict_config_time_settings.get('N', 200))
        if t_sol is None:
            self.t_sol = np.linspace(t_start, t_end, N)
            self.t_start = t_start
            self.t_end = t_end
            self.N = N
        else:
            self.t_sol = t_sol
            self.t_start = t_sol[0]
            self.t_end = t_sol[-1]
            self.N = len(t_sol)
        # print(f"DEBUGGING: t_sol = {self.t_sol}")

    def _setup_model_related(self, **kwargs):
        pass

    def prepare(self, *args, **kwargs):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter
    
    def timesteps(self):
        return self.t_sol

    def run(
        self, i_s: Optional[List[int]] = [0, ], 
        parameters: Optional[Union[Dict[str, Any], List[Any]]] = None,
        raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs
        ):
        results_array = super().run(i_s=i_s, parameters=parameters, raise_exception_on_model_break=raise_exception_on_model_break, *args, **kwargs)
        return results_array

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
        result_dict_inner = {self.time_column_name: self.t_sol, self.index_column_name: unique_run_index, self.qoi_column: model_output} 
        result_df = pd.DataFrame(result_dict_inner)
        # print(f"DEBUGGIN result_df = {result_df}")
        return result_df
    
    def _transform_model_output(self, model_output_df, *args, **kwargs):
        pass
    