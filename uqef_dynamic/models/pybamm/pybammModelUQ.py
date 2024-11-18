import copy
from distutils.util import strtobool
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pybamm
from typing import List, Optional, Dict, Any, Union
import time

# import matplotlib.pyplot as plt
# import seaborn as sns
# from functools import partial

from uqef_dynamic.models.time_dependent_baseclass.time_dependent_model import TimeDependentModel

class pybammModelUQ(TimeDependentModel):
    def __init__(self, configurationObject, inputModelDir, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, inputModelDir, workingDir, *args, **kwargs)

    def _setup(self, **kwargs):
        super()._setup(**kwargs)

    def _setup_model_related(self, **kwargs):
        if self.dict_config_model_settings and self.dict_config_model_settings is not None:
            self.options = self.dict_config_model_settings.get("options", {'surface form': 'differential'})
            self.division_factor = self.dict_config_model_settings.get("division_factor", 4)
            self.current_ooi = self.dict_config_model_settings.get("current_ooi", "Current function [A]")
            self.time_ooi = self.dict_config_model_settings.get("time_ooi", "Time [s]")
            self.voltage_ooi = self.dict_config_model_settings.get("voltage_ooi", "Voltage [V]")  # "Voltage [V]" 'Local voltage [V]', 'Battery voltage [V]', 'Terminal voltage [V]'
        else:
            self.options = {'surface form': 'differential'}
            self.division_factor = 4
            self.current_ooi = "Current function [A]"
            self.time_ooi = "Time [s]"
            self.voltage_ooi = "Voltage [V]"  # "Voltage [V]"
        # import drive cycle from file
        if "model_paths" not in self.configurationObject:
            input_file = self.inputModelDir / self.configurationObject["model_paths"].get(
                        "model_file", "US06.csv")
        else:
            input_file = self.inputModelDir / "US06.csv"
        
        self.dfn = pybamm.lithium_ion.DFN(options=self.options)
        self.model_params = self.dfn.default_parameter_values
        self.drive_cycle = pd.read_csv(
            input_file , comment="#", header=None
        ).to_numpy()
        # create interpolant
        self.current_interpolant = pybamm.Interpolant(self.drive_cycle[:, 0], self.drive_cycle[:, 1]/self.division_factor, pybamm.t)
        # set drive cycle
        self.model_params[self.current_ooi] = self.current_interpolant 
        self.t_sol =  self.drive_cycle[:, 0]
        self.current = self.drive_cycle[:, 1]/self.division_factor

        self.list_of_dates_of_interest = self.t = self.t_sol
        self.t_starting = min(self.t_sol)
        self.t_final  = max(self.t_sol)

        def j0_neg(c_e, c_s_surf, c_s_max, T):
            return 96485.3321 * 1e-10 * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5
        self.j0_neg = j0_neg
    
        def j0_pos(c_e, c_s_surf, c_s_max, T):
            return 96485.3321 * 1e-10 * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5
        self.j0_pos = j0_pos


    def _timespan_setup(self, **kwargs):
        self.spin_up_length = 0
        self.simulation_length = len(self.t_sol)
        self.start_date = self.t_starting
        self.start_date_predictions = self.t_starting
        self.end_date = self.t_final
        self.full_data_range = self.t_sol
        self.simulation_range = self.t_sol

    # def timesteps(self):
    #     return self.t_sol
    
    # def run(
    #     self, i_s: Optional[List[int]] = [0, ], 
    #     parameters: Optional[Union[Dict[str, Any], List[Any]]] = None,
    #     raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs
    #     ):
    #     results_array = super().run(i_s=i_s, parameters=parameters, raise_exception_on_model_break=raise_exception_on_model_break, *args, **kwargs)
    #     return results_array
    
    def _parameters_configuration(self, parameters, take_direct_value, *args, **kwargs):
        model_params = {}  # copy.deepcopy(self.model_params)  # self.model_params.copy()
        model_params["Negative electrode porosity"] = parameters[0]
        model_params["Positive electrode porosity"] = parameters[1]
        model_params["Negative electrode diffusivity [m2.s-1]"] = parameters[2]
        model_params["Positive electrode diffusivity [m2.s-1]"] = parameters[3]
        model_params["Negative particle radius [m]"] = parameters[4]
        model_params["Positive particle radius [m]"] = parameters[5]
        # model_params['Negative electrode exchange-current density [A.m-2]'] = self.j0_neg
        # model_params['Positive electrode exchange-current density [A.m-2]'] =self.j0_pos

        return model_params
    
    def _model_run(self, parameters_dict):
        model_params = copy.deepcopy(self.model_params)
        for elem in parameters_dict:
            model_params[elem] = parameters_dict[elem]

        sim_US06_1 = pybamm.Simulation(
        self.dfn, parameter_values=model_params, solver=pybamm.CasadiSolver(mode="fast")
        )
        sol_US06_1 = sim_US06_1.solve(self.drive_cycle[:, 0]) 
        # print(f"DEBUGGIN sol_US06_1 = {sol_US06_1}")
        return sol_US06_1
    
    def _process_model_output(self, model_output, unique_run_index, *args, **kwargs):
        sol_US06_1 = model_output
        t_sol = sol_US06_1[self.time_ooi].entries
        V_sol = sol_US06_1[self.voltage_ooi](t_sol)
        ## no idea why some simulations only yield around 570 timesteps? 
        ## => too large current density!
        # if len(V_sol) != len(drive_cycle[:,0]): 	
        #     print("Parameter vector non-convergent")
        #     print(sol_US06_1.termination)
        #     return np.array([np.NaN] * len(drive_cycle[:,0]))
        # else:
        #     return V_sol
        result_dict_inner = {self.time_column_name: t_sol, self.index_column_name: unique_run_index, 'current': self.current, self.qoi_column: V_sol} 
        result_df = pd.DataFrame(result_dict_inner)
        # print(f"DEBUGGIN result_df = {result_df}")
        return result_df
        
    def _plotting(self, result_df, unique_run_index, curr_working_dir, *args, **kwargs):
        plt.figure()
        plt.plot(result_df[self.time_column_name].values(), result_df[self.qoi_column].values())
        plt.xlabel("Time / s")
        plt.ylabel("Voltage / V")

