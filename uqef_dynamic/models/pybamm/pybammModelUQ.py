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

# from uqef.model import Model
from uqef_dynamic.utils import utility
from uqef_dynamic.models.time_dependent_baseclass.time_dependent_model import TimeDependentModel

class pybammModelUQ(TimeDependentModel):
    def __init__(self, configurationObject, inputModelDir, workingDir=None, *args, **kwargs):
        super(pybammModelUQ, self).__init__(configurationObject, inputModelDir, workingDir, *args, **kwargs)
        # self.configurationObject = None
        # if configurationObject is None:
        #     pass
        # elif isinstance(configurationObject, dict):
        #     self.configurationObject = configurationObject
        # else:
        #     with open(configurationObject) as f:
        #         self.configurationObject = json.load(f)

        # self.inputModelDir = Path(inputModelDir)  
        # # "~/.local/lib/python3.10/site-packages/pybamm/input/drive_cycles"
        # # "/work/ga45met/anaconda3/envs/py3115_uq/lib/python3.11/site-packages/pybamm/input/drive_cycles/US06.csv"
        # # "/dss/dsshome1/lxc0C/ga45met2/.conda/envs/uq_env/lib/python3.7/site-packages/pybamm/input/drive_cycles/US06.csv"
        
        # if workingDir is None:
        #     workingDir = self.inputModelDir
        # self.workingDir = Path(workingDir)
        # self.workingDir.mkdir(parents=True, exist_ok=True)

        # #####################################
        # # these set of control variables are for UQEF & UQEF-Dynamic framework...
        # #####################################

        # self.uq_method = kwargs.get('uq_method', None)
        # self.raise_exception_on_model_break = kwargs.get('raise_exception_on_model_break', False)
        # if self.uq_method is not None and self.uq_method == "sc":  # always break when running gPCE simulation
        #     self.raise_exception_on_model_break = True
        # self.disable_statistics = kwargs.get('disable_statistics', False)
        # # if not self.disable_statistics:
        # #     self.writing_results_to_a_file = False

        # self._setup(**kwargs)

    def _setup(self, **kwargs):
        super(pybammModelUQ, self)._setup(**kwargs)

    def _setup_model_related(self, **kwargs):
        if self.dict_config_model_settings and self.dict_config_model_settings is not None:
            self.options = self.dict_config_model_settings.get("options", {'surface form': 'differential'})
            self.division_factor = self.dict_config_model_settings.get("division_factor", 4)
            self.current_ooi = self.dict_config_model_settings.get("current_ooi", "Current function [A]")
            self.time_ooi = self.dict_config_model_settings.get("time_ooi", "Time [s]")
            self.voltage_ooi = self.dict_config_model_settings.get("voltage_ooi", "Battery voltage [V]")  # "Voltage [V]" 'Local voltage [V]', 'Battery voltage [V]', 'Terminal voltage [V]'
        else:
            self.options = {'surface form': 'differential'}
            self.division_factor = 4
            self.current_ooi = "Current function [A]"
            self.time_ooi = "Time [s]"
            self.voltage_ooi = "Battery voltage [V]"  # "Voltage [V]"
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

    def _timespan_setup(self, **kwargs):
        self.spin_up_length = 0
        self.simulation_length = len(self.t_sol)
        self.start_date = self.t_starting
        self.start_date_predictions = self.t_starting
        self.end_date = self.t_final
        self.full_data_range = self.t_sol
        self.simulation_range = self.t_sol

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
        model_params = {}  # copy.deepcopy(self.model_params)  # self.model_params.copy()
        model_params["Negative electrode porosity"] = parameters[0]
        model_params["Positive electrode porosity"] = parameters[1]
        model_params["Negative electrode diffusivity [m2.s-1]"] = parameters[2]
        model_params["Positive electrode diffusivity [m2.s-1]"] = parameters[3]
        model_params["Negative particle radius [m]"] = parameters[4]
        model_params["Positive particle radius [m]"] = parameters[5]
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
        result_dict_inner = {utility.TIME_COLUMN_NAME: t_sol, utility.INDEX_COLUMN_NAME: unique_run_index, 'current': self.current, self.qoi_column: V_sol} 
        result_df = pd.DataFrame(result_dict_inner)
        # print(f"DEBUGGIN result_df = {result_df}")
        return result_df
    
    def _transform_model_output(self, model_output_df, *args, **kwargs):
        pass 
    
    def _plotting(self, result_df, unique_run_index, curr_working_dir, *args, **kwargs):
        plt.figure()
        plt.plot(result_df[utility.TIME_COLUMN_NAME].values(), result_df[self.qoi_column].values())
        plt.xlabel("Time / s")
        plt.ylabel("Voltage / V")


    # =================================================

    # def run(
        #     self, i_s: Optional[List[int]] = [0, ], 
        #     parameters: Optional[Union[Dict[str, Any], List[Any]]] = None,
        #     raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs
        #     ):

        # if raise_exception_on_model_break is None:
        #     raise_exception_on_model_break = self.raise_exception_on_model_break
        # take_direct_value = kwargs.get("take_direct_value", False)
        # if self.uq_method == "ensemble":
        #     take_direct_value = True
        # take_direct_value = True
        # createNewFolder = kwargs.get("createNewFolder", False)
        # deleteFolderAfterwards = kwargs.get("deleteFolderAfterwards", True)
        # plotting = kwargs.get("plotting", self.plotting)

        # results_array = []
        # parameter = None  # Initialize parameter here
        # for ip in range(0, len(i_s)):  # for each piece of work
        #     start = time.time()

        #     unique_run_index = i_s[ip]  # i is unique index run

        #     if parameters is not None:
        #         parameter = parameters[ip]
        #     else:
        #         parameter = None  # an unaltered run will be executed

        #     id_dict = {"index_run": unique_run_index}

        #     # this indeed represents the number of parameters considered to be uncertain, later on parameters_dict might
        #     # be extanded with fixed parameters that occure in configurationObject
        #     if parameter is None:
        #         number_of_uncertain_params = 0
        #     elif isinstance(parameter, dict):
        #         number_of_uncertain_params = len(list(parameter.keys()))
        #     else:
        #         number_of_uncertain_params = len(parameter)

        #     # set param values
        #     model_params = copy.deepcopy(self.model_params)  # self.model_params.copy()
        #     model_params["Negative electrode porosity"] = parameter[0]
        #     model_params["Positive electrode porosity"] = parameter[1]
        #     model_params["Negative electrode diffusivity [m2.s-1]"] = parameter[2]
        #     model_params["Positive electrode diffusivity [m2.s-1]"] = parameter[3]
        #     model_params["Negative particle radius [m]"] = parameter[4]
        #     model_params["Positive particle radius [m]"] = parameter[5]
        #     parameters_dict = model_params

        #     try:
        #         sim_US06_1 = pybamm.Simulation(
        #         self.dfn, parameter_values=model_params, solver=pybamm.CasadiSolver(mode="fast")
        #         )
        #         sol_US06_1 = sim_US06_1.solve(self.drive_cycle[:, 0])
        #         t_sol = sol_US06_1["Time [s]"].entries
        #         V_sol = sol_US06_1["Voltage [V]"](t_sol)
        #     except:
        #         print("Parameter vector non-convergent")
        #         print(parameter)
        #         t_sol = np.array([np.NaN] * len(self.drive_cycle[:,0]))
        #         V_sol = np.array([np.NaN] * len(self.drive_cycle[:,0]))
        #         # return np.array([np.NaN] * len(drive_cycle[:,0]))
        #         index_run_and_parameters_dict = {**id_dict, **parameters_dict, "success": False}
        #         result_df = None
        #     else:
        #         index_run_and_parameters_dict = {**id_dict, **parameters_dict, "success": True}

        #         result_dict_inner = {utility.TIME_COLUMN_NAME: t_sol, "Index_run": unique_run_index, 'current': self.current, 'V_sol': V_sol} 
        #         result_df = pd.DataFrame(result_dict_inner)

        #         if plotting:
        #             # plt.figure()
        #             # plt.plot(self.t_sol, self.drive_cycle[:, 1]/self.division_factor)
        #             # plt.xlabel("Time / s")
        #             # plt.ylabel("Current / A")

        #             plt.figure()
        #             plt.plot(t_sol, V_sol)
        #             plt.xlabel("Time / s")
        #             plt.ylabel("Voltage / V")
            
        #     end = time.time()
        #     runtime = end - start

        #     result_dict = {
        #         "run_time": runtime,
        #         "result_time_series":result_df,
        #         "parameters_dict": index_run_and_parameters_dict
        #     }

        #     results_array.append((result_dict, runtime))

        # return results_array



