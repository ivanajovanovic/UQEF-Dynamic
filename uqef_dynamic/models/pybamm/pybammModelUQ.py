import copy
from distutils.util import strtobool
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pybamm
import time
from typing import List, Optional, Dict, Any, Union

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
            #self.options = self.dict_config_model_settings.get("options", {'surface form': 'differential'})
            self.division_factor = self.dict_config_model_settings.get("division_factor", 100.0) # 4
            self.current_ooi = self.dict_config_model_settings.get("current_ooi", "Current function [A]")
            self.time_ooi = self.dict_config_model_settings.get("time_ooi", "Time [s]")
            self.voltage_ooi = self.dict_config_model_settings.get("voltage_ooi", "Voltage [V]")  # "Voltage [V]" 'Local voltage [V]', 'Battery voltage [V]', 'Terminal voltage [V]'
        else:
            #self.options = {'surface form': 'differential'}
            self.division_factor = 100.0 #4
            self.current_ooi = "Current function [A]"
            self.time_ooi = "Time [s]"
            self.voltage_ooi = "Voltage [V]"  # "Voltage [V]"
        # import drive cycle from file
        if "model_paths" not in self.configurationObject:
            input_file = self.inputModelDir / self.configurationObject["model_paths"].get(
                        "model_file", "US06.csv")
        else:
            input_file = self.inputModelDir / "US06.csv"
        
        self.dfn = pybamm.lithium_ion.DFN() #pybamm.lithium_ion.DFN(options=self.options)
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
    
    def _parameters_configuration_6d(self, parameters, take_direct_value, *args, **kwargs):
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

    def _parameters_configuration_24d(self, parameters, take_direct_value, *args, **kwargs):
        model_params = {}  # copy.deepcopy(self.model_params)  # self.model_params.copy()
        
        def j0_neg(c_e, c_s_surf, c_s_max, T):
            return 96485.3321 * parameters[8] * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5

        def j0_pos(c_e, c_s_surf, c_s_max, T):
            return 96485.3321 * parameters[15] * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5

        model_params['Cation transference number'] = parameters[0]
        model_params['Electrolyte conductivity [S.m-1]'] = parameters[1]
        model_params['Electrolyte diffusivity [m2.s-1]'] = parameters[2]
        model_params['Initial concentration in electrolyte [mol.m-3]'] = parameters[3]

        model_params['Negative electrode Bruggeman coefficient (electrode)'] = parameters[4]
        model_params['Negative electrode Bruggeman coefficient (electrolyte)'] = parameters[4]

        model_params['Negative electrode active material volume fraction'] = 1 - parameters[5]
        model_params['Negative electrode conductivity [S.m-1]'] = parameters[6]
        model_params['Negative electrode diffusivity [m2.s-1]'] = parameters[7]
        model_params['Negative electrode exchange-current density [A.m-2]'] = j0_neg
        model_params['Negative electrode porosity'] = parameters[5]
        model_params['Negative electrode thickness [m]'] = parameters[9]
        model_params['Negative particle radius [m]'] = parameters[10]
        model_params['Positive electrode Bruggeman coefficient (electrode)'] = parameters[11]
        model_params['Positive electrode Bruggeman coefficient (electrolyte)'] = parameters[11]
        model_params['Positive electrode active material volume fraction'] = 1 - parameters[12]
        model_params['Positive electrode conductivity [S.m-1]'] = parameters[13]
        model_params['Positive electrode diffusivity [m2.s-1]'] = parameters[14]
        model_params['Positive electrode exchange-current density [A.m-2]'] = j0_pos
        model_params['Positive electrode porosity'] = parameters[12]
        model_params['Positive electrode thickness [m]'] = parameters[16]
        model_params['Positive particle radius [m]'] = parameters[17]
        model_params['Separator Bruggeman coefficient (electrolyte)'] = parameters[18]
        model_params['Separator porosity'] = parameters[19]
        model_params['Separator thickness [m]'] = parameters[20]
        model_params['Thermodynamic factor'] = parameters[21]
        model_params['Maximum concentration in negative electrode [mol.m-3]'] = parameters[22]
        model_params['Maximum concentration in positive electrode [mol.m-3]'] = parameters[23]
        # set the initial concentrations
        model_params['Initial concentration in negative electrode [mol.m-3]'] = parameters[22] * 0.5
        model_params['Initial concentration in positive electrode [mol.m-3]'] = parameters[23] * 0.5

        return model_params
    
    # def _parameters_configuration(self, parameters, take_direct_value, *args, **kwargs):
    #     # return self._parameters_configuration_24d(parameters, take_direct_value, *args, **kwargs)
    #     return super()._parameters_configuration(parameters, take_direct_value, *args, **kwargs)

    def _model_run(self, parameters_dict):
        model_params = copy.deepcopy(self.model_params)
        # for elem in parameters_dict:
        #     model_params[elem] = parameters_dict[elem]

        if len(parameters_dict) == 6:
            model_params['Negative electrode porosity'] = parameters_dict["Negative electrode porosity"]
            model_params['Positive electrode porosity'] = parameters_dict["Positive electrode porosity"]
            model_params['Negative electrode diffusivity [m2.s-1]'] = parameters_dict["Negative electrode diffusivity [m2.s-1]"]
            model_params['Positive electrode diffusivity [m2.s-1]'] = parameters_dict["Positive electrode diffusivity [m2.s-1]"]
            model_params['Negative particle radius [m]'] = parameters_dict["Negative particle radius [m]"]
            model_params['Positive particle radius [m]'] = parameters_dict["Positive particle radius [m]"]
        elif len(parameters_dict) == 24:
            # def j0_neg(c_e, c_s_surf, c_s_max, T):
            #     return 96485.3321 * parameters_dict["Negative electrode exchange-current density [A.m-2]"] * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5

            # def j0_pos(c_e, c_s_surf, c_s_max, T):
            #     return 96485.3321 * parameters_dict["Positive electrode exchange-current density [A.m-2]"] * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5

            # model_params['Cation transference number'] = parameters_dict["Cation transference number"]
            # model_params['Electrolyte conductivity [S.m-1]'] = parameters_dict["Electrolyte conductivity [S.m-1]"]
            # model_params['Electrolyte diffusivity [m2.s-1]'] = parameters_dict["Electrolyte diffusivity [m2.s-1]"]
            # model_params['Initial concentration in electrolyte [mol.m-3]'] = parameters_dict["Initial concentration in electrolyte [mol.m-3]"]
            # model_params['Negative electrode Bruggeman coefficient (electrode)'] = parameters_dict["Negative electrode Bruggeman coefficient (electrode)"]
            # model_params['Negative electrode Bruggeman coefficient (electrolyte)'] = parameters_dict["Negative electrode Bruggeman coefficient (electrode)"]
            # model_params['Negative electrode active material volume fraction'] = 1 - parameters_dict["Negative electrode porosity"]
            # model_params['Negative electrode conductivity [S.m-1]'] = parameters_dict["Negative electrode conductivity [S.m-1]"]
            # model_params['Negative electrode diffusivity [m2.s-1]'] = parameters_dict["Negative electrode diffusivity [m2.s-1]"]
            # model_params['Negative electrode exchange-current density [A.m-2]'] = j0_neg
            # model_params['Negative electrode porosity'] = parameters_dict["Negative electrode porosity"]
            # model_params['Negative electrode thickness [m]'] = parameters_dict["Negative electrode thickness [m]"]
            # model_params['Negative particle radius [m]'] = parameters_dict["Negative particle radius [m]"]
            # model_params['Positive electrode Bruggeman coefficient (electrode)'] = parameters_dict["Positive electrode Bruggeman coefficient (electrode)"]
            # model_params['Positive electrode Bruggeman coefficient (electrolyte)'] = parameters_dict["Positive electrode Bruggeman coefficient (electrode)"]
            # model_params['Positive electrode active material volume fraction'] = 1 - parameters_dict["Positive electrode porosity"]
            # model_params['Positive electrode conductivity [S.m-1]'] = parameters_dict["Positive electrode conductivity [S.m-1]"]
            # model_params['Positive electrode diffusivity [m2.s-1]'] = parameters_dict["Positive electrode diffusivity [m2.s-1]"]
            # model_params['Positive electrode exchange-current density [A.m-2]'] = j0_pos
            # model_params['Positive electrode porosity'] = parameters_dict["Positive electrode porosity"]
            # model_params['Positive electrode thickness [m]'] = parameters_dict["Positive electrode thickness [m]"]
            # model_params['Positive particle radius [m]'] = parameters_dict["Positive particle radius [m]"]
            # model_params['Separator Bruggeman coefficient (electrolyte)'] = parameters_dict["Separator Bruggeman coefficient (electrolyte)"]
            # model_params['Separator porosity'] = parameters_dict["Separator porosity"]
            # model_params['Separator thickness [m]'] = parameters_dict["Separator thickness [m]"]
            # model_params['Thermodynamic factor'] = parameters_dict["Thermodynamic factor"]
            # model_params['Maximum concentration in negative electrode [mol.m-3]'] = parameters_dict["Maximum concentration in negative electrode [mol.m-3]"]
            # model_params['Maximum concentration in positive electrode [mol.m-3]'] = parameters_dict["Maximum concentration in positive electrode [mol.m-3]"]
            # # set the initial concentrations
            # model_params['Initial concentration in negative electrode [mol.m-3]'] = parameters_dict["Maximum concentration in negative electrode [mol.m-3]"] * 0.5
            # model_params['Initial concentration in positive electrode [mol.m-3]'] = parameters_dict["Maximum concentration in positive electrode [mol.m-3]"] * 0.5

            def j0_neg(c_e, c_s_surf, c_s_max, T):
                return 96485.3321 * parameters_dict["$k_{neg}$"] * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5
            def j0_pos(c_e, c_s_surf, c_s_max, T):
                return 96485.3321 * parameters_dict["$k_{pos}$"] * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5
            
            model_params['Cation transference number'] = parameters_dict["$t^+$"]
            model_params['Electrolyte conductivity [S.m-1]'] = parameters_dict["$\\kappa$"]
            model_params['Electrolyte diffusivity [m2.s-1]'] = parameters_dict["$D_l$"]
            model_params['Initial concentration in electrolyte [mol.m-3]'] = parameters_dict["$c_{l,0}$"]
            model_params['Negative electrode Bruggeman coefficient (electrode)'] = parameters_dict["$b_{neg}$"]
            model_params['Negative electrode Bruggeman coefficient (electrolyte)'] = parameters_dict["$b_{neg}$"]
            model_params['Negative electrode active material volume fraction'] = 1 - parameters_dict["$\\varepsilon_{l,neg}$"]
            model_params['Negative electrode conductivity [S.m-1]'] = parameters_dict["$\\sigma_{neg}$"]
            model_params['Negative electrode diffusivity [m2.s-1]'] = parameters_dict["$D_{s,neg}$"]
            model_params['Negative electrode exchange-current density [A.m-2]'] = j0_neg
            model_params['Negative electrode porosity'] = parameters_dict["$\\varepsilon_{l,neg}$"]
            model_params['Negative electrode thickness [m]'] = parameters_dict["$L_{neg}$"]
            model_params['Negative particle radius [m]'] = parameters_dict["$R_{neg}$"]
            model_params['Positive electrode Bruggeman coefficient (electrode)'] = parameters_dict["$b_{pos}$"]
            model_params['Positive electrode Bruggeman coefficient (electrolyte)'] = parameters_dict["$b_{pos}$"]
            model_params['Positive electrode active material volume fraction'] = 1 - parameters_dict["$\\varepsilon_{l,pos}$"]
            model_params['Positive electrode conductivity [S.m-1]'] = parameters_dict["$\\sigma_{pos}$"]
            model_params['Positive electrode diffusivity [m2.s-1]'] = parameters_dict["$D_{s,pos}$"]
            model_params['Positive electrode exchange-current density [A.m-2]'] = j0_pos
            model_params['Positive electrode porosity'] = parameters_dict["$\\varepsilon_{l,pos}$"]
            model_params['Positive electrode thickness [m]'] = parameters_dict["$L_{pos}$"]
            model_params['Positive particle radius [m]'] = parameters_dict["$R_{pos}$"]
            model_params['Separator Bruggeman coefficient (electrolyte)'] = parameters_dict["$b_{sep}$"]
            model_params['Separator porosity'] = parameters_dict["$\\varepsilon_{l,sep}$"]
            model_params['Separator thickness [m]'] = parameters_dict["$L_{sep}$"]
            model_params['Thermodynamic factor'] = parameters_dict["$\Theta$"]
            model_params['Maximum concentration in negative electrode [mol.m-3]'] = parameters_dict["$c_{max,neg}$"]
            model_params['Maximum concentration in positive electrode [mol.m-3]'] = parameters_dict["$c_{max,pos}$"]
            # set the initial concentrations
            model_params['Initial concentration in negative electrode [mol.m-3]'] = parameters_dict["$c_{max,neg}$"] * 0.5
            model_params['Initial concentration in positive electrode [mol.m-3]'] = parameters_dict["$c_{max,pos}$"] * 0.5

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

