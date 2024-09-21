import numpy as np
import matplotlib.pyplot as plotter
import pathlib
import pickle
import os

from uqef_dynamic.utils import colors
from uqef_dynamic.models.time_dependent_baseclass.time_dependent_statistics import TimeDependentStatistics


class IshigamiStatistics(TimeDependentStatistics):

    def __init__(self, configurationObject, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, workingDir, *args, **kwargs)

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

        # if type=="both":
        #     sm1 = vm1/v
        #     sm2 = vm2/v
        #     sm3 = 0.0  # vm3/v

        #     st1 = (vm1 + vm13)/v
        #     st2 = vm2/v
        #     st3 = vm13/v
        #     # Sobol_m_analytical = np.array([0.3138/0.3139, 0.4424/0.4424, 0.0/0.0000], dtype=np.float64)
        #     sobol_m_analytical = np.array([sm1, sm2, sm3], dtype=np.float64)

        #     # Sobol_t_analytical = np.array([0.5574/0.5576, 0.4424/0.4424, 0.2436/0.2437], dtype=np.float64)
        #     sobol_t_analytical = np.array([st1, st2, st3], dtype=np.float64)
        #     return sobol_m_analytical, sobol_t_analytical
        # elif type=="main" or type=="m":
        #     sm1 = vm1/v
        #     sm2 = vm2/v
        #     sm3 = 0.0
        #     return np.array([sm1, sm2, sm3], dtype=np.float64)
        # elif type=="total" or type=="t":
        #     st1 = (vm1 + vm13)/v
        #     st2 = vm2/v
        #     st3 = vm13/v
        #     return np.array([st1, st2, st3], dtype=np.float64)
        # else:
        #     raise ValueError(f"Unknown type {type}.")
    
    def prepare_for_plotting(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):
        pass
    
    def printResults_single_qoi(self, single_qoi_column, dict_time_vs_qoi_stat=None, timestep=-1, **kwargs):
        #dict_time_vs_qoi_stat is self.result_dict[single_qoi_column] 
        if dict_time_vs_qoi_stat is not None:
            local_result_dict = dict_time_vs_qoi_stat[self.timesteps[0]]                     
        elif self.result_dict is not None:
            local_result_dict = self.result_dict[single_qoi_column]
        else:
            raise ValueError("No result dictionary provided.")

        print("STATISTICS INFO: E, Var, Std, etc.")
        if "E" in local_result_dict:
            print(f"E: {local_result_dict['E']}")
        if "Var" in local_result_dict:
            print(f"Var: {local_result_dict['Var']}")
        if "StdDev" in local_result_dict:
            print(f"StdDev: {local_result_dict['StdDev']}")
        if "Skew" in local_result_dict:
            print(f"Skew: {local_result_dict['Skew']}")
        if "Kurt" in local_result_dict:
            print(f"Kurt: {local_result_dict['Kurt']}")
        if "P10" in local_result_dict:
            print(f"P10: {local_result_dict['P10']}")
        if "P90" in local_result_dict:
            print(f"P90: {local_result_dict['P90']}")
     
        # print("\n STATISTICS INFO: gPCE")
        # gPCE, gpce_coeff

        if "qoi_values" in local_result_dict:
            qoi_file = os.path.abspath(os.path.join(str(self.workingDir), "qoi_file.npy"))
            np.save(qoi_file, local_result_dict["qoi_values"])

        print("STATISTICS INFO: Sobol' Indices")
        self._check_if_Sobol_t_computed()
        self._check_if_Sobol_m_computed()
        self._check_if_Sobol_m2_computed()

        # type_of_sobol_indices_computed set to "both", "main", "total"
        # if self._is_Sobol_t_computed and self._is_Sobol_m_computed:
        #     type_of_sobol_indices_computed = "both"
        #     Sobol_m_analytical, Sobol_t_analytical = self.get_analytical_sobol_indices(type="both")
        # elif self._is_Sobol_t_computed:
        #     type_of_sobol_indices_computed = "total"
        #     Sobol_t_analytical = self.get_analytical_sobol_indices(type="total")
        #     Sobol_m_analytical = None
        # elif self._is_Sobol_m_computed:
        #     type_of_sobol_indices_computed = "main"
        #     Sobol_m_analytical = self.get_analytical_sobol_indices(type="main")
        #     Sobol_t_analytical = None

        Sobol_m_analytical = Sobol_t_analytical = None
        Sobol_m_analytical, Sobol_t_analytical = self.get_analytical_sobol_indices()
        print("Sobol_m_analytical: {}".format(Sobol_m_analytical, ".6f"))
        print("Sobol_t_analytical: {}".format(Sobol_t_analytical, ".6f"))
        Sobol_t_error = np.empty(len(self.labels), dtype=np.float64)
        Sobol_m_error = np.empty(len(self.labels), dtype=np.float64)

        if self._is_Sobol_t_computed and Sobol_t_analytical is not None:
            for i in range(len(self.labels)):
                if local_result_dict["Sobol_t"].shape[0] == len(self.timesteps):
                    # print(f"Sobol's Total Index for parameter {self.labels[i]} is: \n")
                    # print(f"Sobol Total Simulation = {(local_result_dict["Sobol_t"].T)[i]} \n")
                    # print(f"Sobol Total Analytical = {Sobol_t_analytical[i]} \n")
                    Sobol_t_error[i] = (local_result_dict["Sobol_t"].T)[i] - Sobol_t_analytical[i]
                else:
                    # print(f"Sobol's Total Index for parameter {self.labels[i]} is: \n")
                    # print(f"Sobol Total Simulation = {local_result_dict["Sobol_t"][i]:.4f} \n")
                    # print(f"Sobol Total Analytical = {Sobol_t_analytical[i]:.4f} \n")
                    Sobol_t_error[i] = local_result_dict["Sobol_t"][i] - Sobol_t_analytical[i]
        if self._is_Sobol_m_computed and Sobol_m_analytical is not None:
            for i in range(len(self.labels)):
                if local_result_dict["Sobol_m"].shape[0] == len(self.timesteps):
                    # print(f"Sobol's Main Index for parameter {self.labels[i]} is: \n")
                    # print(f"Sobol Main Simulation = {(local_result_dict["Sobol_m"].T)[i]:.4f} \n")
                    # print(f"Sobol Main Analytical = {Sobol_m_analytical[i]:.4f} \n")
                    Sobol_m_error[i] = (local_result_dict["Sobol_m"].T)[i] - Sobol_m_analytical[i]
                else:
                    # print(f"Sobol's Main Index for parameter {self.labels[i]} is: \n")
                    # print(f"Sobol Main Simulation = {local_result_dict["Sobol_m"][i]:.4f} \n")
                    # print(f"Sobol Main Analytical = {Sobol_m_analytical[i]:.4f} \n")
                    Sobol_m_error[i] = local_result_dict["Sobol_m"][i] - Sobol_m_analytical[i]

        print("STATISTICS INFO: Sobol' Indices (Error)")
        if self._is_Sobol_t_computed:
            print("Sobol_t: {}".format(local_result_dict["Sobol_t"], ".6f"))
            if Sobol_t_analytical is not None:
                print("Sobol_t_analytical: {}".format(Sobol_t_analytical, ".6f"))
                print("Sobol_t_error: {}".format(Sobol_t_error, ".6f"))
        if self._is_Sobol_m_computed:
            print("Sobol_m: {}".format(local_result_dict["Sobol_m"], ".6f"))
            if Sobol_m_analytical is not None:
                print("Sobol_m_analytical: {}".format(Sobol_m_analytical, ".6f"))
                print("Sobol_m_error: {}".format(Sobol_m_error, ".6f"))
        
        if self._is_Sobol_m2_computed:
            print(f"Sobol_m2_qoi: {local_result_dict['Sobol_m2']}")

        if self._is_Sobol_t_computed:
            sobol_t_qoi_file = os.path.abspath(os.path.join(str(self.workingDir), "sobol_t_qoi_file.npy"))
            np.save(sobol_t_qoi_file, local_result_dict["Sobol_t"])
            sobol_t_qoi_file = os.path.abspath(os.path.join(str(self.workingDir), "sobol_t_error.npy"))
            np.save(sobol_t_qoi_file, Sobol_t_error)

        if self._is_Sobol_m_computed:
            sobol_m_qoi_file = os.path.abspath(os.path.join(str(self.workingDir), "sobol_m_qoi_file.npy"))
            np.save(sobol_m_qoi_file, local_result_dict["Sobol_m"])
            sobol_m_qoi_file = os.path.abspath(os.path.join(str(self.workingDir), "sobol_m_error.npy"))
            np.save(sobol_m_qoi_file, Sobol_m_error)

    def plotResults_single_qoi(self, single_qoi_column, dict_time_vs_qoi_stat=None, timestep=-1, display=False, fileName="",
                               fileNameIdent="", directory="./", fileNameIdentIsFullName=False, safe=True,
                               dict_what_to_plot=None, **kwargs):
        pass
     
    def plotResults(self, timestep=-1, display=False, fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, dict_what_to_plot=None, **kwargs):
        """
        This function plots the statistics of a single, or multiple, QoI.
        Thake a look at the plotResults_single_qoi function for more details.
        """
        pass

