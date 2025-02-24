"""
A set of utility functions for preparing and/or post-processing data for 
Uncertainty Quantification (UQ) & Sensitivity Analysis (SA) runs across different models.

This module includes functions for:

- Transforming parameters
- Plotting
- Calculating various Goodness-of-Fit (GoF), objective, likelihood functions, and metrics
- Working with configuration files and objects (e.g., reading, extracting data, and manipulating configuration files/objects)
- Configuring time settings
- Path-related operations
- Processing and manipulating pandas.DataFrame (the primary data structure)
- Supporting SG analysis
- etc.

@author: Ivana Jovanovic Buha
"""

# Standard library imports
from collections import defaultdict
import datetime
from distutils.util import strtobool
import json
import os
import os.path as osp
from typing import List, Optional, Dict, Any, Union, Tuple

# Third party imports
import chaospy as cp
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import pickle
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
import scipy
from tabulate import tabulate
import time

# for parallel computing
import multiprocessing

from uqef_dynamic.utils import sens_indices_sampling_based_utils

# definition of some 'static' variables, names, etc.
TIME_COLUMN_NAME="TimeStamp"
INDEX_COLUMN_NAME = "Index_run"
QOI_COLUMN_NAME = "Value"  # "model"
GOF = "GOF"
QOI_COLUMN_NAME_CENTERED = QOI_COLUMN_NAME + "_centered"

ARGS_FILE = 'uqsim_args.pkl'
CONFIGURATION_OBJECT_FILE = "configurationObject"
NODES_FILE = "nodes.simnodes.zip"
SIMULATION_PARAMETERS_FILE = "simulation_parameters.npy"
PARAMETERS_FILE = "parameters.pkl"
TIME_INFO_FILE = "time_info.txt"
# Files produced by UQEF-Dynamic.Statistics and Samples class
DF_SIMULATIONS_FILE = "df_simulations.pkl"
DF_SIMULATIONS_CONDITIONED_FILE = "df_simulations_conditioned.pkl"

# DF_SIMULATIONS_FILE = "df_all_simulations.pkl"  # old one
DF_STATE_FILE = "df_state.pkl"
DF_INDEX_PARAMETER_FILE = "df_index_parameter.pkl"
DF_INDEX_PARAMETER_CONDITIONED_FILE = "df_index_parameter_conditioned.pkl"
DF_INDEX_PARAMETER_GOF_FILE = "df_index_parameter_gof.pkl"
DF_INDEX_PARAMETER_GOF_CONDITIONED_FILE = "df_index_parameter_gof_conditioned.pkl"
# Active Subspaces and gradient analysis related files
DICT_APPROX_MATRIX_C_FILE = "dict_approx_matrix_c.pkl"
DICT_MATRIX_C_EIGEN_DECOMPOSITION_FILE = "dict_matrix_c_eigen_decomposition.pkl"
DICT_ACTIVE_SCORES_FILE = "active_scores_dict.pkl"
F_KL_SURROGATE_FILE= "f_kl_surrogate_df.pkl"
DF_TIME_VARYING_GRAD_ANALYSIS_FILE = "df_time_varying_grad_analysis.pkl"
DF_TIME_AGGREGATED_GRAD_ANALYSIS_FILE = "df_time_aggregated_grad_analysis.pkl"
STATISTICS_DICTIONARY_FILE = f"statistics_dictionary_qoi.pkl"
# Optionally, one might save uqef.simulation parameters / nodes / weights
DF_UQSIM_SIMULATION_PARAMETERS_FILE = "df_uqsim_simulation_parameters.pkl"
DF_UQSIM_SIMULATION_NODES_FILE = "df_uqsim_simulation_nodes.pkl"
DF_UQSIM_SIMULATION_WEIGHTS_FILE = "df_uqsim_simulation_weights.pkl"

QOI_ENTRY = 'qoi'
PCE_ENTRY = 'gPCE'
PCE_COEFF_ENTRY = 'gpce_coeff'
QOI_DIST_ENTR = 'qoi_dist'
MEAN_ENTRY = 'E'
E_MINUS_STD_ENTRY = 'E_minus_std'
E_PLUS_STD_ENTRY = 'E_plus_std'
E_MINUS_2STD_ENTRY = 'E_minus_2std'
E_PLUS_2STD_ENTRY = 'E_plus_2std'
VAR_ENTRY = 'Var'
STD_DEV_ENTRY = 'StdDev'
SOBOL_FIRST_ORDER_ENTRY = 'Sobol_m'
SOBOL_SECOND_ORDER_ENTRY = 'Sobol_m2'
SOBOL_TOTAL_ORDER_ENTRY = 'Sobol_t'
P10_ENTRY = 'P10'
P90_ENTRY = 'P90'
SKEW_ENTRY = 'Skew'
KURT_ENTRY = 'Kurt'

DEFAULT_DICT_WHAT_TO_PLOT = {
    "E_minus_std": False, "E_plus_std": False, 
    "E_minus_2std": False, "E_plus_2std": False,
    "P10": False, "P90": False,
    "StdDev": False, "Skew": False, "Kurt": False, "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False
}

DEFAULT_DICT_STAT_TO_COMPUTE = {
    "Var": True, "StdDev": True, 
    "E_minus_std": False, "E_plus_std": False, 
    "E_minus_2std": False, "E_plus_2std": False,
    "P10": True, "P90": True,
    "Skew": False, "Kurt": False, "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True
}
###################################################################################################################
# Paths related functions
###################################################################################################################


class FileError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        return repr(self.message)


def _check_if_file_exists_pathlib(file: pathlib.PosixPath):
    return file.exists()


def _check_is_file_pathlib(file: pathlib.PosixPath):
    return file.is_file()


def check_if_file_exists_pathlib(file_path, message=None):
    if not _check_is_file_pathlib(file_path) or not _check_if_file_exists_pathlib(file_path):
        raise FileError(message)
    else:
        return True


def _check_if_file_exists_str(file_path, message=None):
    if not os.path.isfile(file_path):
        raise FileError(message)
    else:
        return True


def check_if_file_exists(file_path, message=None):
    if message is None:
        message = f'{file_path} is not a file, or it does not exists!'
    if isinstance(file_path, str):
        return _check_if_file_exists_str(file_path, message)
    elif isinstance(file_path, pathlib.PosixPath):
        return check_if_file_exists_pathlib(file_path, message)
    else:
        raise FileError(f"{file_path} is neither a pathlib object nor string!")


def pathlib_to_from_str(path, transfrom_to='str'):
    """
    Take the input path and transform it to transfrom_to type
    transfrom_to: 'str' or 'Path'
    """
    if transfrom_to == 'Path' and isinstance(path, str):
        return pathlib.Path(path)
    elif transfrom_to == 'str' and isinstance(path, pathlib.PosixPath):
        return str(path)
    elif (transfrom_to == 'str' and isinstance(path, str)) or (transfrom_to == 'Path' and
                                                               isinstance(path, pathlib.PosixPath)):
        return path
    else:
        raise ValueError(f"Error in pathlib_to_from_str fucnction got"
                         f"unexpected argument; First argument should be of type string or pathlib.PosixPath")


def _path_to_str(path):
    return str(path)


def _path_to_pathlib(path):
    return pathlib.Path(path)


def get_current_directory():
    #return os.getcwd()
    return pathlib.Path().cwd()


def get_full_path_of_file(file: pathlib.PosixPath):
    return file.resolve()


def get_home_directory():
    return pathlib.Path.home()

###################################################################################################################
# UQEF-Dynamic  specific paths related functions
###################################################################################################################


def get_dict_with_qoi_name_specific_output_file_paths_based_on_workingDir(workingDir, qoi_string=QOI_COLUMN_NAME, **kwargs):
    statistics_dictionary_file = workingDir / f"statistics_dictionary_qoi_{qoi_string}.pkl"
    f_kl_surrogate_coefficients_file = workingDir / f"f_kl_surrogate_coefficients_{qoi_string}.npy"
    f_kl_surrogate_df_file = workingDir / f"f_kl_surrogate_df_{qoi_string}.pkl"
    generalized_sobol_indices_file = workingDir / f"generalized_sobol_indices_{qoi_string}.pkl"
    covariance_matrix_file = workingDir / f"covariance_matrix_{qoi_string}.npy"
    eigenvalues_file = workingDir / f"eigenvalues_{qoi_string}.npy"
    eigenvectors_file = workingDir / f"eigenvectors_{qoi_string}.npy"

    return {
        "statistics_dictionary_file": statistics_dictionary_file,
        "f_kl_surrogate_coefficients_file": f_kl_surrogate_coefficients_file,
        "f_kl_surrogate_df_file": f_kl_surrogate_df_file,
        "generalized_sobol_indices_file": generalized_sobol_indices_file,
        "covariance_matrix_file": covariance_matrix_file,
        "eigenvalues_file": eigenvalues_file,
        "eigenvectors_file": eigenvectors_file,
        }


def get_dict_with_qoi_name_timestamp_specific_output_file_paths_based_on_workingDir(workingDir, qoi_string=QOI_COLUMN_NAME, timestamp=0.0, **kwargs):
    gpce_surrogate_file = workingDir / f"gpce_surrogate_{qoi_string}_{timestamp}.pkl"
    gpce_coeffs_file = workingDir / f"gpce_coeffs_{qoi_string}_{timestamp}.npy"
    generalized_sobol_indices_file = workingDir / f"generalized_sobol_indices_{qoi_string}_{timestamp}.pkl"
    return {
        "gpce_surrogate_file":gpce_surrogate_file, 
        "gpce_coeffs_file":gpce_coeffs_file,
        "generalized_sobol_indices_file": generalized_sobol_indices_file,
        }


def get_dict_with_output_file_paths_based_on_workingDir(workingDir, qoi_string=QOI_COLUMN_NAME, **kwargs):
    """
    Returns a dictionary containing output file paths based on the working directory.

    Parameters:
    - workingDir (str): The working directory.
    - qoi_string (str): The quality of interest string. Default is QOI_COLUMN_NAME="Value".
    - **kwargs: Additional keyword arguments.

    Returns:
    - dict: A dictionary containing the following output file paths:
        - args_file (str): Path to the args file.
        - configuration_object_file (str): Path to the configuration object file.
        - nodes_file (str): Path to the nodes file.
        - simulation_parameters_file (str): Path to the simulation parameters file.
        - parameters_file (str): Path to the parameters file.
        - time_info_file (str): Path to the time info file.
        - df_index_parameter_file (str): Path to the df index parameter file.
        - df_index_parameter_gof_file (str): Path to the df index parameter gof file.
        - df_simulations_file (str): Path to the df simulations file.
        - df_state_file (str): Path to the df state file.
        - f_kl_surrogate_file (str): Path to the f kl surrogate file.
        - statistics_dictionary_file (str): Path to the statistics dictionary file.
        - df_time_varying_grad_analysis_file (str): Path to the df time varying grad analysis file.
        - df_time_aggregated_grad_analysis_file (str): Path to the df time aggregated grad analysis file.
        - dict_approx_matrix_c_file (str): Path to the dict approx matrix c file.
        - dict_matrix_c_eigen_decomposition_file (str): Path to the dict matrix c eigen decomposition file.
        - dict_active_scores_file (str): Path to the dict active scores file.
        - df_uqsim_simulation_parameters_file (str): Path to the df uqsim simulation parameters file.
        - df_uqsim_simulation_nodes_file (str): Path to the df uqsim simulation nodes file.
        - df_uqsim_simulation_weights_file (str): Path to the df uqsim simulation weights file.
    """
    args_file, configuration_object_file, nodes_file, simulation_parameters_file, parameters_file, time_info_file, \
    df_index_parameter_file, df_index_parameter_gof_file, df_simulations_file, df_state_file, \
    f_kl_surrogate_file, statistics_dictionary_file, \
    df_time_varying_grad_analysis_file, df_time_aggregated_grad_analysis_file, \
    dict_approx_matrix_c_file, dict_matrix_c_eigen_decomposition_file, dict_active_scores_file, \
    df_uqsim_simulation_parameters_file, df_uqsim_simulation_nodes_file, df_uqsim_simulation_weights_file = \
                    update_output_file_paths_based_on_workingDir(workingDir, qoi_string=qoi_string, **kwargs)
    return {
        "args_file": args_file, 
        "configuration_object_file": configuration_object_file, 
        "nodes_file": nodes_file,
        "simulation_parameters_file": simulation_parameters_file,
        "parameters_file": parameters_file, 
        "time_info_file": time_info_file,
        "df_index_parameter_file": df_index_parameter_file,
        "df_index_parameter_gof_file": df_index_parameter_gof_file,
        "df_simulations_file": df_simulations_file,
        "df_state_file": df_state_file,
        "f_kl_surrogate_file":f_kl_surrogate_file,
        "statistics_dictionary_file": statistics_dictionary_file,
        "df_time_varying_grad_analysis_file": df_time_varying_grad_analysis_file,
        "df_time_aggregated_grad_analysis_file": df_time_aggregated_grad_analysis_file,
        "dict_approx_matrix_c_file": dict_approx_matrix_c_file,
        "dict_matrix_c_eigen_decomposition_file": dict_matrix_c_eigen_decomposition_file,
        "dict_active_scores_file": dict_active_scores_file,
        "df_uqsim_simulation_parameters_file": df_uqsim_simulation_parameters_file,
        "df_uqsim_simulation_nodes_file": df_uqsim_simulation_nodes_file,
        "df_uqsim_simulation_weights_file": df_uqsim_simulation_weights_file
    }


def update_output_file_paths_based_on_workingDir(workingDir, qoi_string=QOI_COLUMN_NAME, **kwargs):
    """
    Update the output file paths based on the working directory.

    Args:
        workingDir (str): The working directory.
        qoi_string (str, optional): The quality of interest string. Defaults to QOI_COLUMN_NAME="Value".
        **kwargs: Additional keyword arguments for specifying file paths.

    Returns:
        tuple: A tuple containing the updated file paths.

    """
    args_file = workingDir / kwargs.get("args_file", ARGS_FILE)
    configuration_object_file = workingDir / kwargs.get("configuration_object_file", CONFIGURATION_OBJECT_FILE)
    nodes_file = workingDir / kwargs.get("nodes_file", NODES_FILE)
    simulation_parameters_file = workingDir / kwargs.get("simulation_parameters_file", SIMULATION_PARAMETERS_FILE)
    parameters_file = workingDir / kwargs.get("parameters_file", PARAMETERS_FILE)
    time_info_file = workingDir / kwargs.get("time_info_file", TIME_INFO_FILE)

    # Files produced by Samples class
    df_index_parameter_file = workingDir / kwargs.get("df_index_parameter_file", DF_INDEX_PARAMETER_FILE)
    # optional set of files
    df_index_parameter_gof_file = workingDir / kwargs.get("df_index_parameter_gof_file", DF_INDEX_PARAMETER_GOF_FILE)
    df_simulations_file = workingDir / kwargs.get("df_simulations_file", DF_SIMULATIONS_FILE)
    df_state_file = workingDir / kwargs.get("df_state_file", DF_STATE_FILE)

    f_kl_surrogate_file = workingDir / kwargs.get("f_kl_surrogate_file", F_KL_SURROGATE_FILE)

    # Active Subspaces and gradient analysis related files
    df_time_varying_grad_analysis_file = workingDir / kwargs.get("df_time_varying_grad_analysis_file", DF_TIME_VARYING_GRAD_ANALYSIS_FILE)
    df_time_aggregated_grad_analysis_file = workingDir / kwargs.get("df_time_aggregated_grad_analysis_file", DF_TIME_AGGREGATED_GRAD_ANALYSIS_FILE)
    dict_approx_matrix_c_file = workingDir / kwargs.get("dict_approx_matrix_c_file", DICT_APPROX_MATRIX_C_FILE)
    dict_matrix_c_eigen_decomposition_file = workingDir / kwargs.get("dict_matrix_c_eigen_decomposition_file", DICT_MATRIX_C_EIGEN_DECOMPOSITION_FILE)
    dict_active_scores_file = workingDir / kwargs.get("dict_active_scores_file", DICT_ACTIVE_SCORES_FILE)

    # Optionally, one might save uqef.simulation parameters / nodes / weights
    df_uqsim_simulation_parameters_file = workingDir / kwargs.get("df_uqsim_simulation_parameters_file", DF_UQSIM_SIMULATION_PARAMETERS_FILE)
    df_uqsim_simulation_nodes_file = workingDir / kwargs.get("df_uqsim_simulation_nodes_file", DF_UQSIM_SIMULATION_NODES_FILE)
    df_uqsim_simulation_weights_file = workingDir / kwargs.get("df_uqsim_simulation_weights_file", DF_UQSIM_SIMULATION_WEIGHTS_FILE)

    # Files produced by UQEF.Statistics and statistics, single qoi - single file
    statistics_dictionary_file = workingDir / f"statistics_dictionary_qoi_{qoi_string}.pkl"
    # dict_with_qoi_name_specific_output_file_paths = get_dict_with_qoi_name_specific_output_file_paths_based_on_workingDir(workingDir, qoi_string)
    # statistics_dictionary_file = dict_with_qoi_name_specific_output_file_paths["statistics_dictionary_file"]
    # f_kl_surrogate_coefficients_file = dict_with_qoi_name_specific_output_file_paths["f_kl_surrogate_coefficients_file"]
    # f_kl_surrogate_df_file = dict_with_qoi_name_specific_output_file_paths["f_kl_surrogate_df_file"]
    # generalized_sobol_indices_file = = dict_with_qoi_name_specific_output_file_paths["generalized_sobol_indices_file"]

    return args_file, configuration_object_file, nodes_file, simulation_parameters_file, parameters_file, time_info_file, \
    df_index_parameter_file, df_index_parameter_gof_file, df_simulations_file, df_state_file, \
    f_kl_surrogate_file, statistics_dictionary_file, \
    df_time_varying_grad_analysis_file, df_time_aggregated_grad_analysis_file, \
    dict_approx_matrix_c_file, dict_matrix_c_eigen_decomposition_file, dict_active_scores_file, \
    df_uqsim_simulation_parameters_file, df_uqsim_simulation_nodes_file, df_uqsim_simulation_weights_file


# TODO Update this class with new changes
class UQOutputPaths(object):
    def __init__(self, workingDir):
        self.workingDir = workingDir
        self._update_other_paths_based_on_workingDir(workingDir)

    def _update_other_paths_based_on_workingDir(self, workingDir, qoi_string=QOI_COLUMN_NAME, **kwargs):
        # 'global' files

        self.nodes_file = workingDir / kwargs.get("nodes_file", NODES_FILE)
        self.parameters_file = workingDir / kwargs.get("parameters_file", PARAMETERS_FILE)
        self.args_file = workingDir / kwargs.get("args_file", ARGS_FILE)
        self.configuration_object_file = workingDir / kwargs.get("configuration_object_file", CONFIGURATION_OBJECT_FILE)

        # master_configuration_folder = workingDir / "master_configuration"
        self.model_runs_folder = workingDir / "model_runs"
        self.master_configuration_folder = self.model_runs_folder / "master_configuration"

        # Files produced by model - __init__
        # TODO this files are either in workingDir or workingDir/"model_runs" - model_runs_folder
        self.df_measured_file = self.model_runs_folder / "df_measured.pkl"  # model_runs_folder/"df_measured.pkl"
        self.df_past_simulated_file = self.model_runs_folder / "df_past_simulated.pkl"  # "df_simulated.pkl"
        self.df_unaltered_file = self.model_runs_folder / "df_unaltered.pkl"  # "df_unaltered_ergebnis.pkl"
        self.gof_past_sim_meas_file = self.model_runs_folder / "gof_past_sim_meas.pkl"
        self.gof_unaltered_meas_file = self.model_runs_folder / "gof_unaltered_meas.pkl"
        self.gpce_file = self.model_runs_folder / "gpce.pkl"

        self.df_simulations_file = self.model_runs_folder / kwargs.get("df_simulations_file", DF_SIMULATIONS_FILE)
        self.df_index_parameter_gof_file = self.model_runs_folder /  kwargs.get("df_index_parameter_gof_file", DF_INDEX_PARAMETER_GOF_FILE)
        self.df_index_parameter_file = self.model_runs_folder / kwargs.get("df_index_parameter_file", DF_INDEX_PARAMETER_FILE)

        # Files produced by UQEF-Dynamic.Statistics class
        self.statistics_dictionary_file = self.model_runs_folder / f"statistics_dictionary_qoi_{qoi_string}.pkl"
        # self.statistics_dictionary_file = self.model_runs_folder / "statistics_dictionary_qoi_calculateNSE.pkl"

        # Active Subspaces and gradient analysis related files
        self.dict_approx_matrix_c_file = self.model_runs_folder / kwargs.get("dict_approx_matrix_c_file", DICT_APPROX_MATRIX_C_FILE)
        self.dict_matrix_c_eigen_decomposition_file = self.model_runs_folder /kwargs.get("dict_matrix_c_eigen_decomposition_file", DICT_MATRIX_C_EIGEN_DECOMPOSITION_FILE)
        self.df_time_varying_grad_analysis_file = self.model_runs_folder / kwargs.get("df_time_varying_grad_analysis_file", DF_TIME_VARYING_GRAD_ANALYSIS_FILE)
        self.df_time_aggregated_grad_analysis_file = self.model_runs_folder / kwargs.get("df_time_aggregated_grad_analysis_file", DF_TIME_AGGREGATED_GRAD_ANALYSIS_FILE)
        self.dict_active_scores_file = self.model_runs_folder / kwargs.get("dict_active_scores_file", DICT_ACTIVE_SCORES_FILE)

        # Optionally, one might save uqef.simulation parameters / nodes / weights
        self.df_uqsim_simulation_parameters_file = self.model_runs_folder / kwargs.get("df_uqsim_simulation_parameters_file", DF_UQSIM_SIMULATION_PARAMETERS_FILE)
        self.df_uqsim_simulation_nodes_file = self.model_runs_folder / kwargs.get("df_uqsim_simulation_nodes_file", DF_UQSIM_SIMULATION_NODES_FILE)
        self.df_uqsim_simulation_weights_file = self.model_runs_folder / kwargs.get("df_uqsim_simulation_weights_file", DF_UQSIM_SIMULATION_WEIGHTS_FILE)

        self.output_stat_graph_filename = workingDir / "sim-plotly.html"
        self.output_stat_graph_filename = str(self.output_stat_graph_filename)

    def update_specifi_model_run_output_file_paths(self, model_runs_folder, i):
        """
        Note: This is specific for Larsim model!
        These files are outputed in case when
        run_and_save_simulations and always_save_original_model_runs options are set to True
        model_runs_folder = workingDir / "model_runs"
        """
        parameters_Larsim_run_file = model_runs_folder / f"parameters_Larsim_run_{i}.pkl"
        parameters_Larsim_run_failed_file = model_runs_folder / f"parameters_Larsim_run_{i}_failed.pkl"
        df_Larsim_run_file = model_runs_folder / f"df_Larsim_run_{i}.pkl"
        gof_file = model_runs_folder / f"gof_{i}.pkl"
        gradients_matrices_file = model_runs_folder / f"gradients_matrices_{i}.npy"
        df_Larsim_raw_run_fileh = model_runs_folder / f"df_Larsim_raw_run_{i}.pkl"
        return parameters_Larsim_run_file, df_Larsim_run_file, gof_file, gradients_matrices_file, df_Larsim_raw_run_fileh

#####################################
# Create DFs from simulation nodes
#####################################


def get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=None):
    """
    Convert UQEF (simulation) nodes object to a pandas DataFrame.

    Args:
        simulationNodes (object): The UQEF (simulation) nodes object.
        nodes_or_paramters (str, optional): Specifies whether to convert nodes or parameters. Defaults to "nodes".
        params_list (list, optional): List of parameter names. Defaults to None.

    Returns:
        pandas.DataFrame: The converted DataFrame.
    """
    numDim = simulationNodes.nodes.shape[0]
    numSamples = simulationNodes.nodes.shape[1]
    if nodes_or_paramters == "nodes":
        if params_list is None:
            my_ditc = {f"x{i}": simulationNodes.nodes[i, :] for i in range(numDim)}
        else:
            # important assumption about the order of the nodes and their corresponding names
            my_ditc = {f"{params_list[i]}": simulationNodes.nodes[i, :] for i in range(numDim)}
    else:
        if params_list is None:
            my_ditc = {f"x{i}": simulationNodes.parameters[i, :] for i in range(numDim)}
        else:
            # important assumption about the order of the nodes and their corresponding names
            my_ditc = {f"{params_list[i]}": simulationNodes.parameters[i, :] for i in range(numDim)}
    df_nodes = pd.DataFrame(my_ditc)
    return df_nodes


def get_df_from_simulationNodes_list(simulationNodes_list):
     """
     Convert a simulationNodes_list into a pandas DataFrame.

     Parameters:
     simulationNodes_list (numpy.ndarray): The input simulationNodes_list array with shape (d, N).

     Returns:
     pandas.DataFrame: The resulting DataFrame with columns representing each dimension of the simulationNodes_list.

     Example:
     >>> simulationNodes_list = np.array([[1, 2, 3], [4, 5, 6]])
     >>> get_df_from_simulationNodes_list(simulationNodes_list)
         x0  x1
     0   1   4
     1   2   5
     2   3   6
     """
     numDim = simulationNodes_list.shape[0]
     numSamples = simulationNodes_list.shape[1]
     my_ditc = {f"x{i}": simulationNodes_list[i, :] for i in range(numDim)}
     df_nodes = pd.DataFrame(my_ditc)
     return df_nodes


def generate_df_with_nodes_and_weights_from_file(file_path, params_list=None):
    """
    Generate a pandas DataFrame with nodes and weights from a file.

    Args:
        file_path (str): The path to the file containing the nodes and weights.
        params_list (list, optional): A list of parameter names corresponding to the columns of the file. 
                                      If None, default parameter names will be used (x0, x1, x2, ...).

    Returns:
        pandas.DataFrame: A DataFrame containing the nodes and weights, with column names based on the parameter names.

    """
    nodes_and_weights_array = np.loadtxt(file_path, delimiter=',')
    numDim = nodes_and_weights_array.shape[1] - 1
    numSamples = nodes_and_weights_array.shape[0]

    if params_list is None:
        my_ditc = {f"x{i}": nodes_and_weights_array[:, i] for i in range(numDim)}
    else:
        # important assumption about the order of the nodes and their corresponding names
        my_ditc = {f"{params_list[i]}": nodes_and_weights_array[:, i] for i in range(numDim)}
    my_ditc["w"] = nodes_and_weights_array[:, numDim]
    df_nodes_and_weights = pd.DataFrame(my_ditc)
    return df_nodes_and_weights

#####################################
# Some plotting of simulation nodes
#####################################


def plot_2d_matrix_static_from_list(simulationNodes_list, title="Plot nodes"):
    if not isinstance(simulationNodes_list, pd.DataFrame):
        dfsimulationNodes = get_df_from_simulationNodes_list(simulationNodes_list, nodes_or_paramters)
    else:
        dfsimulationNodes = simulationNodes_list

    sns.set(style="ticks", color_codes=True)
    pairplot = sns.pairplot(dfsimulationNodes, vars=list(dfsimulationNodes.columns), corner=True)
    return pairplot.fig
    # plt.title(title, loc='left')


def plot_3d_dynamic(dfsimulationNodes):
    my_columns = list(dfsimulationNodes.columns)
    fig = px.scatter_3d(dfsimulationNodes, x=my_columns[0], y=my_columns[1], z=my_columns[2])
    return fig


def plot_3d_statis(simulationNodes, nodes_or_paramters="nodes"):
    fig = plt.figure()
    axs = fig.gca(projection='3d')
    if nodes_or_paramters == "nodes":
        axs.scatter(list(simulationNodes.nodes[0, :]),
                    list(simulationNodes.nodes[1, :]),
                    list(simulationNodes.nodes[2, :]), marker="o")
        plt.title('Nodes\n')
    else:
        axs.scatter(list(simulationNodes.parameters[0, :]),
                    list(simulationNodes.parameters[1, :]),
                    list(simulationNodes.parameters[2, :]), marker="o")
        plt.title('Parameters\n')
    axs.set_xlabel("x1")
    axs.set_ylabel("x2")
    axs.set_zlabel("x3")
    plt.show()


def plot_2d_matrix_static(simulationNodes, nodes_or_paramters="nodes"):
    if not isinstance(simulationNodes, pd.DataFrame):
        dfsimulationNodes = get_df_from_simulationNodes(simulationNodes, nodes_or_paramters)
    else:
        dfsimulationNodes = simulationNodes

    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(dfsimulationNodes, vars=list(dfsimulationNodes.columns), corner=True)
    plt.title(f"Plot: {nodes_or_paramters}", loc='left')
    plt.show()


def plot_2d_grids(grid_array, title=None, weights=None, 
    xaxis_title="X_1", yaxis_title="X_2",
    scale_factor = 2e3, plot_weights_with_sign=False,
    sizemode='area', marker_symbol='circle'
    ):
    """
    Function for plotting a grid of nodes from 2D distribution
    parameters:
    - grid_array - numpy array of dim Number_of_nodesxdim
    - weights  - numpy array of dim Number_of_nodesx1
    - sizemode, e.g., 'area' or 'diameter'
    """
    if weights is None:
        size = 10
    else:
        weights = np.array(weights)
        if np.any(weights<0) and not plot_weights_with_sign:
            size = np.where(weights < 0, np.abs(weights) * scale_factor, weights * scale_factor)
            #idx = weights > 0
            #size = -weights[~idx]*scale_factor
        else:
            size = weights*scale_factor

    if not plot_weights_with_sign:
        fig = go.Figure(data=go.Scatter(
        x=grid_array[:,0],
        y=grid_array[:,1],
        marker_symbol=marker_symbol,
        mode='markers',  # 'markers' means it will plot points
        marker=dict(
            size=size,       # Adjust the marker size
            color='blue',  # Color of the points
            opacity=0.8,    # Transparency of the points
            sizemode=sizemode  # Adjust how size is applied (default is 'area') 'diameter'
        ),
        text = weights
        ))
    else:
        idx = weights >= 0
        fig = go.Figure(data=go.Scatter(
        x=grid_array[idx,0],
        y=grid_array[idx,1],
        mode='markers',  # 'markers' means it will plot points
        marker=dict(
            size=weights[idx]*scale_factor, # Adjust the marker size
            color='blue',  # Color of the points
            opacity=1.0,    # Transparency of the points
            sizemode=sizemode  # Adjust how size is applied (default is 'area') 'diameter'
        ),
        name="positive weights",
        text = weights[idx]
        ))
        
        fig.add_trace(go.Scatter(
        x=grid_array[~idx,0],
        y=grid_array[~idx,1],
        mode='markers',  # 'markers' means it will plot points
        marker=dict(
            size=np.abs(weights[~idx])*scale_factor, # Adjust the marker size
            color='red',  # Color of the points
            opacity=1.0,    # Transparency of the points
            sizemode=sizemode  # Adjust how size is applied (default is 'area') 'diameter'
        ),
        name="negative weights",
        text = weights[~idx]
        ))
    
    # Add title and labels
    if title is not None:
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            width=800,  # Set the width of the figure
            height=600  # Set the height of the figure
        )
    else:
        fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=800,  # Set the width of the figure
        height=600  # Set the height of the figure
        )

    # Update layout for white background and black zero lines
    # fig.update_layout(
    #     plot_bgcolor='white',  # Set plot background color to white
    #     paper_bgcolor='white',  # Set outer (paper) background color to white
    #     xaxis=dict(
    #         showgrid=False,      # Remove x-axis gridlines
    #         # zeroline=True,       # Show x-axis zero line
    #         showline=True,       # Show bottom x-axis line
    #         # zerolinecolor='black',  # Set x-axis zero line color to black
    #         # zerolinewidth=2      # Set the width of the x-axis zero line
    #         linecolor='black',   # Set left x-axis line color to black
    #         linewidth=2          # Set width of the left x-axis line
    #     ),
    #     yaxis=dict(
    #         showgrid=False,      # Remove y-axis gridlines
    #         # zeroline=True,       # Show y-axis zero line
    #         showline=True,       # Show bottom y-axis line
    #         # zerolinecolor='black',  # Set y-axis zero line color to black
    #         # zerolinewidth=2      # Set the width of the y-axis zero line
    #         linecolor='black',   # Set left y-axis line color to black
    #         linewidth=2          # Set width of the left y-axis line
    #     )
    # )
    return fig

#####################################
# Playing with transformations
#####################################
    

def transformation_of_parameters(samples, distribution_r, distribution_q):
    return transformation_of_parameters_var1(samples, distribution_r, distribution_q)


def transformation_of_parameters_var1(samples, distribution_r, distribution_q):
    """
    :param samples: array of samples from distribution_r
    :param distribution_r: 'standard' distribution
    :param distribution_q: 'user-defined' distribution
    :return: array of samples from distribution_q
    """
    # var 1
    return distribution_q.inv(distribution_r.fwd(samples))


def transformation_of_parameters_var1_1(samples, distribution_q):
    """
    :param samples: array of samples from distribution_r - when distribution_r is U[0,1]
    :param distribution_r: 'standard' distribution
    :param distribution_q: 'user-defined' distribution
    :return: array of samples from distribution_q
    """
    # var 1
    return distribution_q.inv(samples)


def transformation_of_parameters_uniform(samples, distribution_r, distribution_q):
    return transformation_of_parameters_var2(samples, distribution_r, distribution_q)


def transformation_of_parameters_var2(samples, distribution_r, distribution_q):
    """
    :param samples: array of samples from distribution_r, when distribution_r is U[-1,1] or U[0,1]
    :param distribution_r: 'standard' distribution either U[-1,1] or U[0,1]
    :param distribution_q: 'user-defined' distribution
    :return: array of samples from distribution_q
    """
    #distinqush between distribution_r is U[-1,1] or U[0,1]
    dim = len(distribution_r)
    assert len(distribution_r) == len(distribution_q)
    _a = np.empty([dim, 1])
    _b = np.empty([dim, 1])

    for i in range(dim):
        r_lower = distribution_r[i].lower
        r_upper = distribution_r[i].upper
        q_lower = distribution_q[i].lower
        q_upper = distribution_q[i].upper

        if r_lower == -1:
            _a[i] = (q_lower + q_upper) / 2
            _b[i] = (q_upper - q_lower) / 2
        elif r_lower == 0:
            _a[i] = q_lower
            _b[i] = (q_upper - q_lower)

    return _a + _b * samples

#####################################
# Utility functions for processing/manipulating pandas.DataFrame (i.e., main data structure)
#####################################


def transform_column_in_df(df, transformation_function_str, column_name, new_column_name=None):
    """
    Apply a transformation function to a column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be transformed.
        transformation_function_str (str): The transformation function to be applied. Supported options are "log", "log2", "log10", and "exp".
        column_name (str): The name of the column to be transformed.
        new_column_name (str, optional): The name of the new column to store the transformed values. If not provided, the original column will be overwritten.

    Raises:
        NotImplementedError: If the specified transformation function is not supported.

    Returns:
        None; as a side effet update the provided DataFrame with the transformed column.

    """
    if new_column_name is None:
        new_column_name = column_name

    if transformation_function_str == "log":
        df[column_name] = df[column_name].apply(lambda x: x if x > 1e-4 else 1e-4)
        df[new_column_name] = np.log(df[column_name], out=np.zeros_like(df[column_name].values), where=(df[column_name].values!=0))
    elif transformation_function_str == "log2":
        df[column_name] = df[column_name].apply(lambda x: x if x > 1e-4 else 1e-4)
        df[new_column_name] = np.log2(df[column_name], out=np.zeros_like(df[column_name].values), where=(df[column_name].values!=0))
    elif transformation_function_str == "log10":
        df[column_name] = df[column_name].apply(lambda x: x if x > 1e-4 else 1e-4)
        df[new_column_name] = np.log10(df[column_name], out=np.zeros_like(df[column_name].values), where=(df[column_name].values!=0))
    elif transformation_function_str == "exp":
        # df[column_name] = df[column_name].apply(lambda x: x if x > 1e-4 else 1e-4)
        df[new_column_name] = np.exp(df[column_name])
    else:
        # flux_df[new_column_name] = flux_df[single_qoi_column].apply(single_transformation)
        raise NotImplementedError("For now, only log transformation is supported")

#####################################
# Utility for calculating different GoF/Objective/Likelihood functions/metrices
#####################################
from . import objectivefunctions

_all_functions = [objectivefunctions.MAE, objectivefunctions.MSE,
                  objectivefunctions.RMSE, objectivefunctions.NRMSE, objectivefunctions.RSR,
                  objectivefunctions.BIAS, objectivefunctions.PBIAS, objectivefunctions.ROCE,
                  objectivefunctions.NSE, objectivefunctions.LogNSE,
                  objectivefunctions.LogGaussian, objectivefunctions.CorrelationCoefficient,
                  objectivefunctions.KGE]


mapping_gof_names_to_functions = {
    "MAE": objectivefunctions.MAE,
    "MSE": objectivefunctions.MSE,
    "RMSE": objectivefunctions.RMSE,
    "NRMSE": objectivefunctions.NRMSE,
    "RSR": objectivefunctions.RSR,
    "BIAS": objectivefunctions.BIAS,
    "PBIAS": objectivefunctions.PBIAS,
    "ROCE": objectivefunctions.ROCE,
    "NSE": objectivefunctions.NSE,
    "LogNSE": objectivefunctions.LogNSE,
    "LogGaussian": objectivefunctions.LogGaussian,
    "CorrelationCoefficient": objectivefunctions.CorrelationCoefficient,
    "KGE": objectivefunctions.KGE
}


def gof_list_to_function_names(gof_list):
    if gof_list == "all" or gof_list is None:
        gof_list = _all_functions
    else:
        if not isinstance(gof_list, list):
            gof_list = [gof_list,]
        for idx, f in enumerate(gof_list):
            if not callable(f) and f in mapping_gof_names_to_functions:
                gof_list[idx] = mapping_gof_names_to_functions[f]
            elif callable(f) and f in _all_functions:
                continue
            else:
                raise ValueError("Not proper specification of Goodness of Fit function name")
    return gof_list


def _check_if_measured_or_predicted_are_empty(measuredDF, predictedDF, measuredDF_column_name, simulatedDF_column_name, gof_list):
    if measuredDF.empty or predictedDF.empty \
            or measuredDF_column_name not in measuredDF.columns \
            or simulatedDF_column_name not in predictedDF.columns:
        result_dict = {}
        for f in gof_list:
            result_dict[f.__name__] = np.nan
        return True, result_dict
    else:
        return False, None


def dataframe_difference(df1, df2, which=None):
    """Find rows which are different between two DataFrames."""
    comparison_df = df1.merge(df2,
                              indicator=True,
                              how='outer')
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df


def filter_two_DF_on_common_timesteps(DF1, DF2, saveToFile=None, column_name_df1="TimeStamp", column_name_df2=None):
    """
    Note, the function will modify the input DataFrames inplace
    :param DF1: DataFrame
    :param DF2: DataFrame
    :param saveToFile: str
    :param column_name_df1: str
    :param column_name_df2: str
    """
    # really important to check if there are any missing time steps compared to measured array
    # in other words measured and observed values have to be of the same length
    # TODO Should we make a copy of DF1 and DF1 or change the origins?
    # TODO Maybe there is more elegant way to 
    if column_name_df2 is None:
        column_name_df2 = column_name_df1
    df1_column_is_index_column = False
    df2_column_is_index_column = False
    if DF1.index.name == column_name_df1:
        df1_column_is_index_column = True
        DF1 = DF1.reset_index()
        DF1.rename(columns={DF1.index.name: column_name_df1}, inplace=True)
    if DF2.index.name == column_name_df2:
        df2_column_is_index_column = True
        DF2 = DF2.reset_index()
        DF2.rename(columns={DF2.index.name: column_name_df2}, inplace=True)

    list_of_columns_to_drop = [x for x in list(DF1.columns) if x != column_name_df1]
    DF1_timeSteps = DF1.drop(list_of_columns_to_drop, axis=1, errors='ignore')
    list_of_columns_to_drop = [x for x in list(DF2.columns) if x != column_name_df2]
    DF2_timeSteps = DF2.drop(list_of_columns_to_drop, axis=1, errors='ignore')

    diff_df = dataframe_difference(DF1_timeSteps, DF2_timeSteps, which=None)
    if saveToFile is not None:
        diff_df.to_pickle(saveToFile, compression="gzip")

    left_only = diff_df[diff_df['_merge'] == 'left_only']
    right_only = diff_df[diff_df['_merge'] == 'right_only']

    DF1 = DF1[~DF1[column_name_df1].isin(left_only[column_name_df1].values)]
    DF2 = DF2[~DF2[column_name_df2].isin(right_only[column_name_df2].values)]

    if df1_column_is_index_column:
        DF1.set_index(column_name_df1, inplace=True)
    if df2_column_is_index_column:
        DF2.set_index(column_name_df2, inplace=True)

    return DF1, DF2


def calculateGoodnessofFit_simple(measuredDF, simulatedDF, gof_list,
                                  measuredDF_time_column_name="TimeStamp",
                                  measuredDF_column_name='Value',
                                  simulatedDF_time_column_name="TimeStamp",
                                  simulatedDF_column_name='Value',
                                  return_dict=True,
                                  **kwargs):
    """
    Assumption - two columns of interest are aligned with respect to time
    :param measuredDF:
    :param simulatedDF:
    :param gof_list:
    :param measuredDF_column_name:
    :param simulatedDF_column_name:
    :return:
    """
    # calculate mean of the observed - measured discharge
    # mean_gt_discharge = np.mean(measuredDF[measuredDF_column_name].values)

    # TODO Duplicated code
    gof_list = gof_list_to_function_names(gof_list)
    is_empty, result_dict = _check_if_measured_or_predicted_are_empty(measuredDF, simulatedDF,
                                                                      measuredDF_column_name, simulatedDF_column_name,
                                                                      gof_list)
    if is_empty:
        return result_dict

    # DataFrames containing measurements might be longer than the one containing model predictions - alignment is needed
    # It might be as well that one of DataFrames does not contain all the timesteps the other one does
    # therefore, apply one of these two filtering functions
    # assert measuredDF_time_column_name == simulatedDF_time_column_name, "Assertion failed in utility.calculateGoodnessofFit_simple"
    if id(measuredDF) != id(simulatedDF):
        # make copy of the original arguments, since filter_two_DF_on_common_timesteps will modifeid propageted DataFrames inplace
        measuredDF = measuredDF.copy()
        simulatedDF = simulatedDF.copy()
        measuredDF, simulatedDF = filter_two_DF_on_common_timesteps(
            measuredDF, simulatedDF, column_name_df1=measuredDF_time_column_name, column_name_df2=simulatedDF_time_column_name)
    #simulatedDF, measuredDF = align_dataFrames_timewise_2(simulatedDF, measuredDF)

    if return_dict:
        result = dict()
    else:
        result = []

    for f in gof_list:
        try:
            temp_result = f(
                measuredDF=measuredDF, simulatedDF=simulatedDF, 
                measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name=simulatedDF_column_name, **kwargs)
        except:
            temp_result = np.nan

        if return_dict:
            result[f.__name__] = temp_result
        else:
            result.append(temp_result)

    return result


###################################################################################################################
# Functions for working with configuration files and configuration objects
###################################################################################################################

###################################################################################################################
# Time configurations
###################################################################################################################


def parse_datetime_configuration(time_settings_config):
    """
    Reads configuration dictionary and determines the start end end date of the simulation
    Returns a list with two datetime objects
    """
    if time_settings_config is None or not time_settings_config:
        time_settings_config = dict()
        start_dt = None
        end_dt = None
    else:
        if isinstance(time_settings_config, dict):
            if "time_settings" in time_settings_config:
                datas = time_settings_config["time_settings"]
            else:
                datas = time_settings_config
            start_year = datas.get("start_year", None)
            start_month = datas.get("start_month", None)
            start_day = datas.get("start_day", None)
            start_hour = datas.get("start_hour", None)
            start_minute = datas.get("start_minute", None)
            end_year = datas.get("end_year", None)
            end_month = datas.get("end_month", None)
            end_day = datas.get("end_day", None)
            end_hour = datas.get("end_hour", None)
            end_minute = datas.get("end_minute", None)
        elif isinstance(time_settings_config, list) or isinstance(time_settings_config, tuple):
            start_year, start_month, start_day, start_hour, start_minute, \
            end_year, end_month, end_day, end_hour, end_minute = time_settings_config
        else:
            raise Exception(f"[Error] in parse_datetime_configuration: time_settings_config should be "
                            f"dict, list or tuple storing start_year, start_month, etc.")

        if all(var is not None for var in [start_year, start_month, start_day, start_hour, start_minute]):
            start_dt = datetime.datetime(year=start_year, month=start_month, day=start_day,
                                        hour=start_hour, minute=start_minute)
        else:
            start_dt = None

        if all(var is not None for var in [end_year, end_month, end_day, end_hour, end_minute]):
            end_dt = datetime.datetime(year=end_year, month=end_month, day=end_day,
                                    hour=end_hour, minute=end_minute)
        else:
            end_dt = None
            
    return [start_dt, end_dt]


def compute_previous_timestamp(timestamp, resolution, delta=1):
    if timestamp is None:
        return None
    if resolution == "daily":
        # pd.DateOffset(days=1)
        previous_timestamp = pd.to_datetime(timestamp) - pd.Timedelta(days=1)
    elif resolution == "hourly":
        previous_timestamp = pd.to_datetime(timestamp) - pd.Timedelta(h=1)
    elif resolution == "minute":
        previous_timestamp = pd.to_datetime(timestamp) - pd.Timedelta(m=1)
    else:
        previous_timestamp = timestamp - delta
    return previous_timestamp

###################################################################################################################
# Reading, extract data and manipulate configuration file/object
###################################################################################################################


def return_configuration_object(configurationObject):
    if isinstance(configurationObject, str) or isinstance(configurationObject, pathlib.PosixPath):
        try:
            with open(configurationObject) as f:
                return json.load(f)
        except ValueError:
            raise ValueError(
                f"[Error] The .json file {configurationObject} "
                f"which should be Configuration Object does not exist!\n")
    elif isinstance(configurationObject, dict):
        return configurationObject
    else:
        raise ValueError(
            "[Error] You have to specify the Configuration Object or provide the path to the .json file!\n")


def check_if_configurationObject_is_in_right_format_and_return(configurationObject: Union[dict, pd.DataFrame, str, pathlib.PosixPath, Any], raise_error: Optional[bool] = True):
    if isinstance(configurationObject, dict) or isinstance(configurationObject, pd.DataFrame):
        return configurationObject
    elif isinstance(configurationObject, str) or isinstance(configurationObject, pathlib.PosixPath):
        configurationObject = read_configuration_dict_from_json_file(configurationObject)
        return configurationObject
    else:
        if raise_error:
            raise Exception("Error in configure_parameters_value function - configurationObject is not in the expected form!")
        else:
            return None


def read_configuration_object_json(configurationObject, element=None):
    """
    :param configurationObject: posixPath to the json file
    element can be, e.g., "Output", "parameters", "Timeframe", "parameters_settings"
    """
    # with configurationObject.open(encoding="UTF-8") as source:
    with open(configurationObject) as f:
        configuration_object = json.load(f)
    if element is None:
        return configuration_object
    else:
        return configuration_object[element]


def read_configuration_dict_from_json_file(configurationFile):
    """

    :param configurationFile: posixPath to the json file
    :return:
    """
    return read_configuration_object_json(configurationObject=configurationFile)


def read_configuration_object_dill(configurationObject, element=None):
    """
    element can be, e.g., "Output", "parameters", "Timeframe", "parameters_settings"
    """
    with open(configurationObject, 'rb') as f:
        configuration_object = dill.load(f)
    if element is None:
        return configuration_object
    else:
        return configuration_object[element]


def get_configuration_value(key: str, config_dict: dict, default_value: Any, **kwargs) -> Any:
    """
    Get a configuration value from a dictionary or kwargs.

    Args:
        key (str): The key to look for in the dictionary and kwargs.
        config_dict (dict): The dictionary to look for the key in.
        default_value (Any): The default value to return if the key is not found.
        **kwargs: Optional keyword arguments to look for the key in.

    Returns:
        The value associated with the key, or the default value if the key is not found.
    """
    if key in kwargs:
        return kwargs[key]
    else:
        return config_dict.get(key, default_value)


def get_simulation_settings(configurationObject: Union[dict, pd.DataFrame, str, pathlib.PosixPath, Any], key_from_configurationObject="simulation_settings"):
    """
    Get the simulation settings from a configuration object.

    Args:
        configurationObject (dict): The configuration object containing the simulation settings.
        key_from_configurationObject (str): The key to look for in the configuration object.

    Returns:
        dict: The simulation settings, or an empty dictionary if the configuration object is None or doesn't contain the simulation settings.
    """
    configurationObject = check_if_configurationObject_is_in_right_format_and_return(configurationObject, raise_error=False)
    if configurationObject is None:
        return {}

    return configurationObject.get(key_from_configurationObject, {})


def handle_multiple_qoi(qoi, qoi_column, result_dict):
    """
    Handles the logic related to multiple quantities of interest (qoi).

    Args:
        qoi: The quantity of interest.
        qoi_column: The column of the quantity of interest.
        result_dict: The result dictionary where the results are stored.

    Returns:
        The updated result dictionary.
    """
    multiple_qoi = False
    number_of_qois = 1
    if (isinstance(qoi, list) and isinstance(qoi_column, list)) or (qoi == GOF and isinstance(qoi_column, list)):
        multiple_qoi = True
        number_of_qois = len(qoi_column)
    return multiple_qoi, number_of_qois


def handle_transform_model_output(transform_model_output, result_dict, dict_config_simulation_settings):
    """
    Handles the logic related to the 'transform_model_output' configuration.

    Args:
        transform_model_output: The 'transform_model_output' configuration value.
        result_dict: The result dictionary where the results are stored.
        dict_config_simulation_settings: The dictionary containing the simulation settings.

    Returns:
        The updated 'transform_model_output' configuration value.
    """
    if result_dict["multiple_qoi"]:
        try:
            for idx, single_transform_model_output in enumerate(transform_model_output):
                if single_transform_model_output == "None":
                    transform_model_output[idx] = None
        except KeyError:
            transform_model_output = [None] * result_dict["number_of_qois"]
    else:
        if transform_model_output == "None":
            transform_model_output = None

    return transform_model_output


def read_simulation_settings_from_configuration_object_refactored(configurationObject: dict, **kwargs) -> dict:
    # TODO finish this function; idea is to refactor read_simulation_settings_from_configuration_object such that it is more modular
    result_dict = dict()
    dict_config_simulation_settings = get_simulation_settings(configurationObject)

    if "qoi" in kwargs:
        qoi = kwargs['qoi']
    else:
        qoi = dict_config_simulation_settings.get("qoi", "model_output")
    result_dict["qoi"] = qoi

    if "qoi_column" in kwargs:
        qoi_column = kwargs['qoi_column']
    else:
        qoi_column = dict_config_simulation_settings.get("qoi_column", QOI_COLUMN_NAME)
    result_dict["qoi_column"] = qoi_column

    multiple_qoi, number_of_qois = handle_multiple_qoi(qoi, qoi_column, result_dict)
    result_dict["multiple_qoi"] = multiple_qoi
    result_dict["number_of_qois"] = number_of_qois

    transform_model_output = get_configuration_value("transform_model_output", dict_config_simulation_settings, "None", **kwargs)
    transform_model_output = handle_transform_model_output(transform_model_output, result_dict, dict_config_simulation_settings)
    result_dict["transform_model_output"] = transform_model_output

    # handle_read_measured_data
    # handle_qoi_column_measured
    # handle_calculate_GoF
    # handle_objective_function
    # handle_mode_and_center
    # handle_compute_gradients

    return result_dict


def read_simulation_settings_from_configuration_object(configurationObject: Union[dict, pd.DataFrame, str, pathlib.PosixPath, Any], **kwargs) -> dict:
    """
        Reads simulation settings from a configuration object and returns a dictionary of settings.

        This function reads various simulation settings such as 'qoi', 'qoi_column', 'transform_model_output', 
        'read_measured_data', 'qoi_column_measured', 'calculate_GoF', 'objective_function', etc. from the 
        configuration object. For some settings it will first check if the value occures kwargs,
        just then in the configuration object, and if not the default value might be used.

        Args:
            configurationObject (dict): The configuration object containing simulation settings.
            **kwargs: Optional keyword arguments for overriding settings in the configuration object.

        Returns:
            result_dict: A dictionary containing the simulation settings.

        Raises:
            ValueError: If 'read_measured_data' is True but 'qoi_column_measured' is None.
            Exception: If 'mode' is not one of 'continuous', 'sliding_window', 'resampling'.
            Exception: If 'center' is not one of 'center', 'left', 'right'.
            Exception: If 'method' is not one of 'avrg', 'max', 'min'.
        In case configurationObject does not exist or is not in the right format it will return an empty dictionary.
    """
    # TODO Refactor this long function into smaller functions.
    result_dict = dict()

    configurationObject = check_if_configurationObject_is_in_right_format_and_return(configurationObject,
                                                                                     raise_error=False)
    
    # TODO Hmm, should this indeed stand here; or do I want to populte the result_dict with the default values?
    if configurationObject is None:
        return result_dict

    # dict_config_simulation_settings = configurationObject["simulation_settings"]
    dict_config_simulation_settings = configurationObject.get("simulation_settings", dict())

    if "qoi" in kwargs:
        qoi = kwargs['qoi']
    else:
        qoi = dict_config_simulation_settings.get("qoi", "model_output")
    result_dict["qoi"] = qoi

    if "qoi_column" in kwargs:
        qoi_column = kwargs['qoi_column']
    else:
        qoi_column = dict_config_simulation_settings.get("qoi_column", QOI_COLUMN_NAME)
    result_dict["qoi_column"] = qoi_column

    # temp = result_dict["qoi_column"]

    # multiple_qoi = False
    # number_of_qois = 1
    # if (isinstance(qoi, list) and isinstance(qoi_column, list)) or (qoi == GOF and isinstance(qoi_column, list)):
    #     multiple_qoi = True
    #     number_of_qois = len(qoi_column)
    # result_dict["multiple_qoi"] = multiple_qoi
    # result_dict["number_of_qois"] = number_of_qois
    multiple_qoi, number_of_qois = handle_multiple_qoi(qoi, qoi_column, result_dict)
    result_dict["multiple_qoi"] = multiple_qoi
    result_dict["number_of_qois"] = number_of_qois

    result_dict["autoregressive_model_first_order"] = strtobool(dict_config_simulation_settings.get(\
        "autoregressive_model_first_order", "False"))
    result_dict["scale_factor_autoregressive_model_first_order"] = dict_config_simulation_settings.get(\
            "scale_factor_autoregressive_model_first_order", 1.0)

    if "transform_model_output" in kwargs:
        transform_model_output = kwargs['transform_model_output']
    else:
        if result_dict["multiple_qoi"]:
            try:
                transform_model_output = dict_config_simulation_settings["transform_model_output"]
                for idx, single_transform_model_output in enumerate(transform_model_output):
                    if single_transform_model_output == "None":
                        transform_model_output[idx] = None
            except KeyError:
                transform_model_output = [None] * result_dict["number_of_qois"]
        else:
            transform_model_output = dict_config_simulation_settings.get("transform_model_output", "None")
            if transform_model_output == "None":
                transform_model_output = None
    result_dict["transform_model_output"] = transform_model_output

    if "read_measured_data" in kwargs:
        read_measured_data = kwargs['read_measured_data']
    else:
        if result_dict["multiple_qoi"]:
            read_measured_data = []
            try:
                temp = dict_config_simulation_settings["read_measured_data"]
            except KeyError:
                temp = ["False"] * result_dict["number_of_qois"]
            for i in range(result_dict["number_of_qois"]):
                if isinstance(temp[i], str):
                    read_measured_data.append(strtobool(temp[i]))
                else:
                    read_measured_data.append(temp[i])
        else:
            read_measured_data = dict_config_simulation_settings.get("read_measured_data", "False")
            if isinstance(read_measured_data, str):
                read_measured_data = strtobool(read_measured_data)

    if "qoi_column_measured" in kwargs:
        qoi_column_measured = kwargs['qoi_column_measured']
    else:
        if result_dict["multiple_qoi"]:
            try:
                qoi_column_measured = dict_config_simulation_settings["qoi_column_measured"]
                for idx, single_qoi_column_measured in enumerate(qoi_column_measured):
                    if single_qoi_column_measured == "None":
                        qoi_column_measured[idx] = None
            except KeyError:
                qoi_column_measured = [None] * result_dict["number_of_qois"]
        else:
            qoi_column_measured = dict_config_simulation_settings.get("qoi_column_measured", "None")
            if qoi_column_measured == "None":
                qoi_column_measured = None
    result_dict["qoi_column_measured"] = qoi_column_measured

    if result_dict["multiple_qoi"]:
        for idx, single_read_measured_data in enumerate(read_measured_data):
            if single_read_measured_data and qoi_column_measured[idx] is None:
                # raise ValueError
                read_measured_data[idx] = False
    else:
        if read_measured_data and qoi_column_measured is None:
            # raise ValueError
            read_measured_data = False
    result_dict["read_measured_data"] = read_measured_data

    if result_dict["multiple_qoi"]:
        assert len(read_measured_data) == len(qoi_column)
        assert len(read_measured_data) == len(qoi_column_measured)

    calculate_GoF = strtobool(dict_config_simulation_settings.get("calculate_GoF", "False"))
    # self.calculate_GoF has to follow the self.read_measured_data which tells if ground truth data for that qoi is available
    list_calculate_GoF = [False, ]
    if calculate_GoF:
        if result_dict["multiple_qoi"]:
            list_calculate_GoF = [False] * len(read_measured_data)
            for idx, single_read_measured_data in enumerate(read_measured_data):
                list_calculate_GoF[idx] = single_read_measured_data
        else:
            list_calculate_GoF = [read_measured_data, ]
    result_dict["calculate_GoF"] = calculate_GoF
    result_dict["list_calculate_GoF"] = list_calculate_GoF

    objective_function = dict_config_simulation_settings.get("objective_function", [])
    result_dict["objective_function"] = objective_function

    objective_function_qoi = None
    objective_function_names_qoi = None
    list_objective_function_qoi = None
    list_objective_function_names_qoi = None
    if result_dict["qoi"] == GOF:
        # take only those Outputs of Interest that have measured data
        if result_dict["multiple_qoi"]:
            updated_qoi_column = []
            updated_qoi_column_measured = []
            updated_read_measured_data = []
            for idx, single_qoi_column in enumerate(qoi_column):
                if read_measured_data[idx]:
                    updated_qoi_column.append(single_qoi_column)
                    updated_qoi_column_measured.append(qoi_column_measured[idx])
                    updated_read_measured_data.append(True)
            # here, we overwrite qoi_column such that it only contains columns which do have a corresponding measured data
            qoi_column = updated_qoi_column
            qoi_column_measured = updated_qoi_column_measured
            read_measured_data = updated_read_measured_data
            result_dict["qoi_column"] = qoi_column
            result_dict["qoi_column_measured"] = qoi_column_measured
        else:
            if not read_measured_data:
                raise ValueError
        objective_function_qoi = dict_config_simulation_settings.get("objective_function_qoi", "all")
        objective_function_qoi = gof_list_to_function_names(objective_function_qoi)
        if isinstance(objective_function_qoi, list):
            objective_function_names_qoi = [
                single_gof.__name__ if callable(single_gof) else single_gof \
                for single_gof in objective_function_qoi]
            list_objective_function_qoi = objective_function_qoi.copy()
            list_objective_function_names_qoi = objective_function_names_qoi.copy()
        else:
            list_objective_function_qoi = [objective_function_qoi, ]
            if callable(objective_function_qoi):
                objective_function_names_qoi = objective_function_qoi.__name__
            else:
                objective_function_names_qoi = objective_function_qoi
            list_objective_function_names_qoi = [objective_function_names_qoi,]
    result_dict["objective_function_qoi"] = objective_function_qoi
    result_dict["objective_function_names_qoi"] = objective_function_names_qoi
    result_dict["list_objective_function_qoi"] = list_objective_function_qoi
    result_dict["list_objective_function_names_qoi"] = list_objective_function_names_qoi

    # Create a list version of some configuration parameters which might be needed when computing GoF
    # if self.qoi.lower() == GOF.lower() or self.calculate_GoF:
    if not isinstance(qoi_column, list):
        list_qoi_column = [qoi_column, ]
    else:
        list_qoi_column = qoi_column.copy()
    if not isinstance(qoi_column_measured, list):
        list_qoi_column_measured = [qoi_column_measured, ]
    else:
        list_qoi_column_measured = qoi_column_measured.copy()
    if not isinstance(read_measured_data, list):
        list_read_measured_data = [read_measured_data, ]
    else:
        list_read_measured_data = read_measured_data.copy()
    if not isinstance(transform_model_output, list):
        list_transform_model_output = [transform_model_output,]
    else:
        list_transform_model_output = transform_model_output.copy()
    result_dict["list_qoi_column"] = list_qoi_column
    result_dict["list_qoi_column_measured"] = list_qoi_column_measured
    result_dict["list_read_measured_data"] = list_read_measured_data
    result_dict["list_transform_model_output"] = list_transform_model_output

    assert len(list_qoi_column) == len(list_qoi_column_measured)
    assert len(list_qoi_column_measured) == len(list_read_measured_data)

    dict_qoi_column_and_measured_info = defaultdict(tuple)
    for idx, single_qoi_column in enumerate(list_qoi_column):
        dict_qoi_column_and_measured_info[single_qoi_column] = \
            (list_read_measured_data[idx], list_qoi_column_measured[idx], list_transform_model_output[idx])
    result_dict["dict_qoi_column_and_measured_info"] = dict_qoi_column_and_measured_info

    mode = dict_config_simulation_settings.get("mode", "continuous")
    if mode != "continuous" and mode != "sliding_window" and mode != "resampling":
        raise Exception(f"[ERROR] mode should have one of the following values:"
                        f" \"continuous\" or \"sliding_window\" or \"resampling\"")
    interval = dict_config_simulation_settings.get("interval", 24)
    min_periods = dict_config_simulation_settings.get("min_periods", 1)
    method = dict_config_simulation_settings.get("method", "avrg")
    center = dict_config_simulation_settings.get("center", "center")
    if center != "center" and center != "left" and center != "right":
        raise Exception(f"[ERROR:] center should be either \"center\" or \"left\" or \"right\"")
    if method != "avrg" and method != "max" and method != "min":
        raise Exception(f"[ERROR:] method should be either \"avrg\" or \"max\" or \"min\"")
    result_dict["mode"] = mode
    result_dict["method"] = method
    result_dict["interval"] = interval
    result_dict["min_periods"] = min_periods
    result_dict["center"] = center

    compute_gradients = strtobool(dict_config_simulation_settings.get("compute_gradients", "False"))
    result_dict["compute_gradients"] = compute_gradients

    CD = None
    eps_val_global = None
    compute_active_subspaces = False
    save_gradient_related_runs = False
    gradient_analysis = False
    if compute_gradients:
        gradient_method = dict_config_simulation_settings.get("gradient_method", "Forward Difference")
        if gradient_method == "Central Difference":
            CD = 1  # flag for using Central Differences (with 2 * num_evaluations)
        elif gradient_method == "Forward Difference":
            CD = 0  # flag for using Forward Differences (with num_evaluations)
        else:
            raise Exception(f"NUMERICAL GRADIENT EVALUATION ERROR: "
                            f"Only \"Central Difference\" and \"Forward Difference\" supported")
        eps_val_global = dict_config_simulation_settings.get("eps_gradients", 1e-4)

        compute_active_subspaces = strtobool(
            dict_config_simulation_settings.get("compute_active_subspaces", "False"))
        save_gradient_related_runs = strtobool(
            dict_config_simulation_settings.get("save_gradient_related_runs", "False"))
        gradient_analysis = strtobool(
            dict_config_simulation_settings.get("gradient_analysis", "False"))

    result_dict["CD"] = CD
    result_dict["eps_val_global"] = eps_val_global
    result_dict["compute_active_subspaces"] = compute_active_subspaces
    result_dict["save_gradient_related_runs"] = save_gradient_related_runs
    result_dict["gradient_analysis"] = gradient_analysis

    if "condition_results_based_on_metric" in kwargs:
        condition_results_based_on_metric = kwargs['condition_results_based_on_metric']
    else:
        condition_results_based_on_metric = dict_config_simulation_settings.get("condition_results_based_on_metric", None)

    if "condition_results_based_on_metric_value" in kwargs:
        condition_results_based_on_metric_value = kwargs['condition_results_based_on_metric_value']
    else:
        condition_results_based_on_metric_value = dict_config_simulation_settings.get("condition_results_based_on_metric_value", None)
    if "condition_results_based_on_metric_sign" in kwargs:
        condition_results_based_on_metric_sign = kwargs['condition_results_based_on_metric_sign']
    else:
        condition_results_based_on_metric_sign = dict_config_simulation_settings.get("condition_results_based_on_metric_sign", "smaller")
    result_dict["condition_results_based_on_metric"] = condition_results_based_on_metric
    result_dict["condition_results_based_on_metric_value"] = condition_results_based_on_metric_value
    result_dict["condition_results_based_on_metric_sign"] = condition_results_based_on_metric_sign

    return result_dict

# def infer_qoi_column_names(dict_processed_simulation_settings_from_config_file, **kwargs):
#     always_process_original_model_output = kwargs.get("always_process_original_model_output", False)
#     list_qoi_column_processed = []
#
#     if dict_processed_simulation_settings_from_config_file["mode"] == "continuous":
#         if dict_processed_simulation_settings_from_config_file["qoi"] == "GoF":
#             for idx, single_qoi_column in enumerate(
#                     dict_processed_simulation_settings_from_config_file["list_qoi_column"]):
#                 if dict_processed_simulation_settings_from_config_file["list_read_measured_data"][idx]:
#                     for single_objective_function_name_qoi in \
#                             dict_processed_simulation_settings_from_config_file["list_objective_function_names_qoi"]:
#                         new_column_name = single_objective_function_name_qoi + "_" + single_qoi_column
#                         list_qoi_column_processed.append(new_column_name)
#                         # self.additional_qoi_columns_besides_original_model_output = True
#                         # self.qoi_is_a_single_number = True
#                         # TODO in this case QoI is a single number, not a time-series!!!
#         else:
#             # here, model output itself is regarded as a QoI
#             pass
#             # list_qoi_column_processed.append(self.list_qoi_column)
#     elif self.mode == "sliding_window":
#         if self.qoi == "GoF":
#             for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                 if self.list_read_measured_data[idx]:
#                     for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
#                         new_column_name = single_objective_function_name_qoi + "_" + single_qoi_column + \
#                                           "_sliding_window"
#                         list_qoi_column_processed.append(new_column_name)
#                         # self.additional_qoi_columns_besides_original_model_output = True
#         else:
#             for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                 new_column_name = single_qoi_column + "_" + self.method + "_sliding_window"
#                 list_qoi_column_processed.append(new_column_name)
#                 # self.additional_qoi_columns_besides_original_model_output = True
#
#     if self.compute_gradients:
#         if self.gradient_analysis:
#             always_process_original_model_output = True
#             for single_param_name in self.nodeNames:
#                 if self.mode == "continuous":
#                     if self.qoi == "GoF":
#                         for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                             if self.list_read_measured_data[idx]:
#                                 for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
#                                     new_column_name = "d_" + single_objective_function_name_qoi + "_" + \
#                                                       single_qoi_column + "_" + "_d_" + single_param_name
#                                     # list_qoi_column_processed.append(new_column_name)
#                                     self.list_grad_columns.append(new_column_name)
#                                     # self.additional_qoi_columns_besides_original_model_output = True
#                                     # self.qoi_is_a_single_number = True
#                                     # TODO in this case QoI is a single number, not a time-series!!!
#                     else:
#                         for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                             new_column_name = "d_" + single_qoi_column + "_d_" + single_param_name
#                             # list_qoi_column_processed.append(new_column_name)
#                             self.list_grad_columns.append(new_column_name)
#                             # self.additional_qoi_columns_besides_original_model_output = True
#                 elif self.mode == "sliding_window":
#                     if self.qoi == "GoF":
#                         for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                             if self.list_read_measured_data[idx]:
#                                 for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
#                                     new_column_name = "d_" + single_objective_function_name_qoi + "_" + \
#                                                       single_qoi_column + "_sliding_window" \
#                                                       + "_d_" + single_param_name
#                                     # list_qoi_column_processed.append(new_column_name)
#                                     self.list_grad_columns.append(new_column_name)
#                                     # self.additional_qoi_columns_besides_original_model_output = True
#                     else:
#                         for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                             new_column_name = "d_" + single_qoi_column + "_" + self.method + "_sliding_window" + \
#                                               "_d_" + single_param_name
#                             # list_qoi_column_processed.append(new_column_name)
#                             self.list_grad_columns.append(new_column_name)
#                             # self.additional_qoi_columns_besides_original_model_output = True
#
#         elif self.compute_active_subspaces:
#             always_process_original_model_output = True
#             pass
#
#     if self.corrupt_forcing_data:
#         always_process_original_model_output = True
#         pass  # 'precipitation' column is in the results df
#
#     wrong_computation_of_new_qoi_columns = self.additional_qoi_columns_besides_original_model_output and \
#                           len(list_qoi_column_processed) == 0
#     assert not wrong_computation_of_new_qoi_columns
#
#     if self.additional_qoi_columns_besides_original_model_output and len(list_qoi_column_processed) != 0:
#         if always_process_original_model_output:
#             self.list_qoi_column = self.list_original_model_output_columns + list_qoi_column_processed
#         else:
#             self.list_qoi_column = list_qoi_column_processed

#####################################
# Functions related to parameters, mainly reading parameters from configuration object
#####################################


def get_list_of_uncertain_parameters_from_configuration_dict(configurationObject: Union[dict, pd.DataFrame, str, pathlib.PosixPath, Any], raise_error:Optional[bool]=False, uq_method: str="ensemble") -> List[str]:
    list_of_parameters = get_list_of_parameters_dicts_from_configuration_dict(configurationObject, raise_error=raise_error)
    nodeNames = []
    for i in list_of_parameters:
        if uq_method == "ensemble" or i["distribution"] != "None":
            nodeNames.append(i["name"])
    labels = [nodeName.strip() for nodeName in nodeNames]
    return nodeNames


def get_list_of_parameters_dicts_from_configuration_dict(configurationObject: Union[dict, pd.DataFrame, str, pathlib.PosixPath, Any], raise_error:Optional[bool]=False) -> List[dict]:
    configurationObject = check_if_configurationObject_is_in_right_format_and_return(configurationObject,
                                                                                     raise_error=raise_error)
    result_list = []
    if configurationObject is None and not raise_error:
        return result_list
    elif configurationObject is None and raise_error:
        raise Exception("configurationObject is None!")
    result_list = configurationObject.get("parameters", [])
    if not result_list:
        if raise_error:
            raise Exception("Function get_list_of_parameters_dicts_from_configuration_dict could not get list of parameters from configurationObject!")
        else:
            print("Note that the function could not get list of parameters from configurationObject!")
    return result_list


def get_list_of_parameters_dicts_from_configurationObject(configurationObject: Union[dict, List, pd.DataFrame, str, pathlib.PosixPath, Any]) -> List[dict]:
    """
    This function is used to extract the list of parameters from a configuration object.
    It differs from get_list_of_parameters_dicts_from_configuration_dict  and get_list_of_uncertain_parameters_from_configuration_dict 
    in that it can handle a list of parameters as well
    """
    if isinstance(configurationObject, (dict, pd.DataFrame, str, pathlib.PosixPath)):
        return get_list_of_parameters_dicts_from_configuration_dict(configurationObject, raise_error=False)
    elif isinstance(configurationObject, list):
        return configurationObject
    return []


def get_param_info_dict_from_configurationObject(configurationObject: Union[dict, List, pd.DataFrame, str, pathlib.PosixPath, Any])-> Dict[str, dict]:
    """
    :param configurationObject: dictionary storing information about parameters
    :return: dictionary storing just some information about parameters
    """
    list_of_parameters_dict = get_list_of_parameters_dicts_from_configurationObject(configurationObject)
    
    result_dict = dict()

    if not list_of_parameters_dict or list_of_parameters_dict is None:
        raise Exception("Note that the function get_param_info_dict_from_configurationObject will return an empty dict for dict_param_info!")

    for param_entry_dict in list_of_parameters_dict:
        param_name = param_entry_dict.get("name")
        distribution = param_entry_dict.get("distribution", None)
        if "lower_limit" in param_entry_dict:
            lower_limit = param_entry_dict["lower_limit"]
        elif "lower" in param_entry_dict:
            lower_limit = param_entry_dict["lower"]
        else:
            lower_limit = None
        if "upper_limit" in param_entry_dict:
            upper_limit = param_entry_dict["upper_limit"]
        elif "upper" in param_entry_dict:
            upper_limit = param_entry_dict["upper"]
        else:
            upper_limit = None
        # lower_limit = param_entry_dict.get("lower_limit", None)
        # upper_limit = param_entry_dict.get("upper_limit", None)
        default_value = param_entry_dict.get("default", None)
        # parameter_value = param_entry_dict.get("value", None)

        result_dict[param_name] = {
            'distribution': distribution, 'default_value': default_value,
            'lower_limit': lower_limit, 'upper_limit': upper_limit
        }
    return result_dict


def get_param_info_dict(default_par_info_dict: dict, configurationObject: Union[dict, List, pd.DataFrame, str, pathlib.PosixPath, Any]= None)-> Dict[str, dict]:
    """
    This function differs from get_param_info_dict_from_configurationObject in that it also fills in the missing paramter information from default_par_info_dict
    :param default_par_info_dict: dictionary storing information about parameters
    :param configurationObject: dictionary storing information about parameters
    :return: filtered dictionary storing information about parameters obtained both from default_par_info_dict and configurationObject
    """
    result_dict = defaultdict(dict)

    list_of_parameters_dict = get_list_of_parameters_dicts_from_configurationObject(configurationObject)

    if list_of_parameters_dict and list_of_parameters_dict is not None:
        for param_entry_dict in list_of_parameters_dict:
            param_name = param_entry_dict.get("name")
            # list_of_params_names_from_configurationObject.append(param_name)
            distribution = param_entry_dict.get("distribution", None)
            if "lower_limit" in param_entry_dict:
                lower_limit = param_entry_dict["lower_limit"]
            elif "lower" in param_entry_dict:
                lower_limit = param_entry_dict["lower"]
            else:
                lower_limit = None
            if "upper_limit" in param_entry_dict:
                upper_limit = param_entry_dict["upper_limit"]
            elif "upper" in param_entry_dict:
                upper_limit = param_entry_dict["upper"]
            else:
                upper_limit = None
            # lower_limit = param_entry_dict.get("lower_limit", None)
            # upper_limit = param_entry_dict.get("upper_limit", None)
            default = param_entry_dict.get("default", None)
            # parameter_value = param_entry_dict.get("value", None)
            result_dict[param_name] = {
                'distribution': distribution, 'default': default,
                'lower_limit': lower_limit, 'upper_limit': upper_limit
            }

    for single_param_name in default_par_info_dict.keys():
        if single_param_name in result_dict:
            continue
        else:
            param_entry_dict = default_par_info_dict[single_param_name]
            param_name = single_param_name
            distribution = param_entry_dict.get("distribution", None)
            default = param_entry_dict.get("default", None)
            if "lower_limit" in param_entry_dict:
                lower_limit = param_entry_dict["lower_limit"]
            elif "lower" in param_entry_dict:
                lower_limit = param_entry_dict["lower"]
            else:
                lower_limit = None
            if "upper_limit" in param_entry_dict:
                upper_limit = param_entry_dict["upper_limit"]
            elif "upper" in param_entry_dict:
                upper_limit = param_entry_dict["upper"]
            else:
                upper_limit = None
            result_dict[param_name] = {
                'distribution': distribution, 'default': default,
                'lower_limit': lower_limit, 'upper_limit': upper_limit
            }

    return result_dict


def _get_default_value_from_default_par_info_dict(param_name, default_par_info_dict):
    if default_par_info_dict is None:
        raise ValueError("default_par_info_dict is None!")
    if isinstance(default_par_info_dict[param_name], dict):
        return default_par_info_dict[param_name]["default"]
    return default_par_info_dict[param_name]


def get_list_of_parameters_dict_from_default_par_info_dict(default_par_info_dict):
    return [{"name": name, **info} for name, info in default_par_info_dict.items()]


def _extract_parameters_from_list_of_parameters_from_configurationObject(list_of_parameters_dict, default_par_info_dict):
    """
    Extracts parameters from a list of parameters obtained from a configuration object.

    Args:
        list_of_parameters_dict (list): A list of parameters obtained from a configuration object.
        default_par_info_dict (dict): A dictionary containing default parameter information.

    Returns:
        dict: A dictionary containing the extracted parameters.

    """
    parameters_dict = {}
    for single_param in list_of_parameters_dict:
        if "value" in single_param:
            parameters_dict[single_param['name']] = single_param["value"]
        elif "default" in single_param:
            parameters_dict[single_param['name']] = single_param["default"]
        else:
            parameters_dict[single_param['name']] = _get_default_value_from_default_par_info_dict(single_param['name'], default_par_info_dict)
    return parameters_dict


def _get_parameter_value(single_param, parameters=None, uncertain_param_counter=0, take_direct_value=False, default_par_info_dict=None):
    if take_direct_value:
        if parameters is not None:
            return parameters[uncertain_param_counter]
        elif single_param is not None and "name" in single_param:
            return _get_default_value_from_default_par_info_dict(single_param['name'], default_par_info)
        else:
            raise ValueError("parameters is None and single_param is None!")
    elif single_param is not None and "value" in single_param:
        return single_param["value"]
    elif single_param is not None and "default" in single_param:
        return single_param["default"]
    elif single_param is not None and "name" in single_param:
        return _get_default_value_from_default_par_info_dict(single_param['name'], default_par_info_dict)
    else:
        raise ValueError("single_param is None!")

def _handle_no_parameters(configurationObject, default_par_info_dict):
    if configurationObject is not None:
        list_of_parameters_dict = get_list_of_parameters_dicts_from_configurationObject(configurationObject)
        if list_of_parameters_dict:
            return _extract_parameters_from_list_of_parameters_from_configurationObject(list_of_parameters_dict, default_par_info_dict)
    if default_par_info_dict is not None:
        return default_par_info_dict
    else:
        raise ValueError("No parameters and no default_par_info_dict provided. Returning an empty dictionary.")


def _handle_parameters(parameters, configurationObject, default_par_info_dict, take_direct_value):
    """
    This function is used to extract the parameters from the configuration object and assigne them
    the conrete value from parameters (dictionary or list).

    Args:
        parameters (dict, or list): A list of parameters.
        configurationObject (dict): The configuration object containing information about the parameters.
        default_par_info_dict (dict): A dictionary containing default parameter information.
        take_direct_value (bool): A boolean value indicating whether to take the direct value.

    Returns:
        dict: A dictionary containing paramter names as keys and parameter values (either extracted from the parameters
        argument, or from the configuration object values, or from the default parameter information dictionary) as values.

    Important control variable is take_direct_value. If take_direct_value is True, 
    then the function will take the parameter values directly from the parameters variable.     

    The final dictionary will cotain entries for all the parameters which are present in the configurationObject (both cetain and uncertain parameters).
    """
    parameters_dict = {}

    if isinstance(parameters, dict) and take_direct_value:
        parameters_dict = parameters
    else:   
        uncertain_param_counter = 0
        if configurationObject and configurationObject is not None:
            list_of_parameters_dict = get_list_of_parameters_dicts_from_configurationObject(configurationObject)
        elif default_par_info_dict is not None:
            list_of_parameters_dict = get_list_of_parameters_dict_from_default_par_info_dict(default_par_info_dict)
        else:
            # raise ValueError("configurationObject is None and default_par_info_dict is None! Returning an empty dictionary.")
            print("configurationObject is None and default_par_info_dict is None! Returning an empty dictionary.")
            return parameters_dict

        if not isinstance(parameters, list) and isinstance(parameters, (int, float, complex)):
            parameters = [parameters]

        for single_param in list_of_parameters_dict:
            if 'distribution' in single_param and single_param['distribution'] != "None":
                parameters_dict[single_param['name']] = parameters[uncertain_param_counter]
                uncertain_param_counter += 1
            else:
                parameters_dict[single_param['name']] = _get_parameter_value(single_param, parameters, uncertain_param_counter, take_direct_value, default_par_info_dict)
                if take_direct_value:
                    uncertain_param_counter += 1

    return parameters_dict


def parameters_configuration(parameters, configurationObject: Union[dict, List, pd.DataFrame, str, pathlib.PosixPath, Any], default_par_info_dict, take_direct_value: bool = False) -> dict:
    """
    This function is only for legacy purposes. Use configuring_parameter_values instead
    """
    return configuring_parameter_values(parameters, configurationObject, default_par_info_dict, take_direct_value=take_direct_value)


def configuring_parameter_values(parameters, configurationObject: Union[dict, List, pd.DataFrame, str, pathlib.PosixPath, Any]=None, default_par_info_dict: dict=None, take_direct_value: bool = False) -> dict:
    """
    Note: If not take_direct_value and parameters!= None, parameters_dict will contain
    some value for every single parameter in configurationObject (e.g., it might at the end have more entries than the
    input parameters variable). This it (as of the Nov 2024) in contrast to the logic in TimeDependentModel._parameters_configuration,
    where only the values for the uncertain parameters should be present in the parameters_dict returned by this!
    :param parameters:
    :type parameters: dictionary or array storing all uncertain parameters
       in the same order as parameters are listed in configurationObject
    :param configurationObject (dict, List): dictionary/list/path to json dictionary storing information about parameters, 
    or list dictionaries where each dictionary stores information about the particular parameter. If parameters is None, the function will try to extract the parameters from configurationObject!
    :param default_par_info_dict: dictionary storing default information about parameters. If parameters is None, and the configurationObject is either None or does not contian parameter values, the
    function will try to extract the parameters from default_par_info_dict!  
    :param take_direct_value: 
        take_direct_value should be True if parameter_value_dict is a dict with keys being paramter name and values being parameter values;
        if parameter_value_dict is a list of parameter values corresponding to the order of the parameters in the configuration file, then take_direct_value should be False
    :return parameters_dict: dictionary storing parameter names as keys and parameter values as values
    """
    parameters_dict = dict() #defaultdict()  # copy.deepcopy(DEFAULT_PAR_VALUES_DICT)

    if parameters is None:
        parameters_dict = _handle_no_parameters(configurationObject, default_par_info_dict)
    else:
        parameters_dict = _handle_parameters(parameters, configurationObject, default_par_info_dict, take_direct_value)
    
    return parameters_dict


def generate_distributions(configurationObject, algorithm="mc"):
    """
    Generate joint and standard distributions based on the given configuration object.

    Args:
        configurationObject (dict): The configuration object containing information about the parameters and their distributions.
        algorithm (str, optional): The UQ algorithm to use. Options: "mc", "sc", "pce", "kl". Defaults to "mc".

    Returns:
        tuple: A tuple containing the joint distribution, the standard distribution, and a list of parameter names.

    Raises:
        NotImplementedError: If the specified UQ algorithm is not supported yet or if the distribution is not supported yet.

    Note:
        Currently only "Uniform", "LogUniform", and "Normal" are supported
    """
    list_of_single_dist = []
    list_of_single_standard_dist = []
    param_names = []

    for param in configurationObject["parameters"]:
        param_names.append(param["name"])
        if param["distribution"] == "Uniform":
            list_of_single_dist.append(cp.Uniform(param["lower"], param["upper"]))
            if algorithm == "mc" or algorithm == "sampling" or algorithm == "samples":
                list_of_single_standard_dist.append(cp.Uniform(0, 1))
            elif algorithm == "kl" or algorithm == "pce" or algorithm == "sc":
                list_of_single_standard_dist.append(cp.Uniform(-1, 1))
            else:
                raise NotImplementedError(f"Sorry, the algorithm {algorithm} is not supported yet")
        elif param["distribution"] == "LogUniform":
            list_of_single_dist.append(cp.LogUniform(param["lower"], param["upper"]))
            list_of_single_standard_dist.append(cp.LogUniform())
        elif param["distribution"] == "Normal":
            list_of_single_dist.append(cp.Normal(param["mu"], param["sigma"]))
            list_of_single_standard_dist.append(cp.Normal())
        # TODO add a general branch that checks if the distribution exists in chaospy
        else:
            raise NotImplementedError(f"Sorry, the distribution {param['distribution']} is not supported yet")

    joint_dist = cp.J(*list_of_single_dist)
    joint_dist_standard  = cp.J(*list_of_single_standard_dist)

    return joint_dist, joint_dist_standard, param_names

# def parameters_configuration_for_gradient_approximation(
#         parameters_dict, configurationObject, parameter_index_to_perturb, eps_val=1e-4, take_direct_value=False):
#
#     info_dict_on_perturbed_param = dict()
#
#     configurationObject = utility._check_if_configurationObject_is_in_right_format_and_return(configurationObject)
#     uncertain_param_counter = 0
#     for id, single_param in enumerate(configurationObject['parameters']):
#         # TODO if uncertain_param_counter != parameter_index_to_perturb:
#         if id != parameter_index_to_perturb:
#             if single_param['distribution'] != "None" and parameters[uncertain_param_counter] is not None:
#                 parameters_dict[single_param['name']] = parameters[uncertain_param_counter]
#                 uncertain_param_counter += 1
#             else:
#                 if "value" in single_param:
#                     parameters_dict[single_param['name']] = single_param["value"]
#                 elif "default" in single_param:
#                     parameters_dict[single_param['name']] = single_param["default"]
#                 else:
#                     parameters_dict[single_param['name']] = DEFAULT_PAR_VALUES_DICT[single_param['name']]
#         else:
#             if "lower_limit" in single_param:
#                 parameter_lower_limit = single_param["lower_limit"]
#             elif "lower" in single_param:
#                 parameter_lower_limit = single_param["lower"]
#             else:
#                 parameter_lower_limit = None
#
#             if "upper_limit" in single_param:
#                 parameter_upper_limit = single_param["upper_limit"]
#             elif "upper" in single_param:
#                 parameter_upper_limit = single_param["upper"]
#             else:
#                 parameter_upper_limit = None
#
#             if parameter_lower_limit is None or parameter_upper_limit is None:
#                 raise Exception(
#                     'ERROR in configuring_parameter_values: perturb_sinlge_param_around_nominal is set to True but '
#                     'parameter_lower_limit or parameter_upper_limit are not specified!')
#             else:
#                 param_h = eps_val * (parameter_upper_limit - parameter_lower_limit)
#                 parameter_lower_limit += param_h
#                 parameter_upper_limit -= param_h
#
#             if single_param['distribution'] != "None" and parameters[uncertain_param_counter] is not None:
#                 new_parameter_value = parameters[uncertain_param_counter] + param_h
#                 parameters_dict[single_param['name']] = (new_parameter_value, param_h)
#                 uncertain_param_counter += 1
#             else:
#                 if "value" in single_param:
#                     parameters_dict[single_param['name']] = single_param["value"] + param_h
#                 elif "default" in single_param:
#                     parameters_dict[single_param['name']] = single_param["default"] + param_h
#                 else:
#                     parameters_dict[single_param['name']] = DEFAULT_PAR_VALUES_DICT[single_param['name']] + param_h
#
#             info_dict_on_perturbed_param = {
#                 "uncertain_param_counter": uncertain_param_counter, "id": id,
#                 "name": single_param['name'], "param_h": param_h}
#
#     return parameters_dict, info_dict_on_perturbed_param


def update_parameter_dict_for_gradient_computation(parameters, configurationObject, take_direct_value=False,
                                                   perturb_single_param_around_nominal=False,
                                                   parameter_index_to_perturb=0, eps_val=1e-4
                                                   ):
    # TODO Rewrite bigger part of the function above
    # iterate through all the parameters
    list_of_parameters_from_json = configurationObject["parameters"]

    for id, param_entry_dict in enumerate(list_of_parameters_from_json):
        if perturb_single_param_around_nominal and id != parameter_index_to_perturb:
            continue

    parameter_lower_limit = param_entry_dict["lower_limit"] if "lower_limit" in param_entry_dict else None
    parameter_upper_limit = param_entry_dict["upper_limit"] if "upper_limit" in param_entry_dict else None
    param_h = eps_val * (parameter_upper_limit - parameter_lower_limit)
    parameter_lower_limit += param_h
    parameter_upper_limit -= param_h
    raise NotImplementedError

###################################################################################################################
# Reading saved data - produced by UQEF related run/simulation
###################################################################################################################
# Note: think about moving these functions to UQEF-Dynamic/uqef_dynamic/utils/uqPostProcessing.py


def load_uqsim_args_dict(file):
    with open(file, 'rb') as f:
        uqsim_args = pickle.load(f)
    uqsim_args_dict = vars(uqsim_args)
    return uqsim_args_dict


def read_and_print_uqsim_args_file(file):
    with open(file, 'rb') as f:
        uqsim_args = pickle.load(f)
    uqsim_args_temp_dict = vars(uqsim_args)
    print(f"UQSIM.ARGS")
    for key, value in uqsim_args_temp_dict.items():
        print(f"{key}: {value}")
    return uqsim_args


def load_configuration_object(file):
    with open(file, 'rb') as f:
        configurationObject = dill.load(f)
    return configurationObject


def _get_df_simulation_from_file(working_folder, df_all_simulations_name="df_all_simulations.pkl"):
    df_all_simulations = pathlib.Path(working_folder) / df_all_simulations_name
    df_all_simulations = pd.read_pickle(df_all_simulations, compression='gzip')
    return df_all_simulations


def _get_nodes_from_file(working_folder, dill_or_pickle="dill"):
    working_folder = pathlib_to_from_str(working_folder, transfrom_to='Path')
    nodes_dict = working_folder / 'nodes.simnodes'
    with open(nodes_dict, 'rb') as f:
        if dill_or_pickle == "dill":
            simulationNodes = dill.load(f)
        elif dill_or_pickle == "pickle":
            simulationNodes = pickle.load(f)
        else:
            simulationNodes = None
    return simulationNodes


def _get_df_index_parameter_gof_from_file(
        working_folder, df_index_parameter_gof_values_file_name="df_index_parameter_gof_values.pkl"):
    df_index_parameter_gof_values_file = pathlib.Path(working_folder) / df_index_parameter_gof_values_file_name
    df_index_parameter_gof_values = pd.read_pickle(df_index_parameter_gof_values_file, compression='gzip')
    return df_index_parameter_gof_values


def _get_statistics_dict(working_folder, statistics_dictionary_file_name="statistics_dictionary.pkl"):
    statistics_dictionary_file = pathlib.Path(working_folder) / statistics_dictionary_file_name
    statistics_dictionary = None
    if statistics_dictionary_file.is_file():
        with open(statistics_dictionary_file, 'rb') as f:
            statistics_dictionary = pickle.load(f)
    return statistics_dictionary


def create_statistics_dictionary_from_saved_single_qoi_statistics_dictionary(workingDir, list_qoi_column):
    statistics_dictionary = defaultdict(dict)
    for single_qoi in list_qoi_column:
        statistics_dictionary_file_temp = workingDir / f"statistics_dictionary_qoi_{single_qoi}.pkl"
        assert statistics_dictionary_file_temp.is_file()
        with open(statistics_dictionary_file_temp, 'rb') as f:
            statistics_dictionary_temp = pickle.load(f)
        statistics_dictionary[single_qoi] = statistics_dictionary_temp
    return statistics_dictionary


def read_all_saved_simulations(df_all_simulations_file):
    if df_all_simulations_file.is_file():
        # Reading Saved Simulations - Note: This migh be a huge file,
        # especially for MC/Saltelli kind of simulations
        df_simulation_result = pd.read_pickle(df_all_simulations_file, compression="gzip")
    else:
        df_simulation_result = None
    return df_simulation_result

###################################################################################################################
# Plotting params and GoF values - mostly from df_index_parameter or df_index_parameter_gof filtered for a single station
###################################################################################################################


def _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof):
    # TODO think if one should distinguish between only stochastic parameters
    # in that case use get_list_of_uncertain_parameters_from_configuration_dict
    def check_if_column_stores_parameter(x):
        if isinstance(x, tuple):
            x = x[0]
        return x not in list(mapping_gof_names_to_functions.keys()) and not x.startswith("d_") and \
               x.lower() not in ["index_run", "station", "successful_run", "success", "qoi"]
    return [x for x in df_index_parameter_gof.columns.tolist() if check_if_column_stores_parameter(x)]


def _get_gof_columns_df_index_parameter_gof(df_index_parameter_gof):
    def check_if_column_stores_gof(x):
        if isinstance(x, tuple):
            x = x[0]
        return x in list(mapping_gof_names_to_functions.keys()) and x.lower() not in ["index_run", "station", "successful_run", "qoi"]
    return [x for x in df_index_parameter_gof.columns.tolist() if check_if_column_stores_gof(x)]


def _get_grad_columns_df_index_parameter_gof(df_index_parameter_gof):
    return [x for x in df_index_parameter_gof.columns.tolist() if x.startswith("d_")]


def plot_hist_of_gof_values_from_df(df_index_parameter_gof, name_of_gof_column="NSE"):
    return df_index_parameter_gof.hist(name_of_gof_column)


def plot_hist(ax, data, param_name, num_bins=20):
    counts, bins, _ = ax.hist(data, bins=num_bins)
    ax.set_xlabel(param_name)


def plot_log_hist(ax, data, param_name, num_bins=20):
    counts, bins, _ = ax.hist(data, bins=num_bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax.cla()
    counts, bins, _ = ax.hist(data, bins=logbins)
    ax.set_xscale('log')
    ax.set_xlabel(param_name)


def plot_subplot_params_hist_from_df_and_configurationObject(df_index_parameter, configurationObject):
    list_of_params_dict_info = configurationObject['parameters']
    fig, axes = plt.subplots(len(list_of_params_dict_info), 1, figsize=(6,12))
    index = 0
    for single_param_dict in list_of_params_dict_info:
        if single_param_dict["distribution"] == "Uniform" or single_param_dict["distribution"] == "Normal":
            plot_hist(axes[index], df_index_parameter[single_param_dict["name"]], single_param_dict["name"], num_bins=50)
        if single_param_dict["distribution"] == "LogUniform" or single_param_dict["distribution"] == "LogNormal":
            plot_log_hist(axes[index], df_index_parameter[single_param_dict["name"]], single_param_dict["name"], num_bins=50)
        index += 1
    plt.tight_layout()
    return fig


def plot_subplot_params_hist_from_df(df_index_parameter_gof):
    columns_with_parameters = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    fig = make_subplots(rows=1, cols=len(columns_with_parameters))
    for i in range(len(columns_with_parameters)):
        if isinstance(columns_with_parameters[i], tuple):
            param_name = columns_with_parameters[i][0]
            fig.append_trace(
                go.Histogram(
                    x=df_index_parameter_gof[columns_with_parameters[i]],
                    name=param_name
                ), row=1, col=i + 1)
            fig.update_xaxes(title_text=f'{param_name}', row=1, col=i + 1)
        else:
            param_name = columns_with_parameters[i]
            fig.append_trace(
                go.Histogram(
                    x=df_index_parameter_gof[columns_with_parameters[i]],
                    name=param_name
                ), row=1, col=i + 1)
            fig.update_xaxes(title_text=f'{param_name}', row=1, col=i + 1)
    fig.add_annotation(
        dict(
        text='Histrogram',
        x=-0.05,  # Position the label outside the left side of the plot
        y=0.5,
        showarrow=False,
        textangle=-90,  # Rotate the text
        xref='paper',
        yref='paper',
        font=dict(size=16)
        )
        )
    return fig


def generate_mask_based_on_multiple_column_comparison(df, column_name, threshold_value, comparison):
    """
    Generate a boolean mask based on multiple column comparisons.

    Args:
        df (pandas.DataFrame): The DataFrame to generate the mask from.
        column_name (str or list): The name(s) of the column(s) to compare.
        threshold_value (float or list): The threshold value(s) to compare against.
        comparison (str or list): The comparison operator(s) to use for the comparison.

    Returns:
        pandas.Series: A boolean mask indicating which rows satisfy all the conditions.

    Raises:
        ValueError: If the lengths of `column_name`, `threshold_value`, and `comparison` are not the same.

    """
    if not isinstance(column_name, list):
        column_name = [column_name,]
    if not isinstance(threshold_value, list):
        threshold_value = [threshold_value,]
    if not isinstance(comparison, list):
        comparison = [comparison,]
    if len(column_name) != len(threshold_value) or len(column_name) != len(comparison):
        raise ValueError("Length of column_name, threshold_value and comparison should be the same!")
    # Initialize `mask` to True for all rows initially
    mask = pd.Series([True] * len(df), index=df.index)
    for single_column_name, single_threshold_value, single_comparison in zip(column_name, threshold_value, comparison):
        # Generate a mask for the current condition
        current_mask = generate_mask_based_on_column_comparison(df, single_column_name, single_threshold_value, single_comparison)
        # Combine the new mask with the cumulative mask using `&`
        mask = mask & current_mask
    return mask


def generate_mask_based_on_column_comparison(df, column_name, threshold_value, comparison="smaller"):
    """
    comparison should be: "smaller", "greater", "equal", "not_equal", "smaller_or_equal",  "greater_or_equal"
    '>', '<=', '=='
    """
    if comparison not in ["smaller", "greater", "equal", "not_equal", "smaller_or_equal", "greater_or_equal",
    '==', '!=', '<', '>', '<=', '>=']:
        raise Exception("Comparison should be: \"smaller\", \"greater\", \"equal\", \"not_equal\", \"smaller_or_equal\", \"greater_or_equal\" ")
    if comparison == "smaller" or comparison == "<":
        mask = df[column_name] < threshold_value
    elif comparison == "greater" or comparison == ">":
        mask = df[column_name] > threshold_value
    elif comparison == "equal" or comparison == "==":
        mask = df[column_name] == threshold_value
    elif comparison == "not_equal" or comparison == "!=":
        mask = df[column_name] != threshold_value
    elif comparison == "smaller_or_equal" or comparison == "<=":
        mask = df[column_name] <= threshold_value
    elif comparison == "greater_or_equal" or comparison == ">=":
        mask = df[column_name] >= threshold_value
    else:
        raise Exception(\
        "Comparison should be: \"smaller\", \"greater\", \"equal\", \"not_equal\", \"smaller_or_equal\", \"greater_or_equal\" \
        \"==\", \"!=\", \"<\", \">\", \"<=\", \">=\"")
    return mask


def plot_subplot_params_hist_from_df_conditioned(df_index_parameter_gof, name_of_gof_column="NSE",
                                                 threshold_gof_value=0, comparison="smaller"):
    """
    comparison should be: "smaller", "greater", "equal", "not_equal", "smaller_or_equal",  "greater_or_equal"
    """
    mask = generate_mask_based_on_column_comparison(
        df=df_index_parameter_gof, column_name=name_of_gof_column, threshold_value=threshold_gof_value, comparison=comparison)

    columns_with_parameters = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    fig = make_subplots(rows=1, cols=len(columns_with_parameters))
    for i in range(len(columns_with_parameters)):
        if isinstance(columns_with_parameters[i], tuple):
            param_name = columns_with_parameters[i][0]
            fig.append_trace(
                go.Histogram(
                    x=df_index_parameter_gof[mask][columns_with_parameters[i]],
                    name=param_name
                ), row=1, col=i + 1)
            fig.update_xaxes(title_text=f'{param_name}', row=1, col=i + 1)
        else:
            param_name = columns_with_parameters[i]
            fig.append_trace(
                go.Histogram(
                    x=df_index_parameter_gof[mask][columns_with_parameters[i]],
                    name=param_name
                ), row=1, col=i + 1)
            fig.update_xaxes(title_text=f'{param_name}', row=1, col=i + 1)
    fig.add_annotation(
        dict(
        text='Histrogram',
        x=-0.05,  # Position the label outside the left side of the plot
        y=0.5,
        showarrow=False,
        textangle=-90,  # Rotate the text
        xref='paper',
        yref='paper',
        font=dict(size=16)
        )
        )
    return fig

# Problematic function until plotly is updated
# def plot_scatter_matrix_params_vs_gof(
#     df_index_parameter_gof, columns_with_parameters=None, name_of_gof_column="NSE", hover_name="Index_run"):
#     if columns_with_parameters is None:
#         columns_with_parameters = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
#     # fig = px.scatter_matrix(df_index_parameter_gof, dimensions=columns_with_parameters, color=name_of_gof_column)
#     #     # hover_name=hover_name
#     # fig.update_traces(diagonal_visible=False)
#     df = df_index_parameter_gof
#     columns = columns_with_parameters

#     num_columns = len(columns)
#     # Create a subplot grid for scatter matrix
#     fig = make_subplots(rows=num_columns, cols=num_columns,
#                         shared_xaxes=True, shared_yaxes=True,
#                         horizontal_spacing=0.02, vertical_spacing=0.02)

#     for i, x_col in enumerate(columns):
#         for j, y_col in enumerate(columns):
#             if i == j:
#                 # Diagonal cells: Histogram for the column
#                 fig.add_trace(go.Histogram(x=df[x_col], marker=dict(color='lightblue')),
#                             row=i+1, col=j+1)
#             else:
#                 # Off-diagonal cells: Scatter plot for column pairs
#                 fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers',
#                                         marker=dict(size=5, color=df[name_of_gof_column], 
#                                         coloraxis="coloraxis")),
#                             row=i+1, col=j+1)

#             # Update axis labels only on the outer edges
#             if j == 0:
#                 fig.update_yaxes(title_text=y_col, row=i+1, col=j+1)
#             if i == num_columns - 1:
#                 fig.update_xaxes(title_text=x_col, row=i+1, col=j+1)

#     # Update layout
#     fig.update_layout(height=600, width=600, title_text="Custom Scatter Matrix",
#                     showlegend=False)
#     return fig

###################################################################################################################
# Plotting UQ & SA Output
###################################################################################################################


def scatter_3d_params_vs_gof(df_index_parameter_gof, param1, param2, param3, name_of_gof_column="NSE",
                             name_of_index_run_column="Index_run"):
    fig = px.scatter_3d(
        df_index_parameter_gof, x=param1, y=param2, z=param3, color=name_of_gof_column, opacity=0.7,
        hover_data=[name_of_gof_column, name_of_index_run_column]
    )
    return fig


def plot_surface_2d_params_vs_gof(df_index_parameter_gof, param1, param2, num_of_points_in_1d=8,
                                  name_of_gof_column="NSE"):
    # from mpl_toolkits.mplot3d import Axes3D
    # from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    # from matplotlib import cm
    # from matplotlib.ticker import LinearLocator, FormatStrFormatter
    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.gca(projection='3d')

    # x = samples_df_index_parameter_gof['A2'].to_numpy()
    # y = samples_df_index_parameter_gof['BSF'].to_numpy()
    # z = samples_df_index_parameter_gof['RMSE'].to_numpy()

    # X = np.reshape(x, (-1, 8))
    # Y = np.reshape(y, (-1, 8))
    # Z = np.reshape(z, (-1, 8))

    # surf_Rosen = ax.plot_surface(X, Y, Z, cmap=cm.rainbow)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('RMSE')
    # plt.tight_layout()
    x = df_index_parameter_gof[param1].values
    y = df_index_parameter_gof[param2].values
    z = df_index_parameter_gof[name_of_gof_column].values
    X = np.reshape(x, (-1, num_of_points_in_1d))
    Y = np.reshape(y, (-1, num_of_points_in_1d))
    Z = np.reshape(z, (-1, num_of_points_in_1d))
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    fig.update_layout(scene=dict(xaxis_title=param1, yaxis_title=param2, zaxis_title=name_of_gof_column),
                      autosize=False,
                      title='GoF Plot',
                      scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                      width=900, height=500,
                      margin=dict(r=20, b=10, l=10, t=10)
                      )
    return fig


def plot_parallel_params_vs_gof(df_index_parameter_gof, name_of_gof_column="NSE", list_of_params=None):
    if list_of_params is None:
        list_of_params = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    dimensions_columns = list_of_params + [name_of_gof_column,]
    # fig = px.parallel_coordinates(df_index_parameter_gof, color=name_of_gof_column, dimensions=dimensions_columns)
    # Step-by-step Define the dimensions for the parallel coordinates
    dimensions = [
        dict(range=[df_index_parameter_gof[col].min(), df_index_parameter_gof[col].max()],
            label=col,
            values=df_index_parameter_gof[col])
        for col in dimensions_columns
    ]

    # Create the parallel coordinates plot
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=df_index_parameter_gof[name_of_gof_column],
                #colorscale='Viridis',  # Optional color scale
                showscale=True),       # Shows color scale legend
        dimensions=dimensions
    ))
    return fig


def plot_scatter_matrix_params_vs_gof_seaborn(df_index_parameter_gof, columns_with_parameters=None, name_of_gof_column="NSE"):
    if columns_with_parameters is None:
        columns_with_parameters = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(
        df_index_parameter_gof, vars=columns_with_parameters, 
        corner=True, hue=name_of_gof_column
    )
    plt.show()


def scatter_3d_params_static(df_index_parameter_gof, param1, param2, param3):
    fig = plt.figure()
    axs = fig.gca(projection='3d')
    axs.scatter(list(df_index_parameter_gof[param1]),
                list(df_index_parameter_gof[param2]),
                list(df_index_parameter_gof[param3]), marker="o")
    plt.title('Parameters\n')
    axs.set_xlabel(param1)
    axs.set_ylabel(param2)
    axs.set_zlabel(param3)
    plt.show()


def plot_all_model_runs(df_simulation_result: pd.DataFrame, single_qoi_column: str=QOI_COLUMN_NAME, 
                        time_column_name: str=TIME_COLUMN_NAME, index_column_name: str=INDEX_COLUMN_NAME, save_plot: bool=False, workingDir: pathlib.PosixPath=None):
    """
    This function plots all model runs for a single QoI
    :param df_simulation_result: DataFrame storing simulation results; it should contain columns: time, index, QoI
    :param single_qoi_column: name of the column storing the QoI
    :param time_column_name: name of the column storing time
    :param index_column_name: name of the column storing index
    :param save_plot: if True, the plot will be saved
    :param workingDir: directory where the plot will be saved if save_plot is True 
    :return: plotly figure
    """

    total_number_model_runs =  df_simulation_result[index_column_name].nunique()
    list_of_dates_of_interest = df_simulation_result[time_column_name].unique()

    grouped = df_simulation_result.groupby(index_column_name)
    groups = grouped.groups
    keyIter = list(groups.keys())
    fig = go.Figure()
    for key in keyIter:
        df_simulation_result_subset = df_simulation_result.loc[groups[key].values]
        fig.add_trace(
            go.Scatter(
                x=df_simulation_result_subset[time_column_name], 
                y=df_simulation_result_subset[single_qoi_column],
                line_color='LightSkyBlue', mode="lines", opacity=0.3, showlegend=False,
            )
        )

    # model_runs = np.empty((
    #     df_simulation_result[index_column_name].nunique(), df_simulation_result[time_column_name].nunique()
    # ))
    # index_run_local = 0
    # for key in keyIter:
    #     model_runs[index_run_local] = df_simulation_result.loc[groups[key].values, single_qoi_column]
    #     index_run_local += 1
    # lines = [
    #     go.Scatter(
    #         x=list_of_dates_of_interest,
    #         y=single_row,
    #         showlegend=False,
    #         # legendgroup=colours[i],
    #         mode="lines",
    #         line=dict(
    #             color='LightSkyBlue'),
    #         opacity=0.1
    #     )
    #     for single_row in model_runs
    # ]
    # for trace in lines:
    #     fig.add_trace(trace)

    fig.update_traces(mode='lines')
    fig.update_layout(
        title=f"Simulation Results - #runs {total_number_model_runs}",
        xaxis_title="time t",
        yaxis_title=f"{single_qoi_column}",)
    if save_plot and workingDir.is_file():
        fileName = f"model_runs_plot_{single_qoi_column}.html"
        fileName = str(workingDir) + fileName        
        pyo.plot(fig, filename=fileName)
    return fig

# ===================================================================================================================
# Functions for saving the GPCE surrogate model
# ===================================================================================================================
# Note: these set of functions also exist in UQEF-Dynamic/uqef_dynamic/utils/uqPostProcessing.py


def save_gpce_surrogate_model(workingDir, gpce, qoi, timestamp):
    # timestamp = pd.Timestamp(timestamp).strftime('%Y-%m-%d %X')
    fileName = f"gpce_surrogate_{qoi}_{timestamp}.pkl"
    fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
    with open(fullFileName, 'wb') as handle:
        pickle.dump(gpce, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def save_gpce_coeffs(workingDir, coeff, qoi, timestamp):
    fileName = f"gpce_coeffs_{qoi}_{timestamp}.npy"
    fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
    np.save(fullFileName, coeff)


def read_gpce_surrogate_model(workingDir, qoi, timestamp):
    fileName = f"gpce_surrogate_{qoi}_{timestamp}.pkl"
    fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
    with open(fullFileName, 'rb') as f:
        gpce_surrogate = pickle.load(f)
    return gpce_surrogate


def read_gpce_surrogate_model(fullFileName):
    with open(fullFileName, 'rb') as f:
        gpce_surrogate = pickle.load(f)
    return gpce_surrogate

###################################################################################################################
# Running general model
###################################################################################################################

def running_the_model_over_time_and_parameters(model, t: Union[np.array, np.ndarray, List[Union[int, float]]], parameters: np.ndarray, **kwargs) -> np.ndarray:
    """
    This function runs the model over time and parameters
    :model: function
        Model to run
    :t: np.ndarray
        Time array
    :parameters: np.ndarray
        Parameters for the model dimension (dim, number_of_nodes)
    :kwargs: dict
        Additional keyword arguments for the model
    :return: np.ndarray
        Model runs dimension (number_of_nodes, len(t))
    """
    model_runs = np.empty((parameters.shape[1], len(t)))
    for idx, single_node in enumerate(parameters.T):
        model_runs[idx] = model(t, *single_node, **kwargs)
    return model_runs

   
def running_the_model_and_generating_df(
    model, 
    t: Union[np.array, np.ndarray, List[Union[int, float]]], 
    parameters: np.ndarray,
    time_column_name: str=TIME_COLUMN_NAME,
    index_column_name: str= INDEX_COLUMN_NAME
    ) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Runs the model over time and parameters, and generates a DataFrame with the simulation results.

    Args:
        model: The model to be run.
        t: The time array or list.
        parameters: The array of parameters used to stimulate the model. Expected to be of shape dim x number_of_nodes
        time_column_name: The name of the column to store time values in the DataFrame. Defaults to TIME_COLUMN_NAME.
        index_column_name: The name of the column to store index values in the DataFrame. Defaults to INDEX_COLUMN_NAME.

    Returns:
        A tuple containing the model runs as a numpy array and the simulation results as a pandas DataFrame.
    """
    model_runs = running_the_model_over_time_and_parameters(model, t, parameters)
    list_of_single_df = []
    for idx, single_node in enumerate(parameters.T):
        df_temp = pd.DataFrame(model_runs[idx], columns=[QOI_COLUMN_NAME])
        df_temp[time_column_name] = t
        df_temp[index_column_name] = idx
        tuple_column = [tuple(single_node)] * len(df_temp)
        df_temp['Parameters'] = tuple_column
        list_of_single_df.append(df_temp)

    df_simulation_result = pd.concat(
        list_of_single_df, ignore_index=True, sort=False, axis=0)
    return model_runs, df_simulation_result


def running_model_in_parallel_and_generating_df(
    model, 
    run_model_single_parameter_node_routine,
    t: Union[np.array, np.ndarray, List[Union[int, float]]], 
    parameters: np.ndarray,
    list_unique_index_model_run_list: List[int],
    num_processes: int,
    time_column_name: str=TIME_COLUMN_NAME,
    index_column_name: str= INDEX_COLUMN_NAME
    ) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Runs the model over time and parameters (in parallel), and generates a DataFrame with the simulation results.

    Args:
        model: The model to be run.
        run_model_single_parameter_node_routine:
        t: The time array or list.
        parameters: The array of parameters used to stimulate the model. Expected to be of shape dim x number_of_nodes
        list_unique_index_model_run_list:
        num_processes:
        time_column_name: The name of the column to store time values in the DataFrame. Defaults to TIME_COLUMN_NAME.
        index_column_name: The name of the column to store index values in the DataFrame. Defaults to INDEX_COLUMN_NAME.

    Returns:
        A tuple containing the model runs as a numpy array and the simulation results as a pandas DataFrame.
    """
    model_runs = np.empty((parameters.shape[1], len(t)))
    list_of_single_df = []
    def process_particles_concurrently(particles_to_process):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for index_run, y_t_model, parameter_value in \
                    pool.starmap(run_model_single_parameter_node_routine, \
                                    [(model, particle[0], particle[1]) \
                                    for particle in particles_to_process]):
                yield index_run, y_t_model, parameter_value
    for index_run, y_t_model, parameter_value in process_particles_concurrently(\
    zip(parameters.T, list_unique_index_model_run_list)):
        model_runs[index_run] = y_t_model
        df_temp = pd.DataFrame(y_t_model, columns=[QOI_COLUMN_NAME])
        df_temp[time_column_name] = t
        df_temp[index_column_name] = index_run
        tuple_column = [tuple(parameter_value)] * len(df_temp)
        df_temp['Parameters'] = tuple_column
        list_of_single_df.append(df_temp)
    
    df_simulation_result = pd.concat(
        list_of_single_df, ignore_index=True, sort=False, axis=0)
    return model_runs, df_simulation_result


#####################################
# Different utility functions for generating quadrature nodes, polynomial expansion basis, etc.
# Also for computing statistics/conducting postprocessing over either monte carlo samples or building gPCE surrogate model
# These functions are mainly wrappers around the functions from the chaospy library; however that adher to data structure and logic in the UQEF-Dynamic framework
#####################################


def generate_quadrature_nodes_and_weights(joint_dist, quadrature_order: int, rule: str = 'g', growth: bool = False, sparse: bool = False):
    """
    This function generates quadrature nodes and weights using the quadrature rule specified by the user
    rule: str, optional
        Quadrature rule to use. The default is 'g'.
    growth: bool, optional
        If True, use a nested growth rule. The default is False.
    sparse: bool, optional
        If True, use a sparse grid. The default is False.
    return: tuple
        nodes_quad: np.ndarray
            Quadrature nodes dimension (dim, number_of_nodes)
        weights_quad: np.ndarray
            Quadrature weights dimension (dim, number_of_nodes)
    """
    nodes_quad, weights_quad = cp.generate_quadrature(
        quadrature_order, joint_dist, rule=rule, growth=growth, sparse=sparse)
    return nodes_quad, weights_quad


def generate_parameters_from_nodes(nodes_quad, joint_dist_standard, joint_dist):
    """
    This function transforms the quadrature nodes from the standard normal space to the original space of the parameters needed for the model
    return: np.ndarray dimension (dim, number_of_nodes)
    """
    parameters_quad = transformation_of_parameters(nodes_quad, joint_dist_standard, joint_dist)
    return parameters_quad


def generate_polynomial_expansion(joint_dist, order: int, rule: str = 'three_terms_recurrence', poly_normed: bool = True):
    """
    This function generates the polynomial expansion basis
    rule: str, optional
        Quadrature rule to use. The default is 'three_terms_recurrence'. Other options are: 'three_terms_recurrence', 'gram_schmidt', 'cholesky'
    """
    polynomial_expansion, norms = cp.generate_expansion(
        order, joint_dist, rule=rule, normed=poly_normed, retall=True)
    return polynomial_expansion, norms

# =================================================================================================
# Functions for computing PCE over time
# =================================================================================================


def compute_gPCE_over_time(model, t: Union[np.array, np.ndarray, List[Union[int, float]]], expansion_order: int, joint_dist, 
                            parameters: np.ndarray, nodes: np.ndarray, regression: bool = False, weights_quad: np.ndarray = None, poly_rule: str = 'three_terms_recurrence', poly_normed: bool = True):
    """
    This function computes the generalized Polynomial Chaos Expansion (gPCE) over time
    model: function
        Model to run
    t: np.ndarray
        Time array  
    expansion_order: int
        Order of the polynomial expansion
    joint_dist: chaospy.distributions
        Joint distribution of with respect to which the polynomial expansion will be build, e.g., joint_dist_standard
    parameters: np.ndarray 
        Parameters for the model dimension (dim, number_of_nodes)
    nodes: np.ndarray 
        Quadrature nodes dimension (dim, number_of_nodes)
    regression: bool, optional
    weights_quad: np.ndarray 
        Quadrature weights dimension (dim, number_of_nodes)
    poly_rule: str, optional
    poly_normed: bool, optional
    return: np.ndarray
        gPCE over time dimension (len(t), )
    """
    polynomial_expansion, norms = cp.generate_expansion(
        expansion_order, joint_dist, rule=poly_rule, normed=poly_normed, retall=True)
    model_runs = running_the_model_over_time_and_parameters(model, t, parameters)
    model_runs = model_runs.T

    dim = parameters.shape[0]
    number_expansion_coefficients = int(scipy.special.binom(dim+expansion_order, dim))  # cp.dimensions(polynomial_expansion)
    print(f"Total number of expansion coefficients in {dim}D space: {int(number_expansion_coefficients)}")

    gPCE_over_time =  np.empty((len(t),), dtype=object) # np.empty((len(t), number_expansion_coefficients))
    coeff = np.empty((len(t),), dtype=object)
    for idx, _ in enumerate(model_runs):  # for element in t:
        if regression:
            gPCE_over_time[idx] = cp.fit_regression(
                polynomials=polynomial_expansion, abscissas=nodes, evals=model_runs[idx], model=None)
        else:
            gPCE_over_time[idx], coeff[idx] = cp.fit_quadrature(
                orth=polynomial_expansion, nodes=nodes, weights=weights_quad, 
                solves=model_runs[idx], retall=True, norms=norms
                )
    return gPCE_over_time, polynomial_expansion, np.asarray(norms, dtype=np.float64), np.asarray(coeff, dtype=np.float64)


def compute_PSP_over_time(model, t: Union[np.array, np.ndarray, List[Union[int, float]]], joint_dist_standard, joint_dist, \
                           quadrature_order: int, expansion_order: int, rule_quadrature: str = 'g', growth: bool = False, sparse: bool = False, 
                           poly_rule: str = 'three_terms_recurrence', poly_normed: bool = True):
    nodes_quad, weights_quad = generate_quadrature_nodes_and_weights(joint_dist_standard, quadrature_order, rule_quadrature, growth, sparse)
    parameters_quad = generate_parameters_from_nodes(nodes_quad, joint_dist_standard, joint_dist)
    polynomial_expansion, norms = generate_polynomial_expansion(joint_dist_standard, expansion_order, poly_rule, poly_normed)
    model_runs = running_the_model_over_time_and_parameters(model, t, parameters_quad)
    model_runs = model_runs.T
    gPCE_over_time =  np.empty((len(t),), dtype=object) # np.empty((len(t), number_expansion_coefficients))
    coeff = np.empty((len(t),), dtype=object)
    for idx, _ in enumerate(model_runs):  # for element in t:
        gPCE_over_time[idx], coeff[idx] = cp.fit_quadrature(
            orth=polynomial_expansion, nodes=nodes_quad, weights=weights_quad, 
            solves=model_runs[idx], retall=True, norms=norms
            )
    return gPCE_over_time, polynomial_expansion, np.asarray(norms, dtype=np.float64), np.asarray(coeff, dtype=np.float64)


def build_gpce_surrogate_from_coefficients(gpce_coeffs, polynomial_expansion, polynomial_norms=None):
    """
    Builds a Generalized Polynomial Chaos Expansion surrogate model based on the saved coefficients.

    Args:
        gpce_coeffs (dict or np.ndarray): Coefficients of the GPCE expansion. If a dictionary, it should contain
            coefficients for each time stamp. If a numpy array, it should contain coefficients for a particular time stamp.
        polynomial_expansion (): Chaospy polynomial expansion.
        polynomial_norms (): polynomial norms.

    Returns:
        gpce_surrogate (dict or np.ndarray): Surrogate model built using the GPCE method.

    Raises:
        Exception: If gpce_coeffs is neither a dictionary nor a numpy array.

    """
    if isinstance(gpce_coeffs, dict):
        gpce_surrogate = dict()
        for key, single_gpce_coeffs in gpce_coeffs.items():
            gpce_surrogate[key] = sum([c * p for c, p in zip(single_gpce_coeffs, polynomial_expansion)])  #sum(c * p for c, p in zip(single_gpce_coeffs, polynomial_expansion))
    elif isinstance(gpce_coeffs, np.ndarray):
        gpce_surrogate = sum(c * p for c, p in zip(gpce_coeffs, polynomial_expansion))
    else:
        raise Exception("gpce_coeffs should be either a dictionary or a numpy array")
    return gpce_surrogate

# =================================================================================================
# Functions for computing statistics over time
# =================================================================================================


def computing_mc_statistics(
    df_simulation_result: pd.DataFrame, 
    numEvaluations: int, 
    time_column_name: str=TIME_COLUMN_NAME,
    single_qoi_column: str=QOI_COLUMN_NAME,
    compute_only_mean:bool=False,
    compute_Sobol_m:bool=False,
    samples=None,
    dict_stat_to_compute:dict=DEFAULT_DICT_STAT_TO_COMPUTE,
    ) -> Dict[Any, Dict[str, Any]]:
    """
    Parameters:
    - df_simulation_result: pd.DataFrame
        The DataFrame containing the simulation results.
    - numEvaluations: int
        The number of evaluations.
    - time_column_name: str, optional
        The name of the column representing the time points.
    - single_qoi_column: str, optional
        The name of the column representing the quantity of interest (QoI).
        Default is QOI_COLUMN_NAME.
    - compute_only_mean: bool, optional. Default is False.
    - compute_Sobol_m: bool, optional. Default is False.
    - samples: np.array, optional; only relevan if compute_Sobol_m is True
        matrix of parameter samples used to force the model should be of the size NxD. Defaults to None.
    - dict_stat_to_compute(dict, optional): dictionary of entrie what statistics to be computedl can be combined with above variables...

    Returns:
    - result_dict: Dict[Any, Dict[str, Any]]
        A dictionary containing the computed statistics for each time point.
        The keys are the time points, and the values are dictionaries containing the statistics.

    Note:
    - As a side effect this function adds a new column to df_simulation_result with centered simulations.
    - This function assumes that the DataFrame has a column named 'time_column_name' representing the time points.
    """
    single_qoi_column_centered = single_qoi_column + "_centered"
    df_simulation_result[single_qoi_column_centered] = None
    grouped = df_simulation_result.groupby([time_column_name,])
    groups = grouped.groups
    result_dict = defaultdict(dict)
    for key, val_indices in groups.items():
        _, result_dict[key] = computing_mc_statistics_single_date(
            key, val_indices, df_simulation_result, numEvaluations, single_qoi_column, compute_only_mean, compute_Sobol_m, samples,
            dict_stat_to_compute
            )
    return result_dict


def computing_mc_statistics_parallel_in_time(
    num_processes: int,
    df_simulation_result: pd.DataFrame, 
    numEvaluations: int, 
    time_column_name: str=TIME_COLUMN_NAME,
    single_qoi_column: str=QOI_COLUMN_NAME,
    compute_only_mean:bool=False,
    compute_Sobol_m:bool=False,
    samples=None,
    dict_stat_to_compute:dict=DEFAULT_DICT_STAT_TO_COMPUTE,
) -> Dict[Any, Dict[str, Any]]:
    """
    Compute Monte Carlo statistics in parallel for each time point.

    Args:
        num_processes: int
        df_simulation_result (pd.DataFrame): The simulation result dataframe.
        numEvaluations (int): The number of evaluations.
        time_column_name (str, optional): The name of the column representing time. Defaults to TIME_COLUMN_NAME.
        single_qoi_column (str, optional): The name of the single QoI column. Defaults to QOI_COLUMN_NAME.
        compute_only_mean (bool, optional): Whether to compute the rest of the statistics of only mean. Default is False.
        compute_Sobol_m (bool, optional): Whether to compute Sobol indices. Defaults to False.
        samples (np.array, optional): Matrix of parameter samples used to force the model should be of the size NxD.
            only relevan if compute_Sobol_m is True; Defaults to None.
        dict_stat_to_compute(dict, optional): dictionary of entrie what statistics to be computedl can be combined with above variables...
    Returns:
        Dict[Any, Dict[str, Any]]: A dictionary containing the computed statistics for each time point.
    """
    def process_dates_concurrently(groups):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for date, result_dict_single_date in pool.starmap(computing_mc_statistics_single_date, \
                                       [(key, val_indices, df_simulation_result, numEvaluations,
                                       single_qoi_column, compute_only_mean, compute_Sobol_m, samples, dict_stat_to_compute) for key, val_indices in groups.items()]):
                yield date, result_dict_single_date
    
    result_dict = defaultdict(dict)
    grouped = df_simulation_result.groupby([time_column_name,])
    groups = grouped.groups
    for date, result_dict_single_date in process_dates_concurrently(groups):
        result_dict[date] = result_dict_single_date
    return result_dict


def computing_gpce_statistics(
    df_simulation_result: pd.DataFrame, polynomial_expansion, 
    nodes: np.ndarray, weights: np.ndarray, dist, 
    time_column_name: str=TIME_COLUMN_NAME,
    single_qoi_column: str=QOI_COLUMN_NAME, 
    regression:bool=False, store_gpce_surrogate_in_stat_dict:bool=True, save_gpce_surrogate=False,
    compute_only_gpce:bool=False,
    compute_Sobol_t:bool=True, compute_Sobol_m:bool=False,compute_Sobol_m2:bool=False,
    dict_stat_to_compute:dict=DEFAULT_DICT_STAT_TO_COMPUTE,
    polynomial_norms=None,
) -> Dict[Any, Dict[str, Any]]:
    """
    Computes statistics time-wise over all the simulations using gPCE.

    Args:
        df_simulation_result (pd.DataFrame): The simulation result dataframe.
        polynomial_expansion: The polynomial expansion used for gPCE.
        nodes (np.ndarray): The nodes for the quadrature rule.
        weights (np.ndarray): The weights for the quadrature rule.
        dist: The distribution object representing the random variable.
        time_column_name (str, optional): The name of the time column in the dataframe. Defaults to TIME_COLUMN_NAME.
        single_qoi_column (str, optional): The name of the column containing the single quantity of interest. Defaults to QOI_COLUMN_NAME.
        regression (bool, optional): Whether to use regression for gPCE. Defaults to False.
        store_gpce_surrogate_in_stat_dict (bool, optional): Whether to store the gPCE surrogate in the result dictionary. Defaults to True.
        save_gpce_surrogate (bool, optional): Whether to save the gPCE surrogate to a file. Defaults to False.
        compute_only_gpce (bool, optional): Whether to compute only gPCE. It is relevant because computing all the stat takes a lot of time
        Defaults to False.
        compute_Sobol_t (bool, optional): Whether to compute Sobol' indices (total effects). Defaults to True.
        compute_Sobol_m (bool, optional): Whether to compute Sobol' indices (main effects). Defaults to False.
        compute_Sobol_m2 (bool, optional): Whether to compute Sobol' indices (total-order main effects). Defaults to False.
        dict_stat_to_compute(dict, optional): dictionary of entrie what statistics to be computedl can be combined with above variables...
        polynomial_norms(numpy.ndarray, optional): The norms of the polynomial expansion. Defaults to None.
    Returns:
        Dict[Any, Dict[str, Any]]: A dictionary containing the computed statistics for each time stamp.
    
    Note:
        - As a side effect this function adds a new column to df_simulation_result with centered simulations.
    """
    single_qoi_column_centered = single_qoi_column + "_centered"
    df_simulation_result[single_qoi_column_centered] = None
    grouped = df_simulation_result.groupby([time_column_name,])
    groups = grouped.groups
    result_dict = defaultdict(dict)
    for key, val_indices in groups.items():
        _, result_dict[key] = computing_gpce_statistics_single_date(
            key, val_indices, df_simulation_result, polynomial_expansion, nodes, weights, dist,
            single_qoi_column, regression, store_gpce_surrogate_in_stat_dict, save_gpce_surrogate,
            compute_only_gpce,
            compute_Sobol_t, compute_Sobol_m, compute_Sobol_m2, dict_stat_to_compute,
            polynomial_norms)
    return result_dict
        

def computing_gpce_statistics_parallel_in_time(
    num_processes: int,
    df_simulation_result: pd.DataFrame, polynomial_expansion, 
    nodes: np.ndarray, weights: np.ndarray, dist, 
    time_column_name: str=TIME_COLUMN_NAME,
    single_qoi_column: str=QOI_COLUMN_NAME, 
    regression:bool=False, store_gpce_surrogate_in_stat_dict:bool=True, save_gpce_surrogate=False,
    compute_only_gpce:bool=False,
    compute_Sobol_t:bool=True, compute_Sobol_m:bool=False, compute_Sobol_m2:bool=False,
    dict_stat_to_compute:dict=DEFAULT_DICT_STAT_TO_COMPUTE,
    polynomial_norms=None,
) -> Dict[Any, Dict[str, Any]]:
    """
    Computes GPCE statistics in parallel for each time point in the given DataFrame.

    Args:
        num_processes: int,
        df_simulation_result (pd.DataFrame): The DataFrame containing simulation results.
        polynomial_expansion: The polynomial expansion used for GPCE.
        nodes (np.ndarray): The nodes used for GPCE.
        weights (np.ndarray): The weights used for GPCE.
        dist: The distribution used for GPCE.
        time_column_name (str, optional): The name of the column representing time. Defaults to TIME_COLUMN_NAME.
        single_qoi_column (str, optional): The name of the column representing the single quantity of interest. Defaults to QOI_COLUMN_NAME.
        regression (bool, optional): Whether to perform regression. Defaults to False.
        store_gpce_surrogate_in_stat_dict (bool, optional): Whether to store the GPCE surrogate in the statistics dictionary. Defaults to True.
        save_gpce_surrogate (bool, optional): Whether to save the GPCE surrogate. Defaults to False.
        compute_only_gpce (bool, optional): Whether to compute only gPCE. It is relevant because computing all the stat takes a lot of time
        Defaults to False.
        compute_Sobol_t (bool, optional): Whether to compute Sobol indices for total effects. Defaults to True.
        compute_Sobol_m (bool, optional): Whether to compute Sobol indices for main effects. Defaults to False.
        compute_Sobol_m2 (bool, optional): Whether to compute Sobol indices for second-order interactions. Defaults to False.
        dict_stat_to_compute(dict, optional): dictionary of entrie what statistics to be computedl can be combined with above variables...
        polynomial_norms(numpy.ndarray, optional): The norms of the polynomial expansion. Defaults to None.
    Returns:
        Dict[Any, Dict[str, Any]]: A dictionary containing the computed GPCE statistics for each time point.
    """
    def process_dates_concurrently(groups):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for date, result_dict_single_date in pool.starmap(computing_gpce_statistics_single_date, \
                                       [(key, val_indices, df_simulation_result, 
                                       polynomial_expansion, nodes, weights, dist,
                                       single_qoi_column, 
                                       regression, store_gpce_surrogate_in_stat_dict, save_gpce_surrogate, compute_only_gpce,
                                       compute_Sobol_t, compute_Sobol_m, compute_Sobol_m2, dict_stat_to_compute, polynomial_norms) for key, val_indices in groups.items()]):
                yield date, result_dict_single_date
    
    result_dict = defaultdict(dict)
    grouped = df_simulation_result.groupby([time_column_name,])
    groups = grouped.groups
    for date, result_dict_single_date in process_dates_concurrently(groups):
        result_dict[date] = result_dict_single_date
    return result_dict


def statistics_result_dict_to_df(result_dict,  time_column_name: str=TIME_COLUMN_NAME) -> pd.DataFrame:
    """
    Convert the nested dictionary to a pandas DataFrame. Set time column as index column.
    """
    df_stat = pd.DataFrame.from_dict(result_dict, orient='index')
    df_stat.index.name = time_column_name
    return df_stat


def computing_mc_statistics_single_date(
    time_stamp, val_indices, 
    df_simulation_result: pd.DataFrame, 
    numEvaluations: int, 
    single_qoi_column: str=QOI_COLUMN_NAME,
    compute_only_mean:bool=False,
    compute_Sobol_m:bool=False,
    samples=None,
    dict_stat_to_compute:dict=DEFAULT_DICT_STAT_TO_COMPUTE,
    ):
    """
    Parameters:
    - time_stamp: current time stamp
    - val_indices: these are indices in the df_simulation_result that represent all the different
    simulations/run for the current time stamp
    - df_simulation_result: pd.DataFrame
        The DataFrame containing the simulation results.
    - numEvaluations: int
        The number of evaluations.
    - single_qoi_column: str, optional
        The name of the column representing the quantity of interest (QoI).
        Default is QOI_COLUMN_NAME.
    - compute_only_mean: bool, optional. Default is False.
    - compute_Sobol_m: bool, optional. Default is False.
    - samples: np.array, optional; only relevan if compute_Sobol_m is True
        matrix of parameter samples used to force the model should be of the size NxD. Defaults to None.
    - dict_stat_to_compute(dict, optional): dictionary of entrie what statistics to be computedl can be combined with above variables...

    Returns:
    - time_stamp: current time stamp
    - result_dict: Dict[Any, Any]
        A dictionary containing the computed statistics for a single time point.
        The keys are different 'statistic names'.

    Note:
    - As a side effect this function adds a new column to df_simulation_result with centered simulations.
    - This function assumes that the DataFrame has a column named 'time_column_name' representing the time points.
    """
    print(f"Time-stamp being proccessed: {time_stamp}")

    result_dict = computing_mc_statistics_simple(
        val_indices=val_indices, df_simulation_result=df_simulation_result, numEvaluations=numEvaluations,
        single_qoi_column=single_qoi_column, compute_only_mean=compute_only_mean, compute_Sobol_m=compute_Sobol_m, samples=samples,
        dict_stat_to_compute=dict_stat_to_compute,
    )
    
    return time_stamp, result_dict

    
def computing_gpce_statistics_single_date(
    time_stamp, val_indices, df_simulation_result, polynomial_expansion, nodes, weights, dist, single_qoi_column: str=QOI_COLUMN_NAME, 
    regression:bool=False, store_gpce_surrogate_in_stat_dict:bool=True, save_gpce_surrogate=False,
    compute_only_gpce:bool=False,
    compute_Sobol_t:bool=True, compute_Sobol_m:bool=False, compute_Sobol_m2:bool=False,
    dict_stat_to_compute:dict=DEFAULT_DICT_STAT_TO_COMPUTE,
    polynomial_norms=None,
    ):
    """
    Parameters:
    - time_stamp: current time stamp
    - val_indices: these are indices in the df_simulation_result that represent all the different
    simulations/run for the current time stamp
    - df_simulation_result: pd.DataFrame
        The DataFrame containing the simulation results.
    - polynomial_expansion: The polynomial expansion used for gPCE.
    - nodes (np.ndarray): The nodes for the quadrature rule.
    - weights (np.ndarray): The weights for the quadrature rule.
    - dist: The distribution object representing the random variable.
    - single_qoi_column (str, optional): The name of the column containing the single quantity of interest. Defaults to QOI_COLUMN_NAME.
    - regression (bool, optional): Whether to use regression for gPCE. Defaults to False.
    - store_gpce_surrogate_in_stat_dict (bool, optional): Whether to store the gPCE surrogate in the result dictionary. Defaults to True.
    - save_gpce_surrogate (bool, optional): Whether to save the gPCE surrogate to a file. Defaults to False.
    - compute_only_gpce (bool, optional): Whether to compute only gPCE. It is relevant because computing all the stat takes a lot of time
       Defaults to False.
    - compute_Sobol_t (bool, optional): Whether to compute Sobol' indices (total effects). Defaults to True.
    - compute_Sobol_m (bool, optional): Whether to compute Sobol' indices (main effects). Defaults to False.
    - compute_Sobol_m2 (bool, optional): Whether to compute Sobol' indices (total-order main effects). Defaults to False.
    - dict_stat_to_compute(dict, optional): dictionary of entrie what statistics to be computedl can be combined with above variables...
    - polynomial_norms(numpy.ndarray, optional): The norms of the polynomial expansion. Defaults to None.

    Returns:
    - time_stamp: current time stamp
    - result_dict: Dict[Any, Any]
        A dictionary containing the computed statistics for a single time point.
        The keys are different 'statistic names'.

    Note:
    - As a side effect this function adds a new column to df_simulation_result with centered simulations.
    - This function assumes that the DataFrame has a column named 'time_column_name' representing the time points.
    """
    print(f"Time-stamp being proccessed: {time_stamp}")

    result_dict = computing_gpce_statistics_simple(
        polynomial_expansion=polynomial_expansion, nodes=nodes, weights=weights, dist=dist, 
        val_indices=val_indices, 
        df_simulation_result=df_simulation_result, 
        single_qoi_column=single_qoi_column, 
        regression=regression, 
        store_gpce_surrogate_in_stat_dict=store_gpce_surrogate_in_stat_dict,
        save_gpce_surrogate=save_gpce_surrogate,
        compute_only_gpce=compute_only_gpce,
        compute_Sobol_t=compute_Sobol_t, compute_Sobol_m=compute_Sobol_m, compute_Sobol_m2=compute_Sobol_m2,
        dict_stat_to_compute=dict_stat_to_compute,
        polynomial_norms=polynomial_norms,
        )

    return time_stamp, result_dict

# =================================================================================================
# 'Simple' (time is not figuring) functions for computing statistics
# These functions might go to the uqPostprocessing.py or parallel_statistics.py files
# =================================================================================================

def computing_mc_statistics_simple(
    qoi_values=None,
    val_indices=None, 
    df_simulation_result: pd.DataFrame=None, 
    numEvaluations:int=None, 
    single_qoi_column: str=QOI_COLUMN_NAME,
    compute_only_mean:bool=False,
    compute_Sobol_m:bool=False,
    samples=None,
    dict_stat_to_compute:dict=DEFAULT_DICT_STAT_TO_COMPUTE):
    """
    Parameters:
    - qoi_values: np.array, optional
    - val_indices: these are indices in the df_simulation_result that represent all the different
    simulations/run, optional
    - df_simulation_result: pd.DataFrame
        The DataFrame containing the simulation results, optional
    Note: either qoi_values or val_indices and df_simulation_result should be provided!
    - numEvaluations: int
        The number of evaluations, optional
    - single_qoi_column: str, optional
        The name of the column representing the quantity of interest (QoI).
        Default is QOI_COLUMN_NAME.
    - compute_only_mean: bool, optional. Default is False.
    - compute_Sobol_m: bool, optional. Default is False.
    - samples: np.array, optional; only relevan if compute_Sobol_m is True
        matrix of parameter samples used to force the model should be of the size NxD. Defaults to None.
    - dict_stat_to_compute(dict, optional): dictionary of entrie what statistics to be computedl can be combined with above variables...

    Returns:
    - result_dict: Dict[Any, Any]
        A dictionary containing the computed statistics for a single time point.
        The keys are different 'statistic names'.

    Note:
    - As a side effect this function adds a new column to df_simulation_result with centered simulations.
    - This function assumes that the DataFrame has a column named 'time_column_name' representing the time points.
    """
    
    result_dict = {}

    if qoi_values is None:
        if df_simulation_result is not None and val_indices is not None:
            qoi_values = df_simulation_result.loc[val_indices.values][single_qoi_column].values
        else:
            raise ValueError("Either qoi_values or val_indices and df_simulation_result should be provided!")

    mean = result_dict['E'] = np.mean(qoi_values, 0)

    if df_simulation_result is not None and val_indices is not None:
        single_qoi_column_centered = single_qoi_column + "_centered"
        df_simulation_result.loc[val_indices, single_qoi_column_centered] = df_simulation_result.loc[val_indices, single_qoi_column] - mean

    if not compute_only_mean:
        if dict_stat_to_compute.get("Var", False):
            result_dict['Var'] = np.var(qoi_values, ddof=1) 
            # result_dict['Var'] = np.sum((qoi_values - result_dict["E"]) ** 2, \
            # axis=0, dtype=np.float64) / (numEvaluations - 1)
        if dict_stat_to_compute.get("StdDev", False):
            result_dict["StdDev"] = np.std(qoi_values, 0, ddof=1)
        if dict_stat_to_compute.get("Skew", False):
            result_dict["Skew"] = scipy.stats.skew(qoi_values, axis=0, bias=True)
        if dict_stat_to_compute.get("Kurt", False):
            result_dict["Kurt"] = scipy.stats.kurtosis(qoi_values, axis=0, bias=True)

        if dict_stat_to_compute.get("P10", False):
            result_dict["P10"] = np.percentile(qoi_values, 10, axis=0)
            if isinstance(result_dict["P10"], list) and len(result_dict["P10"]) == 1:
                result_dict["P10"] = result_dict["P10"][0]
        if dict_stat_to_compute.get("P90", False):
            result_dict["P90"] = np.percentile(qoi_values, 90, axis=0)
            if isinstance(result_dict["P90"], list) and len(result_dict["P90"]) == 1:
                result_dict["P90"] = result_dict["P90"][0]

        if dict_stat_to_compute.get("E_minus_std", False):
            result_dict["E_minus_std"] = result_dict['E'] - result_dict['StdDev']
        if dict_stat_to_compute.get("E_plus_std", False):
            result_dict["E_plus_std"] = result_dict['E'] + result_dict['StdDev']

    if compute_Sobol_m and samples is not None:
        dim = samples.shape[1]
        if numEvaluations is None:
            numEvaluations = samples.shape[0]
        result_dict["Sobol_m"] = sens_indices_sampling_based_utils.compute_sens_indices_based_on_samples_rank_based(
                samples=samples, Y=qoi_values[:numEvaluations, np.newaxis], D=dim, N=numEvaluations)

    return result_dict

    
def computing_gpce_statistics_simple(
    polynomial_expansion, nodes, weights, dist, 
    qoi_values=None,
    val_indices=None, 
    df_simulation_result: pd.DataFrame=None, 
    single_qoi_column: str=QOI_COLUMN_NAME, 
    regression:bool=False, store_gpce_surrogate_in_stat_dict:bool=True, save_gpce_surrogate=False,
    compute_only_gpce:bool=False,
    compute_Sobol_t:bool=True, compute_Sobol_m:bool=False, compute_Sobol_m2:bool=False,
    dict_stat_to_compute:dict=DEFAULT_DICT_STAT_TO_COMPUTE,
    polynomial_norms=None,
    ):
    """
    Parameters:
    - qoi_values: np.array, optional
    - val_indices: these are indices in the df_simulation_result that represent all the different
    simulations/run, optional
    - df_simulation_result: pd.DataFrame
        The DataFrame containing the simulation results, optional
    Note: either qoi_values or val_indices and df_simulation_result should be provided!
    - polynomial_expansion: The polynomial expansion used for gPCE.
    - nodes (np.ndarray): The nodes for the quadrature rule.
    - weights (np.ndarray): The weights for the quadrature rule.
    - dist: The distribution object representing the random variable.
    - single_qoi_column (str, optional): The name of the column containing the single quantity of interest. Defaults to QOI_COLUMN_NAME.
    - regression (bool, optional): Whether to use regression for gPCE. Defaults to False.
    - store_gpce_surrogate_in_stat_dict (bool, optional): Whether to store the gPCE surrogate in the result dictionary. Defaults to True.
    - save_gpce_surrogate (bool, optional): Whether to save the gPCE surrogate to a file. Defaults to False.
    - compute_only_gpce (bool, optional): Whether to compute only gPCE. It is relevant because computing all the stat takes a lot of time
       Defaults to False.
    - compute_Sobol_t (bool, optional): Whether to compute Sobol' indices (total effects). Defaults to True.
    - compute_Sobol_m (bool, optional): Whether to compute Sobol' indices (main effects). Defaults to False.
    - compute_Sobol_m2 (bool, optional): Whether to compute Sobol' indices (total-order main effects). Defaults to False.
    - dict_stat_to_compute(dict, optional): dictionary of entrie what statistics to be computedl can be combined with above variables...
    - polynomial_norms(numpy.ndarray, optional): The norms of the polynomial expansion. Defaults to None.

    Returns:
    - result_dict: Dict[Any, Any]
        A dictionary containing the computed statistics for a single time point.
        The keys are different 'statistic names'.

    Note:
    - As a side effect this function adds a new column to df_simulation_result with centered simulations.
    - This function assumes that the DataFrame has a column named 'time_column_name' representing the time points.
    """
    result_dict = {}

    if qoi_values is None:
        if df_simulation_result is not None and val_indices is not None:
            qoi_values = df_simulation_result.loc[val_indices.values][single_qoi_column].values
        else:
            raise ValueError("Either qoi_values or val_indices and df_simulation_result should be provided!")

    coeff = None
    if regression:
        qoi_gPCE, coeff = cp.fit_regression(polynomial_expansion, nodes, qoi_values)
    else:
        qoi_gPCE, coeff = cp.fit_quadrature(
            orth=polynomial_expansion, nodes=nodes, weights=weights, 
            solves=qoi_values, retall=True, norms=polynomial_norms
            )

    numPercSamples = 10 ** 5

    if store_gpce_surrogate_in_stat_dict:
        result_dict[PCE_ENTRY] = qoi_gPCE
        result_dict[PCE_COEFF_ENTRY] = coeff  # gpce_coeff

    mean = result_dict[MEAN_ENTRY] = float(cp.E(qoi_gPCE, dist))  # TODO I can as well compute the mean from the samples
    result_dict["E_quad"] = np.dot(qoi_values, weights)
    
    if df_simulation_result is not None and val_indices is not None:
        single_qoi_column_centered = single_qoi_column + "_centered"
        df_simulation_result.loc[val_indices, single_qoi_column_centered] = df_simulation_result.loc[val_indices, single_qoi_column] - mean

    if not compute_only_gpce:
        dict_stat_to_compute[SOBOL_TOTAL_ORDER_ENTRY] = compute_Sobol_t
        dict_stat_to_compute[SOBOL_FIRST_ORDER_ENTRY] = compute_Sobol_m
        dict_stat_to_compute[SOBOL_SECOND_ORDER_ENTRY] = compute_Sobol_m2

        if dict_stat_to_compute.get(VAR_ENTRY, False):
            result_dict[VAR_ENTRY] = float(cp.Var(qoi_gPCE, dist))
        if dict_stat_to_compute.get(STD_DEV_ENTRY, False):
            result_dict[STD_DEV_ENTRY] = float(cp.Std(qoi_gPCE, dist))
        if dict_stat_to_compute.get(SKEW_ENTRY, False):
            result_dict[SKEW_ENTRY] = cp.Skew(qoi_gPCE, dist).round(4)
        if dict_stat_to_compute.get(KURT_ENTRY, False):
            result_dict[KURT_ENTRY] = cp.Kurt(qoi_gPCE, dist)
        if dict_stat_to_compute.get(QOI_DIST_ENTR, False):
            result_dict[QOI_DIST_ENTR] = cp.QoI_Dist(qoi_gPCE, dist)
        if dict_stat_to_compute.get(P10_ENTRY, False):
            result_dict[P10_ENTRY] = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
            if isinstance(result_dict[P10_ENTRY], list) and len(result_dict[P10_ENTRY]) == 1:
                result_dict[P10_ENTRY] = result_dict[P10_ENTRY][0]
        if dict_stat_to_compute.get(P90_ENTRY, False):
            result_dict[P90_ENTRY] = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
            if isinstance(result_dict[P90_ENTRY], list) and len(result_dict[P90_ENTRY]) == 1:
                result_dict[P90_ENTRY] = result_dict[P90_ENTRY][0]
        if compute_Sobol_t:
            result_dict[SOBOL_TOTAL_ORDER_ENTRY] = cp.Sens_t(qoi_gPCE, dist) #np.squeeze(cp.Sens_t(qoi_gPCE, dist))
        if compute_Sobol_m:
            result_dict[SOBOL_FIRST_ORDER_ENTRY] = cp.Sens_m(qoi_gPCE, dist) #np.squeeze(cp.Sens_m(qoi_gPCE, dist))
        if compute_Sobol_m2:
            result_dict[SOBOL_SECOND_ORDER_ENTRY] = cp.Sens_m2(qoi_gPCE, dist) #np.squeeze(cp.Sens_m2(qoi_gPCE, dist))
    return result_dict


def compute_mc_quantity(model, jointDists, jointStandard=None, numSamples=1000, rule="R",
                        sampleFromStandardDist=False, can_model_evaluate_all_vector_nodes=False,
                        read_nodes_from_file=False, rounding=False, round_dec=4, **kwargs):
    """
    Function for computing MC-based E and/or Var (to be regarded as analyitical values)
    Function for computing the expected value and variance of a model using Monte Carlo simulation.
    This function is different compared to computing_mc_statistics_simple because it runs the model
    The function computes the expected value and variance of a model using Monte Carlo simulation (relays on numpy).
    The function uses the Chaospy package to generate the samples.

    Possible additional arguments:
        - parameters_file_name (str): The name of the file containing the parameters.
        - compute_mean (bool): If True, the expected value is computed. Defaults to True.
        - compute_var (bool): If True, the variance is computed. Defaults to True.
    """
    print(f"\n==MC Computation Chaospy==")

    if sampleFromStandardDist and jointStandard is not None:
        dist = jointStandard
    else:
        dist = jointDists

    if read_nodes_from_file:
        parameters_file_name = kwargs.get('parameters_file_name', None)
        if parameters_file_name is None:
            raise
        nodes_and_weights_array = np.loadtxt(parameters_file_name, delimiter=',')
        # TODO what if nodes_and_weights_array do not correspond to dist! However, this is not important for MC
        #labels = [param_name.strip() for param_name in param_names]
        stochastic_dim = len(jointDists) #len(param_names)
        nodes = nodes_and_weights_array[:, :stochastic_dim].T
    else:
        if rounding:
            nodes = dist.sample(size=numSamples, rule=rule).round(round_dec)
        else:
            nodes = dist.sample(size=numSamples, rule=rule)
    nodes = np.array(nodes)

    if sampleFromStandardDist and jointStandard is not None:
        parameters = transformation_of_parameters(
            nodes, jointDists, jointStandard)
    else:
        parameters = nodes

    start_time_model_evaluations = time.time()
    if can_model_evaluate_all_vector_nodes:
        evaluations = model(parameters.T)
    else:
        evaluations = np.array([model(parameter) for parameter in parameters.T])
    end_time_model_evaluations = time.time()

    time_model_evaluations = end_time_model_evaluations - start_time_model_evaluations
    numEvaluations = len(parameters.T)  # len(evaluations)

    print(f"Needed time for model evaluations is: {time_model_evaluations} \n"
          f"for {numEvaluations} number of full model runs;")

    compute_mean = kwargs.get('compute_mean', True)
    compute_var = kwargs.get('compute_var', True)

    start_time_computing_statistics = time.time()
    expectedInterp = None
    varianceInterp = None
    if compute_mean:
        expectedInterp = np.mean(evaluations)
        print(f"MC_expectation = {expectedInterp}")
    if compute_var:
        qoi_values = evaluations - expectedInterp
        # varianceInterp = np.sum(qoi_values ** 2, axis=0, dtype=np.float64) / (numEvaluations - 1)
        varianceInterp = np.var(qoi_values, ddof=1)
        print(f"MC_variance = {varianceInterp}")
    end_time_computing_statistics = time.time()
    time_computing_statistics = end_time_computing_statistics - start_time_computing_statistics
    print(f"Needed time for computing statistics is: {time_computing_statistics};\n")

    return expectedInterp, varianceInterp

# =================================================================================================


def compute_total_sobol_indices_with_n_samples(samples, Y, D, N):
    """
    : param samples: UQEF.Samples.parameters; should be of the size NxD
    :param Y:  function evaluations dim(Y) = (N x t)
    :param D: Stochastic dimension
    :param N: Number of samples
    :return:
    """
    mean = np.mean(Y, axis=0)
    variance = np.var(Y, axis=0, ddof=1)
    denominator = np.where(variance, variance, 1)

    px = samples.argsort(axis=0)  # samples are NxD; this will sort each column and return indices
    pi_j = px.argsort(axis=0) + 1
    argpiinv = (pi_j % N) + 1

    # s_t = np.zeros(D)
    s_t = []
    for j in range(D):
        # numerator = np.mean((A - A_B[j]) ** 2, axis=0) / 2  # np.mean((A-A_B[j].T)**2, -1)
        N_j = px[argpiinv[:, j] - 1, j]
        YN_j = Y[N_j, :]
        # print(f"DEBUGGING - shape of YN_j - {YN_j.shape}")
        numerator = (np.mean(Y * YN_j, axis=0) - mean**2)
        s_t_j = numerator[0] / denominator[0]
        s_t.append(s_t_j)
    return np.array(s_t)

# =================================================================================================
# Utility functions mainly needed for the KL expansion pipeline
# =================================================================================================


def compute_mean_from_df_simulation_result(df_simulation_result, algorithm: str= "samples",weights=None, time_column_name: str=TIME_COLUMN_NAME, single_qoi_column: str=QOI_COLUMN_NAME):
    """
    Compute the mean values from a DataFrame containing simulation results.

    Args:
        df_simulation_result (pandas.DataFrame): The DataFrame containing the simulation results.
        algorithm (str, optional): The algorithm used for computing the mean. Defaults to "samples".
        weights (numpy.ndarray, optional): The weights used for quadrature-based algorithms. Defaults to None.
        time_column_name (str, optional): The name of the column representing time. Defaults to TIME_COLUMN_NAME.
        single_qoi_column (str, optional): The name of the column representing the single quantity of interest. Defaults to QOI_COLUMN_NAME.

    Returns:
        dict: A dictionary where the keys are the time values and the values are the computed mean values.
    """
    grouped = df_simulation_result.groupby([time_column_name,])
    groups = grouped.groups
    mean_dict = {}
    for key, val_indices in groups.items():
        qoi_values = df_simulation_result.loc[val_indices.values][single_qoi_column].values
        # compute mean
        if algorithm == "samples" or algorithm == "sampling" or algorithm == "mc":
            mean = np.mean(qoi_values, 0)
        elif algorithm == "quadrature" or algorithm == "pce" or algorithm == "sc" or algorithm == "kl":
            if weights is None:
                raise ValueError("Weights must be provided for quadrature-based algorithms")
            mean = np.dot(qoi_values, weights)
        else:
            raise ValueError(f"Unknown algorithm - {algorithm}")
        # df_simulation_result.loc[val_indices, single_qoi_column + "_mean"] = mean
        mean_dict[key] = mean
    return mean_dict


def add_centered_qoi_column_to_df_simulation_result(df_simulation_result, algorithm: str= "samples",weights=None, time_column_name: str=TIME_COLUMN_NAME, single_qoi_column: str=QOI_COLUMN_NAME):
    grouped = df_simulation_result.groupby([time_column_name,])
    groups = grouped.groups
    single_qoi_column_centered = single_qoi_column + "_centered"
    for key, val_indices in groups.items():
        qoi_values = df_simulation_result.loc[val_indices.values][single_qoi_column].values
        # compute mean
        if algorithm == "samples" or algorithm == "sampling" or algorithm == "mc":
            mean = np.mean(qoi_values, 0)
        elif algorithm == "quadrature" or algorithm == "pce" or algorithm == "sc" or algorithm == "kl":
            if weights is None:
                raise ValueError("Weights must be provided for quadrature-based algorithms")
            mean = np.dot(qoi_values, weights)
        else:
            raise ValueError(f"Unknown algorithm - {algorithm}")
        df_simulation_result.loc[val_indices, single_qoi_column_centered] = df_simulation_result.loc[val_indices, single_qoi_column] - mean


def center_outputs(N, N_quad, df_simulation_result, weights, single_qoi_column, index_column_name: str=INDEX_COLUMN_NAME, algorithm: str= "samples", time_column_name: str=TIME_COLUMN_NAME):
    centered_outputs = np.empty((N, N_quad))

    single_qoi_column_centered = single_qoi_column + "_centered"
    if single_qoi_column_centered not in df_simulation_result.columns:
        add_centered_qoi_column_to_df_simulation_result(df_simulation_result, algorithm=algorithm, weights=weights, single_qoi_column=single_qoi_column, time_column_name=time_column_name)

    grouped = df_simulation_result.groupby([index_column_name,])
    groups = grouped.groups
    for key, val_indices in groups.items():
        centered_outputs[int(key), :] = df_simulation_result.loc[val_indices, single_qoi_column_centered].values
    # print(f"DEBUGGING - centered_outputs.shape {centered_outputs.shape}")
    return centered_outputs


def compute_covariance_matrix_in_time(N_quad, centered_outputs, weights, algorithm: str= "samples"):
    covariance_matrix = np.empty((N_quad, N_quad))
    for c in range(N_quad):
        for s in range(N_quad):
            if algorithm == "samples" or algorithm == "sampling"  or algorithm == "mc":
                N = centered_outputs.shape[0]  # len(centered_outputs[:, c])
                covariance_matrix[s, c] = 1/(N-1) * \
                np.dot(centered_outputs[:, c], centered_outputs[:, s])
            elif algorithm == "quadrature" or algorithm == "pce" or algorithm == "sc" or algorithm == "kl":
                if weights is None:
                    raise ValueError("Weights must be provided for quadrature-based algorithms")
                covariance_matrix[s, c] = np.dot(weights, centered_outputs[:, c]*centered_outputs[:,s])
            else:
                raise ValueError(f"Unknown algorithm - {algorithm}")
    return covariance_matrix


def plot_covariance_matrix(covariance_matrix, directory_for_saving_plots, filname="covariance_matrix.png"):
    plt.figure()
    plt.imshow(covariance_matrix, cmap='hot', interpolation='nearest')
    # Add a color bar to the side
    plt.colorbar()
    # Add title and labels if needed
    plt.title('Covariance Matrix')
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), filname))
    plt.savefig(fileName)


def save_covariance_matrix(covariance_matrix, directory_for_saving, qoi):
    fileName = f"covariance_matrix_{qoi}.npy"
    fullFileName = os.path.abspath(os.path.join(str(directory_for_saving), fileName))
    np.save(fullFileName, covariance_matrix)


def solve_eigenvalue_problem(covariance_matrix, weights):
    # from scipy.linalg import eig
    # from scipy.linalg import eigh
    from numpy.linalg import eig
    from scipy.linalg import sqrtm

    # Solve Discretized (generalized) Eigenvalue Problem
    # Setting-up the system
    K = covariance_matrix
    # Check if the approximation of the covarriance matrix is symmetric
    cov_matrix_is_symmetric = np.array_equal(covariance_matrix, covariance_matrix.T)
    print(f"Check if the approximation of the covarriance matrix is symmetric - {cov_matrix_is_symmetric}")
    W = np.diag(weights)
    sqrt_W = sqrtm(W)
    LHS = sqrt_W@K@sqrt_W

    # Solving the system
    # B = np.identity(LHS.shape[0])
    # Alternatively one can solve the standard eigenvalue problem with symmetric/Hermitian matrices
    # from numpy.linalg import eigh
    # eigenvalues_h, eigenvectors_h = eigh(LHS)
    eigenvalues, eigenvectors = eig(LHS)
    idx = eigenvalues.argsort()[::-1]   # Sort by descending real part of eigenvalues
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    eigenvalues = sorted_eigenvalues
    eigenvectors = sorted_eigenvectors
    eigenvalues_real = np.asarray([element.real for element in eigenvalues], dtype=np.float64)
    eigenvalues_real_scaled = eigenvalues_real/eigenvalues_real[0]
    final_eigenvectors = np.linalg.inv(sqrt_W)@eigenvectors
    eigenvectors = final_eigenvectors
    return eigenvalues_real, eigenvectors
    # return eigenvalues, eigenvectors


def plot_eigenvalues(eigenvalues, directory_for_saving_plots):
    # Plotting the eigenvalues
    eigenvalues_real = np.asarray([element.real for element in eigenvalues], dtype=np.float64)
    eigenvalues_real_scaled = eigenvalues_real/eigenvalues_real[0]
    plt.figure()
    plt.yscale("log")
    plt.plot(eigenvalues, 'x')
    plt.title('Eigenvalues of the Covariance Operator')
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "eigenvalues_covariance_operator.png"))
    plt.savefig(fileName)
    plt.figure()
    plt.yscale("log")
    plt.plot(eigenvalues_real_scaled, 'x')
    plt.title('Scaled Eigenvalues of the Covariance Operator')
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "scaled_eigenvalues_covariance_operator.png"))
    plt.savefig(fileName)


def setup_kl_expansion_matrix(eigenvalues, N_kl, N, N_quad, weights, centered_outputs, eigenvectors):
    # Approximating the KL Expansion - setting up the matrix
    f_kl_eval_at_params = np.empty((N_kl, N))
    # f_kl_eval_at_params_2 = np.empty((N_kl, N))

    # weights @ centered_outputs[k,:] @ eigenvectors[:,i]
    for i in range(N_kl):
        for k in range(N):
            # TODO There is some Casting complex values to real discards the imaginary part happening here; probably from complex eigenvectors in certain situations...
            f_kl_eval_at_params[i, k] = np.sum(weights * centered_outputs[k,:] * eigenvectors[:,i])
            # f_kl_eval_at_params[i, k] = np.dot(np.dot(weights.ravel(), centered_outputs[k,:].ravel()), eigenvectors[:,i].ravel())
            # f_kl_eval_at_params[i, k] = 0
            # for m in range(N_quad):
            #     f_kl_eval_at_params[i, k] += weights[m]*centered_outputs.T[m,k]*eigenvectors[m,i]
    return f_kl_eval_at_params


def pce_of_kl_expansion(
    N_kl, polynomial_expansion, nodes, weights, f_kl_eval_at_params, regression=False, polynomial_norms=None):
    # PCE of the KL Expansion
    f_kl_surrogate_dict = {}
    # f_kl_surrogate_coefficients = np.empty(N_kl, c_number)
    f_kl_surrogate_coefficients = []
    for i in range(N_kl):
        # TODO Change this data structure, make it that the keys are time-stampes to resamble result_dict_statistics
        f_kl_surrogate_dict[i] = {}
        # print(f"DEBUGGING - f_kl_eval_at_params[{i},:].shape - {f_kl_eval_at_params[i,:].shape}")
        if regression:
            f_kl_gPCE, f_kl_coeff = cp.fit_regression(
                polynomials=polynomial_expansion, abscissas=nodes, 
                evals=f_kl_eval_at_params[i,:], retall=True, 
                model=None,
                )
        else:
            f_kl_gPCE, f_kl_coeff = cp.fit_quadrature(
                orth=polynomial_expansion, nodes=nodes, weights=weights, solves=f_kl_eval_at_params[i,:], retall=True,
                norms=polynomial_norms
                )        
        f_kl_surrogate_dict[i]["gPCE"] = f_kl_gPCE
        f_kl_surrogate_dict[i]["gpce_coeff"] = f_kl_coeff
        # f_kl_surrogate_coefficients[i] = np.asfarray(f_kl_coeff).T
        f_kl_surrogate_coefficients.append(np.asarray(f_kl_coeff, dtype=np.float64)) 
    f_kl_surrogate_coefficients = np.asarray(f_kl_surrogate_coefficients, dtype=np.float64)
    return f_kl_surrogate_dict, f_kl_surrogate_coefficients

# =================================================================================================
# Generlaized Sobol Indices
# =================================================================================================


def computing_generalized_sobol_total_indices_from_kl_expan(
    f_kl_surrogate_coefficients: np.ndarray,
    polynomial_expansion: cp.polynomial,
    weights: np.ndarray,
    param_names: List[str],
    fileName: Union[str, pathlib.PosixPath],
    total_variance=None,
    compute_total_variance_based_on_pce_coefficients=False
):
    """
    Computes the generalized Sobol total indices from a polynomial expansion of the KL expansion coefficints.
    This function computes for the last timestep in result_dict_statistics!
    The current implamantion of the function assumes that the polynomial expansion is normalized.
    One would have to do scaling with norms of the polynomials if they are not normalized.

    Args:
        f_kl_surrogate_coefficients (np.ndarray): 
        polynomial_expansion (cp.polynomial): The polynomial expansion.
        weights (np.ndarray): An array of weights for time quadratures.
        param_names (List[str]): A list of parameter names.
        fileName (str): The name of the file to write the results to.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the computed generalized Sobol total indices, 
        with parameter names as the keys.
        total_variance
        total_variance_based_on_pce_coefficients whic is None if compute_total_variance_based_on_pce_coefficients is False and total_variance is not None
    Raises:
        None
    """
    # TODO Important aspect here is if polynomial_expansion is normalized or not
    dic = polynomial_expansion.todict()
    alphas = []
    for idx in range(len(polynomial_expansion)):
        expons = np.array([key for key, value in dic.items() if value[idx]])
        alphas.append(tuple(expons[np.argmax(expons.sum(1))]))

    index = np.array([any(alpha) for alpha in alphas])  # list of non-constant terms from poly expans

    dict_of_num = defaultdict(list)
    for idx in range(len(alphas[0])):
        dict_of_num[idx] = []

    total_variance_based_on_pce_coefficients = None

    if compute_total_variance_based_on_pce_coefficients or total_variance is None:
        variance_over_kl_terms = []

    for i in range(f_kl_surrogate_coefficients.shape[0]):
        coefficients = np.asarray(f_kl_surrogate_coefficients[i,:], dtype=np.float64)
        if compute_total_variance_based_on_pce_coefficients or total_variance is None:
            variance = np.sum(coefficients[index] ** 2, axis=0) 
            variance_over_kl_terms.append(variance)
        for idx in range(len(alphas[0])):
            index_local = np.array([alpha[idx] > 0 for alpha in alphas])  # Compute the total Sobol indices
            # TODO add for the first order
            # index_local = np.array(
            #     [
            #         bool(alpha[idx] and not any(alpha[:idx] + alpha[idx + 1 :]))
            #         for alpha in alphas
            #     ]
            # )
            dict_of_num[idx].append(np.sum(coefficients[index_local] ** 2, axis=0))

    if compute_total_variance_based_on_pce_coefficients or total_variance is None:
        variance_over_kl_terms = np.asarray(variance_over_kl_terms, dtype=np.float64)
        total_variance_based_on_pce_coefficients = np.sum(variance_over_kl_terms)
        if total_variance is None:
            denum = total_variance_based_on_pce_coefficients
            raise ValueError("Total variance must be provided")
    
    if total_variance is not None:
        denum = total_variance
        
    param_name_generalized_sobol_total_indices = {}
    for idx in range(len(alphas[0])):
        param_name = param_names[idx]
        # num = np.dot(np.asarray(dict_of_num[idx], dtype=np.float64), weights)
        num = np.sum(np.asarray(dict_of_num[idx], dtype=np.float64), axis=0)
        s_tot_generalized = num/denum
        param_name_generalized_sobol_total_indices[param_name] = s_tot_generalized
        print(f"Generalized Total Sobol Index computed based on the PCE of KL expansion for {param_name} is {s_tot_generalized}")
        with open(fileName, 'a') as file:
            # Write each variable to the file followed by a newline character
            file.write(f'{param_name}: {s_tot_generalized}\n')
    return param_name_generalized_sobol_total_indices, total_variance, total_variance_based_on_pce_coefficients


def computing_generalized_sobol_total_indices_from_poly_expan(
    result_dict_statistics: Dict[Any, Dict[str, Any]],
    polynomial_expansion: cp.polynomial,
    weights: np.ndarray,
    param_names: List[str],
    fileName: Union[str, pathlib.PosixPath],
    total_variance=None
):
    """
    Computes the generalized Sobol total indices from a polynomial expansion.
    This function computes for the last timestep in result_dict_statistics!
    The current implamantion of the function assumes that the polynomial expansion is normalized.
    One would have to do scaling with norms of the polynomials if they are not normalized.

    Args:
        result_dict_statistics (Dict[Any, Dict[str, Any]]): A dictionary containing the statistics of the results.
         Important assumtion is that it contains the coefficients of the polynomial expansion under 'gpce_coeff'/PCE_COEFF_ENTRY key over time.
        polynomial_expansion (cp.polynomial): The polynomial expansion.
        weights (np.ndarray): An array of weights for time quadratures.
        param_names (List[str]): A list of parameter names.
        fileName (str): The name of the file to write the results to.

    Returns:
        None

    Raises:
        None
    """
    # TODO Important aspect here is if polynomial_expansion is normalized or not
    dic = polynomial_expansion.todict()
    alphas = []
    for idx in range(len(polynomial_expansion)):
        expons = np.array([key for key, value in dic.items() if value[idx]])
        alphas.append(tuple(expons[np.argmax(expons.sum(1))]))

    index = np.array([any(alpha) for alpha in alphas])

    dict_of_num = defaultdict(list)
    for idx in range(len(alphas[0])):
        dict_of_num[idx] = []

    variance_over_time_array = []

    max_time = max(result_dict_statistics.keys())
    for time_stamp in result_dict_statistics.keys():  
        coefficients = np.asarray(result_dict_statistics[time_stamp][PCE_COEFF_ENTRY], dtype=np.float64)
        # if "Var" in result_dict_statistics[time_stamp]:
        #     variance = result_dict_statistics[time_stamp]['Var']
        # else:
        variance = np.sum(coefficients[index] ** 2, axis=0)
        variance_over_time_array.append(variance)
        for idx in range(len(alphas[0])):
            index_local = np.array([alpha[idx] > 0 for alpha in alphas])      # Compute the total Sobol indices
            # TODO add for the first order
            # index_local = np.array(
            #     [
            #         bool(alpha[idx] and not any(alpha[:idx] + alpha[idx + 1 :]))
            #         for alpha in alphas
            #     ]
            # )
            dict_of_num[idx].append(np.sum(coefficients[index_local] ** 2, axis=0))  # scaling with norm of the polynomial corresponding to the index_local

    variance_over_time_array = np.asarray(variance_over_time_array, dtype=np.float64)
    if total_variance is None:
        denum = np.dot(variance_over_time_array, weights)
    else:
        denum = total_variance
    print(f"[INFO computing_generalized_sobol_total_indices_from_poly_expan] total_variance={denum}")
    for idx in range(len(alphas[0])):
        param_name = param_names[idx]
        num = np.dot(np.asarray(dict_of_num[idx], dtype=np.float64), weights)
        s_tot_generalized = num/denum
        result_dict_statistics[max_time][f"generalized_sobol_total_index_{param_name}"] = s_tot_generalized
        print(f"Generalized Total Sobol Index for {param_name} is {s_tot_generalized}")
        with open(fileName, 'a') as file:
            # Write each variable to the file followed by a newline character
            file.write(f'{param_name}: {s_tot_generalized}\n')


def computing_generalized_sobol_total_indices_from_poly_expan_over_time(
    result_dict_statistics: Dict[Any, Dict[str, Any]],
    polynomial_expansion: cp.polynomial,
    weights: np.ndarray,
    param_names: List[str],
    fileName: Union[str, pathlib.PosixPath],
    total_variance=None,
    look_back_window_size: Union[str, int]='whole',
    resolution: str = "integer",
):
    for time_stamp in result_dict_statistics.keys():
        qoi_time_stamp = time_stamp
        fileName_new = extend_filename_with_timestamp(fileName, qoi_time_stamp)
        print(f"DEBUGGING INFO : {qoi_time_stamp} {fileName_new}")
        computing_generalized_sobol_total_indices_from_poly_expan_single_timesample(
            qoi_time_stamp, result_dict_statistics, polynomial_expansion, weights, param_names, fileName_new, total_variance=None,
            look_back_window_size=look_back_window_size, resolution=resolution,
        )
    
    
def computing_generalized_sobol_total_indices_from_poly_expan_single_timesample(
    qoi_time_stamp: Any,
    result_dict_statistics: Dict[Any, Dict[str, Any]],
    polynomial_expansion: cp.polynomial,
    weights: np.ndarray,
    param_names: List[str],
    fileName: Union[str, pathlib.PosixPath],
    total_variance=None,
    look_back_window_size: Union[str, int] = 'whole',
    resolution: str = "integer"
):
    """
    Computes the generalized Sobol total indices from a polynomial expansion.
    This function computes for the last timestep in result_dict_statistics!
    The current implamantion of the function assumes that the polynomial expansion is normalized.
    One would have to do scaling with norms of the polynomials if they are not normalized.

    Args:
        qoi_time_stamp (Any): The time stamp of the quantity of interest.
        result_dict_statistics (Dict[Any, Dict[str, Any]]): A dictionary containing the statistics of the results.
         Important assumtion is that it contains the coefficients of the polynomial expansion under 'gpce_coeff'/PCE_COEFF_ENTRY key over time.
        polynomial_expansion (cp.polynomial): The polynomial expansion.
        weights (np.ndarray): An array of weights for time quadratures.
        param_names (List[str]): A list of parameter names.
        fileName (str): The name of the file to write the results to.
        look_back_window_size (Union[str, int], optional): The look back window size. Defaults to 'whole'.
        resolution (str, optional): The resolution of the time stamps. Defaults to "integer".

    Returns:
        None

    Raises:
        None
    """
    # TODO Important aspect here is if polynomial_expansion is normalized or not
    dic = polynomial_expansion.todict()
    alphas = []
    for idx in range(len(polynomial_expansion)):
        expons = np.array([key for key, value in dic.items() if value[idx]])
        alphas.append(tuple(expons[np.argmax(expons.sum(1))]))

    index = np.array([any(alpha) for alpha in alphas])

    dict_of_num = defaultdict(list)
    for idx in range(len(alphas[0])):
        dict_of_num[idx] = []

    variance_over_time_array = []

    total_number_of_timestamps = len(result_dict_statistics.keys())
    total_number_of_weights = len(weights)
    # assert total_number_of_timestamps == total_number_of_weights, "The number of timestamps and weights must be the same!"
    number_of_timestamps = 0

    if qoi_time_stamp not in result_dict_statistics.keys():
        raise ValueError(f"The time stamp {qoi_time_stamp} is not in the dictionary of statistics!")

    for time_stamp in result_dict_statistics.keys():
        if time_stamp > qoi_time_stamp:
            continue
        if look_back_window_size!='whole':
            take_current_time_stamp_into_account = True
            take_current_time_stamp_into_account = is_time_stamp_in_look_back_window(time_stamp, qoi_time_stamp, look_back_window_size, resolution)
            if not take_current_time_stamp_into_account:
                continue
        number_of_timestamps += 1
        coefficients = np.asarray(result_dict_statistics[time_stamp][PCE_COEFF_ENTRY], dtype=np.float64)
        variance = np.sum(coefficients[index] ** 2, axis=0)
        variance_over_time_array.append(variance)
        for idx in range(len(alphas[0])):
            index_local = np.array([alpha[idx] > 0 for alpha in alphas])      # Compute the total Sobol indices
            # TODO add for the first order
            # index_local = np.array(
            #     [
            #         bool(alpha[idx] and not any(alpha[:idx] + alpha[idx + 1 :]))
            #         for alpha in alphas
            #     ]
            # )
            dict_of_num[idx].append(np.sum(coefficients[index_local] ** 2, axis=0))  # scaling with norm of the polynomial corresponding to the index_local
    
    if number_of_timestamps==0:
        raise ValueError(f"No time stamp is found in the dictionary of statistics for the time stamp {qoi_time_stamp}")
    
    # update the weights over time
    # TODO - change this eventually
    total_variance = None
    # if total_number_of_weights!=number_of_timestamps:
    if number_of_timestamps==1:
        h = 1
        weights = [1,]
    else:
        # TODO Think about 1 in num?
        h = 1/(number_of_timestamps-1)
        weights = [h for i in range(number_of_timestamps)]
        assert number_of_timestamps==len(weights)
    if len(weights) >= 3:
        weights[0] /= 2
        weights[-1] /= 2
    weights = np.asarray(weights, dtype=np.float64)

    variance_over_time_array = np.asarray(variance_over_time_array, dtype=np.float64)

    if total_variance is None:
        denum = np.dot(variance_over_time_array, weights)
    else:
        denum = total_variance

    for idx in range(len(alphas[0])):
        param_name = param_names[idx]
        num = np.dot(np.asarray(dict_of_num[idx], dtype=np.float64), weights)
        s_tot_generalized = num/denum
        result_dict_statistics[qoi_time_stamp][f"generalized_sobol_total_index_{param_name}"] = s_tot_generalized
        print(f"Generalized Total Sobol Index for {param_name} for {qoi_time_stamp} is {s_tot_generalized}")
        # with open(fileName, 'a') as file:
        #     # Write each variable to the file followed by a newline character
        #     file.write(f'{param_name}: {s_tot_generalized}\n')


def compute_difference_between_two_time_stamps(end_timestamp, start_timestamp, resolution):
    if resolution == "integer":
        return end_timestamp - start_timestamp
    elif resolution in ["daily", "hourly", "minute"]:
        end_timestamp_pdTimestamp = pd.Timestamp(end_timestamp)
        start_timestamp_pdTimestamp = pd.Timestamp(start_timestamp)    
        difference = end_timestamp_pdTimestamp - start_timestamp_pdTimestamp
        if resolution == "daily": 
            return difference.days
        elif resolution == "hourly":
            total_seconds = difference.total_seconds()
            total_hours = total_seconds // 3600  # Total hours
            return total_hours
        elif resolution == "minute":
            total_seconds = difference.total_seconds()
            total_minutes = total_seconds // 60  # Total minutes
            return total_minutes
    else:
        raise ValueError(f"Unknown resolution - {resolution}")


def is_time_stamp_in_look_back_window(time_stamp, qoi_time_stamp, look_back_window_size, resolution):
    if resolution == "integer":
        if qoi_time_stamp - time_stamp > look_back_window_size:
            return False
    elif resolution in ["daily", "hourly", "minute"]:
        qoi_time_stamp_pdTimestamp = pd.Timestamp(qoi_time_stamp)
        time_stamp_pdTimestamp = pd.Timestamp(time_stamp)
        difference = qoi_time_stamp_pdTimestamp - time_stamp_pdTimestamp
        if resolution == "daily":
            if difference.days > look_back_window_size:
                return False
        elif resolution == "hourly":
            total_seconds = difference.total_seconds()
            total_hours = total_seconds // 3600  # Total hours
            if total_hours > look_back_window_size:
                return False
        elif resolution == "minute":
            total_seconds = difference.total_seconds()
            total_minutes = total_seconds // 60  # Total minutes
            if total_minutes > look_back_window_size:
                return False
    else:
        raise ValueError(f"Unknown resolution - {resolution}")
    return True
# =================================================================================================
# Utility for SG analysis
# =================================================================================================


def generate_table_single_rule_over_dim_and_orders_sparse_and_nonsparse(rule, dists, dim, q_orders, growth=None):
    num_nodes = np.zeros((len(dists), len(q_orders), 2), dtype=np.int32)

    if growth is None:
        growth = True if rule == "c" else False

    for dist_i, dist in enumerate(dists):
        for q_i, order in enumerate(q_orders):
            abscissas, weights = cp.generate_quadrature(order, dist, rule=rule, sparse=True, growth=growth)
            num_nodes[dist_i][q_i][0] = len(abscissas.T)
            abscissas, weights = cp.generate_quadrature(order, dist, rule=rule, sparse=False, growth=growth)
            num_nodes[dist_i][q_i][1] = len(abscissas.T)

    table = []
    for dist_i, dist in enumerate(dists):
        table.append(["*", "*", "*", "*"])
        for q_i, order in enumerate(q_orders):
            table.append([rule, dim[dist_i], order, num_nodes[dist_i][q_i][0], num_nodes[dist_i][q_i][1]])
    print(tabulate(table, headers=["rule", "dim", "q", "#nodes sparse tensor", "#nodes full tensor"], numalign="right"))


def generate_table_single_rule_over_dim_and_orders(rule, dists, dim, q_orders, sparse=True, growth=None):
    num_nodes = np.zeros((len(dists), len(q_orders), 1), dtype=np.int32)

    table_column_name = "#nodes sparse tensor" if sparse else "#nodes full tensor"

    if growth is None:
        growth = True if rule == "c" else False

    for dist_i, dist in enumerate(dists):
        for q_i, order in enumerate(q_orders):
            abscissas, weights = cp.generate_quadrature(order, dist, rule=rule, sparse=sparse, growth=growth)
            num_nodes[dist_i][q_i][0] = len(abscissas.T)

    table = []
    for dist_i, dist in enumerate(dists):
        table.append(["*", "*", "*", "*"])
        for q_i, order in enumerate(q_orders):
            table.append([rule, dim[dist_i], order, num_nodes[dist_i][q_i][0]])
    print(tabulate(table, headers=["rule", "dim", "q", table_column_name], numalign="right"))


def plot_2d_matrix_of_nodes_over_orders(rule, dist, orders, sparse=False, growth=None):
    if growth is None:
        growth = True if rule == "c" else False

    for order in orders:
        abscissas, weights = cp.generate_quadrature(order, dist, rule=rule, growth=growth, sparse=sparse)
        # print(order, abscissas.round(3), weights.round(3))

        dimensionality = len(abscissas)

        abscissas = abscissas.T

        dict_for_plotting = {f"x{i}": abscissas[:, i] for i in range(dimensionality)}

        df_nodes_weights = pd.DataFrame(dict_for_plotting)

        sns.set(style="ticks", color_codes=True)
        g = sns.pairplot(df_nodes_weights, vars=list(dict_for_plotting.keys()), corner=True)
        if growth:
            title = f"{rule} points chaospy; order = {order}; sparse={str(sparse)}; #nodes={abscissas.shape[0]}; growth=True"
        else:
            title = f"{rule} points chaospy; order = {order}; sparse={str(sparse)}; #nodes={abscissas.shape[0]}"
        plt.title(title, loc='left')
        plt.show()


def generate_table_over_rules_orders_for_single_dim(rules, dist, dim, q_orders, growth=None):
    num_nodes = np.zeros((len(rules), len(q_orders), 2), dtype=np.int32)

    # produce num_nodes matrix
    for r_i, r in enumerate(rules):
        for q_i, q in enumerate(q_orders):

            if growth is None:
                growth = True if r == "c" else False

            nodes, weights = cp.generate_quadrature(q, dist, rule=r, growth=growth)
            num_nodes[r_i][q_i][0] = len(nodes.T)

            nodes, weights = cp.generate_quadrature(q, dist, rule=r, sparse=True)
            num_nodes[r_i][q_i][1] = len(nodes.T)

    # create table
    table = []
    for r_i, r in enumerate(rules):
        for q_i, q in enumerate(q_orders):
            ok = num_nodes[r_i][q_i][1] < num_nodes[r_i][q_i][0]
            table.append([r, q, num_nodes[r_i][q_i][0], num_nodes[r_i][q_i][1], "ok" if ok else "nok"])

    print(tabulate(table,
                   headers=["rule", "q", "#nodes full tensor", "#nodes sparse", "#nodes sparse < #nodes full tensor"],
                   numalign="right"))

# =================================================================================================
# MISC - Different set of utility functions
# =================================================================================================
def process_dict_set_predictions_to_zero(dict_set_predictions_to_zero, list_qois):
    if not isinstance(dict_set_predictions_to_zero, dict) or not dict_set_predictions_to_zero:
        if isinstance(dict_set_predictions_to_zero, bool):
            bool_dict_set_predictions_to_zero = dict_set_predictions_to_zero
            dict_set_predictions_to_zero = {}
            for single_qoi in list_qois:
                dict_set_predictions_to_zero[single_qoi] = bool_dict_set_predictions_to_zero
        else:
            dict_set_predictions_to_zero = {}
            for single_qoi in list_qois:
                dict_set_predictions_to_zero[single_qoi] = False
    for single_qoi in list_qois:
        if single_qoi not in dict_set_predictions_to_zero:
            dict_set_predictions_to_zero[single_qoi] = False
    return dict_set_predictions_to_zero


def write_dict_to_file(d, filename, indent=0):
    with open(filename, 'w') as f:
        def write_item(d, indent):
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write(" " * indent + f"{key}: {{\n")
                    write_item(value, indent + 4)
                    f.write(" " * indent + "}\n")
                else:
                    f.write(" " * indent + f"{key}: {value}\n")
        write_item(d, indent)


def find_overlap(lists):
    return list(set.intersection(*map(set, lists)))
    

def extend_filename_with_timestamp(file_path, time_stamp):
    file_path = pathlib.Path(file_path)
    directory_structure = file_path.parent
    # Extract the file name without the extension
    file_stem = file_path.stem
    # Extract the file extension
    file_suffix = file_path.suffix
    # Construct the new file name
    new_file_path = directory_structure / f"{file_stem}_{time_stamp}{file_suffix}"
    return new_file_path


def is_nested_dict_empty(nested_dict):
    if not isinstance(nested_dict, dict):
        return False
    if not nested_dict:
        return True
    return all(is_nested_dict_empty(val) for val in nested_dict.values())


def is_nested_dict_empty_or_none(nested_dict):
    if not isinstance(nested_dict, dict):
        return nested_dict is None
    if not nested_dict:
        return True
    return all(is_nested_dict_empty_or_none(val) for val in nested_dict.values())