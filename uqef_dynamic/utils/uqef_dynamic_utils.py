"""
Set of utility functions for postprocessing data for UQ runs of different models which extend UQEF-Dynamic/time_dependent_model.
Many of these functions exist as well as part of time_dependent_statistics.TimeDependentStatistics or in utilty module
but here we are trying to provide a more general set of functions that can be used for postprocessing data from different UQ and SA runs

In general ulity functions present here fit the way the data is saved in UQEF-Dynamic
and time_dependent_model and time_dependent_statistics modules!

@author: Ivana Jovanovic Buha
"""
from collections import defaultdict
import dill
import pickle
import os
import numpy as np
import math
import pathlib
import pandas as pd
import scipy
import seaborn as sns
import time
from typing import List, Optional, Dict, Any, Union, Tuple

# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo

import matplotlib.pyplot as plt
pd.options.plotting.backend = "plotly"

import sys
#sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')

import chaospy as cp

from uqef_dynamic.utils import colors
from uqef_dynamic.utils import utility
from uqef_dynamic.utils import sens_indices_sampling_based_utils

# ============================================================================================
# Whole pipeline for reading the output saved by UQEF-Dynamic simulation and producing dict of interes
# ============================================================================================
def read_output_files_uqef_dynamic(workingDir, printing=False, **kwargs):
    """
    This function reads the output files created by the UQEF-Dynamic pipeline.
    It reads general file produced by all the models (not specific to any model).
    Args:
    - workingDir: pathlib.Path object, path to the working directory
    - printing: bool, if True, print the information about the output files

    Keyword Args:
    - read_saved_simulations: bool, if True, read the simulation results
    - read_saved_states: bool, if True, read the state results

    Returns:
    - results_dict: dict, dictionary with the following keys / values:
        - workingDir: pathlib.Path object, path to the working directory
        - args_files: dict, dictionary sotrying the paths to the output files
           take a look at the function utility.get_dict_with_output_file_paths_based_on_workingDir
        - uqsim_args_dict: dict, dictionary with the UQEF simulation arguments
        - model: str, name of the model
        - inputModelDir: pathlib.Path object, path to the input model directory
        - configurationObject: dict, configuration object
        - simulation_settings_dict: dict, dictionary with the simulation settings
           take a look at the function utility.read_simulation_settings_from_configuration_object
        - simulationNodes: object, UQEF simulation nodes
        - time_info: str, time information; None if missing file
        - params_list: list, list of model uncertain parameters
        - df_index_parameter: pandas DataFrame, index parameter; None is missing
        - df_index_parameter_gof: pandas DataFrame, index parameter goodness-of-fit (GOF); None is missing
        - gof_list: list, list of goodness-of-fit (GOF) measures; None is missing
        - df_simulation_result: pandas DataFrame, simulation results; None is missing
        - df_state: pandas DataFrame, state; None is missing

        - time_model_simulations: str, time for model simulations
        - time_computing_statistics: str, time for computing statistics
        - parameterNames: list, list of parameter names
        - stochasticParameterNames: list, list of stochastic parameter names
        - number_full_model_evaluations: int, number of full model evaluations
        - full_number_quadrature_points
        - plus extra entries from update_dict_with_results_of_interest_based_on_uqsim_args_dict

    """
    if not workingDir.is_dir():
        raise Exception(f"Directory {workingDir} does not exist!")

    results_dict = {}
    
    args_files = utility.get_dict_with_output_file_paths_based_on_workingDir(
        workingDir,
    )
    for key, value in args_files.items():
        globals()[key] = value

    # Load the UQSim args dictionary
    uqsim_args_dict = utility.load_uqsim_args_dict(args_file)
    if printing:
        print("INFO: uqsim_args_dict: ", uqsim_args_dict)
    model = uqsim_args_dict["model"]

    # Maybe to do something like this...
    # if inputModelDir is not None:
    #     if inputModelDir != uqsim_args_dict["inputModelDir"]:
    #         uqsim_args_dict["inputModelDir"] = pathlib.Path(inputModelDir)
    # else:
    #     inputModelDir = uqsim_args_dict["inputModelDir"]
    # inputModelDir = pathlib.Path(inputModelDir)
    inputModelDir = uqsim_args_dict["inputModelDir"]

    # Load the configuration object
    configurationObject = utility.load_configuration_object(configuration_object_file)
    if printing:
        print("configurationObject: ", configurationObject)
    simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)

    # Timing information
    if time_info_file.is_file():
        with open(time_info_file, 'r') as f:
            time_info = f.read() #readlines()?
        if printing:
            print("INFO: time_info: ", time_info)
    else:
        time_info = None

    # Reading Nodes and Parameters
    if not nodes_file.is_file():
        print(f"INFO: simulationNodes file {nodes_file} is missing!!!")
    with open(nodes_file, 'rb') as f:
        simulationNodes = pickle.load(f)
    if printing:
        print("INFO: simulationNodes: ", simulationNodes)
    dim = simulationNodes.distNodes.shape[0]
    # number_model_runs = simulationNodes.nodes.shape[1]  # this is actually, not a final number of model runs for saltelli's simulation
    distStandard = simulationNodes.joinedStandardDists
    dist = simulationNodes.joinedDists
    if printing:
        print(f"INFO: model-{model}; dim - {dim};")

    results_dict["workingDir"]=workingDir
    results_dict["args_files"]=args_files
    results_dict["uqsim_args_dict"]=uqsim_args_dict
    results_dict["model"]=model
    results_dict["inputModelDir"]=inputModelDir
    results_dict["configurationObject"]=configurationObject
    results_dict["simulation_settings_dict"]=simulation_settings_dict
    results_dict["time_info"]=time_info
    results_dict["simulationNodes"]=simulationNodes
    results_dict["dim"]=dim

    if df_index_parameter_file.is_file():
        df_index_parameter = pd.read_pickle(df_index_parameter_file, compression="gzip")
        params_list = utility._get_parameter_columns_df_index_parameter_gof(
            df_index_parameter)
        if printing:
            print(f"INFO: df_index_parameter - {df_index_parameter}")
    else:
        df_index_parameter = None
        params_list = []
        for single_param in configurationObject["parameters"]:
            params_list.append(single_param["name"])
    if printing:
        print(f"INFO: params_list - {params_list} (note - all the parameters)")
    results_dict["df_index_parameter"]=df_index_parameter
    results_dict["params_list"]=params_list

    if df_index_parameter_gof_file.is_file():
        df_index_parameter_gof = pd.read_pickle(df_index_parameter_gof_file, compression="gzip")
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)
        if printing:
            print(f"INFO: df_index_parameter_gof - {df_index_parameter_gof}")
            print(f"INFO: gof_list - {gof_list}")
    else:
        df_index_parameter_gof = None
        gof_list = None
    results_dict["df_index_parameter_gof"]=df_index_parameter_gof
    results_dict["gof_list"]=gof_list

    # Reading/Creating DataFrame based on Simulation Nodes / Parameters
    # df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
    # df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)
 
    read_saved_simulations = kwargs.get('read_saved_simulations', False)
    read_saved_states = kwargs.get('read_saved_states', False)

    if read_saved_simulations and df_simulations_file.is_file():
        df_simulation_result = pd.read_pickle(df_simulations_file, compression="gzip")
        if printing:
            print(f"INFO: df_simulation_result - {df_simulation_result}")
    else:
        df_simulation_result = None
    results_dict["df_simulation_result"]=df_simulation_result

    if read_saved_states and df_state_file.is_file():
        df_state = pd.read_pickle(df_state_file, compression="gzip")
        if printing:
            print(f"INFO: df_state - {df_state}")
    else:
        df_state = None
    results_dict["df_state"]=df_state

    # ========================================================
    # Extra stuff as from read_all_saved_uqef_dynamic_results_and_produce_dict_of_interest
    # Update dict with results of interest based on uqsim_args_dict - add variant, q_order, mc_numevaluations
    update_dict_with_results_of_interest_based_on_uqsim_args_dict(results_dict, uqsim_args_dict)
    
    # Extra Timing information
    if time_info is not None:
        for line in time_info:
            if line.startswith("time_model_simulations"):
                results_dict["time_model_simulations"] = line.split(':')[1].strip()
            elif line.startswith("time_computing_statistics"):
                results_dict["time_computing_statistics"] = line.split(':')[1].strip()

    # whatch-out this might be tricky when not all params are regarded as uncertain!
    param_labeles = utility.get_list_of_uncertain_parameters_from_configuration_dict(
        configurationObject, raise_error=True, uq_method=uqsim_args_dict["uq_method"])
    #print(f"Debugging - params_list: {params_list}; simulationNodes.nodeNames: {simulationNodes.nodeNames}; param_labeles: {param_labeles}")    
    # results_dict["parameterNames"] = params_list  #not simulationNodes.nodeNames, instead better simulationNodes.orderdDistsNames
    results_dict["stochasticParameterNames"] = param_labeles

    simulation_parameters_file = args_files["simulation_parameters_file"]
    if df_simulation_result is not None:
        results_dict["number_full_model_evaluations"] = len(df_simulation_result)
    elif simulation_parameters_file.is_file():
        simulation_parameters = np.load(simulation_parameters_file,  allow_pickle=True)
        #print(f"Debugging - simulation_parameters.shape: {simulation_parameters.shape}")
        results_dict["number_full_model_evaluations"] = simulation_parameters.shape[0]
    else:
        if uqsim_args_dict["uq_method"]!="saltelli":
            results_dict["number_full_model_evaluations"] = simulationNodes.nodes.shape[1]
        else:
            results_dict["number_full_model_evaluations"] = (uqsim_args_dict["mc_numevaluations"]) * (2 + dim)

    if results_dict["variant"] not in ["m1", "m2"]:
        results_dict["full_number_quadrature_points"] = \
        (results_dict["q_order"] + 1) ** dim
    
    # list_qoi_column = simulation_settings_dict.list_qoi_column
    # list_qoi_column = statisticsObject.list_qoi_column
    # ========================================================

    return results_dict


def read_all_saved_uqef_dynamic_results_and_produce_dict_of_interest(workingDir, **kwargs):
    """
    This function builds on top of the read_output_files_uqef_dynamic function.
    It returns similar dictionary as read_all_saved_uqef_dynamic_results_and_produce_dict_of_interest_single_qoi_single_timestamp
    but for all the timestamps.
    Args:
    - workingDir: pathlib.Path object, path to the working directory

    Returns:
    - results_dict: dict, dictionary with the following keys / values:
        - time_model_simulations: str, time for model simulations
        - time_computing_statistics: str, time for computing statistics
        - parameterNames: list, list of parameter names
        - stochasticParameterNames: list, list of stochastic parameter names
        - number_full_model_evaluations: int, number of full model evaluations
        - full_number_quadrature_points
        - plus extra entries from update_dict_with_results_of_interest_based_on_uqsim_args_dict

    """
    dict_with_results_of_interest = defaultdict()
    results_dict = read_output_files_uqef_dynamic(workingDir, **kwargs)
    for key, value in results_dict.items():
        globals()[key] = value

    dim = simulationNodes.distNodes.shape[0]

    # Update dict with results of interest based on uqsim_args_dict - add variant, q_order, mc_numevaluations
    update_dict_with_results_of_interest_based_on_uqsim_args_dict(dict_with_results_of_interest, uqsim_args_dict)

    # Timing information
    if time_info is not None:
        for line in time_info:
            if line.startswith("time_model_simulations"):
                dict_with_results_of_interest["time_model_simulations"] = line.split(':')[1].strip()
            elif line.startswith("time_computing_statistics"):
                dict_with_results_of_interest["time_computing_statistics"] = line.split(':')[1].strip()

    # whatch-out this might be tricky when not all params are regarded as uncertain!
    param_labeles = utility.get_list_of_uncertain_parameters_from_configuration_dict(
        configurationObject, raise_error=True, uq_method=uqsim_args_dict["uq_method"])
    #print(f"Debugging - params_list: {params_list}; simulationNodes.nodeNames: {simulationNodes.nodeNames}; param_labeles: {param_labeles}")    
    dict_with_results_of_interest["parameterNames"] = params_list  #not simulationNodes.nodeNames, instead better simulationNodes.orderdDistsNames
    dict_with_results_of_interest["stochasticParameterNames"] = param_labeles

    simulation_parameters_file = args_files["simulation_parameters_file"]
    if df_simulation_result is not None:
        dict_with_results_of_interest["number_full_model_evaluations"] = len(df_simulation_result)
    elif simulation_parameters_file.is_file():
        simulation_parameters = np.load(simulation_parameters_file,  allow_pickle=True)
        #print(f"Debugging - simulation_parameters.shape: {simulation_parameters.shape}")
        dict_with_results_of_interest["number_full_model_evaluations"] = simulation_parameters.shape[0]
    else:
        if uqsim_args_dict["uq_method"]!="saltelli":
            dict_with_results_of_interest["number_full_model_evaluations"] = simulationNodes.nodes.shape[1]
        else:
            dict_with_results_of_interest["number_full_model_evaluations"] = (uqsim_args_dict["mc_numevaluations"]) * (2 + dim)

    if dict_with_results_of_interest["variant"] not in ["m1", "m2"]:
        dict_with_results_of_interest["full_number_quadrature_points"] = \
        (dict_with_results_of_interest["q_order"] + 1) ** dim

    return dict_with_results_of_interest

# ============================================================================================


def read_all_saved_uqef_dynamic_results_and_produce_dict_of_interest_single_qoi_single_timestamp(workingDir, 
timestamp, qoi_column_name=None, time_column_name=utility.TIME_COLUMN_NAME, plotting=False, model=None, **kwargs):
    """
    TODO This function eventaully should be refactored to be more general and to be able to read all the files for all the timesteps
    TODO Also options for reading analytical values should be added!
    This function is a whole pipeline for reading the output saved by UQEF-Dynamic simulation and producing dict of interest
    :param workingDir: in case it does not exist, the function raises an Exception
    :param timestamp: in case it is None the function raises a NotImplementedError
    :param qoi_column_name: In case it is None, it will get the value based on simulation_settings_dict["list_qoi_column"]

    :param plotting: if True, it will plot some of the results
    :param model: if provided, it will be used to compare the surrogate model with the full model
    :param kwargs: additional arguments that can be passed to the function
        analytical_E - analytical value for the mean
        analytical_Var - analytical value for the variance
        analytical_Sobol_t - analytical value for the Sobol total indices
        analytical_Sobol_m - analytical value for the Sobol main indices
        compare_surrogate_and_original_model - if True, it will compare the surrogate model with the full model
        comparison_surrogate_vs_model_numSamples - number of samples for the comparison
        comparison_surrogate_vs_model_mc_rule - rule for the comparison
        evaluateSurrogateAtStandardDist - if True, it will evaluate the surrogate model at the standard distribution
        read_saved_simulations: bool, if True, read the simulation results
        read_saved_states: bool, if True, read the state results
    :return: dict_with_results_of_interest

    """

    if not workingDir.is_dir():
        raise Exception(f"Directory {workingDir} does not exist!")

    dict_with_results_of_interest = defaultdict()
    dict_with_results_of_interest["outputModelDir"] = workingDir

    args_files = utility.get_dict_with_output_file_paths_based_on_workingDir(workingDir)
    for key, value in args_files.items():
        globals()[key] = value

    # Reading UQEF-Dynamic Output files

    # Load the UQSim args dictionary
    uqsim_args_dict = utility.load_uqsim_args_dict(args_file)

    # Update dict with results of interest based on uqsim_args_dict - add variant, q_order, mc_numevaluations
    update_dict_with_results_of_interest_based_on_uqsim_args_dict(dict_with_results_of_interest, uqsim_args_dict)

    # Load the configuration object
    configurationObject = utility.load_configuration_object(configuration_object_file)
    simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)

    # Timing information
    with open(time_info_file) as f:
        lines = f.readlines()
        for line in lines:
            #print(line)
            if line.startswith("time_model_simulations"):
                dict_with_results_of_interest["time_model_simulations"] = line.split(':')[1].strip()
            elif line.startswith("time_computing_statistics"):
                dict_with_results_of_interest["time_computing_statistics"] = line.split(':')[1].strip()

    ########################################################
    # Reading Simulation Nodes / Parameters
    with open(nodes_file, 'rb') as f:
        simulationNodes = pickle.load(f)
    dim = simulationNodes.distNodes.shape[0] #len(param_labeles)  # this should represent a stochastic dimensionality

    if df_index_parameter_file.is_file():
        df_index_parameter = pd.read_pickle(df_index_parameter_file, compression="gzip")
    else:
        df_index_parameter = None
    if df_index_parameter is not None:
        params_list = utility._get_parameter_columns_df_index_parameter_gof(
            df_index_parameter)
    else:
        params_list = []
        for single_param in configurationObject["parameters"]:
            params_list.append(single_param["name"])

    if df_index_parameter_gof_file.is_file():
        df_index_parameter_gof = pd.read_pickle(df_index_parameter_gof_file, compression="gzip")
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)

    # whatch-out this might be tricky when not all params are regarded as uncertain!
    param_labeles = utility.get_list_of_uncertain_parameters_from_configuration_dict(
        configurationObject, raise_error=True, uq_method=uqsim_args_dict["uq_method"])
    #print(f"Debugging - params_list: {params_list}; simulationNodes.nodeNames: {simulationNodes.nodeNames}; param_labeles: {param_labeles}")    
    dict_with_results_of_interest["parameterNames"] = params_list  #not simulationNodes.nodeNames, instead better simulationNodes.orderdDistsNames
    dict_with_results_of_interest["stochasticParameterNames"] = param_labeles

    # updating dict_with_results_of_interest
    # number_full_model_evaluations will be overwritten below in certain setups 
    # (main issue is to fatch the correct number for the saltelli method)
    dict_with_results_of_interest["number_full_model_evaluations"] = simulationNodes.nodes.shape[1]
    if uqsim_args_dict["uq_method"]=="saltelli":
        dict_with_results_of_interest["number_full_model_evaluations"] = (uqsim_args_dict["mc_numevaluations"]) * (2 + dim)
        # dict_with_results_of_interest["number_full_model_evaluations"] = (simulationNodes.nodes.shape[1]/2) * (2 + dim)
    
    # Reading parameters which were saved to run/stimulate the model
    if simulation_parameters_file.is_file():
        simulation_parameters = np.load(simulation_parameters_file,  allow_pickle=True)
        #print(f"Debugging - simulation_parameters.shape: {simulation_parameters.shape}")
        dict_with_results_of_interest["number_full_model_evaluations"] = simulation_parameters.shape[0]

    if dict_with_results_of_interest["variant"] not in ["m1", "m2"]:
        dict_with_results_of_interest["full_number_quadrature_points"] = \
        (dict_with_results_of_interest["q_order"] + 1) ** dim

    ########################################################
    # TODO Whatch out if params_list or param_labeles should be propagated here 
    # Reading/Creating DataFrame based on Simulation Nodes / Parameters
    df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
    df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)

    if plotting:
        fig = utility.plot_subplot_params_hist_from_df(df_index_parameter)
        fig.update_layout(title="Prior Distribution of the Parameters",)
        fig.show()
        utility.plot_2d_matrix_static(df_nodes, nodes_or_paramters="nodes")
        utility.plot_2d_matrix_static(df_nodes_params, nodes_or_paramters="parameters")

    ########################################################
    # Reading all the simulations
    # maybe this is not alway necessary

    read_saved_simulations = kwargs.get('read_saved_simulations', False)
    read_saved_states = kwargs.get('read_saved_states', False)

    if read_saved_simulations and df_simulations_file.is_file():
        # Reading Saved Simulations - Note: This migh be a huge file,
        # especially for MC/Saltelli kind of simulations
        df_simulation_result = pd.read_pickle(df_simulations_file, compression="gzip")
        dict_with_results_of_interest["number_full_model_evaluations"] = len(df_simulation_result)
    else:
        df_simulation_result = None

    if read_saved_states and df_state_file.is_file():
        df_state = pd.read_pickle(df_state_file, compression="gzip")
        if printing:
            print(f"INFO: df_state - {df_state}")

    ########################################################
    # Specific part for a single QoI and timestep, this should be refactored to be more general
    qoi_column = simulation_settings_dict["qoi_column"]
    list_qoi_column = simulation_settings_dict["list_qoi_column"] 
    if qoi_column_name is None:
        #qoi_column_name = qoi_column
        qoi_column_name = list_qoi_column[0]  # qoi_column_name should be the same as qoi_column
    else:
        if qoi_column_name not in list_qoi_column:
            raise ValueError(f"QoI column name {qoi_column_name} is not in the list of QoI columns {list_qoi_column}")
    dict_with_results_of_interest["qoi_column_name"] = qoi_column_name
    # TODO extraxt somehow info about timestamp
    if timestamp is None:
        raise NotImplementedError
    if isinstance(timestamp, (int, float)) or isinstance(timestamp, str):
        convert_to_pd_timestamp = False
    elif isinstance(timestamp, pd.Timestamp):
        convert_to_pd_timestamp = True
    else:
        raise ValueError(f"Timestamp {timestamp} is not in the correct format")
    dict_with_results_of_interest["timestamp"] = timestamp

    ########################################################
    # Readinga dictionary containing statistics data
    # Reading UQEF-Dynamic Output files specific for some QoI and timestep
    # dict_output_file_paths_qoi = utility.get_dict_with_qoi_name_specific_output_file_paths_based_on_workingDir(workingDir, qoi_column_name)
    # statistics_dictionary_file = dict_output_file_paths_qoi.get("statistics_dictionary_file")
    # Read a Dictionary Containing Statistics Data
    single_timestamp_single_file = uqsim_args_dict.get("instantly_save_results_for_each_time_step", False)  # TODO might be problem if instantly_save_results_for_each_time_step was overwritten
    statistics_dictionary = read_all_saved_statistics_dict(\
        workingDir, list_qoi_column, single_timestamp_single_file, throw_error=True, convert_to_pd_timestamp=convert_to_pd_timestamp)
    statistics_dictionary  = statistics_dictionary[qoi_column_name]
    # extract only stat dict for a particula timestamp
    if pd.Timestamp(timestamp) in statistics_dictionary:
        statistics_dictionary = statistics_dictionary[pd.Timestamp(timestimestamptemp)]
    elif timestamp in statistics_dictionary:
        statistics_dictionary = statistics_dictionary[timestamp]
    else:
        raise ValueError(f"Time-stamp {timestamp} is not in the statistics dictionary")

    ########################################################

    # TODO Hm, think if this is needed 
    dict_with_results_of_interest.update(statistics_dictionary)

    ########################################################
    if "Sobol_m" in statistics_dictionary:
        dict_with_results_of_interest['sobol_m'] = dict_with_results_of_interest.pop('Sobol_m')
        if isinstance(dict_with_results_of_interest['sobol_m'], (list, np.ndarray)):
            for index, single_stoch_param_name in enumerate(dict_with_results_of_interest["stochasticParameterNames"]):
                if f"Sobol_m_{single_stoch_param_name}" in statistics_dictionary:
                    dict_with_results_of_interest[f"sobol_m_{single_stoch_param_name}"] = statistics_dictionary[f"Sobol_m_{single_stoch_param_name}"]
                else:
                    dict_with_results_of_interest[f"sobol_m_{single_stoch_param_name}"] = statistics_dictionary['Sobol_m'][index].round(4)

    if "Sobol_m2" in statistics_dictionary:
        dict_with_results_of_interest['sobol_m2'] = dict_with_results_of_interest.pop('Sobol_m2')
        if isinstance(dict_with_results_of_interest['sobol_m2'], (list, np.ndarray)):
            for index, single_stoch_param_name in enumerate(dict_with_results_of_interest["stochasticParameterNames"]):
                if f"Sobol_m2_{single_stoch_param_name}" in statistics_dictionary:
                    dict_with_results_of_interest[f"sobol_m2_{single_stoch_param_name}"] = statistics_dictionary[f"Sobol_m2_{single_stoch_param_name}"]
                else:
                    dict_with_results_of_interest[f"sobol_m2_{single_stoch_param_name}"] = statistics_dictionary['Sobol_m2'][index].round(4)

    if "Sobol_t" in statistics_dictionary:
        dict_with_results_of_interest['sobol_t'] = dict_with_results_of_interest.pop('Sobol_t')
        if isinstance(dict_with_results_of_interest['sobol_t'], (list, np.ndarray)):
            for index, single_stoch_param_name in enumerate(dict_with_results_of_interest["stochasticParameterNames"]):
                if f"Sobol_t_{single_stoch_param_name}" in statistics_dictionary:
                    dict_with_results_of_interest[f"sobol_t_{single_stoch_param_name}"] = statistics_dictionary[f"Sobol_t_{single_stoch_param_name}"]
                else:
                    dict_with_results_of_interest[f"sobol_t_{single_stoch_param_name}"] = statistics_dictionary['Sobol_t'][index].round(4)
    
    ########################################################
    # Compare analytical values with the computed ones (if analytical values are provided)

    analytical_E = kwargs.get('analytical_E', None)
    analytical_Var = kwargs.get('analytical_Var', None)
    analytical_Sobol_t = kwargs.get('analytical_Sobol_t', None)
    analytical_Sobol_m = kwargs.get('analytical_Sobol_m', None)
    # TODO add options for reading additiona analyitical stat values!

     # TODO eventually extend this such that these qunatitieis can be read from the saved files or computed based on mc simulations
    if analytical_E is not None:
        dict_with_results_of_interest["error_mean"] = abs(analytical_E - statistics_dictionary['E'])
    else:
        print("Analytical values for E are not provided!")

    if analytical_Var is not None:
        dict_with_results_of_interest["error_var"] = abs(analytical_Var - statistics_dictionary['Var'])
    else:    
        ("Analytical values for Var are not provided!")

    if analytical_Sobol_m is not None and "Sobol_m" in statistics_dictionary:
        Sobol_m_error = abs(statistics_dictionary['Sobol_m'] - analytical_Sobol_m)
        dict_with_results_of_interest["sobol_m_error"] = Sobol_m_error
        if isinstance(Sobol_m_error, (list, np.ndarray)):
            for index, single_stoch_param_name in enumerate(dict_with_results_of_interest["stochasticParameterNames"]):
                dict_with_results_of_interest[f"sobol_m_{single_stoch_param_name}_error"] = Sobol_m_error[index].round(4)
    else:
        print("Analytical values for Sobol first order indices are not provided!")
        # TODO Think about how to code this in a more general way!
        # Try to read from additional files being saved, e.g. Sobol indices etc.
        # sobol_m_error_file = workingDir / "sobol_m_error.npy"
        # sobol_m_qoi_file = workingDir / "sobol_m_qoi_file.npy"
    
    if analytical_Sobol_t is not None and "Sobol_t" in statistics_dictionary:
        Sobol_t_error = abs(statistics_dictionary['Sobol_t'] - analytical_Sobol_t)
        dict_with_results_of_interest["sobol_t_error"] = Sobol_t_error
        if isinstance(Sobol_t_error, (list, np.ndarray)):
            for index, single_stoch_param_name in enumerate(dict_with_results_of_interest["stochasticParameterNames"]):
                dict_with_results_of_interest[f"sobol_t_{single_stoch_param_name}_error"] = Sobol_t_error[index].round(4)
    else:
        print("Analytical values for Sobol total indices are not provided!")
        # TODO Think about how to code this in a more general way!
        # Try to read from additional files being saved, e.g. Sobol indices etc.
        # sobol_t_error_file = workingDir / "sobol_t_error.npy"
        # sobol_t_qoi_file = workingDir / "sobol_t_qoi_file.npy"
        
    ########################################################
    # gPCE Surrogate
    # dict_output_file_paths_qoi_time = utility.get_dict_with_qoi_name_timestamp_specific_output_file_paths_based_on_workingDir(workingDir, qoi_column_name, timestamp)
    # gpce_surrogate_file = dict_output_file_paths_qoi_time.get("gpce_surrogate_file")
    # gpce_coeffs_file = dict_output_file_paths_qoi_time.get("gpce_coeffs_file")

    if dict_with_results_of_interest["variant"] not in ["m1", "m2"]:  # TODO or m1 m2 with regression
        # gpce_surrogate_dictionary = read_all_saved_gpce_surrogate_models(workingDir, list_qoi_column, throw_error=False, convert_to_pd_timestamp=convert_to_pd_timestamp)
        # gpce_coeff_dictionary = read_all_saved_gpce_coeffs(workingDir, list_qoi_column, throw_error=False, convert_to_pd_timestamp=convert_to_pd_timestamp)
        # if gpce_surrogate_dictionary is not None:
        #     gpce_surrogate_dictionary = gpce_surrogate_dictionary[qoi_column_name]
        #     if pd.Timestamp(timestamp) in gpce_surrogate_dictionary:
        #         gpce_surrogate = gpce_surrogate_dictionary[pd.Timestamp(timestamp)]
        #     elif timestamp in gpce_surrogate_dictionary:
        #         gpce_surrogate = gpce_surrogate_dictionary[timestamp]
        #     else:
        #         gpce_surrogate = None
        #         print(f"Sorry there is not gpce_surrogate for timestamp - {timestamp}")
        # elif 'gPCE' in statistics_dictionary:
        #     gpce_surrogate = statistics_dictionary['gPCE']
        # else:
        #     print(f"Sorry you did not save gPCE surrogate")

        # if gpce_coeff_dictionary is not None:
        #     gpce_coeff_dictionary = gpce_coeff_dictionary[qoi_column_name]
        #     if pd.Timestamp(timestamp) in gpce_coeff_dictionary:
        #         gpce_coeffs = gpce_coeff_dictionary[pd.Timestamp(timestamp)]
        #     elif timestamp in gpce_coeff_dictionary:
        #         gpce_coeffs = gpce_coeff_dictionary[timestamp]
        #     else:
        #         gpce_coeffs = None
        #         print(f"Sorry there is not gpce_coeff for timestamp - {timestamp}")
        # elif 'gpce_coeff' in statistics_dictionary:
        #     gpce_coeffs = statistics_dictionary['gpce_coeff']
        # else:
        #     print(f"Sorry you did not save coeff of the gpce surrogate (gpce_coeff)")

        # TODO Check if file with data on comparison of surrogate and full model already exists

        compare_surrogate_and_original_model = kwargs.get('compare_surrogate_and_original_model', False)

        if 'gPCE' in statistics_dictionary:
            gpce_surrogate = statistics_dictionary['gPCE']
        else:
            gpce_surrogate = read_single_gpce_surrogate_models(workingDir, qoi_column_name, timestamp, throw_error=False)
            if gpce_surrogate is None:
                print(f"Sorry you did not save gPCE surrogate")
        if 'gpce_coeff' in statistics_dictionary:
            gpce_coeffs = statistics_dictionary['gpce_coeff']
        else:
            gpce_coeffs = read_single_gpce_coeffs(workingDir, qoi_column_name, timestamp, throw_error=False)
            if gpce_coeffs is None:
                print(f"Sorry you did not save coeff of the gpce surrogate (gpce_coeff)")

        if gpce_surrogate is not None:
            dict_with_results_of_interest["max_p_order"] = max(np.linalg.norm(vector, ord=1) for vector in gpce_surrogate.exponents)
            dict_with_results_of_interest["gPCE_num_coeffs"] = gpce_surrogate.exponents.shape[0]
            if plotting:
                indices = gpce_surrogate.exponents
                dimensionality = indices.shape[1]
                number_of_terms = indices.shape[0]
                dict_for_plotting = {f"q_{i+1}":indices[:, i] for i in range(dimensionality)}
                df_nodes_weights = pd.DataFrame(dict_for_plotting)
                sns.set(style="ticks", color_codes=True)
                g = sns.pairplot(df_nodes_weights, vars = list(dict_for_plotting.keys()), corner=True)
                # plt.title(title, loc='left')
                plt.show()

        comparison_surrogate_vs_model_file_name = workingDir / f"comparison_surrogate_vs_model_{qoi_column_name}_{timestamp}.pkl"
        if comparison_surrogate_vs_model_file_name.is_file():
            with open(comparison_surrogate_vs_model_file_name, 'rb') as f:
                dict_result_comparison_model_and_surrogate = pickle.load(f)
            dict_with_results_of_interest.update(dict_result_comparison_model_and_surrogate)
            if 'numSamples' in dict_with_results_of_interest:
                dict_with_results_of_interest['comparison_surrogate_vs_model_numSamples'] = dict_with_results_of_interest.pop('numSamples')
            if 'rule' in dict_with_results_of_interest:
                dict_with_results_of_interest['comparison_surrogate_vs_model_rule'] = dict_with_results_of_interest.pop('rule')
        else:
            if compare_surrogate_and_original_model and gpce_surrogate is not None and model is not None:
                # 5**dim
                numSamples = kwargs.get('comparison_surrogate_vs_model_numSamples', 1000)
                rule = kwargs.get('comparison_surrogate_vs_model_mc_rule', "R")
                evaluateSurrogateAtStandardDist = kwargs.get('evaluateSurrogateAtStandardDist', True)
                dict_result_comparison_model_and_surrogate = compare_surrogate_and_full_model_for_single_qoi_single_timestamp(
                    model, gpce_surrogate, qoi_column_name,  timestamp,
                    jointDists=simulationNodes.joinedDists, 
                    jointStandard=simulationNodes.joinedStandardDists, 
                    numSamples=numSamples, rule=rule,
                    sampleFromStandardDist=False,
                    evaluateSurrogateAtStandardDist=evaluateSurrogateAtStandardDist,
                    read_nodes_from_file=False, 
                    rounding=False, round_dec=4,
                    compute_error=True, return_evaluations=False,
                    write_to_a_file=True, workingDir=workingDir,
                    )
                # dict_with_results_of_interest["comparison_surrogate_vs_model_numSamples"] = numSamples
                # dict_with_results_of_interest["comparison_surrogate_vs_model_mc_rule"] = rule
                dict_with_results_of_interest.update(dict_result_comparison_model_and_surrogate)
                if not 'comparison_surrogate_vs_model_numSamples' in dict_with_results_of_interest:
                    if 'numSamples' in dict_with_results_of_interest:
                        dict_with_results_of_interest['comparison_surrogate_vs_model_numSamples'] = dict_with_results_of_interest.pop('numSamples')
                    else:
                        dict_with_results_of_interest['comparison_surrogate_vs_model_numSamples'] = numSamples
                if not 'comparison_surrogate_vs_model_rule' in dict_with_results_of_interest:
                    if 'rule' in dict_with_results_of_interest:
                        dict_with_results_of_interest['comparison_surrogate_vs_model_rule'] = dict_with_results_of_interest.pop('rule')
                    else:
                        dict_with_results_of_interest['comparison_surrogate_vs_model_rule'] = rule
            else:
                print("Comparison of surrogate and full model is not performed becuase, either (an original) model is not provided or gpce_surrogate is None \
                or compare_surrogate_and_original_model variable was set to False!")

    ########################################################

    return dict_with_results_of_interest

def update_dict_with_method_variant_based_on_uqsim_args_dict(dict_with_results_of_interest, uqsim_args_dict):
    variant = None
    if uqsim_args_dict["regression"]:
        variant = "m3"
        if uqsim_args_dict["uq_method"]== "mc" or uqsim_args_dict["uq_method"]== "saltelli":
            variant = "m3-mc"
        else:
            variant = "m3-sc"
    elif uqsim_args_dict["uq_method"]== "mc":
        variant = "m1"
    elif uqsim_args_dict["uq_method"]=="saltelli":
        variant = "m2"
    elif uqsim_args_dict["uq_method"]=="sc":
        """
        [m4] gPCE+PSP with a full grid and polynomials of total-order
        [m5] gPCE+PSP with sparse grid and polynomials of total-order
        [m6] gPCE+PSP with a full grid and sparse polynomials (hyperbolic truncation)
        [m7] gPCE+PSP with sparse grid and sparse polynomials (hyperbolic truncation)
        """
        if (not uqsim_args_dict["sc_sparse_quadrature"] and not uqsim_args_dict["read_nodes_from_file"]) and uqsim_args_dict["cross_truncation"]==1.0:
            variant = "m4"
        elif (uqsim_args_dict["sc_sparse_quadrature"] or uqsim_args_dict["read_nodes_from_file"]) and uqsim_args_dict["cross_truncation"]==1.0:
            parameters_file = pathlib.Path(uqsim_args_dict["parameters_file"]).name
            if uqsim_args_dict["sc_quadrature_rule"] == "KPU" or parameters_file.startswith('KPU'):
                variant = "m5-kpu"
            elif uqsim_args_dict["sc_quadrature_rule"] == "GQU" or parameters_file.startswith('GQU'):
                variant = "m5-gqu"
            else:
                variant = "m5"
        elif (not uqsim_args_dict["sc_sparse_quadrature"] and not uqsim_args_dict["read_nodes_from_file"]) and uqsim_args_dict["cross_truncation"]<1.0:
            # ct = uqsim_args_dict["cross_truncation"]
            # variant = f"m6-{ct}"
            variant = f"m6"
        elif (uqsim_args_dict["sc_sparse_quadrature"] or uqsim_args_dict["read_nodes_from_file"]) and uqsim_args_dict["cross_truncation"]<1.0:
            parameters_file = pathlib.Path(uqsim_args_dict["parameters_file"]).name
            if uqsim_args_dict["sc_quadrature_rule"] == "KPU" or parameters_file.startswith('KPU'):
                variant = "m7-kpu"
            elif uqsim_args_dict["sc_quadrature_rule"] == "GQU" or parameters_file.startswith('GQU'):
                variant = "m7-gqu"
            else:
                variant = "m7"
    dict_with_results_of_interest["variant"] = variant


def update_dict_with_results_of_interest_based_on_uqsim_args_dict(dict_with_results_of_interest, uqsim_args_dict):
    """
    Update the dictionary with results of interest based on the given UQSim arguments dictionary.

    Args:
        dict_with_results_of_interest (dict): The dictionary to be updated with the results of interest.
        uqsim_args_dict (dict): The UQSim arguments dictionary.

    Add new entries to the dict_with_results_of_interest:
    - variant
    """
    variant = None
    dict_with_results_of_interest["regression"] = uqsim_args_dict["regression"]
    dict_with_results_of_interest["uq_method"] = uqsim_args_dict["uq_method"]
    if uqsim_args_dict["regression"]:
        variant = "m3"
        dict_with_results_of_interest["q_order"] = uqsim_args_dict["sc_q_order"]
        dict_with_results_of_interest["p_order"] = uqsim_args_dict["sc_p_order"]
        dict_with_results_of_interest["read_nodes_from_file"] = uqsim_args_dict["read_nodes_from_file"]
        dict_with_results_of_interest["sc_quadrature_rule"] = uqsim_args_dict["sc_quadrature_rule"]
        dict_with_results_of_interest["mc_numevaluations"] = uqsim_args_dict["mc_numevaluations"]
        dict_with_results_of_interest["sampling_rule"] = uqsim_args_dict["sampling_rule"]
        if uqsim_args_dict["uq_method"]== "mc" or uqsim_args_dict["uq_method"]== "saltelli":
            variant = "m3-mc"
        else:
            variant = "m3-sc"
    elif uqsim_args_dict["uq_method"]== "mc":
        variant = "m1"
        dict_with_results_of_interest["mc_numevaluations"] = uqsim_args_dict["mc_numevaluations"]
        dict_with_results_of_interest["sampling_rule"] = uqsim_args_dict["sampling_rule"]
    elif uqsim_args_dict["uq_method"]=="saltelli":
        variant = "m2"
        dict_with_results_of_interest["mc_numevaluations"] = uqsim_args_dict["mc_numevaluations"]
        dict_with_results_of_interest["sampling_rule"] = uqsim_args_dict["sampling_rule"]
    elif uqsim_args_dict["uq_method"]=="sc":
        """
        [m4] gPCE+PSP with a full grid and polynomials of total-order
        [m5] gPCE+PSP with sparse grid and polynomials of total-order
        [m6] gPCE+PSP with a full grid and sparse polynomials (hyperbolic truncation)
        [m7] gPCE+PSP with sparse grid and sparse polynomials (hyperbolic truncation)
        """
        dict_with_results_of_interest["q_order"] = uqsim_args_dict["sc_q_order"]
        dict_with_results_of_interest["p_order"] = uqsim_args_dict["sc_p_order"]
        dict_with_results_of_interest["read_nodes_from_file"] = uqsim_args_dict["read_nodes_from_file"]
        dict_with_results_of_interest["sc_quadrature_rule"] = uqsim_args_dict["sc_quadrature_rule"]

        if (not uqsim_args_dict["sc_sparse_quadrature"] and not uqsim_args_dict["read_nodes_from_file"]) and uqsim_args_dict["cross_truncation"]==1.0:
            variant = "m4"
        elif (uqsim_args_dict["sc_sparse_quadrature"] or uqsim_args_dict["read_nodes_from_file"]) and uqsim_args_dict["cross_truncation"]==1.0:
            parameters_file = pathlib.Path(uqsim_args_dict["parameters_file"]).name
            if uqsim_args_dict["sc_quadrature_rule"] == "KPU" or parameters_file.startswith('KPU'):
                variant = "m5-kpu"
            elif uqsim_args_dict["sc_quadrature_rule"] == "GQU" or parameters_file.startswith('GQU'):
                variant = "m5-gqu"
            else:
                variant = "m5"
        elif (not uqsim_args_dict["sc_sparse_quadrature"] and not uqsim_args_dict["read_nodes_from_file"]) and uqsim_args_dict["cross_truncation"]<1.0:
            dict_with_results_of_interest["cross_truncation"] = uqsim_args_dict["cross_truncation"]
            # ct = uqsim_args_dict["cross_truncation"]
            # variant = f"m6-{ct}"
            variant = f"m6"
        elif (uqsim_args_dict["sc_sparse_quadrature"] or uqsim_args_dict["read_nodes_from_file"]) and uqsim_args_dict["cross_truncation"]<1.0:
            dict_with_results_of_interest["cross_truncation"] = uqsim_args_dict["cross_truncation"]
            parameters_file = pathlib.Path(uqsim_args_dict["parameters_file"]).name
            if uqsim_args_dict["sc_quadrature_rule"] == "KPU" or parameters_file.startswith('KPU'):
                variant = "m7-kpu"
            elif uqsim_args_dict["sc_quadrature_rule"] == "GQU" or parameters_file.startswith('GQU'):
                variant = "m7-gqu"
            else:
                variant = "m7"
            
    dict_with_results_of_interest["variant"] = variant

# ===================================================================================================================
# Comparing surrogate (i.e., gPCE model) and full model
# if you want to evaluate the surrogate model for multiple timesteps, in parallel, 
# take a look at the code in scinetifc_pipelines/compare_model_and_surrogate.py
# ===================================================================================================================


def compare_surrogate_and_full_model_for_single_qoi_single_timestamp(
    model, surrogate_model, qoi_column_name,  timestamp,
    jointDists, jointStandard=None, numSamples=1000, rule="R",
    sampleFromStandardDist=False,
    evaluateSurrogateAtStandardDist=False,
    read_nodes_from_file=False, 
    rounding=False, round_dec=4,
    compute_error=True, return_evaluations=False, **kwargs):
    """
    This function is used to compare the surrogate model (e.g., gPCE model) and the full model
    for a single QoI and a single timestep

    For a general comparison one should rely on code in scinetifc_pipelines/compare_model_and_surrogate.py module!
    surrogate_model: gPCE model built for a particular QoI and a particular timestep

    Note: idea - if jointStandard is provided, then one expects that the surrogate model has to be evaluated at the set of nodes
    which come from 'standard' distribution; evaluateSurrogateAtStandardDist should be set to True in this case
    """
    write_to_a_file = kwargs.get('write_to_a_file', False)
    if write_to_a_file:
        workingDir = kwargs.get('workingDir', None)
        if workingDir is None:
            write_to_a_file = False
            print("Warning: workingDir is not provided, so the results will not be saved to a file!")

    parameters = generate_parameters_for_mc_simulation(
        jointDists, jointStandard=jointStandard, numSamples=numSamples, rule=rule,
        sampleFromStandardDist=sampleFromStandardDist, read_nodes_from_file=read_nodes_from_file, rounding=rounding, round_dec=round_dec,
        **kwargs
    )
    if evaluateSurrogateAtStandardDist and jointStandard is not None:
        nodes = utility.transformation_of_parameters(
            parameters, jointDists, jointStandard)
    else:
        nodes = parameters
    
    start = time.time()
    surrogate_model_evaluations= surrogate_model(*nodes)
    end = time.time()
    reevaluation_surrogate_model_time = end - start
    print(f"Time needed for evaluating {nodes.shape[1]} \
    gPCE model is: {reevaluation_surrogate_model_time}")

    start = time.time()
    model_runs, _ = run_uqef_dynamic_model_over_parameters_and_process_result(
        model, parameters, qoi_column_name, return_dict_over_timestamps=True)
    original_model_evaluations = model_runs[timestamp]
    end = time.time()
    reevaluation_model_time = end - start
    print(f"Time needed for evaluating {parameters.shape[1]} \
    of the original model is: {reevaluation_model_time}")

    dict_result = {}

    dict_result['numSamples'] = numSamples
    dict_result['rule'] = rule

    if return_evaluations:
        dict_result["original_model_evaluations"] = original_model_evaluations
        dict_result["surrogate_model_evaluations"] = surrogate_model_evaluations

    dict_result["reevaluation_surrogate_model_duration"] = reevaluation_surrogate_model_time
    dict_result["reevaluation_model_duration"] = reevaluation_model_time

    if compute_error:
        error_linf, error_l2, error_l2_scaled = compute_error_bewteen_surrogate_and_full_model_for_single_qoi_single_timestamp(
            original_model_evaluations, surrogate_model_evaluations)
        dict_result["error_model_linf"] = error_linf
        dict_result["error_model_l2"] = error_l2
        dict_result["error_model_l2_scaled"] = error_l2_scaled

    if write_to_a_file:
        file_name = workingDir / f"comparison_surrogate_vs_model_{qoi_column_name}_{timestamp}.pkl"
        # save the results to a file
        # Save dictionary as a binary file
        with open(file_name, 'wb') as f:
            pickle.dump(dict_result, f)

    return dict_result


def compute_error_bewteen_surrogate_and_full_model_for_single_qoi_single_timestamp(
    original_model_evaluations, surrogate_model_evaluations):
    error_linf = np.max(np.abs(original_model_evaluations - surrogate_model_evaluations))
    # error_linf_np = np.linalg.norm(original_model_evaluations - surrogate_model_evaluations, ord=np.inf)
    error_l2 = np.sqrt(np.sum((original_model_evaluations - surrogate_model_evaluations)**2))
    # error_l2_np = np.linalg.norm(original_model_evaluations - surrogate_model_evaluations, ord=2)
    numSamples = len(original_model_evaluations)
    error_l2_scaled = np.sqrt(np.sum((original_model_evaluations - surrogate_model_evaluations)**2)) / math.sqrt(numSamples)
    print(f"Linf Error = {error_linf};")
    print(f"L2 Error = {error_l2}; L2 Error scaled = {error_l2_scaled}")
    return error_linf, error_l2, error_l2_scaled

# ===================================================================================================================
# Utility functions for working with the statistics object / statistics dictionary
# ===================================================================================================================


def extend_statistics_object(statisticsObject, statistics_dictionary, df_simulation_result=None,
                             time_stamp_column=utility.TIME_COLUMN_NAME, get_measured_data=False, get_unaltered_data=False):
    statisticsObject.set_result_dict(statistics_dictionary)
    # Update statisticsObject.list_qoi_column based on columns inside df_simulation_result
    statisticsObject.list_qoi_column = list(statistics_dictionary.keys())
    # assert set(statisticsObject.list_qoi_column) == ...
    # assert set(statisticsObject.list_qoi_column) == ...

    if df_simulation_result is not None:
        statisticsObject.set_timesteps(list(df_simulation_result[time_stamp_column].unique()))
        statisticsObject.set_timesteps_min(df_simulation_result[time_stamp_column].min())
        statisticsObject.set_timesteps_max(df_simulation_result[time_stamp_column].max())
        statisticsObject.set_number_of_unique_index_runs(
            get_number_of_unique_runs(
                df_simulation_result, index_run_column_name=utility.INDEX_COLUMN_NAME)
        )
    else:
        # Read all this data from statistics_dictionary = statisticsObject.result_dict
        statisticsObject.set_timesteps()
        statisticsObject.set_timesteps_min()
        statisticsObject.set_timesteps_max()
        # statisticsObject.set_number_of_unique_index_runs()

    timestepRange = statisticsObject.get_time_range()

    statisticsObject.set_numTimesteps()

    statisticsObject._check_if_Sobol_t_computed()
    statisticsObject._check_if_Sobol_m_computed()
    statisticsObject._check_if_Sobol_m2_computed()

    if get_measured_data:
        statisticsObject.get_measured_data(
            timestepRange=timestepRange, transforme_mesured_data_as_original_model="False")

    if get_unaltered_data:
        statisticsObject.get_unaltered_run_data(timestepRange=timestepRange)


def get_number_of_unique_runs(df, index_run_column_name=utility.INDEX_COLUMN_NAME):
    return df[index_run_column_name].nunique()


def extracting_statistics_df_for_single_qoi(statisticsObject, qoi=utility.QOI_COLUMN_NAME):
    pass


def get_all_timesteps_from_saved_files(workingDir, first_part_of_the_file="statistics"):
    all_files = os.listdir(workingDir)
    list_TimeStamp = set() # []
    for filename in all_files:
        parts = filename.split('_')
        if parts[0] == first_part_of_the_file and parts[-1].endswith(".pkl"):
            single_timestep = parts[-1].split('.')[0]
            list_TimeStamp.add(single_timestep)  # pd.Timestamp(single_timestep)
    return list_TimeStamp


def get_list_of_qois_from_saved_files(workingDir, first_part_of_the_file="statistics"):
    """
    Note: use this function carefully
    """
    all_files = os.listdir(workingDir)
    list_qois = set() # []
    for filename in all_files:
        parts = filename.split('_')
        if parts[0] == first_part_of_the_file and parts[-1].endswith(".pkl"):
            single_qoi = parts[-2]
            list_qois.add(single_qoi)
    return list_qois
    

def read_all_saved_statistics_dict(workingDir, list_qoi_column, single_timestamp_single_file=False, throw_error=True, convert_to_pd_timestamp=True):
    if single_timestamp_single_file:
        list_TimeStamp = get_all_timesteps_from_saved_files(workingDir, first_part_of_the_file = "statistics")
        if not list_TimeStamp:
            single_timestamp_single_file = False
    statistics_dictionary = defaultdict(dict)
    for single_qoi in list_qoi_column:
        if single_timestamp_single_file:
            statistics_dictionary[single_qoi] = dict()
            for single_timestep in list_TimeStamp:
                statistics_dictionary_file_temp = workingDir / f"statistics_dictionary_{single_qoi}_{single_timestep}.pkl"
                if not statistics_dictionary_file_temp.is_file():
                    if throw_error:
                        raise FileNotFoundError(f"The statistics file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist")
                    else:
                        if convert_to_pd_timestamp:
                            statistics_dictionary[single_qoi][pd.Timestamp(single_timestep)] = None
                        else:
                            statistics_dictionary[single_qoi][single_timestep] = None
                        continue
                # assert statistics_dictionary_file_temp.is_file(), \
                #     f"The statistics file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist"
                with open(statistics_dictionary_file_temp, 'rb') as f:
                    statistics_dictionary_temp = pickle.load(f)
                if convert_to_pd_timestamp:
                    statistics_dictionary[single_qoi][pd.Timestamp(single_timestep)] = statistics_dictionary_temp
                else:
                    statistics_dictionary[single_qoi][single_timestep] = statistics_dictionary_temp
                # statistics_dictionary[single_qoi][pd.Timestamp(single_timestep)] = statistics_dictionary_temp
        else:
            statistics_dictionary_file_temp = workingDir / f"statistics_dictionary_qoi_{single_qoi}.pkl"
            if not statistics_dictionary_file_temp.is_file():
                if throw_error:
                    raise FileNotFoundError(f"The statistics file for qoi-{single_qoi} does not exist")
                else:
                    statistics_dictionary[single_qoi] = None
                    continue
            # assert statistics_dictionary_file_temp.is_file(), \
            #                     f"The statistics file for qoi-{single_qoi} does not exist"
            with open(statistics_dictionary_file_temp, 'rb') as f:
                statistics_dictionary_temp = pickle.load(f)
            statistics_dictionary[single_qoi] = statistics_dictionary_temp
    if utility.is_nested_dict_empty_or_none(statistics_dictionary):
        if throw_error:
            raise FileNotFoundError(f"No statistics files found in the working directory")
        else:
            return None
    return statistics_dictionary

# ===================================================================================================================
# Saving statistics related data in the data frame
# ===================================================================================================================


def create_df_from_statistics_data(
    stat_dict, list_qoi_columns, list_of_uncertain_variables, list_measured_qoi_columns,
    set_lower_predictions_to_zero=False, list_measured_fetched=[], df_measured=None,
    time_column_name=utility.TIME_COLUMN_NAME
):
    raise NotImplementedError("This method is not implemented yet!")


def create_df_from_statistics_data_single_qoi(
    stat_dict, qoi_column, list_of_uncertain_variables, measured_qoi_column, 
    set_lower_predictions_to_zero=False, measured_fetched=False, df_measured=None,
    time_column_name=utility.TIME_COLUMN_NAME):
    """
    Creates a pandas DataFrame from the statistics data for a single quantity of interest (qoi).

    Args:
        qoi_column (str): The column name of the quantity of interest.
        set_lower_predictions_to_zero (bool, optional): Flag indicating whether to set lower predictions to zero. Defaults to False.

    Returns:
        pandas.DataFrame: The DataFrame containing the statistics data for the specified qoi.

    Raises:

    Note:
        This method retrieves the statistics data from the result_dict attribute and constructs a DataFrame
        with columns representing different statistical measures such as mean, standard deviation, percentiles, etc.
        The DataFrame also includes the time column and the qoi column.

        If measured data is available (i.e., df_measured is not Noe) the method addes measured data to the final 
        data frame as well, and if compute_measured_normalized_data is True, the method computes
        the normalized measured data and adds it as a column in the DataFrame.

        If the unaltered_computed flag is True, the plan is to add it to the final df
    """
    if not stat_dict:
        return None

    if qoi_column not in stat_dict:
        stat_dict = stat_dict[qoi_column]

    keyIter = list(stat_dict.keys())  # timesteps (?)

    list_of_columns = [keyIter, ]  # self.timesteps (?)
    list_of_columns_names = [utility.TIME_COLUMN_NAME, ]

    if utility.MEAN_ENTRY in stat_dict[keyIter[0]]:
        mean_time_series = [stat_dict[key][utility.MEAN_ENTRY] for key in keyIter]
        list_of_columns.append(mean_time_series)
        list_of_columns_names.append(utility.MEAN_ENTRY)
    if utility.PCE_COEFF_ENTRY in stat_dict[keyIter[0]]:
        list_of_columns.append([stat_dict[key][utility.PCE_COEFF_ENTRY] for key in keyIter])
        list_of_columns_names.append(utility.PCE_COEFF_ENTRY)
    if utility.PCE_ENTRY in stat_dict[keyIter[0]]:
        list_of_columns.append([stat_dict[key][utility.PCE_ENTRY] for key in keyIter])
        list_of_columns_names.append(utility.PCE_ENTRY)
    if utility.VAR_ENTRY in stat_dict[keyIter[0]]:
        std_time_series = [stat_dict[key][utility.VAR_ENTRY] for key in keyIter]
        list_of_columns.append(std_time_series)
        list_of_columns_names.append(utility.VAR_ENTRY)
    if utility.STD_DEV_ENTRY in stat_dict[keyIter[0]]:
        std_time_series = [stat_dict[key][utility.STD_DEV_ENTRY] for key in keyIter]
        list_of_columns.append(std_time_series)
        list_of_columns_names.append(utility.STD_DEV_ENTRY)
    if utility.P10_ENTRY in stat_dict[keyIter[0]]:
        p10_time_series = [stat_dict[key][utility.P10_ENTRY] for key in keyIter]
        list_of_columns.append(p10_time_series)
        list_of_columns_names.append(utility.P10_ENTRY)
    if utility.P90_ENTRY in stat_dict[keyIter[0]]:
        p90_time_series = [stat_dict[key][utility.P90_ENTRY] for key in keyIter]
        list_of_columns.append(p90_time_series)
        list_of_columns_names.append(utility.P90_ENTRY)
    if utility.SKEW_ENTRY in stat_dict[keyIter[0]]:
        list_of_columns.append([stat_dict[key][utility.SKEW_ENTRY] for key in keyIter])
        list_of_columns_names.append(utility.SKEW_ENTRY)
    if utility.KURT_ENTRY in stat_dict[keyIter[0]]:
        list_of_columns.append([stat_dict[key][utility.KURT_ENTRY] for key in keyIter])
        list_of_columns_names.append(utility.KURT_ENTRY)
    if utility.QOI_DIST_ENTR in stat_dict[keyIter[0]]:
        list_of_columns.append([stat_dict[key][utility.QOI_DIST_ENTR] for key in keyIter])
        list_of_columns_names.append(utility.QOI_DIST_ENTR)

    # self._check_if_Sobol_t_computed(keyIter[0], qoi_column=qoi_column)
    # self._check_if_Sobol_m_computed(keyIter[0], qoi_column=qoi_column)
    is_Sobol_t_computed = utility.SOBOL_TOTAL_ORDER_ENTRY in stat_dict[keyIter[0]]
    is_Sobol_m_computed = utility.SOBOL_FIRST_ORDER_ENTRY in stat_dict[keyIter[0]]
    is_Sobol_m2_computed = utility.SOBOL_SECOND_ORDER_ENTRY in stat_dict[keyIter[0]]

    if is_Sobol_m_computed:
        for i in range(len(list_of_uncertain_variables)):
            sobol_m_time_series = [stat_dict[key][utility.SOBOL_FIRST_ORDER_ENTRY][i] for key in keyIter]
            list_of_columns.append(sobol_m_time_series)
            temp = "Sobol_m_" + list_of_uncertain_variables[i]
            list_of_columns_names.append(temp)
    if is_Sobol_m2_computed:
        for i in range(len(list_of_uncertain_variables)):
            sobol_m2_time_series = [stat_dict[key][utility.SOBOL_SECOND_ORDER_ENTRY][i] for key in keyIter]
            list_of_columns.append(sobol_m2_time_series)
            temp = "Sobol_m2_" + list_of_uncertain_variables[i]
            list_of_columns_names.append(temp)
    if is_Sobol_t_computed:
        for i in range(len(list_of_uncertain_variables)):
            sobol_t_time_series = [stat_dict[key][utility.SOBOL_TOTAL_ORDER_ENTRY][i] for key in keyIter]
            list_of_columns.append(sobol_t_time_series)
            temp = "Sobol_t_" + list_of_uncertain_variables[i]
            list_of_columns_names.append(temp)

    if f'generalized_sobol_total_index_{list_of_uncertain_variables[0]}' in stat_dict[keyIter[-1]]:
        for i in range(len(list_of_uncertain_variables)):
            name = f"generalized_sobol_total_index_{list_of_uncertain_variables[i]}"
            generalized_sobol_total_index_values_temp = []
            at_least_one_entry_found = False
            for key in keyIter:
                if name in stat_dict[key]:
                    at_least_one_entry_found = True
                    temp = stat_dict[key][name]
                    generalized_sobol_total_index_values_temp.append(temp)
            if at_least_one_entry_found:
                list_of_columns_names.append(name)
                if len(generalized_sobol_total_index_values_temp)==1:
                    generalized_sobol_total_index_values_temp = generalized_sobol_total_index_values_temp[0]*len(keyIter)
                list_of_columns.append(generalized_sobol_total_index_values_temp)

    if not list_of_columns:
        return None

    df_statistics_single_qoi = pd.DataFrame(list(zip(*list_of_columns)), columns=list_of_columns_names)
    df_statistics_single_qoi[utility.QOI_ENTRY] = qoi_column

    if utility.MEAN_ENTRY in df_statistics_single_qoi.columns:
        if utility.STD_DEV_ENTRY in df_statistics_single_qoi.columns:
            df_statistics_single_qoi["E_minus_std"] = df_statistics_single_qoi[utility.MEAN_ENTRY] - df_statistics_single_qoi[utility.STD_DEV_ENTRY]
            df_statistics_single_qoi["E_plus_std"] = df_statistics_single_qoi[utility.MEAN_ENTRY] + df_statistics_single_qoi[utility.STD_DEV_ENTRY]
            df_statistics_single_qoi["E_minus_2std"] = df_statistics_single_qoi[utility.MEAN_ENTRY] - 2*df_statistics_single_qoi[utility.STD_DEV_ENTRY]
            df_statistics_single_qoi["E_plus_2std"] = df_statistics_single_qoi[utility.MEAN_ENTRY] + 2*df_statistics_single_qoi[utility.STD_DEV_ENTRY]
        elif utility.VAR_ENTRY in df_statistics_single_qoi.columns:
            df_statistics_single_qoi["E_minus_std"] = df_statistics_single_qoi[utility.MEAN_ENTRY] - np.sqrt(df_statistics_single_qoi[utility.VAR_ENTRY])
            df_statistics_single_qoi["E_plus_std"] = df_statistics_single_qoi[utility.MEAN_ENTRY] + np.sqrt(df_statistics_single_qoi[utility.VAR_ENTRY])
            df_statistics_single_qoi["E_minus_2std"] = df_statistics_single_qoi[utility.MEAN_ENTRY] - 2*np.sqrt(df_statistics_single_qoi[utility.VAR_ENTRY])
            df_statistics_single_qoi["E_plus_2std"] = df_statistics_single_qoi[utility.MEAN_ENTRY] + 2*np.sqrt(df_statistics_single_qoi[utility.VAR_ENTRY])

    if set_lower_predictions_to_zero:
        if 'E_minus_std' in df_statistics_single_qoi:
            df_statistics_single_qoi.loc[df_statistics_single_qoi["E_minus_std"] < 0, "E_minus_std"] = 0
        if 'E_minus_2std' in df_statistics_single_qoi:
            df_statistics_single_qoi.loc[df_statistics_single_qoi["E_minus_2std"] < 0, "E_minus_2std"] = 0
        if 'P10' in df_statistics_single_qoi:
            df_statistics_single_qoi['P10'] = df_statistics_single_qoi['P10'].apply(lambda x: max(0, x))

    if measured_fetched and df_measured is not None:
        if qoi_column in list(df_measured["qoi"].unique()):
            # print(f"{qoi_column}")
            df_measured_subset = df_measured.loc[df_measured["qoi"] == qoi_column][[
                time_column_name, "measured"]]
            # df_measured_subset.drop("qoi", inplace=True)
            df_statistics_single_qoi = pd.merge(df_statistics_single_qoi, df_measured_subset,
                                                on=[time_column_name, ], how='left')
        elif measured_qoi_column in list(df_measured["qoi"].unique()):
            df_measured_subset = df_measured.loc[
                df_measured["qoi"] == measured_qoi_column][[
                time_column_name, "measured"]]
            # df_measured_subset.drop("qoi", inplace=True)
            df_statistics_single_qoi = pd.merge(df_statistics_single_qoi, df_measured_subset,
                                                on=[time_column_name, ], how='left')
        else:
            df_statistics_single_qoi["measured"] = np.nan
    else:
        df_statistics_single_qoi["measured"] = np.nan

    return df_statistics_single_qoi


# ===================================================================================================================
# Functions for saving/reading the GPCE surrogate model
# ===================================================================================================================


def save_gpce_surrogate_model(workingDir, gpce, qoi, timestamp):
    # timestamp = pd.Timestamp(timestamp).strftime('%Y-%m-%d %X')
    fileName = f"gpce_surrogate_{qoi}_{timestamp}.pkl"
    fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
    with open(fullFileName, 'wb') as handle:
        pickle.dump(gpce, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def save_all_gpce_surrogate_model(workingDir, gpce_surrogate_dictionary, list_qoi_column=None, timestamps=None):
    """
    Save all GPCE surrogate models for the given list of QOI columns and timestamps.

    Args:
        workingDir (str): The directory where the models will be saved.
        gpce_surrogate_dictionary (dict): A dictionary containing the GPCE surrogate models.
        list_qoi_column (list, optional): The list of QOI columns to save. If None, all columns will be saved.
        timestamps (list, optional): The list of timestamps to save. If None, all timestamps will be saved.

    Returns:
        None
    """
    if list_qoi_column is None:
        list_qoi_column = list(gpce_surrogate_dictionary.keys())
    if timestamps is None:
        timestamps = gpce_surrogate_dictionary[list_qoi_column[0]].keys()
    if not isinstance(list_qoi_column, list):
        list_qoi_column = [list_qoi_column, ]
    if not isinstance(timestamps, list):
        timestamps = [timestamps, ]
    for single_qoi in list_qoi_column:
        for single_timestep in timestamps:
            save_gpce_surrogate_model(workingDir, gpce_surrogate_dictionary[single_qoi][single_timestep], single_qoi, single_timestep)


def save_gpce_coeffs(workingDir, coeff, qoi, timestamp):
    fileName = f"gpce_coeffs_{qoi}_{timestamp}.npy"
    fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
    np.save(fullFileName, coeff)


def save_all_gpce_coeffs(workingDir, gpce_coeff_dictionary, list_qoi_column=None, timestamps=None):
    """
    Save all GPCE coefficients for the given list of QOI columns and timestamps.

    Args:
        workingDir (str): The working directory where the coefficients will be saved.
        gpce_coeff_dictionary (dict): A dictionary containing the GPCE coefficients for different QOI columns and timestamps.
        list_qoi_column (list, optional): A list of QOI columns to save the coefficients for. If not provided, all QOI columns in the dictionary will be saved.
        timestamps (list, optional): A list of timestamps to save the coefficients for. If not provided, all timestamps in the dictionary for the first QOI column will be saved.

    Returns:
        None
    """
    if list_qoi_column is None:
        list_qoi_column = list(gpce_coeff_dictionary.keys())
    if timestamps is None:
        timestamps = gpce_coeff_dictionary[list_qoi_column[0]].keys()
    if not isinstance(list_qoi_column, list):
        list_qoi_column = [list_qoi_column, ]
    if not isinstance(timestamps, list):
        timestamps = [timestamps, ]
    for single_qoi in list_qoi_column:
        for single_timestep in timestamps:
            save_gpce_coeffs(workingDir, gpce_coeff_dictionary[single_qoi][single_timestep], single_qoi, single_timestep)


def read_all_saved_gpce_surrogate_models(workingDir, list_qoi_column, single_timestamp_single_file=False, throw_error=True, 
convert_to_pd_timestamp=True):
    """
    Reads all saved GPCE surrogate models from the specified working directory.

    Args:
        workingDir (str): The path to the working directory where the GPCE surrogate models are saved.
        list_qoi_column (list): A list of QOI (Quantity of Interest) columns.
        single_timestamp_single_file (bool, optional): Whether each timestamp has a separate file. Defaults to False.
        throw_error (bool, optional): Whether to throw an error if a surrogate file is not found. Defaults to True.
        convert_to_pd_timestamp (bool, optional): Whether to convert the timestamps to pandas Timestamp objects. Defaults to True.

    Returns:
        dict: A dictionary containing the GPCE surrogate models for each QOI and timestamp.
        returns None value if `throw_error` is False but something is missing.

    Raises:
        FileNotFoundError: If a GPCE surrogate file is not found and `throw_error` is True.
        FileNotFoundError: If no GPCE surrogate files are found in the working directory and `throw_error` is True.
    """
    # TODO it seems as single_timestamp_single_file is not relevant here anymore!
    list_TimeStamp = get_all_timesteps_from_saved_files(workingDir, first_part_of_the_file="gpce")
    # if list_qoi_column is None:
    #     list_qoi_column = get_list_of_qois_from_saved_files(workingDir, first_part_of_the_file="gpce")
    if not list_TimeStamp:
        if throw_error:
            raise FileNotFoundError(f"No gpce surrogate files found in the working directory")
        else:
            return None
    gpce_surrogate_dictionary = dict()  # defaultdict(dict)
    for single_qoi in list_qoi_column:
        gpce_surrogate_dictionary[single_qoi] = dict()
        for single_timestep in list_TimeStamp:
            gpce_surrogate_file_temp = workingDir / f"gpce_surrogate_{single_qoi}_{single_timestep}.pkl"
            if not gpce_surrogate_file_temp.is_file():
                if throw_error:
                    raise FileNotFoundError(f"The gpce surrogate file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist")
                else:
                    if convert_to_pd_timestamp:
                        gpce_surrogate_dictionary[single_qoi][pd.Timestamp(single_timestep)] = None
                    else:
                        gpce_surrogate_dictionary[single_qoi][single_timestep] = None
                    continue
            # assert gpce_surrogate_file_temp.is_file(), \
            # f"The gpce surrogate file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist"
            with open(gpce_surrogate_file_temp, 'rb') as f:
                gpce_surrogate_temp = pickle.load(f)
            if convert_to_pd_timestamp:
                gpce_surrogate_dictionary[single_qoi][pd.Timestamp(single_timestep)] = gpce_surrogate_temp
            else:
                gpce_surrogate_dictionary[single_qoi][single_timestep] = gpce_surrogate_temp
    if utility.is_nested_dict_empty_or_none(gpce_surrogate_dictionary):
        if throw_error:
            raise FileNotFoundError(f"No gpce surrogate files found in the working directory")
        else:
            return None
    return gpce_surrogate_dictionary


def read_single_gpce_surrogate_models(workingDir, single_qoi, single_timestep, throw_error=True):
    gpce_surrogate_file_temp = workingDir / f"gpce_surrogate_{single_qoi}_{single_timestep}.pkl"
    if gpce_surrogate_file_temp.is_file():
        with open(gpce_surrogate_file_temp, 'rb') as f:
                gpce_surrogate_temp = pickle.load(f)
        return gpce_surrogate_temp
    else:
        if throw_error:
            raise FileNotFoundError(f"The gpce surrogate file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist")
        else:
            return None


def read_all_saved_gpce_coeffs(workingDir, list_qoi_column, single_timestamp_single_file=False, throw_error=True,
                              convert_to_pd_timestamp=True):
    """
    Read all saved GPCE coefficients from the specified working directory.

    Args:
        workingDir (str): The path to the working directory.
        list_qoi_column (list): A list of QOI (Quantity of Interest) columns.
        single_timestamp_single_file (bool, optional): Whether each timestamp has a separate file. Defaults to False.
        throw_error (bool, optional): Whether to throw an error if a coefficients file is not found. Defaults to True.
        convert_to_pd_timestamp (bool, optional): Whether to convert the timestamps to pandas Timestamp objects. 
                                                  Defaults to True.

    Returns:
        dict: A dictionary containing the GPCE coefficients for each QOI and timestamp.

    Raises:
        FileNotFoundError: If a coefficients file is not found and `throw_error` is set to True.
        FileNotFoundError: If no GPCE coefficients files are found in the working directory and `throw_error` is set to True.
    """
    # TODO it seems as single_timestamp_single_file is not relevant here!
    list_TimeStamp = get_all_timesteps_from_saved_files(workingDir, first_part_of_the_file="gpce")
    if not list_TimeStamp:
        if throw_error:
            raise FileNotFoundError(f"No gpce coeff files found in the working directory")
        else:
            return None
    gpce_coeff_dictionary = dict()  # defaultdict(dict)
    # if list_qoi_column is None:
    #     list_qoi_column = get_list_of_qois_from_saved_files(workingDir, first_part_of_the_file="gpce")
    for single_qoi in list_qoi_column:
        gpce_coeff_dictionary[single_qoi] = dict()
        for single_timestep in list_TimeStamp:
            gpce_coeffs_file_temp = workingDir / f"gpce_coeffs_{single_qoi}_{single_timestep}.npy"
            if not gpce_coeffs_file_temp.is_file():
                if throw_error:
                    raise FileNotFoundError(
                        f"The gpce coefficients file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist"
                    )
                else:
                    if convert_to_pd_timestamp:
                        gpce_coeff_dictionary[single_qoi][pd.Timestamp(single_timestep)] = None
                    else:
                        gpce_coeff_dictionary[single_qoi][single_timestep] = None
                    continue
            # assert gpce_coeffs_file_temp.is_file(), \
            # f"The gpce coefficients file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist"
            if convert_to_pd_timestamp:
                gpce_coeff_dictionary[single_qoi][pd.Timestamp(single_timestep)] = np.load(gpce_coeffs_file_temp, allow_pickle=True)
            else:
                gpce_coeff_dictionary[single_qoi][single_timestep] = np.load(gpce_coeffs_file_temp, allow_pickle=True)
    if utility.is_nested_dict_empty_or_none(gpce_coeff_dictionary):
        if throw_error:
            raise FileNotFoundError("No gpce coefficinets files found in the working directory")
        else:
            return None
    return gpce_coeff_dictionary


def read_single_gpce_coeffs(workingDir, single_qoi, single_timestep, throw_error=True):
    gpce_coeffs_file_temp = workingDir / f"gpce_coeffs_{single_qoi}_{single_timestep}.npy"
    if gpce_coeffs_file_temp.is_file():
        return np.load(gpce_coeffs_file_temp, allow_pickle=True)
    else:
        if throw_error:
            raise FileNotFoundError(f"The gpce coefficients file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist")
        else:
            return None


def read_gpce_surrogate_models_and_coefficients(workingDir, list_qoi_column, convert_to_pd_timestamp=False, printing=False):
    """
    This function reads the gPCE surrogate models (polynomials) and gPCE coefficents from saved files.
    Args:
    - workingDir: pathlib.Path object, path to the working directory
    - list_qoi_column: list, list of the quantity of interest (QoI) columns
    - convert_to_pd_timestamp: bool, if True, convert the timestamp to pandas timestamp when reading/creating dictionaries storitng gPCE model-related data
    - printing: bool, if True, print the information about the gPCE surrogate models
    Returns:
    - gpce_surrogate_dictionary: dict, dictionary with the gPCE surrogate models
    - gpce_coeff_dictionary: dict, dictionary with the gPCE coefficients
    """
    gpce_surrogate_dictionary = read_all_saved_gpce_surrogate_models(
        workingDir=workingDir, list_qoi_column=list_qoi_column, throw_error=False, convert_to_pd_timestamp=convert_to_pd_timestamp)
    gpce_coeff_dictionary = read_all_saved_gpce_coeffs(
        workingDir=workingDir, list_qoi_column=list_qoi_column, throw_error=False, convert_to_pd_timestamp=convert_to_pd_timestamp)
    if gpce_surrogate_dictionary is not None and printing:
        print(f"INFO: gpce_surrogate_dictionary - {gpce_surrogate_dictionary}")
    if gpce_coeff_dictionary is not None and printing:
        print(f"INFO: gpce_coeff_dictionary - {gpce_coeff_dictionary}")
    return gpce_surrogate_dictionary, gpce_coeff_dictionary


def get_entry_surrogate_single_qoi_from_statistics_dict(statistics_dictionary, entry_label=utility.PCE_ENTRY, qoi_column_name=None):
    """
    Retrieves a specified entry for a single quantity of interest (QoI) from a statistics dictionary.

    Args:
        entry_label (str): The label of the entry/key to retrieve from the statistics dictionary. Default is utility.PCE_ENTRY.
        statistics_dictionary (dict): The statistics dictionary.
        qoi_column_name (str, optional): The name of the column in the statistics dictionary containing the QoI data.

    Returns:
        dict or None: The surrogate dictionary for the specified entry label, or None if the entry is not found.

    """
    if statistics_dictionary is None:
        return None

    if entry_label in statistics_dictionary:
        return statistics_dictionary[entry_label]
    elif entry_label in statistics_dictionary[list(statistics_dictionary.keys())[0]]:
        gpce_surrogate_dictionary = dict()
        for single_timestamp in statistics_dictionary.keys():
            gpce_surrogate_dictionary[single_timestamp] = statistics_dictionary[single_timestamp][entry_label]
        return gpce_surrogate_dictionary
    elif qoi_column_name is not None and qoi_column_name in statistics_dictionary:
        statistics_dictionary = statistics_dictionary[qoi_column_name]
        if entry_label in statistics_dictionary:
            return statistics_dictionary[entry_label]
        elif entry_label in statistics_dictionary[list(statistics_dictionary.keys())[0]]:
            gpce_surrogate_dictionary = dict()
            for single_timestamp in statistics_dictionary.keys():
                gpce_surrogate_dictionary[single_timestamp] = statistics_dictionary[single_timestamp][entry_label]
            return gpce_surrogate_dictionary
    else:
        return None


def get_entry_surrogate_single_qoi_single_timestamp_from_statistics_dict(
    statistics_dictionary, timestamp, qoi_column_name, entry_label=utility.PCE_ENTRY):
    """
    Retrieves a specified entry for a single quantity of interest (QoI) from a statistics dictionary.

    Args:
        statistics_dictionary (dict): The statistics dictionary.
        timestamp (int or pd.Timestamp):
        qoi_column_name (str): The name of the column in the statistics dictionary containing the QoI data.
        entry_label (str): The label of the entry/key to retrieve from the statistics dictionary. Default is utility.PCE_ENTRY.

    Returns:
        dict or None: The surrogate dictionary for the specified entry label, or None if the entry is not found.

    """
    if statistics_dictionary is None:
        return None

    if entry_label in statistics_dictionary:
        return statistics_dictionary[entry_label]
    elif timestamp in statistics_dictionary.keys() and entry_label in statistics_dictionary[timestamp]:
        return statistics_dictionary[timestamp][entry_label]
    elif qoi_column_name is not None and qoi_column_name in statistics_dictionary:
        statistics_dictionary = statistics_dictionary[qoi_column_name]
        if entry_label in statistics_dictionary:
            return statistics_dictionary[entry_label]
        elif timestamp in statistics_dictionary.keys() and entry_label in statistics_dictionary[timestamp]:
            return statistics_dictionary[timestamp][entry_label]
    else:
        return None

def fetch_gpce_surrogate_single_qoi_single_timestamp(qoi_column_name, timestamp, workingDir=None, 
statistics_dictionary=None, throw_error=False, **kwargs):
    """
        Fetches the gPCE surrogate model for a single QoI singe timestamp
        Parameters:
        - qoi_column_name (str): The name of the QoI column.
        - timestamp (int or pd.Timestamp):
        - workingDir (str, optional): The working directory. Default is None.
        - statistics_dictionary (dict, optional): The statistics dictionary. Default is None.
        - throw_error (bool, optional): Whether to throw an error if the surrogate model cannot be fetched. Default is False.
        - **kwargs: Additional keyword arguments.

    """
    convert_to_pd_timestamp = kwargs.get('convert_to_pd_timestamp', False)
    single_timestamp_single_file = kwargs.get('single_timestamp_single_file', False)

    if workingDir is None and statistics_dictionary is None:
        raise ValueError("Both workingDir and statistics_dictionary are None!")
    
    # first try to read from the statistics_dictionary
    if statistics_dictionary is not None:
        gpce_surrogate_dictionary = get_entry_surrogate_single_qoi_single_timestamp_from_statistics_dict(
            statistics_dictionary=statistics_dictionary, timestamp=timestamp, qoi_column_name=qoi_column_name, 
            entry_label=utility.PCE_ENTRY)
        if gpce_surrogate_dictionary is not None:
            return gpce_surrogate_dictionary

    if workingDir is not None:
        # Try to read from the saved files
        gpce_surrogate_dictionary = read_single_gpce_surrogate_models(
            workingDir=workingDir, single_qoi=qoi_column_name, single_timestep=timestamp, throw_error=False)
        if gpce_surrogate_dictionary is not None:
            return gpce_surrogate_dictionary
        else:
            # Try to read saved statistics dictionary and read it from there
            statistics_dictionary = read_all_saved_statistics_dict(\
                workingDir=workingDir, 
                list_qoi_column=[qoi_column_name,], 
                single_timestamp_single_file=single_timestamp_single_file, 
                throw_error=False, 
                convert_to_pd_timestamp=convert_to_pd_timestamp)
            # Now try the once again to read from the statistics_dictionary
            statistics_dictionary = statistics_dictionary[qoi_column_name][timestamp]
            return get_entry_surrogate_single_qoi_single_timestamp_from_statistics_dict(
                statistics_dictionary=statistics_dictionary, timestamp=timestamp, qoi_column_name=qoi_column_name, 
                entry_label=utility.PCE_ENTRY)

    return None

def fetch_gpce_surrogate_single_qoi(qoi_column_name, workingDir=None, 
statistics_dictionary=None, throw_error=False, **kwargs):
    """
    Fetches the gPCE surrogate model for a single QoI.

    Parameters:
    - qoi_column_name (str): The name of the QoI column.
    - workingDir (str, optional): The working directory. Default is None.
    - statistics_dictionary (dict, optional): The statistics dictionary. Default is None.
    - throw_error (bool, optional): Whether to throw an error if the surrogate model cannot be fetched. Default is False.
    - **kwargs: Additional keyword arguments.

    Keyword Arguments:
    - convert_to_pd_timestamp (bool, optional): Whether to convert the timestamps to pandas Timestamp objects. Default is False.
    - single_timestamp_single_file (bool, optional): Whether the statistics dictionary contains a single timestamp in a single file. Default is False.

    Returns:
    - gpce_surrogate_dictionary (dict): The gPCE surrogate model dictionary. Might be nested dictionary

    Raises:
    - ValueError: If both workingDir and statistics_dictionary are None.

    Notes:
    - This function tries to fetch the gPCE surrogate model for a single QoI in multiple ways. 
    It first tries to read from the statistics_dictionary, and if not found, it tries to read from the saved files. 
    If the surrogate model is still not found, it tries to read the saved statistics dictionary and fetches the surrogate model from there.
    """
    convert_to_pd_timestamp = kwargs.get('convert_to_pd_timestamp', False)
    single_timestamp_single_file = kwargs.get('single_timestamp_single_file', False)

    if workingDir is None and statistics_dictionary is None:
        raise ValueError("Both workingDir and statistics_dictionary are None!")
    
    # first try to read from the statistics_dictionary
    if statistics_dictionary is not None:
        gpce_surrogate_dictionary = get_entry_surrogate_single_qoi_from_statistics_dict(
            statistics_dictionary, utility.PCE_ENTRY, qoi_column_name)
        if gpce_surrogate_dictionary is not None:
            return gpce_surrogate_dictionary

    if workingDir is not None:
        # Try to read from the saved files
        gpce_surrogate_dictionary = read_all_saved_gpce_surrogate_models(
            workingDir=workingDir, 
            list_qoi_column=[qoi_column_name,], 
            throw_error=False, 
            convert_to_pd_timestamp=convert_to_pd_timestamp)
        if gpce_surrogate_dictionary is not None:
            return gpce_surrogate_dictionary
        else:
            # Try to read saved statistics dictionary and read it from there
            statistics_dictionary = read_all_saved_statistics_dict(\
                workingDir=workingDir, 
                list_qoi_column=[qoi_column_name,], 
                single_timestamp_single_file=single_timestamp_single_file, 
                throw_error=False, 
                convert_to_pd_timestamp=convert_to_pd_timestamp)
            # Now try the once again to read from the statistics_dictionary
            return get_entry_surrogate_single_qoi_from_statistics_dict(statistics_dictionary, utility.PCE_ENTRY, qoi_column_name)

    return None


def fetch_gpce_coeff_single_qoi_single_timestamp(qoi_column_name, timestamp, workingDir=None, 
statistics_dictionary=None, throw_error=False, **kwargs):
    """
            Fetches the coefficints of the gPCE surrogate model for a single QoI singe timestamp
        Parameters:
        - qoi_column_name (str): The name of the QoI column.
        - timestamp (int or pd.Timestamp):
        - workingDir (str, optional): The working directory. Default is None.
        - statistics_dictionary (dict, optional): The statistics dictionary. Default is None.
        - throw_error (bool, optional): Whether to throw an error if the surrogate model cannot be fetched. Default is False.
        - **kwargs: Additional keyword arguments.

    """
    convert_to_pd_timestamp = kwargs.get('convert_to_pd_timestamp', False)
    single_timestamp_single_file = kwargs.get('single_timestamp_single_file', False)

    if workingDir is None and statistics_dictionary is None:
        raise ValueError("Both workingDir and statistics_dictionary are None!")
    
    # first try to read from the statistics_dictionary
    if statistics_dictionary is not None:
        gpce_coeff_dictionary = get_entry_surrogate_single_qoi_single_timestamp_from_statistics_dict(
            statistics_dictionary=statistics_dictionary, timestamp=timestamp, qoi_column_name=qoi_column_name, 
            entry_label=utility.PCE_COEFF_ENTRY)
        if gpce_coeff_dictionary is not None:
            return gpce_coeff_dictionary

    if workingDir is not None:
        # Try to read from the saved files
        gpce_coeff_dictionary = read_single_gpce_coeffs(
            workingDir=workingDir, single_qoi=qoi_column_name, single_timestep=timestamp, throw_error=False)
        if gpce_coeff_dictionary is not None:
            return gpce_coeff_dictionary
        else:
            # Try to read saved statistics dictionary and read it from there
            statistics_dictionary = read_all_saved_statistics_dict(\
                workingDir=workingDir, 
                list_qoi_column=[qoi_column_name,], 
                single_timestamp_single_file=single_timestamp_single_file, 
                throw_error=False, 
                convert_to_pd_timestamp=convert_to_pd_timestamp)
            # Now try the once again to read from the statistics_dictionary
            statistics_dictionary = statistics_dictionary[qoi_column_name][timestamp]
            return get_entry_surrogate_single_qoi_single_timestamp_from_statistics_dict(
                statistics_dictionary=statistics_dictionary, timestamp=timestamp, qoi_column_name=qoi_column_name, 
                entry_label=utility.PCE_COEFF_ENTRY)

    return None


def fetch_gpce_coeff_single_qoi(qoi_column_name, workingDir=None, 
statistics_dictionary=None, throw_error=False, **kwargs):
    """
    Fetches the coefficints of the gPCE surrogate model for a single QoI.

    Parameters:
    - qoi_column_name (str): The name of the QoI column.
    - workingDir (str, optional): The working directory. Default is None.
    - statistics_dictionary (dict, optional): The statistics dictionary. Default is None.
    - throw_error (bool, optional): Whether to throw an error if the gPCE coefficients cannot be fetched. Default is False.
    - **kwargs: Additional keyword arguments.

    Keyword Arguments:
    - convert_to_pd_timestamp (bool, optional): Whether to convert the timestamps to pandas Timestamp objects. Default is False.
    - single_timestamp_single_file (bool, optional): Whether the statistics dictionary contains a single timestamp in a single file. Default is False.

    Returns:
    - gpce_coeff_dictionary (dict): The gPCE coefficients dictionary for the specified QoI.

    Raises:
    - ValueError: If both workingDir and statistics_dictionary are None.

    Notes:
    - This function tries to fetch the gPCE surrogate model for a single QoI in multiple ways. 
    It first tries to read from the statistics_dictionary. 
    If the gPCE coefficients are not found in the statistics_dictionary, it then tries to read from the saved files. 
    If the gPCE coefficients are still not found, it tries to read the saved statistics dictionary and fetch the gPCE coefficients from there.

    """
    convert_to_pd_timestamp = kwargs.get('convert_to_pd_timestamp', False)
    single_timestamp_single_file = kwargs.get('single_timestamp_single_file', False)

    if workingDir is None and statistics_dictionary is None:
        raise ValueError("Both workingDir and statistics_dictionary are None!")
    
    # first try to read from the statistics_dictionary
    if statistics_dictionary is not None:
        gpce_coeff_dictionary = get_entry_surrogate_single_qoi_from_statistics_dict(
            statistics_dictionary, utility.PCE_COEFF_ENTRY, qoi_column_name)
        if gpce_coeff_dictionary is not None:
            return gpce_coeff_dictionary

    if workingDir is not None:
        # Try to read from the saved files
        gpce_coeff_dictionary = read_all_saved_gpce_coeffs(
            workingDir=workingDir, 
            list_qoi_column=[qoi_column_name,], 
            throw_error=False, 
            convert_to_pd_timestamp=convert_to_pd_timestamp)
        if gpce_coeff_dictionary is not None:
            return gpce_coeff_dictionary
        else:
            # Try to read saved statistics dictionary and read it from there
            statistics_dictionary = read_all_saved_statistics_dict(\
                workingDir=workingDir, 
                list_qoi_column=[qoi_column_name,], 
                single_timestamp_single_file=single_timestamp_single_file, 
                throw_error=False, 
                convert_to_pd_timestamp=convert_to_pd_timestamp)
            # Now try the once again to read from the statistics_dictionary
            return get_entry_surrogate_single_qoi_from_statistics_dict(statistics_dictionary, utility.PCE_COEFF_ENTRY, qoi_column_name)

    return None


###################################################################################################################
# Running UQEF-Dynamic model
###################################################################################################################
# Note: think about moving these functions to UQEF-Dynamic/uqef_dynamic/utils/uqPostProcessing.py


def run_uqef_dynamic_model_over_parameters(model, parameters: np.ndarray, raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs) -> List[Tuple[Dict[str, Any], float]]:
    """
    This function runs the model over parameters
    :model: an instance of UQEF-Dynamic Model - time_dependent_model, with a __call__ method being a wrapper for the model.run method
    :parameters: np.ndarray
        Parameters for the model dimension (dim, number_of_nodes)
    :raise_exception_on_model_break: bool
        If True, the function will raise an exception if the model breaks
    :args: tuple

    :kwargs: dict
        Additional keyword arguments for the model 

    Returns: 
    List[Tuple[Dict[str, Any], float]]: A list of tuples for each model run; each tuple is in the form (result_dict, runtime); 
    - result_dict is a dictionary that might contain the following key-value entries (depending on the configuration file):
    - ("result_time_series", flux_df): a dataframe containing the model output for the time period specified in the configuration file.
    - ("state_df", state_df): a dataframe containing the model state for the time period specified in the configuration file.
    - ("gof_df", index_parameter_gof_DF): a dataframe containing the goodness-of-fit values for the time period specified in the configuration file.
    - ("parameters_dict", index_run_and_parameters_dict): a dictionary containing the parameter values for the time period specified in the configuration file.
    - ("run_time", runtime): the runtime of a single model run; should have the same value as runtime variable
    - ("grad_matrix", gradient_matrix_dict): a dictionary containing the gradient vectors for the time period specified in the configuration file.
    """
    i_s = np.arange(parameters.shape[1])
    results_array = model(i_s, parameters.T, raise_exception_on_model_break, *args, **kwargs)    
    return results_array


def uqef_dynamic_model_run_results_array_to_dataframe(results_array,
    extract_only_qoi_columns=False, qoi_columns=[utility.QOI_COLUMN_NAME], time_column_name: str = utility.TIME_COLUMN_NAME, index_column_name: str = utility.INDEX_COLUMN_NAME):
    """
    This function converts the results array to a DataFrame
    :results_array: List[Tuple[Dict[str, Any], float]]: A list of tuples for each model run; each tuple is in the form (result_dict, runtime); 
    Take a look at the description of the function run_uqef_dynamic_model_over_parameters for more information.
    :extract_only_qoi_columns: bool
        If True, only the columns specified in qoi_columns + time_column_name + INDEX_COLUMN_NAME will be extracted from the result DataFrame
    return: 
        - df_simulation_result: pd.DataFrame
        DataFrame containing the simulation results; 
        if "result_time_series" is not in the result_dict, the DataFrame will be None
        - df_index_parameter_values: pd.DataFrame
        DataFrame containing the parameter values
        if "parameters_dict" is not in the result_dict, the DataFrame will be None
        - df_index_parameter_gof_values: pd.DataFrame
        DataFrame containing the goodness-of-fit values and parameter values
        if "gof_df" is not in the result_dict, the DataFrame will be None
        - dict_of_approx_matrix_c: dict
        Dictionary containing the approximated gradient matrix C
        - dict_of_matrix_c_eigen_decomposition: dict
        Dictionary containing the eigenvalues and eigenvectors of the gradient matrix C
        if "grad_matrix" is not in the result_dict, both dictionaries will be None
        - df_state_results: pd.DataFrame
        DataFrame containing the state results
        if "state_df" is not in the result_dict, the DataFrame will be None
    """
    qoi_columns = qoi_columns + [time_column_name, index_column_name]
    list_of_single_df = []
    list_index_parameters_dict = []
    list_of_single_index_parameter_gof_df = []
    list_of_gradient_matrix_dict = []
    list_of_single_state_df = []
    for index_run, single_result_tuple in enumerate(results_array, ):        
        result_dict = single_result_tuple[0]
        runtime = single_result_tuple[1]
        
        if "result_time_series" in result_dict:
            df_result = result_dict["result_time_series"]
            df_result = process_df_simulation_result(df_result, extract_only_qoi_columns, qoi_columns, time_column_name)
            list_of_single_df.append(df_result)
        if "parameters_dict" in result_dict:
            parameters_dict = result_dict["parameters_dict"]
            list_index_parameters_dict.append(parameters_dict)
        if "gof_df" in result_dict:
            gof_df = result_dict["gof_df"]
            list_of_single_index_parameter_gof_df.append(gof_df)
        if "grad_matrix" in result_dict:
            gradient_matrix_dict = result_dict["grad_matrix"]
            list_of_gradient_matrix_dict.append(gradient_matrix_dict)
        if "state_df" in result_dict:
            state_df = result_dict["state_df"]
            state_df = process_df_simulation_result(state_df, time_column_name=time_column_name)
            list_of_single_state_df.append(state_df)

    if list_of_single_df:
        df_simulation_result = pd.concat(list_of_single_df, ignore_index=True, sort=False, axis=0)
    else:
        df_simulation_result = None

    if list_index_parameters_dict:
        df_index_parameter_values = pd.DataFrame(list_index_parameters_dict)
    else:
        df_index_parameter_values = None

    if list_of_single_index_parameter_gof_df:
        df_index_parameter_gof_values = pd.concat(list_of_single_index_parameter_gof_df,
                                                       ignore_index=True, sort=False, axis=0)
    else:
        df_index_parameter_gof_values = None

    if list_of_gradient_matrix_dict:
        dict_of_approx_matrix_c, dict_of_matrix_c_eigen_decomposition = process_list_of_gradient_matrix_dict(list_of_gradient_matrix_dict)
    else:
        dict_of_approx_matrix_c = None
        dict_of_matrix_c_eigen_decomposition = None

    if list_of_single_state_df:
        df_state_results = pd.concat(list_of_single_state_df, ignore_index=True, sort=False, axis=0)
    else:
        df_state_results = None
    return df_simulation_result, df_index_parameter_values, df_index_parameter_gof_values, \
    dict_of_approx_matrix_c, dict_of_matrix_c_eigen_decomposition, df_state_results


def process_df_simulation_result(
    df_result, extract_only_qoi_columns=False, qoi_columns=[utility.QOI_COLUMN_NAME,], time_column_name: str = utility.TIME_COLUMN_NAME):
    if isinstance(df_result, pd.DataFrame) and df_result.index.name == time_column_name:
        df_result = df_result.reset_index()
        df_result.rename(columns={df_result.index.name: time_column_name}, inplace=True)
    if time_column_name not in list(df_result.columns):
        raise Exception(f"Error in Samples class - {time_column_name} is not in the "
                        f"columns of the DataFrame")
    if extract_only_qoi_columns:
        df_result = df_result[qoi_columns]
    return df_result


def process_list_of_gradient_matrix_dict(list_of_gradient_matrix_dict) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """ 
    This function is used to process the list of gradient matrix dictionaries
    and compute the average gradient matrix and its eigen decomposition.
    return: dict_of_approx_matrix_c, dict_of_matrix_c_eigen_decomposition

    """
    gradient_matrix_dict = defaultdict(list)
    dict_of_approx_matrix_c = defaultdict(list)
    dict_of_matrix_c_eigen_decomposition = defaultdict(list)

    for single_gradient_matrix_dict in list_of_gradient_matrix_dict:
        for key, value in single_gradient_matrix_dict.items():
            gradient_matrix_dict[key].append(np.array(value))

    for key in gradient_matrix_dict.keys():
        dict_of_approx_matrix_c[key] = \
            sum(gradient_matrix_dict[key]) / len(gradient_matrix_dict[key])
        dict_of_matrix_c_eigen_decomposition[key] = np.linalg.eigh(dict_of_approx_matrix_c[key])
        # np.linalg.eig(dict_of_approx_matrix_c[key])
    return dict_of_approx_matrix_c, dict_of_matrix_c_eigen_decomposition


def run_uqef_dynamic_model_over_parameters_and_process_result(
    model, parameters: np.ndarray, 
    qoi_column_name: str = utility.QOI_COLUMN_NAME, time_column_name: str = utility.TIME_COLUMN_NAME, index_column_name: str = utility.INDEX_COLUMN_NAME,
    return_dict_over_timestamps=False
    ):
    """
    This function runs the model over parameters
    :model: an instance of UQEF-Dynamic
    :parameters: np.ndarray
        Parameters for the model dimension (dim, number_of_nodes)
    :return_dict_over_timestamps: bool
        If True, the function will return a dictionary with keys being the timestamps and values list of model run over different parameter values
    return:
    List[List[float]]: A list of values of model runs (for different parameter values) over different time steps
    or a dictionary with keys being the timestamps and values being the list of model runs over different parameter
    """
    results_array = run_uqef_dynamic_model_over_parameters(
        model, parameters, raise_exception_on_model_break=True)
    df_simulation_result, _, _, _, _, _ = uqef_dynamic_model_run_results_array_to_dataframe(
        results_array,  extract_only_qoi_columns=True, qoi_columns=[qoi_column_name,], 
        time_column_name=time_column_name, index_column_name=index_column_name)
    df_simulation_result.sort_values(
        by=[index_column_name, time_column_name], ascending=[True, True], inplace=True, kind='quicksort',
        na_position='last'
        )
    groups = groupby_df_simulation_results(df_simulation_result, columns_to_group_by=[time_column_name,])
    t = list(groups.keys())

    if return_dict_over_timestamps:
        model_runs = {}
    else:
        model_runs = np.empty((len(t),), dtype=object)

    for idx, key in enumerate(t):
        if return_dict_over_timestamps:
            model_runs[key]= df_simulation_result.loc[groups[key].values][qoi_column_name].values
        else:
            model_runs[idx] = df_simulation_result.loc[groups[key].values][qoi_column_name].values
    return model_runs, t

# ==============================================================================================================
# Function for computing PCE and PCE-based statistics on UQEF-Dynamic model and data structure
#  by running the model over parameters and timesteps and processing the results
# For simple functions refer to utility module
# ==============================================================================================================


def groupby_df_simulation_results(df_simulation_result, columns_to_group_by: list=[]):
    if not columns_to_group_by:
        columns_to_group_by = [utility.TIME_COLUMN_NAME,]
    grouped = df_simulation_result.groupby(columns_to_group_by)
    return grouped.groups


def compute_gPCE_for_uqef_dynamic_model(model, expansion_order: int, joint_dist, \
    parameters: np.ndarray, nodes: np.ndarray, regression: bool = False, \
    weights_quad: np.ndarray = None, poly_rule: str = 'three_terms_recurrence', poly_normed: bool = True, \
    qoi_column_name: str = utility.QOI_COLUMN_NAME, time_column_name: str = utility.TIME_COLUMN_NAME, index_column_name: str = utility.INDEX_COLUMN_NAME,
    return_dict_over_timestamps=False
    ):
    """
    This function computes the generalized Polynomial Chaos Expansion (gPCE)
    This happens in time_dependent_statistics, however, there the computations are executed in parallel
    model: function
       This function relys on the UQEF-Dynamice kind of model!
    expansion_order: int
        Order of the polynomial expansion
    joint_dist: chaospy.distributions
        Joint distribution of with respect to which the polynomial expansion will be build
    parameters: np.ndarray 
        Parameters for the model dimension (dim, number_of_nodes)
    nodes: np.ndarray 
        Quadrature nodes dimension (dim, number_of_nodes)
        This might be the same as parameters
    regression: bool, optional
    weights_quad: np.ndarray 
        Quadrature weights dimension (dim, number_of_nodes)
    poly_rule: str, optional
    poly_normed: bool, optional
    return_dict_over_timestamps: bool
        If True, the function will return a dictionary with keys being the timestamps and values being the computed statistics
    return: np.ndarray
        gPCE over time dimension (len(t), )
    """
    dim = parameters.shape[0]
    number_expansion_coefficients = int(scipy.special.binom(dim+expansion_order, dim))  # cp.dimensions(polynomial_expansion)
    print(f"Total number of expansion coefficients in {dim}D space: {int(number_expansion_coefficients)}")

    polynomial_expansion, norms = cp.generate_expansion(
        expansion_order, joint_dist, rule=poly_rule, normed=poly_normed, retall=True)
    
    model_runs, timestamps = run_uqef_dynamic_model_over_parameters_and_process_result(model, parameters, qoi_column_name, time_column_name, index_column_name)
    # results_array = run_uqef_dynamic_model_over_parameters(
    #     model, parameters, raise_exception_on_model_break=True)
    # df_simulation_result, _, _, _, _, _ = uqef_dynamic_model_run_results_array_to_dataframe(
    #     results_array,  extract_only_qoi_columns=True, qoi_columns=[qoi_column_name,], 
    #     time_column_name=time_column_name, index_column_name=index_column_name)
    # df_simulation_result.sort_values(
    #     by=[index_column_name, time_column_name], ascending=[True, True], inplace=True, kind='quicksort',
    #     na_position='last'
    #     )
    # groups = groupby_df_simulation_results(df_simulation_result, columns_to_group_by=[time_column_name,])
    # t = list(groups.keys())
    # model_runs = [
    #     df_simulation_result.loc[groups[key].values][qoi_column_name].values for key in t
    #     ]

    if return_dict_over_timestamps:
        # if return_dict_over_timestamps is True, return results in a form of a dictionary over time (timestamps)
        gPCE_over_time = {} 
        coeff =  {}
    else:
        # otherwise return results in a form of numpy arrays over time (consecutive timestamps)
        gPCE_over_time =  np.empty((len(model_runs),), dtype=object) # np.empty((len(t), number_expansion_coefficients))
        coeff = np.empty((len(model_runs),), dtype=object)

    for idx, _ in enumerate(model_runs):  # for element in t:
        if regression:
            if return_dict_over_timestamps:
                gPCE_over_time[timestamps[idx]] = cp.fit_regression(polynomial_expansion, nodes, model_runs[idx])
            else:
                gPCE_over_time[idx] = cp.fit_regression(polynomial_expansion, nodes, model_runs[idx])
        else:
            if return_dict_over_timestamps:
                gPCE_over_time[timestamps[idx]], coeff[timestamps[idx]] = cp.fit_quadrature(polynomial_expansion, nodes, weights_quad, model_runs[idx], retall=True)
            else:
                gPCE_over_time[idx], coeff[idx] = cp.fit_quadrature(polynomial_expansion, nodes, weights_quad, model_runs[idx], retall=True)
    return gPCE_over_time, polynomial_expansion, np.asarray(norms, dtype=np.float64), coeff


def compute_PSP_for_uqef_dynamic_model(model, joint_dist, \
    quadrature_order: int, expansion_order: int, 
    sampleFromStandardDist: bool = False, joint_dist_standard=None,
    rule_quadrature: str = 'g', growth: bool = False, sparse: bool = False, \
    poly_rule: str = 'three_terms_recurrence', poly_normed: bool = True, \
    qoi_column_name: str = utility.QOI_COLUMN_NAME, time_column_name: str = utility.TIME_COLUMN_NAME, index_column_name: str = utility.INDEX_COLUMN_NAME, 
    return_dict_over_timestamps=False
    ):
    """
    This function computes the Pseudo-Spectra Projection (PSP)
    This happens in time_dependent_statistics, however, there the computations are executed in parallel
    Take a look at the definition of the function compute_gPCE_for_uqef_dynamic_model since the arguments are the same
    """
    
    if sampleFromStandardDist:
        if joint_dist_standard is None:
            raise Exception("joint_dist_standard should be provided if sampleFromStandardDist is True")
        nodes_quad, weights_quad = utility.generate_quadrature_nodes_and_weights(joint_dist_standard, quadrature_order, rule_quadrature, growth, sparse)
        parameters_quad = utility.generate_parameters_from_nodes(nodes_quad, joint_dist_standard, joint_dist)
        polynomial_expansion, norms = utility.generate_polynomial_expansion(joint_dist_standard, expansion_order, poly_rule, poly_normed)
    else:
        nodes_quad, weights_quad = utility.generate_quadrature_nodes_and_weights(joint_dist, quadrature_order, rule_quadrature, growth, sparse)
        parameters_quad = nodes_quad
        polynomial_expansion, norms = utility.generate_polynomial_expansion(joint_dist, expansion_order, poly_rule, poly_normed)
    
    model_runs, timestamps = run_uqef_dynamic_model_over_parameters_and_process_result(model, parameters_quad, qoi_column_name, time_column_name, index_column_name)
    # results_array = run_uqef_dynamic_model_over_parameters(
    #     model, parameters_quad, raise_exception_on_model_break=True)
    # df_simulation_result, _, _, _, _, _ = uqef_dynamic_model_run_results_array_to_dataframe(
    #     results_array,  extract_only_qoi_columns=True, qoi_columns=[qoi_column_name,], 
    #     time_column_name=time_column_name, index_column_name=index_column_name)
    # df_simulation_result.sort_values(
    #     by=[index_column_name, time_column_name], ascending=[True, True], inplace=True, kind='quicksort',
    #     na_position='last'
    #     )
    # groups = groupby_df_simulation_results(df_simulation_result, columns_to_group_by=[time_column_name,])
    # t = list(groups.keys())
    # model_runs = [
    #     df_simulation_result.loc[groups[key].values][qoi_column_name].values for key in t
    #     ]
    
    #print(f"Debugging model_runs - {model_runs}")
    #print(f"Debugging len(model_runs) - {len(model_runs)}")

    if return_dict_over_timestamps:
        # if return_dict_over_timestamps is True, return results in a form of a dictionary over time (timestamps)
        gPCE_over_time = {} 
        coeff =  {}
    else:
        # otherwise return results in a form of numpy arrays over time (consecutive timestamps)
        gPCE_over_time =  np.empty((len(model_runs),), dtype=object) # np.empty((len(t), number_expansion_coefficients))
        coeff = np.empty((len(model_runs),), dtype=object)
    for idx, _ in enumerate(model_runs):  # for element in t:
        if return_dict_over_timestamps:
            gPCE_over_time[timestamps[idx]], coeff[timestamps[idx]] = cp.fit_quadrature(polynomial_expansion, nodes_quad, weights_quad, model_runs[idx], retall=True)
        else:
            gPCE_over_time[idx], coeff[idx] = cp.fit_quadrature(polynomial_expansion, nodes_quad, weights_quad, model_runs[idx], retall=True)
    # return gPCE_over_time, polynomial_expansion, np.asarray(norms), np.asarray(coeff)
    return gPCE_over_time, polynomial_expansion, np.asarray(norms, dtype=np.float64), coeff


def compute_PSP_for_uqef_dynamic_model_ionuts_approach(model, joint_dist, joint_dist_standard,\
    quadrature_order: int, expansion_order: int, 
    rule_quadrature: str = 'g', growth: bool = False, sparse: bool = False, \
    poly_rule: str = 'three_terms_recurrence', poly_normed: bool = True, \
    qoi_column_name: str = utility.QOI_COLUMN_NAME, time_column_name: str = utility.TIME_COLUMN_NAME, index_column_name: str = utility.INDEX_COLUMN_NAME,
    return_dict_over_timestamps=False
    ):
    """
    Whatchout this is only for experimenting; it is not proven to provide valid results!
    """
    nodes_quad, weights_quad = utility.generate_quadrature_nodes_and_weights(joint_dist_standard, quadrature_order, rule_quadrature, growth, sparse)
    parameters_quad = utility.generate_parameters_from_nodes(nodes_quad, joint_dist_standard, joint_dist)
    polynomial_expansion, norms = utility.generate_polynomial_expansion(joint_dist, expansion_order, poly_rule, poly_normed) # this is the difference

    model_runs, timestamps = run_uqef_dynamic_model_over_parameters_and_process_result(model, parameters_quad, qoi_column_name, time_column_name, index_column_name)
    # results_array = run_uqef_dynamic_model_over_parameters(
    #     model, parameters_quad, raise_exception_on_model_break=True)
    # df_simulation_result, _, _, _, _, _ = uqef_dynamic_model_run_results_array_to_dataframe(
    #     results_array,  extract_only_qoi_columns=True, qoi_columns=[qoi_column_name,], 
    #     time_column_name=time_column_name, index_column_name=index_column_name)
    # df_simulation_result.sort_values(
    #     by=[index_column_name, time_column_name], ascending=[True, True], inplace=True, kind='quicksort',
    #     na_position='last'
    #     )
    # groups = groupby_df_simulation_results(df_simulation_result, columns_to_group_by=[time_column_name,])
    # t = list(groups.keys())
    # model_runs = [
    #     df_simulation_result.loc[groups[key].values][qoi_column_name].values for key in t
    #     ]
    
    #print(f"Debugging model_runs - {model_runs}")
    #print(f"Debugging len(model_runs) - {len(model_runs)}")

    if return_dict_over_timestamps:
        # if return_dict_over_timestamps is True, return results in a form of a dictionary over time (timestamps)
        gPCE_over_time = {} 
        coeff =  {}
    else:
        # otherwise return results in a form of numpy arrays over time (consecutive timestamps)
        gPCE_over_time =  np.empty((len(model_runs),), dtype=object) # np.empty((len(t), number_expansion_coefficients))
        coeff = np.empty((len(model_runs),), dtype=object)
    for idx, _ in enumerate(model_runs):  # for element in t:
        if return_dict_over_timestamps:
            gPCE_over_time[timestamps[idx]], coeff[timestamps[idx]]= cp.fit_quadrature(polynomial_expansion, nodes_quad, weights_quad, model_runs[idx], retall=True)
        else:
            gPCE_over_time[idx], coeff[idx] = cp.fit_quadrature(polynomial_expansion, nodes_quad, weights_quad, model_runs[idx], retall=True)
    # return gPCE_over_time, polynomial_expansion, np.asarray(norms), np.asarray(coeff)
    return gPCE_over_time, polynomial_expansion, np.asarray(norms, dtype=np.float64), coeff


# ==============================================================================================================
# Function for computing MC-based statistics
#  by running the model over parameters and timesteps and processing the results
# ==============================================================================================================

def run_uq_mc_sim_and_compute_mc_stat_for_uqef_dynamic_model(
    model, jointDists, 
    jointStandard=None, numSamples=1000, rule="R",
    sampleFromStandardDist=False,
    read_nodes_from_file=False, 
    rounding=False, round_dec=4,
    qoi_column_name: str = utility.QOI_COLUMN_NAME, time_column_name: str = utility.TIME_COLUMN_NAME, index_column_name: str = utility.INDEX_COLUMN_NAME,
    return_dict_over_timestamps=False,
    **kwargs
    ):
    """     
    return_dict_over_timestamps: bool
        If True, the function will return a dictionary with keys being the timestamps and values being the computed statistics
    Possible additional arguments:
        - parameters_file_name (str): The name of the file containing the parameters.
        - optional argumetns in the compute_mc_stat_for_uqef_dynamic_model function
            - compute_mean: bool
            - compute_var: bool
            - compute_std: bool
            - compute_skew: bool
            - compute_kurt: bool
            - compute_p10: bool
            - compute_p90: bool
            - compute_Sobol_m: bool
    Take a look at compute_mc_stat_for_uqef_dynamic_model function for a description of a return tuple
    """
    parameters = generate_parameters_for_mc_simulation(\
    jointDists, jointStandard, numSamples, rule, \
    sampleFromStandardDist, read_nodes_from_file, rounding, round_dec, **kwargs)
    return compute_mc_stat_for_uqef_dynamic_model(model, parameters, qoi_column_name, time_column_name, index_column_name, return_dict_over_timestamps, **kwargs)


def generate_parameters_for_mc_simulation(jointDists, jointStandard=None, numSamples=1000, rule="R",
    sampleFromStandardDist=False, read_nodes_from_file=False, rounding=False, round_dec=4,
    **kwargs):
    """
    This function generates the parameters for the Monte Carlo simulation
    return: np.ndarray
        Parameters for the model dimension (dim, number_of_nodes)
    """
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
        sampleFromStandardDist = False
        # TODO Be carefule, for now there is no transformation of parameters if they are read from some file
    else:
        if rounding:
            nodes = dist.sample(size=numSamples, rule=rule).round(round_dec)
        else:
            nodes = dist.sample(size=numSamples, rule=rule)
    nodes = np.array(nodes)

    if sampleFromStandardDist and jointStandard is not None:
        parameters = utility.transformation_of_parameters(
            nodes, jointStandard, jointDists)
    else:
        parameters = nodes
    return parameters


def compute_mc_stat_for_uqef_dynamic_model(model, parameters: np.ndarray,
    qoi_column_name: str = utility.QOI_COLUMN_NAME, time_column_name: str = utility.TIME_COLUMN_NAME, index_column_name: str = utility.INDEX_COLUMN_NAME,
    return_dict_over_timestamps=False,
    **kwargs
    ):
    """
    This function computes the MC statistics for a UQEF-Dynamic model type, based on already generated parameters

    This happens in time_dependent_statistics, however, there the computations are executed in parallel
    model: function
       This function relys on the UQEF-Dynamice kind of model!
    parameters: np.ndarray 
        Parameters for the model dimension (dim, number_of_nodes)
    qoi_column_name: str
        The name of the column containing the quantity of interest
    time_column_name: str
        The name of the column containing the time
    index_column_name: str
        The name of the column containing the index
    return_dict_over_timestamps: bool
        If True, the function will return a dictionary with keys being the timestamps and values being the computed statistics
    :kwargs: dict
        Additional keyword arguments for
        - compute_mean: bool
        - compute_var: bool
        - compute_std: bool
        - compute_skew: bool
        - compute_kurt: bool
        - compute_p10: bool
        - compute_p90: bool
        - compute_Sobol_m: bool
    return: list of numpy arrays, which are either populated with statistcs over time, or are mepty
     E_over_time, Var_over_time, StdDev_over_time, Skew_over_time, Kurt_over_time, P10_over_time, P90_over_time, sobol_m_over_time
    """
    dim = parameters.shape[0]
    numEvaluations = parameters.shape[1] #len(parameters.T)

    model_runs, timestamps = run_uqef_dynamic_model_over_parameters_and_process_result(
        model, parameters, qoi_column_name, time_column_name, index_column_name, return_dict_over_timestamps=False)
    number_of_time_steps = len(model_runs)
    compute_mean = kwargs.get('compute_mean', True)
    compute_var = kwargs.get('compute_var', True)
    compute_std = kwargs.get('compute_std', True)
    compute_skew = kwargs.get('compute_skew', True)
    compute_kurt = kwargs.get('compute_kurt', True)
    compute_p10 = kwargs.get('compute_p10', True)
    compute_p90 = kwargs.get('compute_p90', True)
    compute_Sobol_m = kwargs.get('compute_Sobol_m', False)

    if return_dict_over_timestamps:
        # if return_dict_over_timestamps is True, return results in a form of a dictionary over time (timestamps)
        E_over_time = {}
        Var_over_time = {}
        StdDev_over_time = {}
        Skew_over_time = {}
        Kurt_over_time = {}
        P10_over_time = {}
        P90_over_time = {}
        sobol_m_over_time = {}
    else:
        # otherwise return results in a form of numpy arrays over time (consecutive timestamps)
        E_over_time =  np.empty((number_of_time_steps,), dtype=object)
        Var_over_time =  np.empty((number_of_time_steps,), dtype=object)
        StdDev_over_time =  np.empty((number_of_time_steps,), dtype=object)
        Skew_over_time =  np.empty((number_of_time_steps,), dtype=object)
        Kurt_over_time =  np.empty((number_of_time_steps,), dtype=object)
        P10_over_time =  np.empty((number_of_time_steps,), dtype=object)
        P90_over_time =  np.empty((number_of_time_steps,), dtype=object)
        sobol_m_over_time =  np.empty((number_of_time_steps,), dtype=object)

    start_time_computing_statistics = time.time()
    for idx, _ in enumerate(model_runs):  # for element in t:
        qoi_values = model_runs[idx]
        if compute_mean:
            E = np.mean(qoi_values, 0)
            if return_dict_over_timestamps:
                E_over_time[timestamps[idx]] = E
            else:
                E_over_time[idx] = E
        if compute_var:
            if return_dict_over_timestamps:
                temp_values = qoi_values - E_over_time[timestamps[idx]]
                Var_over_time[timestamps[idx]] = np.var(temp_values, ddof=1)
            else:
                temp_values = qoi_values - E_over_time[idx]
                Var_over_time[idx] = np.var(temp_values, ddof=1)
                # Var_over_time[idx] = np.var(qoi_values, ddof=1)
        if compute_std:
            if return_dict_over_timestamps:
                StdDev_over_time[timestamps[idx]] = np.std(qoi_values, 0, ddof=1)
            else:
                StdDev_over_time[idx] = np.std(qoi_values, 0, ddof=1)
        if compute_skew:
            if return_dict_over_timestamps:
                Skew_over_time[timestamps[idx]] = scipy.stats.skew(qoi_values, axis=0, bias=True)
            else:
                Skew_over_time[idx] = scipy.stats.skew(qoi_values, axis=0, bias=True)
        if compute_kurt:
            if return_dict_over_timestamps:
                Kurt_over_time[timestamps[idx]] = scipy.stats.kurtosis(qoi_values, axis=0, bias=True)
            else:
                Kurt_over_time[idx] = scipy.stats.kurtosis(qoi_values, axis=0, bias=True)
        if compute_p10:
            if return_dict_over_timestamps:
                P10_over_time[timestamps[idx]] = np.percentile(qoi_values, 10, axis=0)
                if isinstance(P10_over_time[timestamps[idx]], list) and len(P10_over_time[timestamps[idx]]) == 1:
                    P10_over_time[timestamps[idx]] = P10_over_time[timestamps[idx]][0]
            else:
                P10_over_time[idx] = np.percentile(qoi_values, 10, axis=0)
                if isinstance(P10_over_time[idx], list) and len(P10_over_time[idx]) == 1:
                    P10_over_time[idx] = P10_over_time[idx][0]
        if compute_p90:
            if return_dict_over_timestamps:
                P90_over_time[timestamps[idx]] = np.percentile(qoi_values, 90, axis=0)
                if isinstance(P90_over_time[timestamps[idx]], list) and len(P90_over_time[timestamps[idx]]) == 1:
                    P90_over_time[timestamps[idx]] = P90_over_time[timestamps[idx]][0]
            else:
                P90_over_time[idx] = np.percentile(qoi_values, 90, axis=0)
                if isinstance(P90_over_time[idx], list) and len(P90_over_time[idx]) == 1:
                    P90_over_time[idx] = P90_over_time[idx][0]
        if compute_Sobol_m and parameters is not None:
            if return_dict_over_timestamps:
                sobol_m_over_time[timestamps[idx]]= sens_indices_sampling_based_utils.compute_sens_indices_based_on_samples_rank_based(
                    samples=parameters.T, Y=qoi_values[:numEvaluations, np.newaxis], D=dim, N=numEvaluations)
            else:
                sobol_m_over_time[idx] = sens_indices_sampling_based_utils.compute_sens_indices_based_on_samples_rank_based(
                    samples=parameters.T, Y=qoi_values[:numEvaluations, np.newaxis], D=dim, N=numEvaluations)
        end_time_computing_statistics = time.time()
        time_computing_statistics = end_time_computing_statistics - start_time_computing_statistics
        print(f"Needed time for computing statistics is: {time_computing_statistics};\n")

    return E_over_time, Var_over_time, StdDev_over_time, Skew_over_time, Kurt_over_time, P10_over_time, P90_over_time, sobol_m_over_time

# ==============================================================================================================
# Functions for computing goodness-of-fit functions for different statistics time-signals produced by UQ and SA simulations
# ==============================================================================================================


def compute_gof_over_different_time_series(df_statistics, objective_function="MAE", qoi_column="Q",
                                           measuredDF_column_names="measured"):
    """
    This function will run only for a single qoi
    """
    if not isinstance(qoi_column, list):
        qoi_column = [qoi_column, ]

    if not isinstance(measuredDF_column_names, list):
        measuredDF_column_names = [measuredDF_column_names, ]

    if not isinstance(objective_function, list):
        objective_function = [objective_function, ]

    result_dict = defaultdict(dict)
    for idx, single_qoi in enumerate(qoi_column):
        result_dict[single_qoi] = defaultdict(dict)
        df_statistics_single_qoi = df_statistics.loc[df_statistics['qoi'] == single_qoi]

        measuredDF_column_name = measuredDF_column_names[idx]

        if measuredDF_column_name not in df_statistics_single_qoi.columns:
            continue

        for single_objective_function in objective_function:
            if not callable(single_objective_function) and single_objective_function in utility.mapping_gof_names_to_functions:
                single_objective_function = utility.mapping_gof_names_to_functions[single_objective_function]
            elif not callable(
                    single_objective_function) and single_objective_function not in utility.mapping_gof_names_to_functions \
                    or callable(single_objective_function) and single_objective_function not in utility._all_functions:
                raise ValueError("Not proper specification of Goodness of Fit function name")

            gof_means_unalt = None
            gof_means_mean = None
            gof_means_mean_m_std = None
            gof_means_mean_p_std = None
            gof_means_p10 = None
            gof_means_p90 = None
            if 'unaltered' in df_statistics_single_qoi.columns:
                gof_means_unalt = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='unaltered')
            if 'E' in df_statistics_single_qoi.columns:
                gof_means_mean = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='E')
            if 'E_minus_std' in df_statistics_single_qoi.columns:
                gof_means_mean_m_std = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='E_minus_std')
            if 'E_plus_std' in df_statistics_single_qoi.columns:
                gof_means_mean_p_std = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='E_plus_std')
            if 'P10' in df_statistics_single_qoi.columns:
                gof_means_p10 = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='P10')
            if 'P90' in df_statistics_single_qoi.columns:
                gof_means_p90 = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='P90')

            result_dict[single_qoi][single_objective_function]["gof_means_unalt"] = gof_means_unalt
            result_dict[single_qoi][single_objective_function]["gof_means_mean"] = gof_means_mean
            result_dict[single_qoi][single_objective_function]["gof_means_mean_m_std"] = gof_means_mean_m_std
            result_dict[single_qoi][single_objective_function]["gof_means_mean_p_std"] = gof_means_mean_p_std
            result_dict[single_qoi][single_objective_function]["gof_means_p10"] = gof_means_p10
            result_dict[single_qoi][single_objective_function]["gof_means_p90"] = gof_means_p90

            print(f"{single_objective_function} - Comparing measured time series and different computed time series"
                  f"\n gof_means_unalt:{gof_means_unalt} \ngof_means_mean:{gof_means_mean} \n"
                  f"gof_means_mean_m_std:{gof_means_mean_m_std} \ngof_means_mean_p_std:{gof_means_mean_p_std} \n"
                  f"gof_means_p10:{gof_means_p10} \ngof_means_p90:{gof_means_p90} \n")
            

def redo_all_statistics(
        workingDir, get_measured_data=False, get_unaltered_data=False, station="MARI", uq_method="sc", plotting=False):
    raise NotImplementedError

###################################################################################################################
# Set of different functions for analyzing df_statistics DataFrame
    # produced as part of UQ simulation - these function may go to utility as well?
###################################################################################################################


def describe_df_statistics(df_statistics_and_measured, single_qoi=None):
    if single_qoi is None:
        result = df_statistics_and_measured.describe(include=np.number)
        print(result)
    else:
        df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured.loc[
            df_statistics_and_measured['qoi'] == single_qoi]
        result = df_statistics_and_measured_single_qoi_subset.describe(include=np.number)
        print(f"{single_qoi}\n\n")
        print(result)


def _get_sensitivity_indices_subset_of_big_df_statistics(df_statistics, single_qoi, param_names,
        si_type="Sobol_t", list_of_columns_to_keep=[]):
    """
    Get a subset of the DataFrame `df_statistics` containing statistics and measured values for a single quantity of interest (`single_qoi`),
    along with sensitivity indices for specified parameter names (`param_names`).

    Args:
        df_statistics (pandas.DataFrame): The DataFrame containing statistics and measured values.
        single_qoi (str): The quantity of interest to filter the DataFrame on.
        param_names (str or list): The parameter names to include in the sensitivity indices.
            If a single string is provided, it will be converted to a list.
        si_type (str, optional): The type of sensitivity index to include.
            Valid options are "Sobol_t", "Sobol_m", and "Sobol_m2".
            Defaults to "Sobol_t".
        list_of_columns_to_keep (list, optional): The list of column names to keep in the subset DataFrame.
            Defaults to []

    Returns:
        pandas.DataFrame: The subset (VIEW) of `df_statistics` containing the specified columns.

    """

    # TODO allow user to specify which column names
    # list_of_columns_to_keep = ["measured", "precipitation", "temperature"]
    list_of_columns_to_keep = list_of_columns_to_keep.copy()
    if not isinstance(param_names, list):
        param_names = [param_names,]
    for param_name in param_names:
        if si_type == "Sobol_t":
            list_of_columns_to_keep.append(f"Sobol_t_{param_name}")
        elif si_type == "Sobol_m":
            list_of_columns_to_keep.append(f"Sobol_m_{param_name}")
        elif si_type == "Sobol_m2":
            list_of_columns_to_keep.append(f"Sobol_m2_{param_name}")
    # df_statistics_and_measured_single_qoi = df_statistics.loc[df_statistics['qoi'] == single_qoi]
    # df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured_single_qoi[list_of_columns_to_keep]
    # return df_statistics_and_measured_single_qoi_subset
    return df_statistics.loc[df_statistics['qoi'] == single_qoi, list_of_columns_to_keep]


def describe_sensitivity_indices_single_qoi_under_some_condition(
        df_statistics, single_qoi, param_names,
        si_type="Sobol_t", condition_columns=None, condition_value=None, condition_sign="equal",
        list_of_columns_to_keep=[]):
    """
    Describes sensitivity indices for a single quantity of interest (QoI) under a given condition.

    Parameters:
        df_statistics (DataFrame): The DataFrame containing the statistics.
        single_qoi (str): The name of the single quantity of interest (QoI).
        param_names (list): The list of parameter names.
        si_type (str, optional): The type of sensitivity index. Defaults to "Sobol_t".
        condition_columns (str or list, optional): The column(s) to apply the condition on. Defaults to None.
        condition_value (int or float, optional): The value to compare against. Defaults to None.
        condition_sign (str, optional): The condition sign to use for comparison. Defaults to "equal".
        list_of_columns_to_keep (list, optional): The list of columns to keep in the subset. Defaults to []

    Raises:
        Exception: If condition_sign is not one of the following strings: 
        "smaller", "greater", "equal", "not_equal", "smaller_or_equal", "greater_or_equal",
        '==', '!=', '<', '>', '<=', '>=', "less than", "less than or equal", "greater than", "greater than or equal".

    Returns:
        None
    """
    df_statistics_and_measured_single_qoi_subset = _get_sensitivity_indices_subset_of_big_df_statistics(
        df_statistics, single_qoi, param_names, si_type, list_of_columns_to_keep)

    # print(f"[DEBUGGING] {id(df_statistics_and_measured_single_qoi_subset)} {id(df_statistics)}")
    if condition_columns is None or condition_value is None:
        result_describe = df_statistics_and_measured_single_qoi_subset.describe(include=np.number)
        # print(f"{result_describe}")
    else:
        if condition_sign not in ["smaller", "greater", "equal", "not_equal", "smaller_or_equal", "greater_or_equal",
        '==', '!=', '<', '>', '<=', '>=', "less than", "less than or equal", "greater than", "greater than or equal"]:
        # if condition_sign not in ["equal", "greater than", "greater than or equal", "less than", "less than or equal"]:
            raise Exception(f"Error in describe_sensitivity_indices_single_qoi_under_some_condition "
                            f"method - condition_sign should be one of the following strings: equal"
                            f"/greater than/greater than or equal/less than/less than or equal")
        if condition_sign == "equal" or condition_sign == "==":
            # df_subset = df_statistics_and_measured_single_qoi_subset[
            #     df_statistics_and_measured_single_qoi_subset[condition_columns] == condition_value].copy()
            # result_describe = df_subset.describe(include=np.number)
            result_describe = df_statistics_and_measured_single_qoi_subset[\
            df_statistics_and_measured_single_qoi_subset[condition_columns] == condition_value].describe(include=np.number)
            # print(f"{result_describe}")
        elif condition_sign == "greater than" or condition_sign == "greater" or condition_sign == ">":
            # df_subset = df_statistics_and_measured_single_qoi_subset[
            #     df_statistics_and_measured_single_qoi_subset[condition_columns] > condition_value].copy()
            # result_describe = df_subset.describe(include=np.number)
            result_describe = df_statistics_and_measured_single_qoi_subset[\
            df_statistics_and_measured_single_qoi_subset[condition_columns] > condition_value].describe(include=np.number)
            # print(f"{result_describe}")
        elif condition_sign == "greater than or equal" or condition_sign == "greater_or_equal" or condition_sign == '>=':
            # df_subset = df_statistics_and_measured_single_qoi_subset[
            #     df_statistics_and_measured_single_qoi_subset[condition_columns] >= condition_value].copy()
            # result_describe = df_subset.describe(include=np.number)
            result_describe = df_statistics_and_measured_single_qoi_subset[\
            df_statistics_and_measured_single_qoi_subset[condition_columns] >= condition_value].describe(include=np.number)
            # print(f"{result_describe}")
        elif condition_sign == "less than" or condition_sign == "smaller" or condition_sign == '<':
            # df_subset = df_statistics_and_measured_single_qoi_subset[
            #     df_statistics_and_measured_single_qoi_subset[condition_columns] < condition_value].copy()
            # result_describe = df_subset.describe(include=np.number)
            result_describe = df_statistics_and_measured_single_qoi_subset[\
            df_statistics_and_measured_single_qoi_subset[condition_columns] < condition_value].describe(include=np.number)
            # print(f"{result_describe}")
        elif condition_sign == "less than or equal" or condition_sign == "smaller_or_equal" or condition_sign == '<=':
            # df_subset = df_statistics_and_measured_single_qoi_subset[
            #     df_statistics_and_measured_single_qoi_subset[condition_columns] <= condition_value].copy()
            # result_describe = df_subset.describe(include=np.number)
            result_describe = df_statistics_and_measured_single_qoi_subset[\
            df_statistics_and_measured_single_qoi_subset[condition_columns] <= condition_value].describe(include=np.number)
            # print(f"{result_describe}")
        elif condition_sign == "not_equal" or condition_sign == '!=':
            result_describe = df_statistics_and_measured_single_qoi_subset[\
            df_statistics_and_measured_single_qoi_subset[condition_columns] != condition_value].describe(include=np.number)
    return result_describe


def compute_df_statistics_columns_correlation(
        df_statistics, single_qoi, param_names,
        only_sensitivity_indices_columns=True, si_type="Sobol_t", plot=True, list_of_columns_to_keep=[]):
    """
    Computes the correlation matrix between columns of a DataFrame containing statistical data.
    If plot plots the correlation matrix
    Args:
        df_statistics (pandas.DataFrame): The DataFrame containing the statistical data.
        single_qoi (str): The name of the quantity of interest (QoI) to consider.
        param_names (list): The names of the parameters.
        only_sensitivity_indices_columns (bool, optional): Whether to consider only the columns related to sensitivity indices. Defaults to True.
        si_type (str, optional): The type of sensitivity indices to consider. Defaults to "Sobol_t".
        plot (bool, optional): Whether to plot the correlation matrix. Defaults to True.
        list_of_columns_to_keep (list, optional): The list of columns to keep when considering only sensitivity indices columns. Defaults to [].

    Returns:
        None
    """
    if only_sensitivity_indices_columns:
        df_statistics_and_measured_single_qoi_subset = _get_sensitivity_indices_subset_of_big_df_statistics(
            df_statistics, single_qoi, param_names, si_type, list_of_columns_to_keep)
    else:
        # df_statistics_and_measured_single_qoi = df_statistics.loc[df_statistics['qoi'] == single_qoi]
        df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured_single_qoi
        df_statistics_and_measured_single_qoi_subset = df_statistics.loc[df_statistics['qoi'] == single_qoi]

    corr_df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured_single_qoi_subset.corr()
    print(f"Correlation matrix: {corr_df_statistics_and_measured_single_qoi_subset} \n")
    fig = None
    if plot:
        sns.set(style="darkgrid")
        mask = np.triu(np.ones_like(corr_df_statistics_and_measured_single_qoi_subset, dtype=bool))
        fig, axs = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr_df_statistics_and_measured_single_qoi_subset, mask=mask, square=True, annot=True,
                    linewidths=.5)
        plt.show()
    return corr_df_statistics_and_measured_single_qoi_subset, fig


def validate_condition(df, condition_results_based_on_metric, condition_results_based_on_metric_value):
    if df is None or condition_results_based_on_metric is None or condition_results_based_on_metric_value is None:
        raise Exception(f"Error in Statistics.handle_condition - it is not possible to condition df-index-parameter-gof-values on the column {condition_results_based_on_metric}")
    if isinstance(condition_results_based_on_metric, str):
        if condition_results_based_on_metric not in df.columns:
            raise Exception(f"Error in Statistics.handle_condition - the column {condition_results_based_on_metric} is not in the df-index-parameter-gof-values")
    elif isinstance(condition_results_based_on_metric, list):
        for single_condition in condition_results_based_on_metric:
            if single_condition not in df.columns:
                raise Exception(f"Error in Statistics.handle_condition - the column {single_condition} is not in the df-index-parameter-gof-values")


def filter_df_simulation_result_based_on_gof_condition(
    df_simulation_result: pd.DataFrame, df_index_parameter_gof: pd.DataFrame, 
    condition_results_based_on_metric: Union[str, list], condition_results_based_on_metric_value: Union[int, float, list], 
    condition_results_based_on_metric_sign: Union[str, list], 
    index_column_name: str=utility.INDEX_COLUMN_NAME, time_column_name: str=utility.TIME_COLUMN_NAME) -> pd.DataFrame:
    """
    Apply a condition to filter the data (model runs - df_simulation_result) 
    based on a given column (metric/gof/likelihood) and the value.
    The analysis is performed based on df_index_parameter_gof_values pd.DataFrame.

    Args:
        df_simulation_result
        df_index_parameter_gof
        condition_results_based_on_metric (str or list): The name(s) of the column(s) to compare.
        condition_results_based_on_metric_value (float or list): The threshold value(s) to compare against.
        condition_results_based_on_metric_sign (str or list): The comparison operator(s) to use for the comparison(
            e.g., '==', '!=', '<', '>', '<=', '>=', "smaller", "greater", "equal", "not_equal", "smaller_or_equal",  "greater_or_equal").
    Returns:
        Data Frame containing filtered data
    """
    validate_condition(df_index_parameter_gof, condition_results_based_on_metric, condition_results_based_on_metric_value)
    mask = utility.generate_mask_based_on_multiple_column_comparison(
        df=df_index_parameter_gof, column_name=condition_results_based_on_metric, 
        threshold_value=condition_results_based_on_metric_value, comparison=condition_results_based_on_metric_sign)

    list_of_index_runs_to_keep = df_index_parameter_gof[mask][index_column_name].tolist()

    filtered_df_simulation_result = df_simulation_result[df_simulation_result[index_column_name].isin(list_of_index_runs_to_keep)]
    filtered_df_simulation_result.sort_values(
        by=[index_column_name, time_column_name], ascending=[True, True], 
        inplace=True, kind='quicksort', na_position='last'
    )
    return filtered_df_simulation_result

###################################################################################################################
# Set of functions for plotting and or computing (KDE-based) CDFs and PDFs - these function may go to utility as well?
###################################################################################################################


def plot_cdfs_of_parameters_sensitivity_indices(df_statistics, single_qoi, param_names, si_type="Sobol_t"):
    df_statistics_and_measured_single_qoi_subset = _get_sensitivity_indices_subset_of_big_df_statistics(
        df_statistics, single_qoi, param_names, si_type, list_of_columns_to_keep=[])
    num_param = len(param_names)

    t = np.linspace(0, 1.0, 1000)
    fig, axs = plt.subplots(1, num_param, figsize=(20, 10))
    for i in range(num_param):
        param_name = param_names[i]
        if si_type == "Sobol_t":
            param_eval = df_statistics_and_measured_single_qoi_subset[f"Sobol_t_{param_name}"].values
        elif si_type == "Sobol_m":
            param_eval = df_statistics_and_measured_single_qoi_subset[f"Sobol_m_{param_name}"].values
        elif si_type == "Sobol_m2":
            param_eval = df_statistics_and_measured_single_qoi_subset[f"Sobol_m2_{param_name}"].values
        distribution = cp.GaussianKDE(param_eval, h_mat=0.005 ** 2)
        axs[i,].plot(t, distribution.cdf(t), label=f"KDE CDF {si_type} {param_name}")
        plt.setp(axs[i,], xlabel=f'{si_type}_{param_name}')
        axs[i,].grid()
    # plt.legend()
    plt.setp(axs[0], ylabel='CDF')
    plt.show()


def single_param_single_qoi_sensitivity_indices_GaussianKDE(
        df_statistics, single_qoi, param_name, si_type="Sobol_t", plot=True, plot_pdf_or_cdf="pdf",
        h_mat=0.02, alpha=0.5):
    df_statistics_and_measured_single_qoi_subset = _get_sensitivity_indices_subset_of_big_df_statistics(
        df_statistics, single_qoi, param_name, si_type, list_of_columns_to_keep=[])
    column_to_keep = None
    if si_type == "Sobol_t":
        column_to_keep = f"Sobol_t_{param_name}"
    elif si_type == "Sobol_m":
        column_to_keep = f"Sobol_m_{param_name}"
    elif si_type == "Sobol_m2":
        column_to_keep = f"Sobol_m2_{param_name}"
    samples = df_statistics_and_measured_single_qoi_subset[column_to_keep].values

    t = np.linspace(0, 1.0, 1000)
    distribution = cp.GaussianKDE(samples, h_mat=h_mat ** 2)

    if plot:
        if plot_pdf_or_cdf == "pdf":
            plt.hist(samples, bins=100, density=True, alpha=alpha)
            plt.plot(t, distribution.pdf(t), label=f"PDF - {si_type} {single_qoi} {param_name}")
        elif plot_pdf_or_cdf == "cdf":
            plt.plot(t, distribution.cdf(t), label=f"CDF - {si_type} {single_qoi} {param_name}")
        else:
            raise Exception()
        plt.legend()
        plt.show()

    return distribution


def gof_values_GaussianKDE(df_index_parameter_gof, gof_list=None, plot=True, plot_pdf_or_cdf="pdf"):
    result_dic_of_distributions = defaultdict()
    if gof_list is None:
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)

    if plot:
        fig, axs = plt.subplots(1, len(gof_list), figsize=(20, 10))

    for i in range(len(gof_list)):
        single_gof = gof_list[i]
        min_single_gof = df_index_parameter_gof[single_gof].min() - abs(df_index_parameter_gof[single_gof].min())*0.001
        max_single_gof = df_index_parameter_gof[single_gof].max() + abs(df_index_parameter_gof[single_gof].max())*0.001
        t = np.linspace(min_single_gof, max_single_gof, 1000)
        gof_eval = df_index_parameter_gof[single_gof].values
        distribution = cp.GaussianKDE(gof_eval, h_mat=0.005 ** 2)
        result_dic_of_distributions[single_gof] = distribution
        if plot:
            if plot_pdf_or_cdf == "pdf":
                axs[i,].hist(gof_eval, bins=100, density=True, alpha=0.5)
                axs[i,].plot(t, distribution.pdf(t), label=f"KDE PDF {single_gof} {gof_list}")
            elif plot_pdf_or_cdf == "cdf":
                axs[i,].plot(t, distribution.cdf(t), label=f"KDE CDF {single_gof} {gof_list}")
            else:
                raise Exception()
            plt.setp(axs[i,], xlabel=f'{single_gof}')
            axs[i,].grid()
    if plot:
        plt.setp(axs[0], ylabel=f'{plot_pdf_or_cdf}')
        plt.show()

    return result_dic_of_distributions

###################################################################################################################
# Set of functions for plotting - these function may go to utility as well?
###################################################################################################################


def plot_heatmap_si_single_qoi(qoi_column, si_df, si_type="Sobol_t", uq_method="sc", time_column_name="TimeStamp"):
    reset_index_at_the_end = False
    if si_df.index.name != time_column_name:
        si_df.set_index(time_column_name, inplace=True)
        reset_index_at_the_end = True

    si_columns_to_plot = [x for x in si_df.columns.tolist() if x != 'measured' and x != 'measured_norm' and x != 'qoi']
    si_columns_to_label = [single_column.split("_", 2)[2] for single_column in si_columns_to_plot]

    # fig = px.imshow(si_df[si_columns_to_plot].T, labels=dict(y='Parameters', x='Dates'))

    if 'qoi' in si_df.columns.tolist():
        fig = px.imshow(si_df.loc[si_df['qoi'] == qoi_column][si_columns_to_plot].T,
                        # color_continuous_scale='Inferno',
                        y = si_columns_to_label,
                        labels=dict(y='Parameters', x='Dates'))
    else:
        fig = px.imshow(si_df[si_columns_to_plot].T,
                        # color_continuous_scale='Inferno',
                        y = si_columns_to_label,
                        labels=dict(y='Parameters', x='Dates'))

    if reset_index_at_the_end:
        si_df.reset_index(inplace=True)
        si_df.rename(columns={si_df.index.name: time_column_name}, inplace=True)

    fig.update_xaxes(
        tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
        dtick="M2"                     # Set tick interval to 1 day for denser ticks
    )
    
    return fig


def plot_si_and_normalized_measured_time_signal_single_qoi(
        qoi_column, si_df, si_type="Sobol_t",
        observed_column_normalized="measured_norm", uq_method="sc", plot_forcing_data=False):

    si_columns = [x for x in si_df.columns.tolist() if x != 'measured' and x != 'measured_norm']

    fig = px.line(si_df, x=si_df.index, y=si_columns)

    if observed_column_normalized in si_df.columns.tolist() and not si_df[observed_column_normalized].isnull().all():
        fig.add_trace(go.Scatter(x=si_df.index,
                                 y=si_df[observed_column_normalized],
                                 fill='tozeroy', name=f"Normalized {qoi_column}"))
    # if plot_forcing_data:
    #     pass
    return fig


def plotting_function_single_qoi(
        df, single_qoi, subplot_titles=None, dict_what_to_plot=None, directory="./", fileName="simulation_big_plot"
):

    n_rows = 1
    starting_row_for_predicted_data = 1

    if dict_what_to_plot is None:
        dict_what_to_plot = {
            "E_minus_std": True, "E_plus_std": True, 
            "E_minus_2std": True, "E_plus_2std": True,
            "P10": True, "P90": True,
            "StdDev": True, "Skew": False, "Kurt": False,
        }

    if dict_what_to_plot.get("StdDev", False) and 'StdDev' in df.columns:
        n_rows += 1
    if dict_what_to_plot.get("Skew", False) and 'Skew' in df.columns:
        n_rows += 1
    if dict_what_to_plot.get("Kurt", False) and 'Kurt' in df.columns:
        n_rows += 1

    if subplot_titles is None:
        subplot_titles = ("Model Output",)
        if dict_what_to_plot.get("StdDev", False) and 'StdDev' in df.columns:
            subplot_titles = subplot_titles + ("Standard Deviation",)
        if dict_what_to_plot.get("Skew", False) and 'Skew' in df.columns:
            subplot_titles = subplot_titles + ("Skew",)
        if dict_what_to_plot.get("Kurt", False) and 'Kurt' in df.columns:
            subplot_titles = subplot_titles + ("Kurt",)

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        horizontal_spacing=0.01, vertical_spacing=0.04
    )
    fig.add_trace(
        go.Scatter(
            x=df['TimeStamp'], y=df['E'],
            text=df['E'],
            name=f"Mean predicted {single_qoi}", mode='lines'
        ),
        row=starting_row_for_predicted_data, col=1
    )

    if dict_what_to_plot.get("E_minus_std", False) and 'E_minus_std' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E_minus_std'],
                name=f'E_minus_std',
                text=df['E_minus_std'], mode='lines', showlegend=False, line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
            ),
            row=starting_row_for_predicted_data, col=1
        )
    if dict_what_to_plot.get("E_plus_std", False) and 'E_plus_std' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E_plus_std'],
                name=f'E+-std',
                text=df['E_plus_std'],
                mode='lines', fill='tonexty', showlegend=True, line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
            ),
            row=starting_row_for_predicted_data, col=1
        )

    if dict_what_to_plot.get("E_minus_2std", False) and 'E_minus_2std' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E_minus_2std'],
                name=f'E_minus_2std',
                text=df['E_minus_2std'], mode='lines', showlegend=False, line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
            ),
            row=starting_row_for_predicted_data, col=1
        )
    if dict_what_to_plot.get("E_plus_2std", False) and 'E_plus_2std' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E_plus_2std'],
                name=f'E+-2std',
                text=df['E_plus_2std'],
                mode='lines', fill='tonexty', showlegend=True, line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
            ),
            row=starting_row_for_predicted_data, col=1
        )

    if dict_what_to_plot.get("P10", False) and 'P10' in df.columns:
        fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["P10"],
                                 name=f'P10',
                                 line_color='rgba(255, 165, 0, 0.3)', mode='lines', showlegend=False),
                      row=starting_row_for_predicted_data, col=1)
    if dict_what_to_plot.get("P90", False) and 'P90' in df.columns:
        fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["P90"],
                                 name=f'P10-P90',
                                 line_color='rgba(255, 165, 0, 0.3)', mode='lines', fill='tonexty', showlegend=True),
                      row=starting_row_for_predicted_data, col=1)

    starting_row_for_predicted_data += 1

    if dict_what_to_plot.get("StdDev", False) and 'StdDev' in df.columns:
        fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["StdDev"],
                                 name=f'std. dev of {single_qoi}', line_color='darkviolet', mode='lines'),
                      row=starting_row_for_predicted_data, col=1)
        starting_row_for_predicted_data += 1

    if dict_what_to_plot.get("Skew", False) and 'Skew' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['Skew'],
                text=df['Skew'], name=f"Skewness of {single_qoi}", mode='markers'
            ),
            row=starting_row_for_predicted_data, col=1
        )
        starting_row_for_predicted_data += 1

    if dict_what_to_plot.get("Kurt", False) and 'Kurt' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['Kurt'],
                text=df['Kurt'], name=f"Kurtosis of {single_qoi}", mode='markers'
            ),
            row=starting_row_for_predicted_data, col=1
        )
        starting_row_for_predicted_data += 1

    fig.update_layout(height=600, width=800,
                      title_text=f"Detailed plot of most important time-series - QoI {single_qoi}")
    timesteps_min = min(df['TimeStamp'])
    timesteps_max = max(df['TimeStamp'])
    fig.update_layout(
        xaxis=dict(
            rangemode='normal',
            range=[timesteps_min, timesteps_max],
            type="date"
        ),
        yaxis=dict(
            rangemode='normal',  # Ensures the range is not padded for markers
            autorange=True       # Auto-range is enabled
        )
    )
    fig.update_layout(
        # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        showlegend=True,
        # template="plotly_white",
    )
    # fig.update_layout(xaxis=dict(type="date"))
    # fig.update_xaxes(
    #     tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
    #     dtick="M1"                     # Set tick interval to 1 day for denser ticks
    # )
    fig.update_layout(title=None)
    fig.update_layout(
        margin=dict(
            t=10,  # Top margin
            b=10,  # Bottom margin
            l=20,  # Left margin
            r=20   # Right margin
        )
    )
    if not str(directory).endswith("/"):
        directory = str(directory) + "/"
    plot_filename = pathlib.Path(directory)  / f"{fileName}.pdf"
    fig.write_image(str(plot_filename), format="pdf", width=1100,)
    fileName = str(directory) + f"{fileName}.html"
    pyo.plot(fig, filename=fileName, auto_open=False)

    return fig
# ============================================================================================
# plotting functions for HBV model
# ============================================================================================
# TODO Think about moving this to some model specific utility module

def plot_parameters_sensitivity_indices_vs_temp_prec_measured(df_statistics, single_qoi, param_names, si_type="Sobol_t"):
    """
    Note: important assumption is that the df_statistics DataFrame has columns with names like: "measured", "precipitation", "temperature"
    This is a specific function tailored for the HBV model and other environmental models/hydrologic models
    :param df_statistics:
    :param single_qoi:
    :param param_names:
    :param si_type:
    :return:
    """
    list_of_columns_to_keep = ["measured", "precipitation", "temperature"]
    df_statistics_and_measured_single_qoi_subset = _get_sensitivity_indices_subset_of_big_df_statistics(
        df_statistics, single_qoi, param_names, si_type, list_of_columns_to_keep)
    num_param = len(param_names)
    fig, axs = plt.subplots(3, num_param, figsize=(15, 8))
    # param = configurationObject["parameters"][0]["name"]
    # param_num = 0
    # param_counter = 0
    for i, ax in enumerate(axs.flat):
        param_num = i % num_param
        param = param_names[param_num]
        # if i % 3 == 0:
        if i < num_param:
            if si_type == "Sobol_t":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_t_{param}", "measured"]].values.T)
                # ax.scatter(*df_statistics_and_measured_single_qoi_subset[["measured", f"sobol_t_{param}", ]].values.T)
            elif si_type == "Sobol_m":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_m_{param}", "measured"]].values.T)
            elif si_type == "Sobol_m2":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_m2_{param}", "measured"]].values.T)
            # elif i % 3 == 1:
        elif i>=num_param and i<2*num_param:
            if si_type == "Sobol_t":
                ax.scatter(
                    *df_statistics_and_measured_single_qoi_subset[[f"Sobol_t_{param}", "precipitation"]].values.T)
            elif si_type == "Sobol_m":
                ax.scatter(
                    *df_statistics_and_measured_single_qoi_subset[[f"Sobol_m_{param}", "precipitation"]].values.T)
            elif si_type == "Sobol_m2":
                ax.scatter(
                    *df_statistics_and_measured_single_qoi_subset[[f"Sobol_m2_{param}", "precipitation"]].values.T)
            # elif i % 3 == 2:
        else:
            if si_type == "Sobol_t":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_t_{param}", "temperature"]].values.T)
            elif si_type == "Sobol_m":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_m_{param}", "temperature"]].values.T)
            elif si_type == "Sobol_m2":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_m2_{param}", "temperature"]].values.T)
        # param_counter += 1
    #     ax.set_title(f'{param}')
    # set labels
    for i in range(num_param):
        param = param_names[i]
        plt.setp(axs[-1, i], xlabel=f'{si_type} {param}')
        # plt.setp(axs[:, i], ylabel=f'{param}')
    plt.setp(axs[0, 0], ylabel='measured')
    plt.setp(axs[1, 0], ylabel='precipitation')
    plt.setp(axs[2, 0], ylabel='temperature')
    return fig
    # plt.setp(axs[0, :], xlabel='measured')
    # plt.setp(axs[1, :], xlabel='precipitation')
    # plt.setp(axs[2, :], xlabel='temperature')


def plot_parameters_sensitivity_indices_vs_temp_prec_measured_plotly(
    df, single_qoi, param_names, forcing_measured_columns, si_type="Sobol_t"):
    df =_get_sensitivity_indices_subset_of_big_df_statistics(
        df_statistics=df, 
        single_qoi=single_qoi, 
        param_names=param_names.copy(), 
        si_type=si_type, 
        list_of_columns_to_keep=forcing_measured_columns.copy()
    )

    num_columns = len(param_names)
    number_rows = len(forcing_measured_columns)

    # print(f"num_columns={num_columns}")
    # print(f"number_rows={number_rows}")

    # Create a subplot grid
    fig = make_subplots(rows=number_rows, cols=num_columns, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.01, vertical_spacing=0.04)

    # Populate the subplot grid with scatter plots
    for i, x_col in enumerate(forcing_measured_columns):
        for j, y_col in enumerate(param_names):
            fig.add_trace(
                go.Scatter(x=df[f"{si_type}_{y_col}"], y=df[x_col], mode='markers', marker=dict(size=5)),
                row=i+1, col=j+1
            )
            # Update axes titles
            if j == 0:
                fig.update_yaxes(title_text=x_col, row=i+1, col=j+1)
            if i == number_rows-1:
                fig.update_xaxes(title_text=y_col, row=i+1, col=j+1)

    # Update layout
    fig.update_layout(title=f"UQEF-Dynamic Scatter Plot -{si_type}", height=800, width=1000, showlegend=False)
    return fig


def plot_forcing_mean_predicted_and_observed_all_qoi(statisticsObject, directory="./", fileName="simulation_big_plot.html"):
    """
    At the moment this is tailored for HBV model
    :param df:
    :return:
    """
    measured_column_names_set = set()
    for single_qoi in statisticsObject.list_qoi_column:
        measured_column_names_set.add(statisticsObject.dict_corresponding_original_qoi_column[single_qoi])

    total_number_of_rows = 2 + len(statisticsObject.list_qoi_column) + len(measured_column_names_set)
    fig = make_subplots(
        rows=total_number_of_rows, cols=1, shared_xaxes=True
        #     subplot_titles=tuple(statisticsObject.list_qoi_column)
    )
    n_row = 3

    fig.add_trace(
        go.Bar(
            x=statisticsObject.forcing_df.index, y=statisticsObject.forcing_df['precipitation'],
            text=statisticsObject.forcing_df['precipitation'],
            name="Precipitation"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=statisticsObject.forcing_df.index, y=statisticsObject.forcing_df['temperature'],
            text=statisticsObject.forcing_df['temperature'],
            name="Temperature", mode='lines'
        ),
        row=2, col=1
    )

    if statisticsObject.df_statistics is None or statisticsObject.df_statistics.empty:
        raise Exception(f"You are trying to call a plotting utiltiy function whereas "
                        f"statisticsObject.df_statistics object is still not computed - make sure to first call"
                        f"statisticsObject.create_df_from_statistics_data")

    measured_column_names_set = set()
    for single_qoi in statisticsObject.list_qoi_column:
        df_statistics_single_qoi = statisticsObject.df_statistics.loc[
            statisticsObject.df_statistics['qoi'] == single_qoi]
        corresponding_measured_column = statisticsObject.dict_corresponding_original_qoi_column[single_qoi]
        if corresponding_measured_column not in measured_column_names_set:
            measured_column_names_set.add(corresponding_measured_column)
            fig.add_trace(
                go.Scatter(
                    x=df_statistics_single_qoi['TimeStamp'],
                    y=df_statistics_single_qoi['measured'],
                    name=f"Observed {corresponding_measured_column}", mode='lines'
                ),
                row=n_row, col=1
            )
            n_row += 1

        fig.add_trace(
            go.Scatter(
                x=df_statistics_single_qoi['TimeStamp'],
                y=df_statistics_single_qoi['E'],
                text=df_statistics_single_qoi['E'],
                name=f"Mean predicted {single_qoi}", mode='lines'),
            row=n_row, col=1
        )
        n_row += 1

    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_layout(height=600, width=800, title_text="Detailed plot of most important time-series")
    fig.update_layout(xaxis=dict(type="date"))
    fig.update_layout(
        # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        showlegend=True,
        # template="plotly_white",
    )
    fig.update_xaxes(
        tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
        dtick="M1"                     # Set tick interval to 1 day for denser ticks
    )
    if not str(directory).endswith("/"):
        directory = str(directory) + "/"
    if fileName is None:
        fileName = "datailed_plot_all_qois.html"
    fileName = directory + fileName
    pyo.plot(fig, filename=fileName)
    return fig, n_row


def plotting_function_single_qoi_hbv(
        df, single_qoi, qoi="Q", subplot_titles=None, dict_what_to_plot=None, directory="./", fileName="simulation_big_plot.html"
):
    """
    At the moment this is tailored for HBV model

    Note: qoi argument is relevant only when == "GoF"/"gof"
    :param df:
    :return:
    """
    n_rows = 3
    starting_row_for_predicted_data = 3

    if dict_what_to_plot is None:
        dict_what_to_plot = {
            "E_minus_std": False, "E_plus_std": False, "P10": False, "P90": False,
            "StdDev": False, "Skew": False, "Kurt": False
        }

    if not isinstance(qoi, list) and qoi.lower() == "gof":
        n_rows += 1
        starting_row_for_predicted_data = 4
        # subplot_titles = ("Precipitation", "Temperature", "Measured Streamflow", "Mean", "Skew", "Kurt")

    if dict_what_to_plot.get("StdDev", False) and 'StdDev' in df.columns:
        n_rows += 1
    if dict_what_to_plot.get("Skew", False) and 'Skew' in df.columns:
        n_rows += 1
    if dict_what_to_plot.get("Kurt", False) and 'Kurt' in df.columns:
        n_rows += 1

    if subplot_titles is None:
        subplot_titles = ("Precipitation [mm/day]", "Temperature [°C]")
        if not isinstance(qoi, list) and qoi.lower() == "gof":
            subplot_titles = subplot_titles + ("Measured Streamflow",)
            subplot_titles = subplot_titles + ("Predicted",)
        else:
            subplot_titles= subplot_titles + ("Measured and Predicted Data",)
        if dict_what_to_plot.get("StdDev", False) and 'StdDev' in df.columns:
            subplot_titles = subplot_titles + ("Standard Deviation",)
        if dict_what_to_plot.get("Skew", False) and 'Skew' in df.columns:
            subplot_titles = subplot_titles + ("Skew",)
        if dict_what_to_plot.get("Kurt", False) and 'Kurt' in df.columns:
            subplot_titles = subplot_titles + ("Kurt",)

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        horizontal_spacing=0.01, vertical_spacing=0.04
    )

    fig.add_trace(
        go.Bar(
            x=df['TimeStamp'], y=df['precipitation'],
            text=df['precipitation'],
            name="Precipitation [mm/day]",
            marker_color='red',
            showlegend=False,
            # mode="lines",
            #         line=dict(
            #             color='LightSkyBlue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['TimeStamp'], y=df['temperature'],
            text=df['temperature'],
            name="Temperature [°C]", mode='lines', #mode='lines+markers'
            marker_color='blue',
            showlegend=False,
        ),
        row=2, col=1
    )

    # Hardcoded
    if not df['measured'].isna().all():
    # if single_qoi == "Q_cms":
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['measured'],
                name="Measured Streamflow [m^3/s]", mode='lines',
                # line=dict(color='green'),
            ),
            row=starting_row_for_predicted_data, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=df['TimeStamp'], y=df['E'],
            text=df['E'],
            name=f"Mean predicted {single_qoi}", mode='lines'
        ),
        row=starting_row_for_predicted_data, col=1
    )

    if dict_what_to_plot.get("E_minus_std", False) and 'E_minus_std' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E_minus_std'],
                name=f'E_minus_std',
                text=df['E_minus_std'], mode='lines', showlegend=False, line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
            ),
            row=starting_row_for_predicted_data, col=1
        )
    if dict_what_to_plot.get("E_plus_std", False) and 'E_plus_std' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E_plus_std'],
                name=f'E+-std',
                text=df['E_plus_std'],
                mode='lines', fill='tonexty', showlegend=True, line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
            ),
            row=starting_row_for_predicted_data, col=1
        )

    if dict_what_to_plot.get("E_minus_2std", False) and 'E_minus_2std' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E_minus_2std'],
                name=f'E_minus_2std',
                text=df['E_minus_2std'], mode='lines', showlegend=False, line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
            ),
            row=starting_row_for_predicted_data, col=1
        )
    if dict_what_to_plot.get("E_plus_2std", False) and 'E_plus_2std' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E_plus_2std'],
                name=f'E+-2std',
                text=df['E_plus_2std'],
                mode='lines', fill='tonexty', showlegend=True, line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
            ),
            row=starting_row_for_predicted_data, col=1
        )

    if dict_what_to_plot.get("P10", False) and 'P10' in df.columns:
        fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["P10"],
                                 name=f'P10',
                                 line_color='rgba(255, 165, 0, 0.3)', mode='lines', showlegend=False),
                      row=starting_row_for_predicted_data, col=1)
    if dict_what_to_plot.get("P90", False) and 'P90' in df.columns:
        fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["P90"],
                                 name=f'P10-P90',
                                 line_color='rgba(255, 165, 0, 0.3)', mode='lines', fill='tonexty', showlegend=True),
                      row=starting_row_for_predicted_data, col=1)

    starting_row_for_predicted_data += 1

    if dict_what_to_plot.get("StdDev", False) and 'StdDev' in df.columns:
        fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["StdDev"],
                                 name=f'std. dev of {single_qoi}', line_color='darkviolet', mode='lines'),
                      row=starting_row_for_predicted_data, col=1)
        starting_row_for_predicted_data += 1

    if dict_what_to_plot.get("Skew", False) and 'Skew' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['Skew'],
                text=df['Skew'], name=f"Skewness of {single_qoi}", mode='markers'
            ),
            row=starting_row_for_predicted_data, col=1
        )
        starting_row_for_predicted_data += 1

    if dict_what_to_plot.get("Kurt", False) and 'Kurt' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['Kurt'],
                text=df['Kurt'], name=f"Kurtosis of {single_qoi}", mode='markers'
            ),
            row=starting_row_for_predicted_data, col=1
        )
        starting_row_for_predicted_data += 1

    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_layout(height=600, width=800,
                      title_text=f"Detailed plot of most important time-series - QoI {single_qoi}")
    timesteps_min = min(df['TimeStamp'])
    timesteps_max = max(df['TimeStamp'])
    fig.update_layout(
        xaxis=dict(
            rangemode='normal',
            range=[timesteps_min, timesteps_max],
            type="date"
        ),
        # yaxis=dict(
        #     rangemode='normal',  # Ensures the range is not padded for markers
        #     autorange=True       # Auto-range is enabled
        # )
    )
    fig.update_layout(
        # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        showlegend=True,
        # template="plotly_white",
    )
    # fig.update_layout(xaxis=dict(type="date"))
    fig.update_xaxes(
        tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
        dtick="M2"                     # Set tick interval to 1 day for denser ticks
    )
    if not str(directory).endswith("/"):
        directory = str(directory) + "/"
    fileName = str(directory) + fileName
    pyo.plot(fig, filename=fileName, auto_open=False)
    return fig