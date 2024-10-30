"""
@author: Ivana Jovanovic Buha
This file is used to read different output files created by the UQEF-Dynamic pipeline.
It reads general file produced by all the models (not specific to any model).
"""
import os
from distutils.util import strtobool
import dill
import numpy as np
import sys
import pathlib
import pandas as pd
import pickle
import time

sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')
from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqef_dynamic_utils
from uqef_dynamic.utils import create_stat_object


def read_output_files_uqef_dynamic(workingDir, qoi_string=None):
    dict_output_file_paths = utility.get_dict_with_output_file_paths_based_on_workingDir(workingDir)
    
    args_file = dict_output_file_paths.get("args_file")
    configuration_object_file = dict_output_file_paths.get("configuration_object_file")
    nodes_file = dict_output_file_paths.get("nodes_file")
    parameters_file = dict_output_file_paths.get("parameters_file")
    time_info_file = dict_output_file_paths.get("time_info_file")
    df_index_parameter_file = dict_output_file_paths.get("df_index_parameter_file")
    df_index_parameter_gof_file = dict_output_file_paths.get("df_index_parameter_gof_file")
    df_simulations_file = dict_output_file_paths.get("df_simulations_file")
    df_state_file = dict_output_file_paths.get("df_state_file")

    df_time_varying_grad_analysis_file = dict_output_file_paths.get("df_time_varying_grad_analysis_file")
    df_time_aggregated_grad_analysis_file = dict_output_file_paths.get("df_time_aggregated_grad_analysis_file")
    dict_approx_matrix_c_file = dict_output_file_paths.get("dict_approx_matrix_c_file")
    dict_matrix_c_eigen_decomposition_file = dict_output_file_paths.get("dict_matrix_c_eigen_decomposition_file")

    # Load the UQSim args dictionary
    uqsim_args_dict = utility.load_uqsim_args_dict(args_file)
    print("INFO: uqsim_args_dict: ", uqsim_args_dict)
    model = uqsim_args_dict["model"]

    # Load the configuration object
    configurationObject = utility.load_configuration_object(workingDir)
    print("configurationObject: ", configurationObject)
    simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)

    # Reading Nodes and Parameters
    with open(nodes_file, 'rb') as f:
        simulationNodes = pickle.load(f)
    print("INFO: simulationNodes: ", simulationNodes)
    dim = simulationNodes.nodes.shape[0]
    model_runs = simulationNodes.nodes.shape[1]
    distStandard = simulationNodes.joinedStandardDists
    dist = simulationNodes.joinedDists
    print(f"INFO: model-{model}; dim - {dim}; model_runs - {model_runs}")

    with open(time_info_file, 'r') as f:
        time_info = f.read()
    print("INFO: time_info: ", time_info)

    # Load the statistics object
    # statistics_dictionary_file = utility.get_dict_with_qoi_name_specific_output_file_paths_based_on_workingDir(\
    # workingDir, qoi_string=qoi_string)
    statisticsObject = create_stat_object.create_statistics_object(
        configuration_object=configurationObject, uqsim_args_dict=uqsim_args_dict, \
        workingDir=workingDir, model=model)
    statistics_dictionary = uqef_dynamic_utils.read_all_saved_statistics_dict(\
        workingDir=workingDir, list_qoi_column=statisticsObject.list_qoi_column, 
        single_timestamp_single_file=uqsim_args_dict.get("instantly_save_results_for_each_time_step", False), 
        throw_error=False
        )
    print(f"INFO: statistics_dictionary - {statistics_dictionary}")

    if df_index_parameter_file.is_file():
        df_index_parameter = pd.read_pickle(df_index_parameter_file, compression="gzip")
        params_list = utility._get_parameter_columns_df_index_parameter_gof(
            df_index_parameter)
        print(f"INFO: df_index_parameter - {df_index_parameter}")
    else:
        params_list = []
        for single_param in configurationObject["parameters"]:
            params_list.append(single_param["name"])
    print(f"INFO: params_list - {params_list} (note - all the parameters)")

    if df_index_parameter_gof_file.is_file():
        df_index_parameter_gof = pd.read_pickle(df_index_parameter_gof_file, compression="gzip")
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)
        print(f"INFO: df_index_parameter_gof - {df_index_parameter_gof}")
        print(f"INFO: gof_list - {gof_list}")

    if df_simulations_file.is_file():
        df_simulation_result = pd.read_pickle(df_simulations_file, compression="gzip")
        print(f"INFO: df_simulation_result - {df_simulation_result}")

    if df_state_file.is_file():
        df_state = pd.read_pickle(df_state_file, compression="gzip")
        print(f"INFO: df_state - {df_state}")

    gpce_surrogate_dictionary = uqef_dynamic_utils.read_all_saved_gpce_surrogate_models(
        workingDir=workingDir, list_qoi_column=statisticsObject.list_qoi_column, throw_error=False, convert_to_pd_timestamp=False)
    gpce_coeff_dictionary = uqef_dynamic_utils.read_all_saved_gpce_coeffs(
        workingDir=workingDir, list_qoi_column=statisticsObject.list_qoi_column, throw_error=False, convert_to_pd_timestamp=False)
    if gpce_surrogate_dictionary is not None:
        print(f"INFO: gpce_surrogate_dictionary - {gpce_surrogate_dictionary}")
    if gpce_coeff_dictionary is not None:
        print(f"INFO: gpce_coeff_dictionary - {gpce_coeff_dictionary}")


if __name__ == '__main__':
    workingDir = pathlib.Path('/gpfs/scratch/pr63so/ga45met2/hbvsask_runs/pce_deltq_3d_short_oldman')
    read_output_files_uqef_dynamic(workingDir)
