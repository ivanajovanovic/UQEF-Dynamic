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
from uqef_dynamic.utils import uqPostprocessing
from uqef_dynamic.utils import create_stat_object


def main(workingDir):
    dict_output_file_paths = utility.get_dict_with_output_file_paths_based_on_workingDir(workingDir)
    
    args_file = dict_output_file_paths.get("args_file")
    configuration_object_file = dict_output_file_paths.get("configuration_object_file")
    nodes_file = dict_output_file_paths.get("nodes_file")
    parameters_file = dict_output_file_paths.get("parameters_file")
    time_info_file = dict_output_file_paths.get("time_info_file")
    df_all_index_parameter_file = dict_output_file_paths.get("df_all_index_parameter_file")
    df_all_index_parameter_gof_file = dict_output_file_paths.get("df_all_index_parameter_gof_file")
    df_all_simulations_file = dict_output_file_paths.get("df_all_simulations_file")
    df_state_results_file = dict_output_file_paths.get("df_state_results_file")

    statistics_dictionary_file = dict_output_file_paths.get("statistics_dictionary_file")  # Note: this seems not to be relevant anymore

    df_time_varying_grad_analysis_file = dict_output_file_paths.get("df_time_varying_grad_analysis_file")
    df_time_aggregated_grad_analysis_file = dict_output_file_paths.get("df_time_aggregated_grad_analysis_file")
    dict_of_approx_matrix_c_file = dict_output_file_paths.get("dict_of_approx_matrix_c_file")
    dict_of_matrix_c_eigen_decomposition_file = dict_output_file_paths.get("dict_of_matrix_c_eigen_decomposition_file")

    # Load the UQSim args dictionary
    # uqsim_args_dict = utility.load_uqsim_args_dict(args_file)
    with open(args_file, 'rb') as f:
        uqsim_args = pickle.load(f)
    uqsim_args_dict = vars(uqsim_args)
    print("INFO: uqsim_args_dict: ", uqsim_args_dict)
    model = uqsim_args_dict["model"]

    # Load the configuration object
    # configuration_object = utility.load_configuration_object(workingDir)
    with open(configuration_object_file, 'rb') as f:
        configurationObject = dill.load(f)
    print("configurationObject: ", configurationObject)

    # Reading Nodes and Parameters
    with open(nodes_file, 'rb') as f:
    #     simulationNodes = dill.load(f)
        simulationNodes = pickle.load(f)
    print("INFO: simulationNodes: ", simulationNodes)
    dim = simulationNodes.nodes.shape[0]
    model_runs = simulationNodes.nodes.shape[1]
    dist = simulationNodes.joinedStandardDists
    print(f"INFO: model-{model}; dim - {dim}; model_runs - {model_runs}")

    with open(time_info_file, 'r') as f:
        time_info = f.read()
    print("INFO: time_info: ", time_info)

    # Load the statistics object
    statisticsObject = create_stat_object.create_statistics_object(
        configurationObject, uqsim_args_dict, workingDir, model=model)
    statistics_dictionary = uqPostprocessing.read_all_saved_statistics_dict(\
        workingDir, statisticsObject.list_qoi_column, uqsim_args_dict.get("instantly_save_results_for_each_time_step", False), throw_error=False)
    print(f"INFO: statistics_dictionary - {statistics_dictionary}")


    if df_all_index_parameter_file.is_file():
        df_index_parameter = pd.read_pickle(df_all_index_parameter_file, compression="gzip")
        params_list = utility._get_parameter_columns_df_index_parameter_gof(
            df_index_parameter)
        print(f"INFO: df_index_parameter - {df_index_parameter}")
        print(f"INFO: params_list - {params_list}")

    if df_all_index_parameter_gof_file.is_file():
        df_index_parameter_gof = pd.read_pickle(df_all_index_parameter_gof_file, compression="gzip")
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)
        print(f"INFO: df_index_parameter_gof - {df_index_parameter_gof}")
        print(f"INFO: gof_list - {gof_list}")

    if df_all_simulations_file.is_file():
        df_simulation_result = pd.read_pickle(df_all_simulations_file, compression="gzip")
        print(f"INFO: df_simulation_result - {df_simulation_result}")

    if df_state_results_file.is_file():
        df_state = pd.read_pickle(df_state_results_file, compression="gzip")
        print(f"INFO: df_state - {df_state}")

    gpce_surrogate_dictionary = uqPostprocessing.read_all_saved_gpce_surrogate_models(workingDir, statisticsObject.list_qoi_column, throw_error=False)
    gpce_coeff_dictionary = uqPostprocessing.read_all_saved_gpce_coeffs(workingDir, statisticsObject.list_qoi_column, throw_error=False)
    if gpce_surrogate_dictionary is not None:
        print(f"INFO: gpce_surrogate_dictionary - {gpce_surrogate_dictionary}")
    if gpce_coeff_dictionary is not None:
        print(f"INFO: gpce_coeff_dictionary - {gpce_coeff_dictionary}")


if __name__ == '__main__':
    workingDir = pathlib.Path('/gpfs/scratch/pr63so/ga45met2/hbvsask_runs/pce_deltq_3d_short_oldman')
    main(workingDir)
