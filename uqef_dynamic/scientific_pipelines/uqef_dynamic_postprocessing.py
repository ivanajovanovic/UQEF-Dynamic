"""
This file is used to compute the statistics for gPCE surrogate model; 
@author: Ivana Jovanovic Buha
"""
import inspect
import json
import os
import subprocess
from distutils.util import strtobool
import dill
import math
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import sys
import pathlib
import pandas as pd
import pickle
import time
from collections import defaultdict

# for parallel computing
import multiprocessing

# for message passing
from mpi4py import MPI
# import threading
import psutil

# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo

import chaospy as cp
import uqef

linux_cluster_run = True
# sys.path.insert(0, os.getcwd())
if linux_cluster_run:
    sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')
else:
    sys.path.insert(0, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')

from uqef_dynamic.utils import parallel_statistics
from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqef_dynamic_utils
from uqef_dynamic.utils import create_stat_object
from uqef_dynamic.utils import colors

#####################################
### MPI infos:
#####################################
# # Initialize MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# # Get the number of threads
# # num_threads = threading.active_count()

def read_surrogate_model_single_working_dir(workingDir, statisticsObject, single_timestamp_single_file,
surrogate_type, recompute_generalized_sobol_indices, polynomial_expansion=None, polynomial_norms=None):
    # ========================================================
    # Read the gPCE / KL surrogate model and its coefficients
    # ========================================================

    list_qois = statisticsObject.list_qoi_column
    statistics_pdTimesteps = statisticsObject.pdTimesteps

    gpce_surrogate_dict_over_qois= defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})
    gpce_coeff_dict_over_qois = defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})
    # kl_surrogate_dict_over_qois = defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})
    kl_coeff_dict_over_qois = {single_qoi:None for single_qoi in list_qois}
    kl_surrogate_df_dict_over_qois = {single_qoi:None for single_qoi in list_qois}

    for single_qoi in list_qois:
        gpce_surrogate = None
        gpce_coeffs = None

        if surrogate_type=='pce':
            gpce_surrogate = uqef_dynamic_utils.fetch_gpce_surrogate_single_qoi(
                qoi_column_name=single_qoi, workingDir=workingDir,
                statistics_dictionary=statisticsObject.result_dict, 
                throw_error=False, single_timestamp_single_file=single_timestamp_single_file)

            if gpce_surrogate is None or recompute_generalized_sobol_indices:
                gpce_coeffs = uqef_dynamic_utils.fetch_gpce_coeff_single_qoi(
                    qoi_column_name=single_qoi, workingDir=workingDir,
                    statistics_dictionary=statisticsObject.result_dict, 
                    throw_error=False, single_timestamp_single_file=single_timestamp_single_file)
                
                if gpce_surrogate is None and gpce_coeffs is None and recompute_generalized_sobol_indices:
                    raise Exception(f"Error - not possible to recompute generalized_sobol_indices when both \
                        gpce_surrogate and gpce_coeffs are missing")

                if gpce_surrogate is None and gpce_coeffs is not None:
                    gpce_surrogate = utility.build_gpce_surrogate_from_coefficients(
                        gpce_coeffs, polynomial_expansion, polynomial_norms)

                if recompute_generalized_sobol_indices:
                    if gpce_coeffs is None:
                        # TODO implement this
                        # gpce_coeffs = utility.compute_gpce_coefficients(
                        #     gpce_surrogate, polynomial_expansion, polynomial_norms)
                        raise NotImplementedError
                    if isinstance(gpce_coeffs, dict) and single_qoi in gpce_coeffs:
                        gpce_coeffs = gpce_coeffs[single_qoi]
                    gpce_coeff_dict_over_qois[single_qoi] = gpce_coeffs

            if isinstance(gpce_surrogate, dict) and single_qoi in gpce_surrogate:
                gpce_surrogate = gpce_surrogate[single_qoi]

            # Make sure that in the end gpce_surrogate is a dictionary ovet timestamps
            if isinstance(gpce_surrogate, np.ndarray) or isinstance(gpce_surrogate, Polynomial):
                gpce_surrogate = {0: gpce_surrogate}  # {statistics_pdTimesteps[0]: gpce_surrogate}

            if gpce_surrogate is None or not isinstance(gpce_surrogate, dict):
                raise Exception(f"Sorry but there is not gPCE model not saved or not of the required type!")

            if statistics_pdTimesteps!=list(gpce_surrogate.keys()):
                print("Watch-out - The timestamps of the statistics and the gPCE surrogate do not match!")
                statistics_pdTimesteps = list(gpce_surrogate.keys())

            # if printing:
            #     temp = gpce_surrogate[statistics_pdTimesteps[0]]
            #     print(f"Qoi - {single_qoi}\n gpce_surrogate for a first timestamp - {temp} \n")
            #     if gpce_coeffs and gpce_coeffs is not None:
            #         temp = gpce_coeffs[statistics_pdTimesteps[0]]
            #         print(f"Qoi - {single_qoi}\n gpce_coeffs - {temp} \n")

            gpce_surrogate_dict_over_qois[single_qoi] = gpce_surrogate

        elif surrogate_type=='kl+pce':
            # read eigenvalues
            fileName = f"eigenvalues_{single_qoi}.npy"
            # fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
            fullFileName = workingDir / fileName
            if fullFileName.is_file():
                eigenvalues = np.load(fullFileName)
                eigenvalues_real = np.asarray([element.real for element in eigenvalues], dtype=np.float64)
                eigenvalues_real_scaled = eigenvalues_real/eigenvalues_real[0]
            else:
                eigenvalues = None
                eigenvalues_real = None
                eigenvalues_real_scaled = None

            # read eigenvectors
            fileName = f"eigenvectors_{single_qoi}.npy"
            # fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
            fullFileName = workingDir / fileName
            if fullFileName.is_file():
                eigenvectors = np.load(fullFileName)
            else:
                eigenvectors = None

            if eigenvalues is None or eigenvectors is None:
                raise Exception("Sorry, \
                    evaluation of the  surrogate (type KL+PCE) can not be performed since eigenvalues or eigenvectors are not saved!")

            # reading the coefficents of PCE approax of the KL 
            fileName = f"f_kl_surrogate_coefficients_{single_qoi}.npy"
            # fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
            fullFileName = workingDir / fileName
            if fullFileName.is_file():
                gpce_coeffs = np.load(fullFileName, allow_pickle=True)
            else:
                gpce_coeffs = None

            if isinstance(gpce_coeffs, dict) and single_qoi in gpce_coeffs:
                gpce_coeffs = gpce_coeffs[single_qoi]
            kl_coeff_dict_over_qois[single_qoi] = gpce_coeffs

            # reading f_kl_surrogate_df
            fileName = f"f_kl_surrogate_df_{single_qoi}.pkl"
            # fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
            fullFileName = workingDir / fileName
            if fullFileName.is_file():
                f_kl_surrogate_df = pd.read_pickle(fullFileName, compression="gzip")
                # Iterating with iterrows()
                f_kl_surrogate_df['eigenvectors'] = None  # Initially, set to None or np.nan
                f_kl_surrogate_df['eigenvectors'] = f_kl_surrogate_df['eigenvectors'].astype(object)  # Explicitly set dtype to object
                for index, row in f_kl_surrogate_df.iterrows():
                    f_kl_surrogate_df.at[index, 'eigenvalues'] = eigenvalues[index]
                    f_kl_surrogate_df.at[index, 'eigenvectors'] = np.array(eigenvectors[:,index]) #eigenvectors[index]
                    # print(f"Index: {index}, gPCE: {type(row['gPCE'])}, gPCE: {row['gPCE'].shape}; gpce_coeff: {type(row['gpce_coeff'])} gpce_coeff: {row['gpce_coeff'].shape}")
            else:
                f_kl_surrogate_df = None
                if gpce_coeffs is not None:
                    pass
                    # gpce_surrogate = utility.build_gpce_surrogate_from_coefficients(
                    #     gpce_coeffs, polynomial_expansion, polynomial_norms)
                    # TODO populate f_kl_surrogate_df 
            kl_surrogate_df_dict_over_qois[single_qoi] = f_kl_surrogate_df
            
            if gpce_coeffs is None and f_kl_surrogate_df is None:
                raise Exception("Sorry, \
                    evaluation of the  surrogate (type KL+PCE) can not be performed since PCE surrogate of the KL modes and/or the coefficients were not saved!")

            # if printing:
            #     if gpce_coeffs is not None:
            #         print(f"kl_coeff_dict_over_qois[{single_qoi}] = {gpce_coeffs}")
            #         print(f"kl_coeff_dict_over_qois[{single_qoi}].shape = {gpce_coeffs.shape}")
            #     if f_kl_surrogate_df is not None:
            #         print(f"kl_surrogate_df_dict_over_qois[{single_qoi}] = {f_kl_surrogate_df}")

            # gpce_surrogate = utility.build_gpce_surrogate_from_coefficients(
            #             gpce_coeffs, polynomial_expansion, polynomial_norms)

            # kl_surrogate_dict_over_qois[single_qoi] = gpce_surrogate
        else:
            raise Exception(f"Sorry, the surrogate type {surrogate_type} is not implemented; it can be either 'pce' or 'kl+pce'!")

    return gpce_surrogate_dict_over_qois, gpce_coeff_dict_over_qois, kl_coeff_dict_over_qois, kl_surrogate_df_dict_over_qois


def _postprocess_kl_expansion_or_generalized_sobol_indices_computation_from_results_single_qoi(
    single_qoi_column, compute_generalized_sobol_indices_from_kl_expansion, compute_generalized_sobol_indices_over_time,
    workingDir, gpce_coeff_dict, gpce_surrogate_dict, polynomial_expansion, weights_time, labels, look_back_window_size, resolution):
    """
    Postprocesses the KL expansion or generalized Sobol indices (computed for the final timestamp or time-wise) 
    computation results for a single quantity of interest (QoI).

    Args:
        single_qoi_column (str): The name of the single quantity of interest (QoI) column.

    Returns:
        dicti over timestamps/or maybe containing info only for the last time stamp with the computed statistics 
        (f"generalized_sobol_total_index_{param_name}")
    """
    result_dict = {}

    dict_with_gpce_coeff_and_generalized_indices = {}
    for single_time_stamp in gpce_coeff_dict.keys():
        dict_with_gpce_coeff_and_generalized_indices[single_time_stamp] = {}
        dict_with_gpce_coeff_and_generalized_indices[single_time_stamp][utility.PCE_COEFF_ENTRY] = gpce_coeff_dict[single_time_stamp]
        # if gpce_surrogate_dict is not None and single_time_stamp in gpce_surrogate_dict:
        #     dict_with_gpce_coeff_and_generalized_indices[single_time_stamp][utility.PCE_ENTRY] = gpce_surrogate_dict[single_time_stamp]

    # recompute weights_time based on gpce_coeff_dict.keys()
    timesteps_max = max(gpce_coeff_dict.keys())
    timesteps_min = min(gpce_coeff_dict.keys())
    N_quad = len(list(gpce_coeff_dict.keys()))
    if N_quad > 1:
        # TODO Think about 1 in num?
        h = (timesteps_max - timesteps_min)/(N_quad-1) #1/(N_quad-1)
        weights_time = [h for i in range(N_quad)]
        if len(weights_time) >= 3:
            weights_time[0] /= 2
            weights_time[-1] /= 2
        weights_time = np.asarray(weights_time, dtype=np.float32)
    else:
        weights_time = np.asarray([1.0], dtype=np.float32)

    if compute_generalized_sobol_indices_from_kl_expansion:
        raise Exception("Sorry, recomputation of the generalized sobol indices bas on kl expansion is still not implemented!")
        # # TOOD implement this by propagating f_kl_surrogate_coefficients Var_kl_approx
        # fileName = workingDir / f"recomputed_generalized_sobol_indices_{single_qoi_column}.pkl"
        # param_name_generalized_sobol_total_indices = utility.computing_generalized_sobol_total_indices_from_kl_expan(
        #     f_kl_surrogate_coefficients, polynomial_expansion, weights_time, labels, fileName, total_variance=Var_kl_approx)
        # print(f"INFO: computation of generalized S.S.I based on KL+gPCE(MC) finished...")
        # last_time_step = max(gpce_coeff_dict.keys())  #last_time_step = list(gpce_coeff_dict.keys())[-1]
        # for param_name in labels:
        #     dict_with_gpce_coeff_and_generalized_indices[last_time_step][f"generalized_sobol_total_index_{param_name}"] = \
        #         param_name_generalized_sobol_total_indices[param_name]
    else: 
        fileName = workingDir / f"recomputed_generalized_sobol_indices_{single_qoi_column}.pkl"
        if compute_generalized_sobol_indices_over_time:
            utility.computing_generalized_sobol_total_indices_from_poly_expan_over_time(
                result_dict_statistics=dict_with_gpce_coeff_and_generalized_indices, 
                polynomial_expansion=polynomial_expansion, weights=weights_time, param_names=labels,
                fileName=fileName, look_back_window_size=look_back_window_size, resolution=resolution)
            print(f"INFO: computation of (over time) generalized S.S.I based on PCE finished...")
        else:
            # the computation of the generalized Sobol indices is done only for the last time step
            utility.computing_generalized_sobol_total_indices_from_poly_expan(
                result_dict_statistics=dict_with_gpce_coeff_and_generalized_indices, 
                polynomial_expansion=polynomial_expansion, 
                weights=weights_time, 
                param_names=labels,
                fileName=fileName)
            print(f"INFO: computation of (over time) generalized S.S.I based on PCE finished...")
    for single_time_stamp in dict_with_gpce_coeff_and_generalized_indices.keys():
        del dict_with_gpce_coeff_and_generalized_indices[single_time_stamp][utility.PCE_COEFF_ENTRY]
    return dict_with_gpce_coeff_and_generalized_indices

# TODO Change so there are alternative how to extract mean (i.e, beside via df_statistics_and_measured)
def evaluate_kl_and_pce_surrogate_model(kl_surrogate_df_dict_over_qois, list_qois, nodes, statistics_pdTimesteps, df_statistics_and_measured):
    """
    Evaluate the KL and PCE surrogate model for given inputs.

    Args:
        kl_surrogate_df_dict_over_qois (dict): A dictionary containing KL surrogate dataframes for each QoI.
        list_qois (list): A list of QoIs.
        nodes (numpy.ndarray): The nodes for evaluation.
        statistics_pdTimesteps (list): A list of timesteps for statistics.
        df_statistics_and_measured (pandas.DataFrame): The dataframe containing statistics (mean column/utility.MEAN_ENTRY!) and measured data.

    Returns:
        dict: A dictionary containing the evaluated surrogate model for each QoI and timestep.
    """
    surrogate_evaluated_dict_over_qois = {single_qoi:{} for single_qoi in list_qois}
    for single_qoi in list_qois:
        f_kl_surrogate_df = kl_surrogate_df_dict_over_qois[single_qoi]
        if f_kl_surrogate_df is None:
            raise Exception(f"Error - kl_surrogate_df_dict_over_qois[{single_qoi}] is None")
        surrogate_evaluated_dict_over_qois[single_qoi] = {}
        # matrix N_kl x N_nodes evaluation of the KL coefficents (PCE surrogates of the KL coefficents) at the nodes
        evaluation_of_kl_coefficients = np.empty((len(f_kl_surrogate_df), nodes.shape[1]))
        for index, row in f_kl_surrogate_df.iterrows():
            evaluation_of_kl_coefficients[index] = evaluate_gPCE_model_single_qoi_single_date(gPCE_model=row['gPCE'], nodes=nodes)

        for date_index, date in enumerate(statistics_pdTimesteps):
            list_of_values_over_nodes = []
            for node_idx in range(nodes.shape[1]):
                temp_value = df_statistics_and_measured.loc[\
                    (df_statistics_and_measured[utility.QOI_ENTRY] == single_qoi) & (df_statistics_and_measured[utility.TIME_COLUMN_NAME] == date)][utility.MEAN_ENTRY].values[0]
                for index, row in f_kl_surrogate_df.iterrows():
                    temp_value += evaluation_of_kl_coefficients[index, node_idx] * row['eigenvectors'][date_index]
                    #temp_value += evaluation_of_kl_coefficients[index, node_idx] * row['eigenvectors'][date_index]*np.sqrt(row['eigenvalues'])
                # TODO Do I want to compare temp_value with original model run?
                list_of_values_over_nodes.append(temp_value)
            surrogate_evaluated_dict_over_qois[single_qoi][date] = np.array(list_of_values_over_nodes)
    return surrogate_evaluated_dict_over_qois

def evaluate_gPCE_model_single_qoi_single_date(gPCE_model, nodes):
    return np.array(gPCE_model(*nodes))

# TODO add measured data to surrogate reevaluations
# TODO run model in parallel for new nodes; observe times
# TODO compare model runs vs surrogate runs - RMSE, MAE, etc., time, if measured available; observe times

# TODO add forcing data to surrogate reevaluations
# TODO when workingDir is a list - make one 'big' Statistics Object and then (re)compute / (re)plot statistics
# TODO recomputation of the generalized sobol indices based on pce surrogate...

def main(workingDir=None, inputModelDir=None, directory_for_saving_plots=None, 
    surrogate_type="pce", single_timestamp_single_file=False, printing=False, plotting=True, model=None, **kwargs):
    """
    Main function for computing the statistics for gPCE surrogate model
    The default is that the gPCE surrogate model and computed coefficients are saved in the corresponding files in the workingDir;  
    If that is not the case, then the function tries to recreate a Statistics Object and read them from the saved statistics dictionary
    """
    # TODO Parallelization does not seem to work properly - it is not faster than the serial version
    # Initialize MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start = time.time()

        dict_with_time_info = {}

        if workingDir is None:
            workingDir = pathlib.Path(os.getcwd())
        print(f"workingDir={workingDir}")

        add_measured_data=kwargs.get('add_measured_data', False)
        add_forcing_data=kwargs.get('add_forcing_data', False)
        read_saved_simulations = kwargs.get('read_saved_simulations', False)
        read_saved_states = kwargs.get('read_saved_states', False)
        set_up_statistics_from_scratch = kwargs.get('set_up_statistics_from_scratch', False)
        recompute_gpce = kwargs.get('recompute_gpce', False)
        recompute_statistics = kwargs.get('recompute_statistics', False)
        reevaluate_surrogate = kwargs.get('reevaluate_surrogate', False)
        reevaluate_original_model = kwargs.get('reevaluate_original_model', False)
        recompute_sobol_indices = kwargs.get('recompute_sobol_indices', False)
        recompute_generalized_sobol_indices = kwargs.get('recompute_generalized_sobol_indices', False)

        replot_statistics_from_statistics_object = kwargs.get('replot_statistics_from_statistics_object', False)
        dict_what_to_plot = kwargs.get('dict_what_to_plot', utility.DEFAULT_DICT_WHAT_TO_PLOT)
            
        inputModelDir_function_input_argument = inputModelDir  # This is because it will be overwritten by the inputModelDir from read_output_files_uqef_dynamic
        workingDir_function_input_argument = workingDir  # This is because it will be overwritten by the workingDir from read_output_files_uqef_dynamic, or maybe not

        set_lower_predictions_to_zero = kwargs.get('set_lower_predictions_to_zero', False)
        set_mean_prediction_to_zero = kwargs.get('set_mean_prediction_to_zero', False)
        correct_sobol_indices = kwargs.get('correct_sobol_indices', False)

        if isinstance(workingDir, list):
            df_statistics_and_measured_list = []
            statisticsObject_list = []
            list_of_list_qois = []
            list_of_statistics_pdTimesteps = []
            si_t_df_list = []
            si_m_df_list = []
            df_simulation_result_list = []
            df_index_parameter_list = []
            list_of_gpce_surrogate_dict_over_qois = []
            list_of_gpce_coeff_dict_over_qois = []

            dict_of_kl_coeff_dict_over_qois_over_workingDir = defaultdict(dict, {single_workingDir:{} for single_workingDir in workingDir})
            dict_of_kl_surrogate_df_dict_over_qois_over_workingDir =  defaultdict(dict, {single_workingDir:{} for single_workingDir in workingDir})

            for single_workingDir in workingDir:
                statisticsObject, df_statistics_and_measured,  si_t_df, si_m_df, df_simulation_result, df_index_parameter, _, uqsim_args_dict, simulationNodes= \
                create_stat_object.get_df_statistics_and_df_si_from_saved_files(
                    workingDir=single_workingDir, 
                    inputModelDir=inputModelDir_function_input_argument, 
                    set_lower_predictions_to_zero=set_lower_predictions_to_zero,
                    set_mean_prediction_to_zero=set_mean_prediction_to_zero,
                    correct_sobol_indices=correct_sobol_indices,
                    read_saved_simulations=read_saved_simulations, 
                    read_saved_states=read_saved_states, 
                    instantly_save_results_for_each_time_step=single_timestamp_single_file,
                    add_measured_data=add_measured_data,
                    add_forcing_data=add_forcing_data,
                    transform_measured_data_as_original_model=True,
                )
                statisticsObject_list.append(statisticsObject)
                if df_statistics_and_measured is not None:
                    df_statistics_and_measured_list.append(df_statistics_and_measured)
                if si_t_df is not None:
                    si_t_df_list.append(si_t_df)
                if si_m_df_list is not None:
                    si_m_df_list.append(si_m_df)
                if df_simulation_result is not None:
                    df_simulation_result_list.append(df_simulation_result)
                if df_index_parameter is not None:
                    df_index_parameter_list.append(df_index_parameter)

                list_of_list_qois.append(statisticsObject.list_qoi_column)
                list_of_statistics_pdTimesteps.append(statisticsObject.pdTimesteps)

                # ========================================================
                # Read the gPCE / KL surrogate model and its coefficients
                # ========================================================
                # statisticsObject.prepareForScStatistics(
                #     simulationNodes, order=uqsim_args_dict['sc_p_order'], 
                #     poly_normed=uqsim_args_dict['sc_poly_normed'], 
                #     poly_rule=uqsim_args_dict['sc_poly_rule'], 
                #     regression=uqsim_args_dict['regression'], 
                #     cross_truncation=uqsim_args_dict['cross_truncation']
                # )
                # polynomial_expansion = statisticsObject.polynomial_expansion
                # polynomial_norms = statisticsObject.polynomial_norms

                jointDists = simulationNodes.joinedDists
                jointStandard = simulationNodes.joinedStandardDists

                if statisticsObject.sampleFromStandardDist:
                    polynomial_expansion, polynomial_norms = cp.generate_expansion(
                        order=uqsim_args_dict['sc_p_order'], dist=jointStandard, rule=uqsim_args_dict['sc_poly_rule'], normed=uqsim_args_dict['sc_poly_normed'],
                        graded=True, reverse=True, cross_truncation=uqsim_args_dict['cross_truncation'], retall=True)
                else:
                    polynomial_expansion, polynomial_norms = cp.generate_expansion(
                        order=uqsim_args_dict['sc_p_order'], dist=jointDists, rule=uqsim_args_dict['sc_poly_rule'], normed=uqsim_args_dict['sc_poly_normed'],
                        graded=True, reverse=True, cross_truncation=uqsim_args_dict['cross_truncation'], retall=True)

                gpce_surrogate_dict_over_qois, gpce_coeff_dict_over_qois, kl_coeff_dict_over_qois, kl_surrogate_df_dict_over_qois = \
                    read_surrogate_model_single_working_dir(
                    workingDir, statisticsObject, single_timestamp_single_file,
                    surrogate_type, recompute_generalized_sobol_indices, polynomial_expansion, polynomial_norms)
                list_of_gpce_surrogate_dict_over_qois.append(gpce_surrogate_dict_over_qois)
                list_of_gpce_coeff_dict_over_qois.append(gpce_coeff_dict_over_qois)

                dict_of_kl_coeff_dict_over_qois_over_workingDir[single_workingDir] = kl_coeff_dict_over_qois
                dict_of_kl_surrogate_df_dict_over_qois_over_workingDir[single_workingDir] = kl_surrogate_df_dict_over_qois
 
            if model is None:
                model = uqsim_args_dict["model"]
            if model != uqsim_args_dict["model"]:
                raise Exception("The model is not the same for all working directories!")

            # TODO WIP
            # TODO Maybe to create one general statisticsObject - used later on for plotting, and statistics...
            # statistics_dictionary = join all the statistics_dictionary     
            #     single_statistics_dictionary = uqef_dynamic_utils.read_all_saved_statistics_dict(\
            #         workingDir=workingDir, list_qoi_column=statisticsObject.list_qoi_column, 
            #         single_timestamp_single_file=uqsim_args_dict.get("instantly_save_results_for_each_time_step", False), 
            #         throw_error=False
            #         )
            # uqef_dynamic_utils.extend_statistics_object(
            #     statisticsObject=statisticsObject, 
            #     statistics_dictionary=statistics_dictionary, 
            #     df_simulation_result=df_simulation_result,
            #     get_measured_data=add_measured_data, 
            #     get_unaltered_data=add_forcing_data
            # )
            # statisticsObject.set_df_statistics_and_measured()
            # df_statistics_and_measured = statisticsObject.merge_df_statistics_data_with_measured_and_forcing_data(
            #     add_measured_data=add_measured_data, add_forcing_data=add_forcing_data, transform_measured_data_as_original_model=transform_measured_data_as_original_model)

            list_qois = utility.find_overlap(list_of_list_qois)
            statistics_pdTimesteps = utility.find_overlap(list_of_statistics_pdTimesteps)
            
            if printing:
                print(f"list_qois - {list_qois}")
                print(f"statistics_pdTimesteps - {statistics_pdTimesteps}")

            gpce_surrogate_dict_over_qois= defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})
            gpce_coeff_dict_over_qois = defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})

            for single_qoi in list_qois:
                for d in list_of_gpce_surrogate_dict_over_qois:
                    gpce_surrogate_dict_over_qois[single_qoi].update(d[single_qoi])
                for d in list_of_gpce_coeff_dict_over_qois:
                    gpce_coeff_dict_over_qois[single_qoi].update(d[single_qoi])

            # if statistics_pdTimesteps!=list(gpce_surrogate_dict_over_qois[list_qois[0]].keys()):
            #     print("Watch-out - The timestamps of the statistics and the gPCE surrogate do not match!")
            #     statistics_pdTimesteps = list(gpce_surrogate_dict_over_qois[list_qois[0]].keys())

            if df_statistics_and_measured_list:
                df_statistics_and_measured = pd.concat(df_statistics_and_measured_list, ignore_index=True)
                df_statistics_and_measured = df_statistics_and_measured.sort_values(by=utility.TIME_COLUMN_NAME)
                df_statistics_and_measured_generalized = df_statistics_and_measured[[col for col \
                    in df_statistics_and_measured.columns if col.startswith("generalized_sobol_total_index_")]]
                timestamp_min = df_statistics_and_measured[utility.TIME_COLUMN_NAME].min()
                timestamp_max = df_statistics_and_measured[utility.TIME_COLUMN_NAME].max()
                print(f"timestamp_min={timestamp_min}; timestamp_max={timestamp_max}")
                statistics_pdTimesteps =  sorted(df_statistics_and_measured[utility.TIME_COLUMN_NAME].unique()) #df_statistics_and_measured. # TODO!!!
            else:
                df_statistics_and_measured = None
                df_statistics_and_measured_generalized = None
                timestamp_min = None
                timestamp_max = None
                statistics_pdTimesteps = None # TODO Maybe infere it in some other way...
            if si_m_df_list:
                si_m_df = pd.concat(si_m_df_list, ignore_index=True)
                si_m_df = si_m_df.sort_values(by=utility.TIME_COLUMN_NAME)
            else:
                si_m_df = None
            if si_t_df_list:
                si_t_df = pd.concat(si_t_df_list, ignore_index=True)
                si_t_df = si_t_df.sort_values(by=utility.TIME_COLUMN_NAME)
            else:
                si_t_df = None  
            if df_simulation_result_list:
                df_simulation_result = pd.concat(df_simulation_result_list, ignore_index=True)
                df_simulation_result = df_simulation_result.sort_values(by=utility.TIME_COLUMN_NAME)
            # TODO I neeed one general simulationNodes and statisticsObject!!!
            # simulationNodes = None

        else:
            # ============================================================
            if set_up_statistics_from_scratch:
                raise NotImplementedError
            # TODO add option to specify config_file/configurationObject and args_file/uqsim_args_dict
            # set df_simulation_result=None; create also simulationNodes
            # and to procedee with a new simulation, i.e, not to read from the saved one...
            # ============================================================
            # Get all files /data/ dataframe / paths saved by UQEF-Dynamic        
            results_dict = uqef_dynamic_utils.read_output_files_uqef_dynamic(
                workingDir, read_saved_simulations=read_saved_simulations)

            # This code will see the following variables retuned in results_dict:
            # workingDir, args_files, uqsim_args_dict, model, inputModelDir, configurationObject
            # simulation_settings_dict, simulationNodes, time_info, params_list, df_index_parameter, 
            # df_index_parameter_gof, gof_list, df_simulation_result, df_state, time_model_simulations,
            # time_computing_statistics, parameterNames, stochasticParameterNames, number_full_model_evaluations, 
            # full_number_quadrature_points, ...
            for key, value in results_dict.items():
                print(f"DEBUGGING {key}-{value}")
                # globals()[key] = value
                # locals()[key] = value
                workingDir = results_dict['workingDir']
                args_files = results_dict['args_files']
                uqsim_args_dict = results_dict['uqsim_args_dict']
                model = results_dict['model']
                inputModelDir = results_dict['inputModelDir']
                configurationObject = results_dict['configurationObject']
                simulation_settings_dict = results_dict['simulation_settings_dict']
                time_info = results_dict['time_info']
                simulationNodes = results_dict['simulationNodes']
                dim = results_dict['dim']
                df_index_parameter = results_dict['df_index_parameter']
                params_list = results_dict['params_list']
                df_index_parameter_gof = results_dict['df_index_parameter_gof']
                gof_list = results_dict['gof_list']
                df_simulation_result = results_dict['df_simulation_result']
                df_state = results_dict['df_state']
                time_model_simulations = results_dict['time_model_simulations']
                time_computing_statistics = results_dict['time_computing_statistics']

                stochasticParameterNames = results_dict['stochasticParameterNames']
                number_full_model_evaluations = results_dict['number_full_model_evaluations']
                variant = results_dict['variant']
                # there are more entries in results_dict added by uqef_dynamic_utils.update_dict_with_results_of_interest_based_on_uqsim_args_dict

            if printing:
                print(f"TIME INFO FROM STATISTICS - {time_info}")

            # This is due to the fact that sometimes instantly_save_results_for_each_time_step variable/fleg tends to be overwritten in the Statistics class, due to different set-ups
            instantly_save_results_for_each_time_step = single_timestamp_single_file
            if single_timestamp_single_file is None:
                single_timestamp_single_file = uqsim_args_dict["instantly_save_results_for_each_time_step"] 
            uqsim_args_dict["instantly_save_results_for_each_time_step"] = single_timestamp_single_file

            if inputModelDir_function_input_argument is not None:
                if inputModelDir_function_input_argument != uqsim_args_dict["inputModelDir"]:
                    uqsim_args_dict["inputModelDir"] = pathlib.Path(inputModelDir_function_input_argument)
            else:
                inputModelDir_function_input_argument = uqsim_args_dict["inputModelDir"]
            inputModelDir_function_input_argument = pathlib.Path(inputModelDir_function_input_argument)
            inputModelDir = inputModelDir_function_input_argument

            if model is None:
                model = uqsim_args_dict["model"]
            if model != uqsim_args_dict["model"]:
                raise Exception("The model is not the same for all working directories!")

            statisticsObject = create_stat_object.create_and_extend_statistics_object(
                configurationObject, uqsim_args_dict, workingDir, model, 
                df_simulation_result=df_simulation_result
            )

            # Add measured Data and/or forcing
            # This might be relevant for plotting in the end
            df_statistics_and_measured = statisticsObject.merge_df_statistics_data_with_measured_and_forcing_data(
                add_measured_data=add_measured_data, 
                add_forcing_data=add_forcing_data, 
                transform_measured_data_as_original_model=True)

            list_qois = statisticsObject.list_qoi_column
            statistics_pdTimesteps = statisticsObject.pdTimesteps

            si_t_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_t")
            si_m_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_m")

            if si_m_df is not None:
                si_m_df.sort_values(by=statisticsObject.time_column_name, ascending=True, inplace=True)
                if correct_sobol_indices:
                    si_columns_to_plot = [x for x in si_m_df.columns.tolist() if x != 'measured' \
                                                and x != 'measured_norm' and x != 'qoi' and x!= statisticsObject.time_column_name]
                    for single_column in si_columns_to_plot: 
                        si_m_df[single_column] = si_m_df[single_column].apply(lambda x: max(0, x))
            if si_t_df is not None:
                si_t_df.sort_values(by=statisticsObject.time_column_name, ascending=True, inplace=True)
                if correct_sobol_indices:
                    si_columns_to_plot = [x for x in si_t_df.columns.tolist() if x != 'measured' \
                                                and x != 'measured_norm' and x != 'qoi' and x!= statisticsObject.time_column_name]
                    for single_column in si_columns_to_plot: 
                        si_t_df[single_column] = si_t_df[single_column].apply(lambda x: max(0, x))

            # ==========================================
            # All above can be computed in this way as well..
            # statisticsObject, df_statistics_and_measured,  si_t_df, si_m_df, df_simulation_result, df_index_parameter, _, uqsim_args_dict, simulationNodes= \
            #     create_stat_object.get_df_statistics_and_df_si_from_saved_files(
            #         workingDir, 
            #         inputModelDir=inputModelDir, 
            #         set_lower_predictions_to_zero=set_lower_predictions_to_zero,
            #         set_mean_prediction_to_zero=set_mean_prediction_to_zero,
            #         correct_sobol_indices=correct_sobol_indices,
            #         read_saved_simulations=read_saved_simulations, 
            #         read_saved_states=read_saved_states, 
            #         instantly_save_results_for_each_time_step=single_timestamp_single_file,
            #         add_measured_data=add_measured_data,
            #         add_forcing_data=add_forcing_data,
            #         transform_measured_data_as_original_model=True,
            #     )

            # ==========================================

            if set_lower_predictions_to_zero:
                if 'E_minus_std' in df_statistics_and_measured:
                    df_statistics_and_measured['E_minus_std'] = df_statistics_and_measured['E_minus_std'].apply(lambda x: max(0, x))        
                if 'E_minus_2std' in df_statistics_and_measured:
                    df_statistics_and_measured['E_minus_2std'] = df_statistics_and_measured['E_minus_2std'].apply(lambda x: max(0, x))        
                if 'P10' in df_statistics_and_measured:
                    df_statistics_and_measured['P10'] = df_statistics_and_measured['P10'].apply(lambda x: max(0, x))
            if set_mean_prediction_to_zero:
                df_statistics_and_measured['E'] = df_statistics_and_measured['E'].apply(lambda x: max(0, x)) 

            # ==========================================
            statisticsObject.prepareForScStatistics(
                simulationNodes, order=uqsim_args_dict['sc_p_order'], 
                poly_normed=uqsim_args_dict['sc_poly_normed'], 
                poly_rule=uqsim_args_dict['sc_poly_rule'], 
                regression=uqsim_args_dict['regression'], 
                cross_truncation=uqsim_args_dict['cross_truncation']
            )
            polynomial_expansion = statisticsObject.polynomial_expansion
            polynomial_norms = statisticsObject.polynomial_norms

            # ========================================================
            # Read the gPCE / KL surrogate model and its coefficients
            # ========================================================
            gpce_surrogate_dict_over_qois, gpce_coeff_dict_over_qois, kl_coeff_dict_over_qois, kl_surrogate_df_dict_over_qois = \
                read_surrogate_model_single_working_dir(
                workingDir, statisticsObject, single_timestamp_single_file,
                surrogate_type, recompute_generalized_sobol_indices, polynomial_expansion, polynomial_norms)

        # ==========================================

        dict_set_lower_predictions_to_zero = kwargs.get('dict_set_lower_predictions_to_zero', {})
        dict_set_mean_prediction_to_zero = kwargs.get('dict_set_mean_prediction_to_zero', {})
        dict_set_lower_predictions_to_zero = utility.process_dict_set_predictions_to_zero(dict_set_lower_predictions_to_zero, list_qois) 
        dict_set_mean_prediction_to_zero = utility.process_dict_set_predictions_to_zero(dict_set_mean_prediction_to_zero, list_qois) 

        # ==========================================
        # print(f"DEBUGGING - statisticsObject.result_dict.keys()={statisticsObject.result_dict.keys()}")  # should be a list of list_qoi_column
        # for single_qoi in statisticsObject.list_qoi_column:
        #     print(f"DEBUGGING - single_qoi={single_qoi}; statisticsObject.result_dict[single_qoi].keys()={statisticsObject.result_dict[single_qoi].keys()}")

        if printing:
            print(f"inputModelDir from model - {inputModelDir}; inputModelDir_function_input_argument-{inputModelDir_function_input_argument}")
            print(f"workingDir from model - {workingDir}; workingDir_function_input_argument-{workingDir_function_input_argument}")

        # if printing:
        #     print(f"statisticsObject.df_statistics-{statisticsObject.df_statistics}")
        #     print(f"statisticsObject.forcing_df-{statisticsObject.forcing_df}")
        #     print(f"statisticsObject.df_measured-{statisticsObject.df_measured}")
        #     print(f"df_statistics_and_measured-{df_statistics_and_measured}")
        #     print(f"df_simulation_result-{df_simulation_result}")
        #     print(f"statistics_pdTimesteps-{statistics_pdTimesteps}")

        # ========================================================
        # TODO This part depends on existance of the simulationNodes object and statisticsObject!!!
        # ========================================================
        jointDists = simulationNodes.joinedDists
        jointStandard = simulationNodes.joinedStandardDists
        evaluateSurrogateAtStandardDist = statisticsObject.sampleFromStandardDist  # uqsim_args_dict['sampleFromStandardDist']

        if reevaluate_surrogate or reevaluate_original_model:
            number_of_samples = kwargs.get('number_of_samples', 1000)
            sampling_rule = kwargs.get('sampling_rule', "random")
            sample_new_nodes_from_standard_dist = kwargs.get('sample_new_nodes_from_standard_dist', True)
            read_new_nodes_from_file = kwargs.get('read_new_nodes_from_file', False)
            rounding = kwargs.get('rounding', False)
            round_dec = kwargs.get('round_dec', 4)
            
            parameters = uqef_dynamic_utils.generate_parameters_for_mc_simulation(
                jointDists=jointDists, numSamples=number_of_samples, rule=sampling_rule,
                sampleFromStandardDist=sample_new_nodes_from_standard_dist, 
                read_nodes_from_file=read_new_nodes_from_file, rounding=rounding, round_dec=round_dec,
            )
            if evaluateSurrogateAtStandardDist and jointStandard is not None:
                nodes = utility.transformation_of_parameters(
                    parameters, jointDists, jointStandard)
            else:
                nodes = parameters

            list_unique_generate_new_samples = range(parameters.shape[1]) # unique index for each sample
            number_of_samples_generated = parameters.shape[1]

            if printing:
                print(f"nodes.shape={nodes.shape}")

        if recompute_statistics:
            compute_other_stat_besides_pce_surrogate = kwargs.get('compute_other_stat_besides_pce_surrogate', False)
            compute_Sobol_t = kwargs.get('compute_Sobol_t', False)
            compute_Sobol_m = kwargs.get('compute_Sobol_m', False)
            compute_Sobol_m2 = kwargs.get('compute_Sobol_m2', False)
            dict_stat_to_compute = kwargs.get('dict_stat_to_compute', utility.DEFAULT_DICT_STAT_TO_COMPUTE)
        
        # ========================================================

        end_time_reading_all_saved_data = time.time()
        dict_with_time_info["time_reading_all_saved_data"] = end_time_reading_all_saved_data - start

        start_time_parallel_computing = time.time()

    else:
        list_qois = None
        gpce_surrogate_dict_over_qois= None
        gpce_coeff_dict_over_qois = None
        nodes = None
        parameters = None
        list_unique_generate_new_samples = None
        number_of_samples = None
        statistics_pdTimesteps = None
        workingDir = None
        reevaluate_surrogate = False
        reevaluate_original_model = None
        recompute_statistics = False
        jointDists = None
        jointStandard = None
        evaluateSurrogateAtStandardDist = None
        compute_other_stat_besides_pce_surrogate = None
        compute_Sobol_t = None
        compute_Sobol_m = None
        compute_Sobol_m2 = None
        dict_stat_to_compute = None
        configurationObject = None
        uqsim_args_dict = None

    list_qois = comm.bcast(list_qois, root=0)
    gpce_surrogate_dict_over_qois = comm.bcast(gpce_surrogate_dict_over_qois, root=0)
    
    statistics_pdTimesteps = comm.bcast(statistics_pdTimesteps, root=0)
    dates_to_process = statistics_pdTimesteps
    workingDir = comm.bcast(workingDir, root=0)

    # TODO This is probably unnecessary to broadcast...
    reevaluate_surrogate = comm.bcast(reevaluate_surrogate, root=0)
    recompute_statistics = comm.bcast(recompute_statistics, root=0)
    reevaluate_original_model = comm.bcast(reevaluate_original_model, root=0)

    # Split the dates among processes
    chunk_size = len(dates_to_process) // size
    remainder = len(dates_to_process) % size
    start_index = rank * chunk_size
    end_index = start_index + chunk_size if rank < size - 1 else len(dates_to_process)
    # end_index = (rank + 1) * chunk_size + (1 if rank < remainder else 0)

    # Distribute dates to processes
    my_dates = dates_to_process[start_index:end_index]

    # Split the nodes/parametres among processes & Distribute the nodes/parametres to processes
    # if reevaluate_original_model:
    nodes = comm.bcast(nodes, root=0)
    parameters = comm.bcast(parameters, root=0)
    
    # We need this to recreate model object
    configurationObject = comm.bcast(configurationObject, root=0)  #TODO Problem when workingDir is a list
    uqsim_args_dict = comm.bcast(uqsim_args_dict, root=0)
    list_unique_generate_new_samples = range(parameters.shape[1]) # unique index for each sample
    number_of_samples = parameters.shape[1]
    
    # Split the parameters among processes
    chunk_size = len(list_unique_generate_new_samples) // size
    remainder = len(list_unique_generate_new_samples) % size
    start_index = rank * chunk_size
    end_index = start_index + chunk_size if rank < size - 1 else len(list_unique_generate_new_samples)
    my_parametres = parameters[:, start_index:end_index]
    my_nodes = nodes[:, start_index:end_index]
    my_indices = list(range(start_index, end_index))
    # maybe I need also access to workingDir, args_files, uqsim_args_dict, model, inputModelDir, configurationObject \
    # and simulation_settings_dict to be able to create my own model object...

    memory_usage_history = []

    # ============================
    # Computation to be done in parallel
    # ============================

    if recompute_statistics:
        if rank == 0:
            start_time_recompute_statistics = time.time()
            print(f"Recomputing the statistics...")
        jointDists = comm.bcast(jointDists, root=0)
        jointStandard = comm.bcast(jointStandard, root=0)
        evaluateSurrogateAtStandardDist = comm.bcast(evaluateSurrogateAtStandardDist, root=0)

        compute_other_stat_besides_pce_surrogate = comm.bcast(compute_other_stat_besides_pce_surrogate, root=0)
        compute_Sobol_t = comm.bcast(compute_Sobol_t, root=0)
        compute_Sobol_m = comm.bcast(compute_Sobol_m, root=0)
        compute_Sobol_m2 = comm.bcast(compute_Sobol_m2, root=0)
        dict_stat_to_compute = comm.bcast(dict_stat_to_compute, root=0)

        # Start the computation of the additional statistics based on the save gPCE surrogate model
        if evaluateSurrogateAtStandardDist:
            distStandard = jointStandard
        else:
            distStandard = jointDists

        gpce_statistics_dict_over_qois = {single_qoi:{} for single_qoi in list_qois}
        for single_qoi in list_qois:
            gpce_statistics_dict_over_qois[single_qoi] = {}
            for date in my_dates:
                local_result_dict = {}
                parallel_statistics.calculate_stats_gpce(
                    local_result_dict=local_result_dict, qoi_gPCE=gpce_surrogate_dict_over_qois[single_qoi][date], 
                    dist=distStandard, compute_other_stat_besides_pce_surrogate=compute_other_stat_besides_pce_surrogate,
                    compute_Sobol_t=compute_Sobol_t, compute_Sobol_m=compute_Sobol_m, compute_Sobol_m2=compute_Sobol_m2, dict_stat_to_compute=dict_stat_to_compute)
                gpce_statistics_dict_over_qois[single_qoi][date] = local_result_dict

        all_results_gpce_statistics_recomputed = comm.gather(gpce_statistics_dict_over_qois, root=0)
        if rank == 0:
            end_time_recompute_statistics = time.time()
            print(f"Done with recomputing the statistics...")
            dict_with_time_info["time_paralle_statistics_recomputation"] = end_time_recompute_statistics - start_time_recompute_statistics

    # TODO This will change in case workingDir is a list
    if reevaluate_original_model:
        if rank == 0:
            start_time_reevaluating_original_model = time.time()
            print(f"Reevaluating the original model...")
        # create model; TODO think about new workingDir/directory_for_saving_plots
        modelObject = create_stat_object.create_model_object(
                configuration_object=configurationObject, uqsim_args_dict=uqsim_args_dict, workingDir=directory_for_saving_plots, model=None, 
                time_column_name=utility.TIME_COLUMN_NAME, index_column_name=utility.INDEX_COLUMN_NAME
            )
        results = modelObject(i_s=my_indices, parameters=my_parametres.T, raise_exception_on_model_break=False)
        # print(f"Results - {results}")
        # # processing the collected result
        df_simulation_result_local, df_index_parameter_local, _, _, _, _ =  uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe(\
            results_array=results, extract_only_qoi_columns=False, qoi_columns=modelObject.list_qoi_column, 
            time_column_name=utility.TIME_COLUMN_NAME, index_column_name=utility.INDEX_COLUMN_NAME)  #time_column_name=modelObject.time_column_name, index_column_name= modelObject.index_column_name
        # print(f"Rank {rank} - Final df_index_parameter_local - {df_index_parameter_local}")
        # print(f"Rank {rank} - Final DF - {df_simulation_result_local}")
        # Gather results from all processes
        all_df_index_parameter_model_evaluated = comm.gather(df_index_parameter_local, root=0)
        all_results_model_evaluated = comm.gather(df_simulation_result_local, root=0)
        
        if rank == 0:
            end_time_reevaluating_original_model = time.time()
            dict_with_time_info["time_parallel_original_model_reevaluations"] = end_time_reevaluating_original_model - start_time_reevaluating_original_model
        
        # if reevaluate_surrogate:
        #     nodes = comm.bcast(nodes, root=0)
        #     if rank == 0:
        #         start_time_reevaluating_reevaluate_surrogate = time.time()
        #         # combin all_df_index_parameter_values_model_evaluated into one
        #         df_index_parameter_reevaluated = pd.concat(all_df_index_parameter_model_evaluated, ignore_index=True, sort=False, axis=0)
        #         # filter only those rows wit 'successful_run' column being True
        #         if 'successful_run' in df_index_parameter_reevaluated.columns:
        #             df_index_parameter_reevaluated = df_index_parameter_reevaluated[df_index_parameter_reevaluated['successful_run']==True]
            
        #     if surrogate_type=='pce':
        #         print(f"Reevaluating the surrogate model...")
        #         # TODO Populate all_results_surrogate_evaluated

    if reevaluate_surrogate:  #and not reevaluate_original_model
        if rank == 0:
            start_time_reevaluating_reevaluate_surrogate = time.time()
            print(f"Reevaluating the surrogate model...")
        if surrogate_type=='pce':
            gpce_surrogate_evaluated_dict_over_qois = {single_qoi:{} for single_qoi in list_qois} #defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})

            # Monitor memory usage at regular intervals
            # Process dates
            for single_qoi in list_qois:
                gpce_surrogate_evaluated_dict_over_qois[single_qoi] = {}
                for date in my_dates:
                    gpce_surrogate_evaluated_dict_over_qois[single_qoi][date] = \
                        evaluate_gPCE_model_single_qoi_single_date(
                            gPCE_model=gpce_surrogate_dict_over_qois[single_qoi][date], nodes=nodes)

            # Gather results from all processes
            all_results_surrogate_evaluated = comm.gather(gpce_surrogate_evaluated_dict_over_qois, root=0)          
    
    # Query memory usage and record it
    memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # Memory usage in MB
    memory_usage_history.append(memory_usage)
    
    # ============================
    # Combine results on rank 0; or do some computation on rank 0
    # ============================

    if rank == 0:
        # TODO Do modification to the mean and surrogate model evaluations
        #  if the autoregressive mode is activated
        
        # ========================================================
        # Finalize recomputation of the original model
        # ========================================================
        if reevaluate_original_model:
            df_model_reevaluated = pd.concat(all_results_model_evaluated, ignore_index=True, sort=False, axis=0)
            df_index_parameter_reevaluated = pd.concat(all_df_index_parameter_model_evaluated, ignore_index=True, sort=False, axis=0)
            if printing:
                print(f"df_model_reevaluated={df_model_reevaluated}")
                print(f"df_index_parameter_reevaluated={df_index_parameter_reevaluated}")
            
            # Post process model evaluations if necessary
            # TODO add option to propagate dictionary storing boolean value for each single qoi
            if set_lower_predictions_to_zero:
                for single_qoi in list_qois:
                    if single_qoi in df_model_reevaluated:
                        df_model_reevaluated[single_qoi] = df_model_reevaluated[single_qoi].apply(lambda x: max(0, x)) 

            # Save newly created DataFrame with the (re)evaluated original model
            if df_model_reevaluated is not None:
                df_model_reevaluated.to_pickle(
                    os.path.abspath(os.path.join(str(directory_for_saving_plots), "df_model_reevaluated.pkl")), compression="gzip")
            if df_index_parameter_reevaluated is not None:
                df_index_parameter_reevaluated.to_pickle(
                    os.path.abspath(os.path.join(str(directory_for_saving_plots), "df_index_parameter_reevaluated.pkl")), compression="gzip")
        # ========================================================
        # Finalize recomputation of the surrogate model
        # ========================================================
        compare_surrogate_and_original_model_runs = False
        if reevaluate_original_model and reevaluate_surrogate and df_model_reevaluated is not None:
            compare_surrogate_and_original_model_runs = True

        if reevaluate_surrogate:
            if surrogate_type=='pce':
                combined_results = {single_qoi:{} for single_qoi in list_qois} #defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})

                for idx, result in enumerate(all_results_surrogate_evaluated):
                    for single_qoi in list_qois:
                        # print(f"{idx} - {single_qoi} - {list(result[single_qoi].keys())}")
                        combined_results[single_qoi].update(result[single_qoi])

                # surrogate_evaluated_dict_over_qois now contains the combined results for all dates processed in parallel on rank 0
                surrogate_evaluated_dict_over_qois = combined_results

                # TODO Add to model runs = Compare to model runs


                # if printing:
                #     for single_qoi in list_qois:
                #         print(f"type(surrogate_evaluated_dict_over_qois[{single_qoi}]) = {type(surrogate_evaluated_dict_over_qois[single_qoi])}")
                #         print(f"surrogate_evaluated_dict_over_qois[{single_qoi}] = {surrogate_evaluated_dict_over_qois[single_qoi]}")
                #         print(f"{list(surrogate_evaluated_dict_over_qois[single_qoi].keys())}")
                #         # print(f"surrogate_evaluated_dict_over_qois[{single_qoi}][{my_dates[0]}] = {surrogate_evaluated_dict_over_qois[single_qoi][my_dates[0]]}")
                #         #Timestamp('2007-08-14 00:00:00')
                end_time_parallel_computing = time.time()
                dict_with_time_info["time_parallel_pce_surrogate_reevaluations"] = end_time_parallel_computing - start_time_reevaluating_reevaluate_surrogate #start_time_parallel_computing

            elif surrogate_type=='kl+pce':
                surrogate_evaluated_dict_over_qois = evaluate_kl_and_pce_surrogate_model(
                    kl_surrogate_df_dict_over_qois, list_qois, nodes, statistics_pdTimesteps, df_statistics_and_measured)

                # if printing:
                #     for single_qoi in list_qois:
                #         print(f"surrogate_evaluated_dict_over_qois[{single_qoi}] = {surrogate_evaluated_dict_over_qois[single_qoi]}")
                #         keys_list = list(surrogate_evaluated_dict_over_qois[single_qoi].keys())
                #         print(f"{list(surrogate_evaluated_dict_over_qois[single_qoi].keys())}")
                #         print(f"surrogate_evaluated_dict_over_qois[{single_qoi}][keys[0]] = {surrogate_evaluated_dict_over_qois[single_qoi][keys_list[0]]}")
                end_time_parallel_computing = time.time()
                dict_with_time_info["time_kl_surrogate_reevaluations"] = end_time_parallel_computing - start_time_reevaluating_reevaluate_surrogate #start_time_parallel_computing
            else:
                raise Exception(f"Sorry, the surrogate type {surrogate_type} is not implemented; it can be either 'pce' or 'kl+pce'!")
                
            # Create a new DataFrame with the evaluated surrogate model
            dict_with_errors_over_qois = {}
            list_of_single_qoi_dfs = []
            for single_qoi in list_qois:
                # create a df from surrogate_evaluated_dict_over_qois[single_qoi] (dictionary over dates) containing the following columns:
                # utility.TIME_COLUMN_NAME, utility.INDEX_COLUMN_NAME, utility.QOI_COLUMN_NAME, utility.QOI_ENTRY
                records = []
                for date, array_of_values in surrogate_evaluated_dict_over_qois[single_qoi].items():
                    for i, single_surrogate_eval in enumerate(array_of_values):
                        records.append((date, i, single_surrogate_eval))
                # Step 2: Create a DataFrame from the records
                df_surrogate_reevaluated_single_qoi = pd.DataFrame(records, columns=[utility.TIME_COLUMN_NAME, utility.INDEX_COLUMN_NAME, utility.QOI_COLUMN_NAME])
                if df_surrogate_reevaluated_single_qoi is not None:
                    # Add qoi column if it does not exist
                    if utility.QOI_ENTRY not in df_surrogate_reevaluated_single_qoi.columns:
                        df_surrogate_reevaluated_single_qoi[utility.QOI_ENTRY] = single_qoi
                    # TODO add option to propagate dictionary storing boolean value for each single qoi
                    if set_lower_predictions_to_zero:
                        df_surrogate_reevaluated_single_qoi[utility.QOI_COLUMN_NAME] = df_surrogate_reevaluated_single_qoi[utility.QOI_COLUMN_NAME].apply(lambda x: max(0, x)) 
                    list_of_single_qoi_dfs.append(df_surrogate_reevaluated_single_qoi)
                    # Comparing with the original model run
                    if compare_surrogate_and_original_model_runs:
                        df_model_reevaluated_subset = df_model_reevaluated[[single_qoi, utility.TIME_COLUMN_NAME, utility.INDEX_COLUMN_NAME]]
                        merged_df = pd.merge(df_surrogate_reevaluated_single_qoi, df_model_reevaluated_subset, on=[utility.TIME_COLUMN_NAME, utility.INDEX_COLUMN_NAME])
                        # Calculate the squared error for each row
                        merged_df['squared_error'] = (merged_df[single_qoi] - merged_df[utility.QOI_COLUMN_NAME])**2

                        # Compute RMSE for each time step by grouping on 'time_column'
                        rmse_over_time = merged_df.groupby(utility.TIME_COLUMN_NAME)['squared_error'] \
                                                    .mean() \
                                                    .apply(np.sqrt) \
                                                    .reset_index(name='RMSE')
                        rmse_over_time.to_pickle(
                            os.path.abspath(os.path.join(str(directory_for_saving_plots), f"rmse_over_time_{single_qoi}.pkl")), compression="gzip")
                        
                        # Compute overall RMSE across all time steps
                        overall_linf = merged_df['squared_error'].max()
                        print(f"DEBUGGING - len(merged_df)-{len(merged_df)}")
                        overall_l2_scaled = np.sqrt(np.sum(merged_df['squared_error'])) / math.sqrt(len(merged_df))
                        overall_rmse = np.sqrt(merged_df['squared_error'].mean())
                        print(f"Overall errors for {single_qoi}: RMSE={overall_rmse}; 1-norm={overall_linf}; 2-norm={overall_l2_scaled}")
                        dict_with_errors_over_qois[single_qoi] = {}
                        dict_with_errors_over_qois[single_qoi]['overall_rmse'] = overall_rmse
                        dict_with_errors_over_qois[single_qoi]['overall_linf'] = overall_linf
                        dict_with_errors_over_qois[single_qoi]['overall_l2_scaled'] = overall_l2_scaled

            if list_of_single_qoi_dfs:
                df_surrogate_reevaluated= pd.concat(list_of_single_qoi_dfs, axis=0)
                df_surrogate_reevaluated.sort_values(by=utility.TIME_COLUMN_NAME, ascending=True, inplace=True)
            else:
                df_surrogate_reevaluated = None
            if printing:
                print(f"df_surrogate_reevaluated={df_surrogate_reevaluated}")

            # TODO Check if surrogate evaluations should be corrected - set to zero when smaller than zero?

            # Save newly created DataFrame with the evaluated surrogate model
            if df_surrogate_reevaluated is not None:
                df_surrogate_reevaluated.to_pickle(
                    os.path.abspath(os.path.join(str(directory_for_saving_plots), "df_surrogate_reevaluated.pkl")), compression="gzip")

        # ========================================================
        # Finalize recomputation of the statistics
        # ========================================================
        if recompute_statistics:
            combined_results = {single_qoi:{} for single_qoi in list_qois} #defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})
            for idx, result in enumerate(all_results_gpce_statistics_recomputed):
                for single_qoi in list_qois:
                    combined_results[single_qoi].update(result[single_qoi])

            gpce_statistics_dict_over_qois = combined_results

            # Check if the 'autoregressive mode' mode is activated - then (re)compute the mean value to correspond to the mean of final QoI
            # if strtobool(configurationObject["simulation_settings"]["autoregressive_model_first_order"]):
            #     previous_timestamp  = utility.compute_previous_timestamp(date, resolution="daily")
            #     temp_E = local_result_dict["E"]  # float(cp.E(temp_gpce_model, distStandard))
            #     temp_E += 0.8*df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==previous_timestamp]['measured'].values[0]
            #         # print(f"E original={local_result_dict['E']}")
            #         # print(f"Measured={df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==single_timestamp]['measured'].values[0]}")
            #     print(f"{single_qoi}-{single_timestamp}-E recomputed={temp_E}\n")

            list_of_single_qoi_dfs = []
            for single_qoi in list_qois:
                # Create a new DataFrame with the re-computed statistics
                df_recomputed_statistics_single_qoi = uqef_dynamic_utils.create_df_from_statistics_data_single_qoi(
                    stat_dict=gpce_statistics_dict_over_qois, 
                    qoi_column=single_qoi, 
                    list_of_uncertain_variables=statisticsObject.labels, 
                    measured_qoi_column=statisticsObject.dict_qoi_column_and_measured_info[single_qoi][1],
                    set_lower_predictions_to_zero=set_lower_predictions_to_zero, 
                    measured_fetched=statisticsObject.dict_qoi_column_and_measured_info[single_qoi][0], 
                    df_measured=statisticsObject.df_measured,
                    time_column_name=utility.TIME_COLUMN_NAME)
                if df_recomputed_statistics_single_qoi is not None:
                    if utility.QOI_ENTRY not in df_recomputed_statistics_single_qoi.columns:
                        df_recomputed_statistics_single_qoi[utility.QOI_ENTRY] = single_qoi
                    list_of_single_qoi_dfs.append(df_recomputed_statistics_single_qoi)
            if list_of_single_qoi_dfs:
                df_recomputed_statistics_and_measured = pd.concat(list_of_single_qoi_dfs, axis=0)
                df_recomputed_statistics_and_measured.sort_values(by=utility.TIME_COLUMN_NAME, ascending=True, inplace=True)
            else:
                df_recomputed_statistics_and_measured = None

            if df_recomputed_statistics_and_measured is not None:
                df_recomputed_statistics_and_measured.to_pickle(
                    os.path.abspath(os.path.join(str(directory_for_saving_plots), "df_recomputed_statistics_and_measured.pkl")), compression="gzip")

            # end_time_parallel_computing = time.time()
            # dict_with_time_info["time_paralle_statistics_recomputation"] = end_time_parallel_computing - start_time_parallel_computing

        # ========================================================
        # Finalize recomputation of the generalized Sobol indices
        # ========================================================
        if recompute_generalized_sobol_indices:
            start_time_generalized_si_recomputation = time.time()
            if statisticsObject.weights_time is None:
                # This is already called in create_stat_object.create_and_extend_statistics_object() -> \
                # uqef_dynamic_utils.extend_statistics_object -> statisticsObject.set_timesteps -> statisticsObject.set_weights_time
                statisticsObject.set_weights_time()

            # use gpce_coeff_dict_over_qois and gpce_surrogate_dict_over_qois; polynomial_expansion, polynomial_norms
            compute_generalized_sobol_indices_from_kl_expansion = kwargs.get('compute_generalized_sobol_indices_from_kl_expansion', False)
            compute_generalized_sobol_indices_over_time = kwargs.get('compute_generalized_sobol_indices_over_time', False)
            look_back_window_size = kwargs.get('look_back_window_size', 'whole')
            resolution = statisticsObject.resolution  #kwargs.get('resolution', 'integer')
            generalized_total_sobol_indices_dict_over_qois = {single_qoi:{} for single_qoi in list_qois}
            
            if printing:
                print(f"statisticsObject.weights_time-{statisticsObject.weights_time}")
                print(f"statisticsObject.labels-{statisticsObject.labels}")

            list_of_single_qoi_dfs = []
            for single_qoi in list_qois:
                # for date in statistics_pdTimesteps:
                dict_with_generalized_indices = _postprocess_kl_expansion_or_generalized_sobol_indices_computation_from_results_single_qoi(
                    single_qoi_column=single_qoi, 
                    compute_generalized_sobol_indices_from_kl_expansion=compute_generalized_sobol_indices_from_kl_expansion, 
                    compute_generalized_sobol_indices_over_time=compute_generalized_sobol_indices_over_time,
                    workingDir=directory_for_saving_plots, 
                    gpce_coeff_dict=gpce_coeff_dict_over_qois[single_qoi], 
                    gpce_surrogate_dict=gpce_surrogate_dict_over_qois[single_qoi], 
                    polynomial_expansion=polynomial_expansion,
                    weights_time=statisticsObject.weights_time, labels=statisticsObject.labels, 
                    look_back_window_size=look_back_window_size, resolution=resolution
                )
                generalized_total_sobol_indices_dict_over_qois[single_qoi] = dict_with_generalized_indices
                
                df_with_generalized_indices_single_qoi = uqef_dynamic_utils.create_df_from_generalized_total_sobol_indices_single_qoi(
                    stat_dict=dict_with_generalized_indices, 
                    qoi_column=single_qoi, 
                    list_of_uncertain_variables=statisticsObject.labels, 
                    measured_qoi_column=statisticsObject.dict_qoi_column_and_measured_info[single_qoi][1],
                    set_lower_predictions_to_zero=set_lower_predictions_to_zero, 
                    measured_fetched=statisticsObject.dict_qoi_column_and_measured_info[single_qoi][0], 
                    df_measured=df_statistics_and_measured, #statisticsObject.df_measured,
                    time_column_name=utility.TIME_COLUMN_NAME)
                
                if df_with_generalized_indices_single_qoi is not None:
                    if "qoi" not in df_with_generalized_indices_single_qoi.columns:
                        df_with_generalized_indices_single_qoi["qoi"] = single_qoi
                    list_of_single_qoi_dfs.append(df_with_generalized_indices_single_qoi)
                
                if printing:
                    print(f"generalized_total_sobol_indices_dict_over_qois for single qoi={single_qoi}- {dict_with_generalized_indices}")

            if list_of_single_qoi_dfs:
                df_generalized_total_sobol_indices_dict_and_measured = pd.concat(list_of_single_qoi_dfs, axis=0)
                df_generalized_total_sobol_indices_dict_and_measured.sort_values(by=utility.TIME_COLUMN_NAME, ascending=True, inplace=True)
            else:
                df_generalized_total_sobol_indices_dict_and_measured = None

            if printing:
                print(f"df_generalized_total_sobol_indices_dict_and_measured={df_generalized_total_sobol_indices_dict_and_measured}")

            if df_generalized_total_sobol_indices_dict_and_measured is not None:
                fileName = f"df_recomputed_generalized_total_sobol_indices_dict_and_measured.pkl"
                # if look_back_window_size != 'whole':
                fileName = f"df_recomputed_generalized_total_sobol_indices_dict_{look_back_window_size}_and_measured.pkl"
                df_generalized_total_sobol_indices_dict_and_measured.to_pickle(
                    os.path.abspath(os.path.join(str(directory_for_saving_plots), fileName)), compression="gzip")

            end_time_generalized_si_recomputation = time.time()
            dict_with_time_info["time_generalized_si_recomputation"] = end_time_generalized_si_recomputation - start_time_generalized_si_recomputation

            # Add (re-computed) generalized S.I to statisticsObject.result_dict      
            # if f'generalized_sobol_total_index_{self.labels[0]}' in self.result_dict[qoi_column][keyIter[-1]]:
            #     for i in range(len(self.labels)):
            #         name = f"generalized_sobol_total_index_{self.labels[i]}"
            #         generalized_sobol_total_index_values_temp = []
            #         at_least_one_entry_found = False
            #         for key in keyIter:
            #             if name in self.result_dict[qoi_column][key]:
            #                 at_least_one_entry_found = True
            #                 temp = self.result_dict[qoi_column][key][name]
            #                 generalized_sobol_total_index_values_temp.append(temp)
            #         if at_least_one_entry_found:
            #             list_of_columns_names.append(name)
            #             if len(generalized_sobol_total_index_values_temp)==1:
            #                 generalized_sobol_total_index_values_temp = generalized_sobol_total_index_values_temp[0]*len(keyIter)
            #             list_of_columns.append(generalized_sobol_total_index_values_temp)

    # Print memory usage history
    print(f"Memory Usage History (Rank {rank}): {memory_usage_history}")
    
    # ============================
    # Plotting and final post-processing in the main process
    # ============================
    # TODO Change when workingDir is a list - statisticsObject.df_measured; Things depend again on statisticsObject
    if rank == 0:
        end = time.time()
        runtime = end - start
        
        dict_with_time_info["total_runtime"] = runtime

        print(f"INFO ABOUT THE RUNTIME - Number of MPI processes={size} runtime={runtime}")
        print(f"INFO ABOUT ALL THE RUNTIME - Number of MPI processes={size} dict_with_time_info={dict_with_time_info}")

        dict_with_time_info_path = directory_for_saving_plots / 'dict_with_time_info.txt'
        with open(dict_with_time_info_path, 'w') as text_file:
            for key, value in dict_with_time_info.items():
                text_file.write(f'{key}: {value}\n')
        
        if dict_with_errors_over_qois:
            dict_with_error_path = directory_for_saving_plots / 'errors_surrogate_vs_model_over_qois.txt'
            utility.write_dict_to_file(dict_with_errors_over_qois, filename=dict_with_error_path)

        if plotting:
            if directory_for_saving_plots is None:
                directory_for_saving_plots = workingDir

            if replot_statistics_from_statistics_object:
                for single_qoi in list_qois:
                    filename = directory_for_saving_plots / f"reploted_statistics_{single_qoi}.html"
                    fig = statisticsObject._plotStatisticsDict_plotly_single_qoi(
                        single_qoi_column=single_qoi, dict_time_vs_qoi_stat=None, 
                        window_title='Forward UQ & SA', 
                        filename=str(filename), display=False,
                        dict_what_to_plot=dict_what_to_plot,
                        measured=add_measured_data,
                        forcing=add_forcing_data,
                        )
                        # filename = pathlib.Path(filename)
                    filename_pdf = pathlib.Path(filename).with_suffix(".pdf")
                    fig.write_image(str(filename_pdf), format="pdf", width=1000,)

            # TODO Add measured and forcing data if available? 
            # TODO Add option to plot model re-runs...
            # Plotting Model and Surrogate Model Runs + Some Statistics
            if reevaluate_surrogate and df_surrogate_reevaluated is not None:
                for single_qoi in list_qois:
                    df_surrogate_reevaluated_single_qoi = df_surrogate_reevaluated.loc[df_surrogate_reevaluated[utility.QOI_ENTRY]==single_qoi]
                    number_unique_surrogate_model_runs = df_surrogate_reevaluated_single_qoi[utility.INDEX_COLUMN_NAME].nunique()
                    grouped = df_surrogate_reevaluated_single_qoi.groupby(utility.INDEX_COLUMN_NAME)
                    groups = grouped.groups
                    keyIter = list(groups.keys())
                    subplot_titles = []
                    n_rows = 1
                    if df_statistics_and_measured is not None:
                        n_rows += 1
                        subplot_titles.append(f"Statistics {single_qoi}")
                    if df_simulation_result is not None:
                        n_rows += 1
                        subplot_titles.append(f"Model Runs {single_qoi} (from original UQ simulations)")
                    if reevaluate_original_model and df_model_reevaluated is not None: 
                        n_rows += 1
                        subplot_titles.append(f"Orignal Model Runs {single_qoi} (in the same nodes as a surrogate)")
                    subplot_titles.append(f"Surrogate Model Runs {single_qoi}")
                    # if n_rows > 1:
                    fig = make_subplots(
                            rows=n_rows, cols=1,
                            subplot_titles=subplot_titles,
                            shared_xaxes=False,
                            vertical_spacing=0.1
                    )
                    # else:
                    #     fig = go.Figure()
                    #     window_title = f"Surrogate Model Runs {single_qoi}"
                    #     fig.update_layout(title_text=window_title)
                    counter = 0
                    for key in keyIter:
                        if counter < 100:
                            counter += 1
                            df_surrogate_reevaluated_single_qoi_subset = df_surrogate_reevaluated_single_qoi.loc[groups[key].values]
                            fig.add_trace(
                                go.Scatter(
                                    x=df_surrogate_reevaluated_single_qoi_subset[utility.TIME_COLUMN_NAME], 
                                    y=df_surrogate_reevaluated_single_qoi_subset[utility.QOI_COLUMN_NAME],
                                    line_color='LightSkyBlue', mode="lines", opacity=0.3, showlegend=False,
                                ), row=n_rows, col=1
                            )
                        else:
                            break

                    if reevaluate_original_model and df_model_reevaluated is not None:
                        df_model_reevaluated_subset = df_model_reevaluated[[single_qoi, utility.TIME_COLUMN_NAME, utility.INDEX_COLUMN_NAME]]
                        grouped = df_model_reevaluated_subset.groupby(utility.INDEX_COLUMN_NAME)
                        groups = grouped.groups
                        keyIter = list(groups.keys())
                        counter = 0
                        for key in keyIter:
                            if counter < 100:
                                counter += 1
                                df_model_reevaluated_subset_single_run = df_model_reevaluated_subset.loc[groups[key].values]
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_model_reevaluated_subset_single_run[utility.TIME_COLUMN_NAME], 
                                        y=df_model_reevaluated_subset_single_run[single_qoi],
                                        line_color='LightSkyBlue', mode="lines", opacity=0.3, showlegend=False,
                                    ), row=n_rows-1, col=1
                                )
                            else:
                                break

                    if df_statistics_and_measured is not None:
                        df_statistics_and_measured_subset = df_statistics_and_measured.loc[df_statistics_and_measured[utility.QOI_ENTRY] == single_qoi]
                        # add mean data to each subgraph
                        showlegend_mean = True
                        for index_row in range(1, n_rows+1):
                            fig.add_trace(
                                go.Scatter(
                                    x=df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME], 
                                    y=df_statistics_and_measured_subset[utility.MEAN_ENTRY],
                                    name=f'E[{single_qoi}]',
                                    line_color='Green', mode="lines", showlegend=showlegend_mean,
                                ), row=index_row, col=1
                            )
                            showlegend_mean = False
                        if dict_what_to_plot.get("E_minus_std", False) and "E_minus_std" in df_statistics_and_measured_subset:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME],
                                    y=df_statistics_and_measured_subset[utility.E_MINUS_STD_ENTRY],
                                    name='mean - std. dev', mode='lines', showlegend=False, line_color='rgba(200, 200, 200, 0.4)',), row=1, col=1
                                )
                        if dict_what_to_plot.get("E_plus_std", False) and "E_plus_std" in df_statistics_and_measured_subset:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME],
                                    y=df_statistics_and_measured_subset[utility.E_MINUS_STD_ENTRY],
                                    name='mean +- std. dev', mode='lines', fill='tonexty', showlegend=True, line_color='rgba(200, 200, 200, 0.4)',), row=1, col=1
                                )
                        if dict_what_to_plot.get("E_minus_2std", False) and "E_minus_2std" in df_statistics_and_measured_subset:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME],
                                    y=df_statistics_and_measured_subset[utility.E_MINUS_2STD_ENTRY],
                                    name='mean - std. dev', mode='lines', showlegend=False, line_color='rgba(200, 200, 200, 0.4)',), row=1, col=1
                                )
                        if dict_what_to_plot.get("E_plus_2std", False) and "E_plus_2std" in df_statistics_and_measured_subset:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME],
                                    y=df_statistics_and_measured_subset[utility.E_PLUS_2STD_ENTRY],
                                    name='mean +- 2*std. dev', mode='lines', fill='tonexty', showlegend=True, line_color='rgba(200, 200, 200, 0.4)',), row=1, col=1
                                )
                        if dict_what_to_plot.get("P10", False) and "P10" in df_statistics_and_measured_subset:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME],
                                    y=df_statistics_and_measured_subset[utility.P10_ENTRY],
                                    name='10th percentile', line_color='rgba(128,128,128, 0.3)', showlegend=False), row=1, col=1
                                )
                        if dict_what_to_plot.get("P90", False) and "P90" in df_statistics_and_measured_subset:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME],
                                    y=df_statistics_and_measured_subset[utility.P90_ENTRY],
                                    name='10th percentile-90th percentile', mode='lines', fill='tonexty', showlegend=True, line=dict(color='rgba(128,128,128, 0.3)'), fillcolor='rgba(128,128,128, 0.3)',), row=1, col=1
                                )

                    if df_simulation_result is not None:
                        df_simulation_result_subset = df_simulation_result[[utility.TIME_COLUMN_NAME, utility.INDEX_COLUMN_NAME, single_qoi]]
                        number_unique_model_runs = df_simulation_result_subset[utility.INDEX_COLUMN_NAME].nunique()
                        grouped = df_simulation_result_subset.groupby(utility.INDEX_COLUMN_NAME)
                        groups = grouped.groups
                        keyIter = list(groups.keys())
                        counter = 0
                        for key in keyIter:
                            if counter < 100:
                                counter += 1
                                df_simulation_result_subset_subset = df_simulation_result_subset.loc[groups[key].values]
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_simulation_result_subset_subset[utility.TIME_COLUMN_NAME], 
                                        y=df_simulation_result_subset_subset[single_qoi],
                                        line_color='LightGreen', mode="lines", opacity=0.2, showlegend=False,
                                    ), row=2, col=1
                                )
                            else:
                                break
                            
                    fig.update_layout(width=1000)
                    fig.update_layout(title_text="Model and Surrogate Model Runs")
                    plot_filename = directory_for_saving_plots / f"surrogate_model_runs_{single_qoi}.html"
                    pyo.plot(fig, filename=str(plot_filename), auto_open=False)
                    plot_filename = directory_for_saving_plots / f"surrogate_model_runs_{single_qoi}.pdf"
                    fig.write_image(str(plot_filename), format="pdf", width=1000,)

            # plotting_generalized_indices = kwargs.get('plotting_generalized_indices', False)
            # n_rows = len(list_qois)
            # subplot_titles = list_qois
            # fig = make_subplots(
            #     rows=n_rows, cols=1,
            #     subplot_titles=subplot_titles,
            #     shared_xaxes=False,
            #     vertical_spacing=0.04
            # )
            # # relevant here  - 
            # # df_statistics_and_measured; statisticsObject.df_statistics/statisticsObject.result_dict
            # # gpce_statistics_dict_over_qois/df_recomputed_statistics_and_measured;
            # # generalized_total_sobol_indices_dict_over_qois/df_generalized_total_sobol_indices_dict_and_measured
            # # si_t_df, si_t_df
            # qoi_idx = 0
            # for single_qoi in statisticsObject.list_qoi_column:
            #     showlegend = True
            #     if qoi_idx > 0:
            #         showlegend = False
            #     # df_statistics_single_qoi = statisticsObject.df_statistics.loc[
            #     #     statisticsObject.df_statistics['qoi'] == single_qoi]
            #     # df = df_statistics_and_measured.loc[
            #     #     df_statistics_and_measured['qoi'] == single_qoi] 
            #     # print(f"dict_qoi_column_and_measured_info - {single_qoi} - {statisticsObject.dict_qoi_column_and_measured_info[single_qoi]}")
            #     if plotting_generalized_indices and df_generalized_total_sobol_indices_dict_and_measured is not None:
            #         df = df_generalized_total_sobol_indices_dict_and_measured.loc[
            #             df_generalized_total_sobol_indices_dict_and_measured['qoi'] == single_qoi] 
            #         df_generalized = df[[col for col in df.columns if col.startswith("generalized_sobol_total_index_")]]
            #         substring = "generalized_sobol_total_index_"
            #         for inx, single_column in enumerate(df_generalized.columns):
            #             current_parameter_name = single_column.split(substring, 1)[1]
            #             fig.add_trace(
            #                 go.Scatter(
            #                     x=df[utility.TIME_COLUMN_NAME], y=df_generalized[single_column],
            #                     name=current_parameter_name, mode='lines',
            #                     line=dict(color=colors.COLORS[inx]),
            #                     showlegend=showlegend
            #                 ),
            #                 row=qoi_idx+1, col=1
            #             )
            #         qoi_idx +=1
            #     # TODO Plot - mean; mean+-std; measured; up to 100 realizations of the surrogate; 
            #     # TODO Plot standard Sobol indices; if saved or re-computed (heatmaps)
            #     # TODO Plot generazlied time-wise Sobol indices - generalized_total_sobol_indices_dict_over_qois[single_qoi] (heatmaps?)
            #     # TODO Plot re-computed generazlied time-wise Sobol indices with some other time window generalized_total_sobol_indices_dict_over_qois (heatmaps?)
            # plot_filename = directory_for_saving_plots / f"SA_and_measured_data_total_generalized_60_days_look_back_window_size.pdf"
            # fig.update_layout(title=None)
            # fig.update_layout(
            #     margin=dict(
            #         t=20,  # Top margin
            #         b=10,  # Bottom margin
            #         l=20,  # Left margin
            #         r=20   # Right margin
            #     )
            # )
            # fig.write_image(str(plot_filename), format="pdf", width=1100,) 
        # =================================
        # All above - step-by-step
        # =================================

        # dict_output_file_paths = utility.get_dict_with_output_file_paths_based_on_workingDir(workingDir)
        # args_file = dict_output_file_paths.get("args_file")
        # configuration_object_file = dict_output_file_paths.get("configuration_object_file")
        # nodes_file = dict_output_file_paths.get("nodes_file")
        # parameters_file = dict_output_file_paths.get("parameters_file")
        # time_info_file = dict_output_file_paths.get("time_info_file")
        # df_index_parameter_file = dict_output_file_paths.get("df_index_parameter_file")
        # df_index_parameter_gof_file = dict_output_file_paths.get("df_index_parameter_gof_file")
        # df_simulations_file = dict_output_file_paths.get("df_simulations_file")
        # df_state_file = dict_output_file_paths.get("df_state_file")

        # # Load the UQSim args dictionary
        # uqsim_args_dict = utility.load_uqsim_args_dict(args_file)
        # print("INFO: uqsim_args_dict: ", uqsim_args_dict)
        # model = uqsim_args_dict["model"]
        # inputModelDir = uqsim_args_dict["inputModelDir"]

        # # Load the configuration object
        # configurationObject = utility.load_configuration_object(configuration_object_file)
        # print("configurationObject: ", configurationObject)
        # simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)

        # if model == "hbvsask":
        #     inputModelDir = uqsim_args_dict["inputModelDir"]
        #     basis = configurationObject['model_settings']['basis']
        
        # # Reading Nodes and Parameters
        # with open(nodes_file, 'rb') as f:
        #     simulationNodes = pickle.load(f)
        # print("INFO: simulationNodes: ", simulationNodes)
        # dim = simulationNodes.nodes.shape[0]
        # model_runs = simulationNodes.nodes.shape[1]
        # distStandard = simulationNodes.joinedStandardDists
        # dist = simulationNodes.joinedDists
        # print(f"INFO: model-{model}; dim - {dim}; model_runs - {model_runs}")
        # df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
        # df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)

        # if time_info_file.is_file():
        #     with open(time_info_file, 'r') as f:
        #         time_info = f.read()
        #     print("INFO: time_info: ", time_info)

        # # Reading Prameters and GoF Computed Data
        # if df_index_parameter_file.is_file():
        #     df_index_parameter = pd.read_pickle(df_index_parameter_file, compression="gzip")
        #     params_list = utility._get_parameter_columns_df_index_parameter_gof(
        #         df_index_parameter)
        #     print(f"INFO: df_index_parameter - {df_index_parameter}")
        # else:
        #     params_list = []
        #     for single_param in configurationObject["parameters"]:
        #         params_list.append(single_param["name"])
        # print(f"INFO: params_list - {params_list} (note - all the parameters)")

        # if df_index_parameter_gof_file.is_file():
        #     df_index_parameter_gof = pd.read_pickle(df_index_parameter_gof_file, compression="gzip")
        #     gof_list = utility._get_gof_columns_df_index_parameter_gof(
        #         df_index_parameter_gof)
        #     print(f"INFO: df_index_parameter_gof - {df_index_parameter_gof}")
        #     print(f"INFO: gof_list - {gof_list}")
        # else:
        #     gof_list = None

        # # or in case of a big simulation, skip reading df_simulation_result
        # df_simulation_result = None
        # if read_all_saved_simulations_file and df_simulations_file.is_file():
        #     # Reading Saved Simulations - Note: This migh be a huge file,
        #     # especially for MC/Saltelli kind of simulations
        #     df_simulation_result = pd.read_pickle(df_simulations_file, compression="gzip")
        #     print(f"INFO: df_simulation_result - {df_simulation_result}")

        # # Re-create Statistics Object and DataFrame Object That contains all the Statistics Data
        # statisticsObject = create_stat_object.create_statistics_object(
        #     configuration_object=configurationObject, uqsim_args_dict=uqsim_args_dict, \
        #     workingDir=workingDir, model=model)

        # # Recreate statisticsObject.result_dict
        # statistics_dictionary = uqef_dynamic_utils.read_all_saved_statistics_dict(\
        #     workingDir=workingDir, list_qoi_column=statisticsObject.list_qoi_column, 
        #     single_timestamp_single_file=uqsim_args_dict.get("instantly_save_results_for_each_time_step", False), 
        #     throw_error=False
        #     )

        # # print(f"DEBUGGING - statistics_dictionary.keys()={statistics_dictionary.keys()}")  # should be a list of list_qoi_column
        # for single_qoi in statisticsObject.list_qoi_column:
        #     print(f"DEBUGGING - single_qoi={single_qoi}; statistics_dictionary[single_qoi].keys()={statistics_dictionary[single_qoi].keys()}")

        # # Once you have satistics_dictionary extend StatisticsObject...
        # uqef_dynamic_utils.extend_statistics_object(
        #     statisticsObject=statisticsObject, 
        #     statistics_dictionary=statistics_dictionary, 
        #     df_simulation_result=df_simulation_result,
        #     get_measured_data=False, 
        #     get_unaltered_data=False
        # )

        # # Add measured Data
        # if model == "larsim":
        #     raise NotImplementedError
        # elif model == "hbvsask":
        #     # This is hard-coded for HBV
        #     statisticsObject.inputModelDir_basis = inputModelDir / basis
        #     statisticsObject.get_measured_data(
        #         timestepRange=(statisticsObject.timesteps_min, statisticsObject.timesteps_max),
        #         transforme_mesured_data_as_original_model="False")
        # else:
        #     raise NotImplementedError

        # # Add forcing Data
        # statisticsObject.get_forcing_data(time_column_name=utility.TIME_COLUMN_NAME)

        # # Create a Pandas.DataFrame
        # statisticsObject.create_df_from_statistics_data()

        # # Merge Everything into a single DataFrame
        # df_statistics_and_measured = pd.merge(
        #     statisticsObject.df_statistics, statisticsObject.forcing_df, 
        #     left_on=statisticsObject.time_column_name, right_index=True)
        # df_statistics_and_measured[utility.TIME_COLUMN_NAME] = pd.to_datetime(df_statistics_and_measured[utility.TIME_COLUMN_NAME])
        # df_statistics_and_measured = df_statistics_and_measured.sort_values(by=utility.TIME_COLUMN_NAME)

        # print(df_statistics_and_measured)

        # ========================================================
        # Read the gPCE surrogate model and its coefficients
        # ========================================================

        # # In case gPCE surrogate and the coefficeints are not saved in the stat_dictionary but as a separate files
        # try_reading_gPCE_from_statisticsObject = False
        # try_reading_gPCE_coeff_from_statisticsObject = False
        # gpce_surrogate_dictionary = uqef_dynamic_utils.read_all_saved_gpce_surrogate_models(workingDir, statisticsObject.list_qoi_column, throw_error=False)
        # if gpce_surrogate_dictionary is None:
        #     try_reading_gPCE_from_statisticsObject = True
        # gpce_coeff_dictionary = uqef_dynamic_utils.read_all_saved_gpce_coeffs(workingDir, statisticsObject.list_qoi_column, throw_error=False)
        # if gpce_coeff_dictionary is None:
        #     try_reading_gPCE_coeff_from_statisticsObject = True

        # extended_result_dict = defaultdict(dict)
        # for single_qoi in statisticsObject.list_qoi_column:
        #     extended_result_dict[single_qoi] = {}
        #     print(f"Computation for single_qoi={single_qoi} is just starting!")
        #     df_statistics_and_measured_subset = df_statistics_and_measured[df_statistics_and_measured['qoi']==single_qoi]
        #     # print(f"DEBUGGING - single_qoi={single_qoi}; df_statistics_and_measured_subset--{df_statistics_and_measured_subset}")
        #     # print(f"DEBUGGING - single_qoi={single_qoi}; df_statistics_and_measured_subset.columns--{df_statistics_and_measured_subset.columns}")

        #     # gpce_surrogate_dictionary_subset = gpce_surrogate_dictionary[single_qoi]
        #     # gpce_coeff_dictionary_subset = gpce_coeff_dictionary[single_qoi]
        #     # print(f"DEBUGGING - single_qoi={single_qoi}; gpce_surrogate_dictionary_subset--{gpce_surrogate_dictionary_subset}; gpce_coeff_dictionary_subset--{gpce_coeff_dictionary_subset}")

        #     statistics_pdTimesteps_to_process = []
        #     # for single_timestamp in statistics_pdTimesteps:
        #     # for single_timestamp in [pd.Timestamp('2007-04-24 00:00:00'), pd.Timestamp('2007-04-25 00:00:00'),\
        #     # pd.Timestamp('2007-04-26 00:00:00'), pd.Timestamp('2007-04-27 00:00:00'), pd.Timestamp('2007-04-28 00:00:00'), \
        #     # pd.Timestamp('2007-04-29 00:00:00'), pd.Timestamp('2007-04-30 00:00:00')]:
        #     for single_timestamp in [pd.Timestamp('2005-10-06 00:00:00'), ]: # 2007-04-24
        #         print(f"{single_qoi} - Started computation for date {single_timestamp}")
        #     #     exists = df_statistics_and_measured['TimeStamp'].isin([single_timestamp]).any()
        #         if not single_timestamp in df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME].values:
        #             print(f"{single_qoi}-Sorry there is no {single_timestamp} in the statistics dataframe!")
        #             continue
                
        #         statistics_pdTimesteps_to_process.append(single_timestamp)

        #         # ========================================================
        #         # Fatch the gPCE surrogate model and its coefficients; either from the saved files or from the statisticsObject
        #         # ========================================================

        #         if not try_reading_gPCE_from_statisticsObject:
        #             try:
        #                 if gpce_surrogate_dictionary[single_qoi] is None or gpce_surrogate_dictionary[single_qoi][single_timestamp] is None:
        #                     try_reading_gPCE_from_statisticsObject = True
        #                 else:
        #                     temp_gpce_model = gpce_surrogate_dictionary[single_qoi][single_timestamp]  
        #                     # print(f"DEBUGGING - gPCE surrogate model was read from saved file")
        #                     # print(f"DEBUGGING - {type(temp_gpce_model)}")     
        #             except:
        #                 try_reading_gPCE_from_statisticsObject = True

        #         if try_reading_gPCE_from_statisticsObject:
        #             if 'gPCE' in statisticsObject.result_dict[single_qoi][single_timestamp].keys():
        #                 temp_gpce_model = statisticsObject.result_dict[single_qoi][single_timestamp]['gPCE']
        #             elif 'gPCE' in df_statistics_and_measured_subset.columns:
        #                 temp_gpce_model = df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==single_timestamp]['gPCE'].values[0]
        #             else:
        #                 raise ValueError(f"{single_qoi}-{single_timestamp}-gPCE surrogate model is not found in files in the working directory, nor in the statistics dictionary or in the statistics dataframe")
        #         # print(f"DEBUGGING - gPCE surrogate model for date {single_timestamp} - {temp_gpce_model}")

        #         if not try_reading_gPCE_coeff_from_statisticsObject:
        #             try:
        #                 if gpce_coeff_dictionary[single_qoi] is None or gpce_coeff_dictionary[single_qoi][single_timestamp] is None:
        #                     try_reading_gPCE_coeff_from_statisticsObject = True
        #                 else:
        #                     temp_gpce_coeff = gpce_coeff_dictionary[single_qoi][single_timestamp]  
        #                     # print(f"DEBUGGING - gPCE surrogate model was read from saved file")
        #                     # print(f"DEBUGGING - {type(temp_gpce_model)}")     
        #             except:
        #                 try_reading_gPCE_coeff_from_statisticsObject = True

        #         if try_reading_gPCE_coeff_from_statisticsObject:
        #             if 'gpce_coeff' in statisticsObject.result_dict[single_qoi][single_timestamp].keys():
        #                 temp_gpce_coeff = statisticsObject.result_dict[single_qoi][single_timestamp]['gpce_coeff']
        #             elif 'gPCE' in df_statistics_and_measured_subset.columns:
        #                 temp_gpce_coeff = df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==single_timestamp]['gpce_coeff'].values[0]
        #             else:
        #                 raise ValueError(f"{single_qoi}-{single_timestamp}-Coeff of the gPCE surrogate model were not found in files in the working directory, nor in the statistics dictionary or in the statistics dataframe")
        #         # print(f"DEBUGGING - gPCE coefficients model for date {single_timestamp} - {temp_gpce_coeff}")

        #         # ========================================================

        #         # Check if the mean value is computed and saved in the statistics dictionary
        #         if 'E' in df_statistics_and_measured_subset.columns:
        #             temp_E = df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==single_timestamp]['E'].values[0]
        #             print(f"{single_qoi}-{single_timestamp}-Reading mean from saved statistics dictionary E={temp_E} \n")
                
        #         # Start the computation of the additional statistics
        #         start = time.time()
        #         local_result_dict = {}
        #         qoi_gPCE = temp_gpce_model
        #         start = time.time()
        #         parallelStatistics.calculate_stats_gpce(
        #             local_result_dict, qoi_gPCE, distStandard, compute_other_stat_besides_pce_surrogate=True,
        #             compute_Sobol_t=True, compute_Sobol_m=True)
        #         local_result_dict
        #         extended_result_dict[single_qoi][single_timestamp] = local_result_dict
        #         end = time.time()
        #         duration = end - start
        #         print(f"{single_qoi}-{single_timestamp}-Time needed for statistics computation for single date {single_timestamp} in {dim}D space, with in total {model_runs} executions of model runs is {duration}")
        #         print(f"{single_qoi}-{single_timestamp}-local_result_dict={local_result_dict} \n")

        #         # Check if the 'autoregressive mode' mode is activated - then (re)compute the mean value to correspond to the mean of final QoI
        #         if strtobool(configurationObject["simulation_settings"]["autoregressive_model_first_order"]):
        #             previous_timestamp  = utility.compute_previous_timestamp(single_timestamp, resolution="daily")
        #             temp_E = local_result_dict["E"]  # float(cp.E(temp_gpce_model, distStandard))
        #             temp_E += 0.8*df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==previous_timestamp]['measured'].values[0]
        #                 # print(f"E original={local_result_dict['E']}")
        #                 # print(f"Measured={df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==single_timestamp]['measured'].values[0]}")
        #             print(f"{single_qoi}-{single_timestamp}-E recomputed={temp_E}\n")

        # # TODO Extend statisticsObject.result_dict, i.e., merge statisticsObject.result_dict and extended_result_dict[single_qoi][single_timestamp]
        # # TODO Rely on statisticsObject plotting and re-computing stat dataframe subroutines


if __name__ == '__main__':

    # HBV-SASK model
    # 7D Sparse-gPCE l=7, p=2 2007 deltaQ_cms - 203
    workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/paper_hydro_uq_sim/hbv_uq_cm2.0220')
    # 3D Sparse-gPCE l=7, p=3 2005-2007 deltaQ_cms
    workingDir = pathlib.Path('/gpfs/scratch/pr63so/ga45met2/hbvsask_runs/pce_deltq_3d_short_oldman')
    workingDir = pathlib.Path('/gpfs/scratch/pr63so/ga45met2/hbvsask_runs/pce_deltq_3d_longer_oldman')
    basis_workingDir = pathlib.Path('/work/ga45met/paper_uqef_dynamic_sim/hbvsask_runs_lxc_autumn_24')
    workingDir = basis_workingDir / 'hbv_uq_cm4.0069'
    workingDir = [
        basis_workingDir / 'hbv_uq_cm4.0067',
        basis_workingDir / 'hbv_uq_cm4.0068',
        basis_workingDir / 'hbv_uq_cm4.0069',
    ]

    workingDir = [
        basis_workingDir / 'hbv_uq_cm4.0080',
        basis_workingDir / 'hbv_uq_cm4.0079',
        basis_workingDir / 'hbv_uq_cm4.0078',
    ]
    inputModelDir = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
    directory_for_saving_plots = pathlib.Path('/work/ga45met/paper_uqef_dynamic_sim/hbv_sask/mc_gpce_p4_ct07_100000_lhc_nse02_2004_2007_oldman')
    directory_for_saving_plots = pathlib.Path('/work/ga45met/paper_uqef_dynamic_sim/hbv_sask/mc_gpce_p5_ct07_30000_lhc_2004_2007_oldman')
    
    # Linux Cluster
    basis_workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/hbvsask_runs/kl_analysis')
    workingDir = basis_workingDir / 'mc_kl40_gpce_10d_p3_ct10_10000lhc_oldman_2007'
    inputModelDir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data")
    directory_for_saving_plots = workingDir
    
    set_lower_predictions_to_zero = True
    dict_set_lower_predictions_to_zero = True
    add_measured_data=True
    add_forcing_data=True
    single_timestamp_single_file = False

    # battery model
    # config_file = '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/uqef_dynamic/models/pybamm/configuration_battery_24_shot_names.json'
    # workingDir = pathlib.Path('/work/ga45met/uqef_dynamic_runs/battery_results/cm4_runs/mc_kl10_p4_ct07_24d_10000_random')
    # workingDir = pathlib.Path('/work/ga45met/uqef_dynamic_runs/battery_results/cm4_runs/mc_kl10_p4_ct07_24d_100000_random')
    # # workingDir = pathlib.Path('/work/ga45met/uqef_dynamic_runs/battery_results/cm4_runs/mc_kl10_p5_ct07_24d_10000_random')
    # workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/battery_runs/mc_kl10_p3_ct07_24d_10000_random')
    # workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/battery_runs/mc_kl10_p4_ct07_24d_100000_random')
    # workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/battery_runs/mc_kl10_p2_ct07_24d_10000_random')
    # inputModelDir = pathlib.Path('/dss/dsshome1/lxc0C/ga45met2/.conda/envs/my_uq_env/lib/python3.11/site-packages/pybamm/input/drive_cycles')
    # directory_for_saving_plots = workingDir
    # set_lower_predictions_to_zero = False
    # dict_set_lower_predictions_to_zero = False
    # add_measured_data=False
    # add_forcing_data=False
    # single_timestamp_single_file = False

    set_up_statistics_from_scratch = False

    recompute_gpce = False
    recompute_statistics = False
    reevaluate_original_model = True 
    reevaluate_surrogate = True
    recompute_sobol_indices = False
    recompute_generalized_sobol_indices = False

    read_saved_simulations = True

    surrogate_type='kl+pce' # 'pce' | 'kl+pce'

    compute_other_stat_besides_pce_surrogate = True
    compute_Sobol_t = True
    compute_Sobol_m = True
    compute_Sobol_m2 = False
    dict_stat_to_compute = {
         "Var": True, "StdDev": True, "P10": True, "P90": True,
         "E_minus_std": False, "E_plus_std": False,
         "Skew": False, "Kurt": False, "Sobol_m": compute_Sobol_m, "Sobol_m2": compute_Sobol_m2, "Sobol_t": compute_Sobol_t
         }

    compute_generalized_sobol_indices_from_kl_expansion = False
    compute_generalized_sobol_indices_over_time = True
    look_back_window_size = 365 #'whole'

    printing = True
    plotting = True
    replot_statistics_from_statistics_object = True
    dict_what_to_plot = {
            "E_minus_2std": False, "E_plus_2std": False,
            "E_minus_std": False, "E_plus_std": False, 
            "P10": True, "P90": True,
            "StdDev": True, "Skew": False, "Kurt": False, "Sobol_m": compute_Sobol_m, "Sobol_m2": compute_Sobol_m2, "Sobol_t": compute_Sobol_t
    }
    plotting_generalized_indices = True

    main(
        workingDir=workingDir,
        inputModelDir=inputModelDir,
        directory_for_saving_plots=directory_for_saving_plots,
        surrogate_type=surrogate_type,
        single_timestamp_single_file=single_timestamp_single_file,
        add_measured_data=add_measured_data,
        add_forcing_data=add_forcing_data,
        read_saved_simulations=read_saved_simulations,
        set_up_statistics_from_scratch=set_up_statistics_from_scratch,
        printing=printing, 
        plotting=plotting,
        replot_statistics_from_statistics_object=replot_statistics_from_statistics_object,
        plotting_generalized_indices=plotting_generalized_indices,
        dict_what_to_plot=dict_what_to_plot,
        recompute_gpce = recompute_gpce,
        recompute_statistics = recompute_statistics,
        compute_other_stat_besides_pce_surrogate = compute_other_stat_besides_pce_surrogate,
        reevaluate_surrogate = reevaluate_surrogate,
        reevaluate_original_model=reevaluate_original_model,
        recompute_sobol_indices = recompute_sobol_indices,
        recompute_generalized_sobol_indices = recompute_generalized_sobol_indices,
        compute_generalized_sobol_indices_from_kl_expansion = compute_generalized_sobol_indices_from_kl_expansion,
        compute_generalized_sobol_indices_over_time = compute_generalized_sobol_indices_over_time,
        look_back_window_size=look_back_window_size,
        compute_Sobol_t = compute_Sobol_t,
        compute_Sobol_m = compute_Sobol_m,
        compute_Sobol_m2 = compute_Sobol_m2,
        dict_stat_to_compute = dict_stat_to_compute,
        dict_set_lower_predictions_to_zero=dict_set_lower_predictions_to_zero,
        set_lower_predictions_to_zero=set_lower_predictions_to_zero,
    )