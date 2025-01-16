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

linux_cluster_run = False
# sys.path.insert(0, os.getcwd())
if linux_cluster_run:
    sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')
else:
    sys.path.insert(0, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')

from uqef_dynamic.utils import parallel_statistics
from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqef_dynamic_utils
from uqef_dynamic.utils import create_stat_object

#####################################
### MPI infos:
#####################################
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Get the number of threads
# num_threads = threading.active_count()

def _postprocess_kl_expansion_or_generalized_sobol_indices_computation_from_results_single_qoi(
    single_qoi_column, compute_generalized_sobol_indices_from_kl_expansion, compute_generalized_sobol_indices_over_time,
    workingDir, gpce_coeff_dict, gpce_surrogate_dict, polynomial_expansion, weights_time, labels, look_back_winodw_size, resolution):
    """
    Postprocesses the KL expansion or generalized Sobol indices (computer for the final timestamp or time-vise) 
    computation results for a single quantity of interest (QoI).

    Args:
        single_qoi_column (str): The name of the single quantity of interest (QoI) column.

    Returns:
        dicti over timestamps/or maybe containing info only for the last time stamp with the computed statistics 
        (f"generalized_sobol_total_index_{param_name}")
    """
    result_dict = {}

    temp_structure_storing_gpce_and_coeff = {}
    for single_time_stamp in gpce_coeff_dict.keys():
        temp_structure_storing_gpce_and_coeff[single_time_stamp] = {}
        temp_structure_storing_gpce_and_coeff[single_time_stamp][utility.PCE_COEFF_ENTRY] = gpce_coeff_dict[single_time_stamp]
        # if gpce_surrogate_dict is not None and single_time_stamp in gpce_surrogate_dict:
        #     temp_structure_storing_gpce_and_coeff[single_time_stamp][utility.PCE_ENTRY] = gpce_surrogate_dict[single_time_stamp]

    if compute_generalized_sobol_indices_from_kl_expansion:
        raise Exception("Sorry, recomputation of the generalized sobol indices bas on kl expansion is still not implemented!")
        # # TOOD implement this by propagating f_kl_surrogate_coefficients Var_kl_approx
        # fileName = workingDir / f"recomputed_generalized_sobol_indices_{single_qoi_column}.pkl"
        # param_name_generalized_sobol_total_indices = utility.computing_generalized_sobol_total_indices_from_kl_expan(
        #     f_kl_surrogate_coefficients, polynomial_expansion, weights_time, labels, fileName, total_variance=Var_kl_approx)
        # print(f"INFO: computation of generalized S.S.I based on KL+gPCE(MC) finished...")
        # last_time_step = max(gpce_coeff_dict.keys())  #last_time_step = list(gpce_coeff_dict.keys())[-1]
        # for param_name in labels:
        #     temp_structure_storing_gpce_and_coeff[last_time_step][f"generalized_sobol_total_index_{param_name}"] = \
        #         param_name_generalized_sobol_total_indices[param_name]
    else: 
        fileName = workingDir / f"recomputed_generalized_sobol_indices_{single_qoi_column}.pkl"
        if compute_generalized_sobol_indices_over_time:
            utility.computing_generalized_sobol_total_indices_from_poly_expan_over_time(
                result_dict_statistics=temp_structure_storing_gpce_and_coeff, 
                polynomial_expansion=polynomial_expansion, weights=weights_time, param_names=labels,
                fileName=fileName, look_back_winodw_size=look_back_winodw_size, resolution=resolution)
            print(f"INFO: computation of (over time) generalized S.S.I based on PCE finished...")
        else:
            # the computation of the generalized Sobol indices is done only for the last time step
            utility.computing_generalized_sobol_total_indices_from_poly_expan(
                result_dict_statistics=temp_structure_storing_gpce_and_coeff, 
                polynomial_expansion=polynomial_expansion, 
                weights=weights_time, 
                param_names=labels,
                fileName=fileName)
            print(f"INFO: computation of (over time) generalized S.S.I based on PCE finished...")
    for single_time_stamp in temp_structure_storing_gpce_and_coeff.keys():
        del temp_structure_storing_gpce_and_coeff[single_time_stamp][utility.PCE_COEFF_ENTRY]
    return temp_structure_storing_gpce_and_coeff


# TODO - make difference between gPCE surrogate and KL+(gPCE) surrrogate
def evaluate_gPCE_model_single_qoi_single_date(gPCE_model, nodes):
    return np.array(gPCE_model(*nodes))

def main(workingDir=None, inputModelDir=None, directory_for_saving_plots=None,  
    single_timestamp_single_file=False, printing=False, plotting=True, **kwargs):
    """
    Main function for computing the statistics for gPCE surrogate model
    The default is that the gPCE surrogate model and computed coefficients are saved in the corresponding files in the workingDir;  
    If that is not the case, then the function tries to recreate a Statistics Object and read them from the saved statistics dictionary
    """
    if rank == 0:
        start = time.time()

        if workingDir is None:
            workingDir = pathlib.Path(os.getcwd())
        print(f"workingDir={workingDir}")

        add_measured_data=kwargs.get('add_measured_data', False)
        add_forcing_data=kwargs.get('add_forcing_data', False)
        read_saved_simulations = kwargs.get('read_saved_simulations', False)

        recompute_gpce = kwargs.get('recompute_gpce', False)
        recompute_statistics = kwargs.get('recompute_statistics', False)
        reevaluate_surrogate = kwargs.get('reevaluate_surrogate', False)
        recompute_sobol_indices = kwargs.get('recompute_sobol_indices', False)
        recompute_generalized_sobol_indices = kwargs.get('recompute_generalized_sobol_indices', False)

        inputModelDir_function_input_argument = inputModelDir  # This is because it will be overwritten by the inputModelDir from read_output_files_uqef_dynamic
        workingDir_function_input_argument = workingDir  # This is because it will be overwritten by the workingDir from read_output_files_uqef_dynamic, or maybe not

        # Get all files /data/ dataframe / paths saved by UQEF-Dynamic        
        results_dict = uqef_dynamic_utils.read_output_files_uqef_dynamic(
            workingDir, read_saved_simulations=read_saved_simulations)
        for key, value in results_dict.items():
            globals()[key] = value

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

        statisticsObject = create_stat_object.create_and_extend_statistics_object(
            configurationObject, uqsim_args_dict, workingDir, model, 
            df_simulation_result=df_simulation_result
        )

        # print(f"DEBUGGING - statisticsObject.result_dict.keys()={statisticsObject.result_dict.keys()}")  # should be a list of list_qoi_column
        for single_qoi in statisticsObject.list_qoi_column:
            print(f"DEBUGGING - single_qoi={single_qoi}; statisticsObject.result_dict[single_qoi].keys()={statisticsObject.result_dict[single_qoi].keys()}")

        # Add measured Data and/or forcing
        # This might be relevant for plotting in the end
        if printing:
            print(f"inputModelDir from model - {inputModelDir}; inputModelDir_function_input_argument-{inputModelDir_function_input_argument}")
            print(f"workingDir from model - {workingDir}; workingDir_function_input_argument-{workingDir_function_input_argument}")

        df_statistics_and_measured = statisticsObject.merge_df_statistics_data_with_measured_and_forcing_data(
            add_measured_data=add_measured_data, add_forcing_data=add_forcing_data, transform_measured_data_as_original_model=True)

        statistics_pdTimesteps = statisticsObject.pdTimesteps

        # ==========================================
        set_lower_predictions_to_zero = kwargs.get('set_lower_predictions_to_zero', False)
        set_mean_prediction_to_zero = kwargs.get('set_mean_prediction_to_zero', False)
        correct_sobol_indices = kwargs.get('correct_sobol_indices', False)

        if set_lower_predictions_to_zero:
            if 'E_minus_std' in df_statistics_and_measured:
                df_statistics_and_measured['E_minus_std'] = df_statistics_and_measured['E_minus_std'].apply(lambda x: max(0, x))        
            if 'E_minus_2std' in df_statistics_and_measured:
                df_statistics_and_measured['E_minus_2std'] = df_statistics_and_measured['E_minus_2std'].apply(lambda x: max(0, x))        
            if 'P10' in df_statistics_and_measured:
                df_statistics_and_measured['P10'] = df_statistics_and_measured['P10'].apply(lambda x: max(0, x))
        if set_mean_prediction_to_zero:
            df_statistics_and_measured['E'] = df_statistics_and_measured['E'].apply(lambda x: max(0, x)) 

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

        # if printing:
        #     print(f"statisticsObject.df_statistics-{statisticsObject.df_statistics}")
        #     print(f"statisticsObject.forcing_df-{statisticsObject.forcing_df}")
        #     print(f"statisticsObject.df_measured-{statisticsObject.df_measured}")
        #     print(f"df_statistics_and_measured-{df_statistics_and_measured}")
        #     print(f"df_simulation_result-{df_simulation_result}")
        #     print(f"statistics_pdTimesteps-{statistics_pdTimesteps}")

        # ========================================================
        jointDists = simulationNodes.joinedDists
        jointStandard = simulationNodes.joinedStandardDists
        evaluateSurrogateAtStandardDist = statisticsObject.sampleFromStandardDist  # uqsim_args_dict['sampleFromStandardDist']

        if reevaluate_surrogate:
            number_of_samples = kwargs.get('number_of_samples', 1000)
            sampling_rule = kwargs.get('sampling_rule', "latin_hypercube")
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

            if printing:
                print(f"nodes.shape={nodes.shape}")

        if recompute_statistics:
            compute_other_stat_besides_pce_surrogate = kwargs.get('compute_other_stat_besides_pce_surrogate', False)
            compute_Sobol_t = kwargs.get('compute_Sobol_t', False)
            compute_Sobol_m = kwargs.get('compute_Sobol_m', False)
            compute_Sobol_m2 = kwargs.get('compute_Sobol_m2', False)
            dict_stat_to_compute = kwargs.get('dict_stat_to_compute', utility.DEFAULT_DICT_STAT_TO_COMPUTE)
        # ========================================================
        # Read the gPCE surrogate model and its coefficients
        # ========================================================
        list_qois = statisticsObject.list_qoi_column
        gpce_surrogate_dict_over_qois= defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})
        gpce_coeff_dict_over_qois = defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})

        for single_qoi in statisticsObject.list_qoi_column:
            gpce_surrogate = None
            gpce_coeffs = None

            gpce_surrogate = uqef_dynamic_utils.fetch_gpce_surrogate_single_qoi(
                qoi_column_name=single_qoi, workingDir=workingDir,
                statistics_dictionary=statisticsObject.result_dict, 
                throw_error=False, single_timestamp_single_file=single_timestamp_single_file)

            if gpce_surrogate is None or recompute_generalized_sobol_indices:
                gpce_coeffs = uqef_dynamic_utils.fetch_gpce_coeff_single_qoi(
                    qoi_column_name=single_qoi, workingDir=workingDir,
                    statistics_dictionary=statisticsObject.result_dict, 
                    throw_error=False, single_timestamp_single_file=single_timestamp_single_file)
                
                statisticsObject.prepareForScStatistics(
                    simulationNodes, order=uqsim_args_dict['sc_p_order'], 
                    poly_normed=uqsim_args_dict['sc_poly_normed'], 
                    poly_rule=uqsim_args_dict['sc_poly_rule'], 
                    regression=uqsim_args_dict['regression'], 
                    cross_truncation=uqsim_args_dict['cross_truncation']
                )
                polynomial_expansion = statisticsObject.polynomial_expansion
                polynomial_norms = statisticsObject.polynomial_norms

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

            if statisticsObject.pdTimesteps!=list(gpce_surrogate.keys()):
                print("Watch-out - The timestamps of the statistics and the gPCE surrogate do not match!")
                statistics_pdTimesteps = list(gpce_surrogate.keys())

            if printing:
                temp = gpce_surrogate[statistics_pdTimesteps[0]]
                print(f"Qoi - {single_qoi}\n gpce_surrogate for a first timestamp - {temp} \n")
                if gpce_coeffs and gpce_coeffs is not None:
                    temp = gpce_coeffs[statistics_pdTimesteps[0]]
                    print(f"Qoi - {single_qoi}\n gpce_coeffs - {temp} \n")

            gpce_surrogate_dict_over_qois[single_qoi] = gpce_surrogate
        
    else:
        list_qois = None
        gpce_surrogate_dict_over_qois= None
        gpce_coeff_dict_over_qois = None
        nodes = None
        statistics_pdTimesteps = None
        workingDir = None
        reevaluate_surrogate = False
        recompute_statistics = False
        jointDists = None
        jointStandard = None
        evaluateSurrogateAtStandardDist = None
        compute_other_stat_besides_pce_surrogate = None
        compute_Sobol_t = None
        compute_Sobol_m = None
        compute_Sobol_m2 = None
        dict_stat_to_compute = None

    list_qois = comm.bcast(list_qois, root=0)
    gpce_surrogate_dict_over_qois = comm.bcast(gpce_surrogate_dict_over_qois, root=0)
    
    statistics_pdTimesteps = comm.bcast(statistics_pdTimesteps, root=0)
    dates_to_process = statistics_pdTimesteps
    workingDir = comm.bcast(workingDir, root=0)

    reevaluate_surrogate = comm.bcast(reevaluate_surrogate, root=0)
    recompute_statistics = comm.bcast(recompute_statistics, root=0)

    # Split the dates among processes
    chunk_size = len(dates_to_process) // size
    remainder = len(dates_to_process) % size
    start_index = rank * chunk_size
    end_index = start_index + chunk_size if rank < size - 1 else len(dates_to_process)
    # end_index = (rank + 1) * chunk_size + (1 if rank < remainder else 0)

    # Distribute dates to processes
    my_dates = dates_to_process[start_index:end_index]

    memory_usage_history = []

    if reevaluate_surrogate:
        nodes = comm.bcast(nodes, root=0)
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
        all_results_gpce_surrogate_evaluated = comm.gather(gpce_surrogate_evaluated_dict_over_qois, root=0)
    
    if recompute_statistics:
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

    # Query memory usage and record it
    memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # Memory usage in MB
    memory_usage_history.append(memory_usage)

    # Combine results on rank 0
    if rank == 0:
        # TODO Do modification to the mean and surrogate model evaluations
        #  if the autoregressive mode is activated

        if reevaluate_surrogate:
            combined_results = {single_qoi:{} for single_qoi in statisticsObject.list_qoi_column} #defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})

            for idx, result in enumerate(all_results_gpce_surrogate_evaluated):
                for single_qoi in statisticsObject.list_qoi_column:
                    # print(f"{idx} - {single_qoi} - {list(result[single_qoi].keys())}")
                    combined_results[single_qoi].update(result[single_qoi])

            # gpce_surrogate_evaluated_dict_over_qois now contains the combined results for all dates processed in parallel on rank 0
            gpce_surrogate_evaluated_dict_over_qois = combined_results

            # TODO Add to model runs = Compare to model runs

            # Create a new DataFrame with the evaluated surrogate model

            if printing:
                for single_qoi in statisticsObject.list_qoi_column:
                    print(f"type(gpce_surrogate_evaluated_dict_over_qois[{single_qoi}]) = {type(gpce_surrogate_evaluated_dict_over_qois[single_qoi])}")
                    print(f"gpce_surrogate_evaluated_dict_over_qois[{single_qoi}] = {gpce_surrogate_evaluated_dict_over_qois[single_qoi]}")
                    print(f"{list(gpce_surrogate_evaluated_dict_over_qois[single_qoi].keys())}")
                    # print(f"gpce_surrogate_evaluated_dict_over_qois[{single_qoi}][{my_dates[0]}] = {gpce_surrogate_evaluated_dict_over_qois[single_qoi][my_dates[0]]}")
                    #Timestamp('2007-08-14 00:00:00')

        if recompute_statistics:
            combined_results = {single_qoi:{} for single_qoi in statisticsObject.list_qoi_column} #defaultdict(dict, {single_qoi:{} for single_qoi in list_qois})
            for idx, result in enumerate(all_results_gpce_statistics_recomputed):
                for single_qoi in statisticsObject.list_qoi_column:
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
            for single_qoi in statisticsObject.list_qoi_column:
                # Create a new DataFrame with the re-computed statistics
                df_recomputed_statistics_single_qoi = uqef_dynamic_utils.create_df_from_statistics_data_single_qoi(
                    stat_dict=gpce_statistics_dict_over_qois, 
                    qoi_column=single_qoi, 
                    list_of_uncertain_variables=statisticsObject.labels, 
                    measured_qoi_column=statisticsObject.dict_qoi_column_and_measured_info[single_qoi][],
                    set_lower_predictions_to_zero=set_lower_predictions_to_zero, 
                    measured_fetched=statisticsObject.dict_qoi_column_and_measured_info[single_qoi][], 
                    df_measured=statisticsObject.df_measured,
                    time_column_name=utility.TIME_COLUMN_NAME)
                if df_recomputed_statistics_single_qoi is not None:
                    list_of_single_qoi_dfs.append(df_recomputed_statistics_single_qoi)
            if list_of_single_qoi_dfs:
                df_recomputed_statistics_and_measured = pd.concat(list_of_single_qoi_dfs, axis=0)
                df_recomputed_statistics_and_measured.sort_values(by=utility.TIME_COLUMN_NAME, ascending=True, inplace=True)
            else:
                df_recomputed_statistics_and_measured = None

        if recompute_generalized_sobol_indices:
            if statisticsObject.weights_time is None:
                # This is already called in create_stat_object.create_and_extend_statistics_object() -> \
                # uqef_dynamic_utils.extend_statistics_object -> statisticsObject.set_timesteps -> statisticsObject.set_weights_time
                statisticsObject.set_weights_time()

            # use gpce_coeff_dict_over_qois and gpce_surrogate_dict_over_qois; polynomial_expansion, polynomial_norms
            compute_generalized_sobol_indices_from_kl_expansion = kwargs.get('compute_generalized_sobol_indices_from_kl_expansion', False)
            compute_generalized_sobol_indices_over_time = kwargs.get('compute_generalized_sobol_indices_over_time', False)
            look_back_winodw_size = kwargs.get('look_back_winodw_size', 'whole')
            resolution = statisticsObject.resolution  #kwargs.get('resolution', 'integer')
            generalized_total_sobol_indices_dict = {single_qoi:{} for single_qoi in list_qois}
            if printing:
                print(f"statisticsObject.weights_time-{statisticsObject.weights_time}")
                print(f"statisticsObject.labels-{statisticsObject.labels}")

            for single_qoi in list_qois:
                # for date in statistics_pdTimesteps:
                temp_structure_storing_gpce_and_coeff = _postprocess_kl_expansion_or_generalized_sobol_indices_computation_from_results_single_qoi(
                    single_qoi, compute_generalized_sobol_indices_from_kl_expansion, compute_generalized_sobol_indices_over_time,
                    workingDir, gpce_coeff_dict_over_qois[single_qoi], gpce_surrogate_dict_over_qois[single_qoi], polynomial_expansion,
                    statisticsObject.weights_time, statisticsObject.labels, look_back_winodw_size, resolution
                )
                generalized_total_sobol_indices_dict[single_qoi] = temp_structure_storing_gpce_and_coeff

                if printing:
                    print(f"{single_qoi} - generalized_total_sobol_indices_dict - {temp_structure_storing_gpce_and_coeff}")

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
    
    # Plotting and final post-processing in the main process
    if rank == 0:
        end = time.time()
        runtime = end - start
        print(f"INFO ABOUT THE RUNTIME - Number of MPI processes={size} runtime={runtime}")

        if plotting:
            dict_what_to_plot = {
                "E_minus_2std": False, "E_plus_2std": False,
                "E_minus_std": False, "E_plus_std": False, 
                "P10": True, "P90": True,
                "StdDev": True, "Skew": False, "Kurt": False, "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False
            }
            n_rows = len(list_qois)
            subplot_titles = list_qois
            # fig = make_subplots(
            #     rows=n_rows, cols=1,
            #     subplot_titles=subplot_titles,
            #     shared_xaxes=False,
            #     vertical_spacing=0.04
            # )
            # relevant here  - 
            # gpce_statistics_dict_over_qois; df_statistics_and_measured; statisticsObject.df_statistics/statisticsObject.result_dict
            # si_t_df, si_t_df
            for single_qoi in statisticsObject.list_qoi_column:
                # df_statistics_single_qoi = statisticsObject.df_statistics.loc[
                #     statisticsObject.df_statistics['qoi'] == single_qoi]
                df = df_statistics_and_measured.loc[
                    df_statistics_and_measured['qoi'] == single_qoi] 
                print(f"dict_qoi_column_and_measured_info - {single_qoi} - {statisticsObject.dict_qoi_column_and_measured_info[single_qoi]}")
                # TODO Plot - mean; mean+-std; measured; up to 100 realizations of the surrogate; 
                # TODO Plot standard Sobol indices; if saved or re-computed (heatmaps)
                # TODO Plot generazlied time-wise Sobol indices - generalized_total_sobol_indices_dict[single_qoi] (heatmaps?)
                # TODO Plot re-computed generazlied time-wise Sobol indices with some other time window generalized_total_sobol_indices_dict (heatmaps?)
        
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
        #     # for single_timestamp in statisticsObject.pdTimesteps:
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

    # 7D Sparse-gPCE l=7, p=2 2007 deltaQ_cms - 203
    workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/paper_hydro_uq_sim/hbv_uq_cm2.0220')
    # 3D Sparse-gPCE l=7, p=3 2005-2007 deltaQ_cms
    workingDir = pathlib.Path('/gpfs/scratch/pr63so/ga45met2/hbvsask_runs/pce_deltq_3d_short_oldman')
    workingDir = pathlib.Path('/gpfs/scratch/pr63so/ga45met2/hbvsask_runs/pce_deltq_3d_longer_oldman')

    basis_workingDir = pathlib.Path('/work/ga45met/paper_uqef_dynamic_sim/hbvsask_runs_lxc_autumn_24')
    workingDir = basis_workingDir / 'hbv_uq_cm4.0069'
    inputModelDir = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
    directory_for_saving_plots = pathlib.Path('/work/ga45met/paper_uqef_dynamic_sim/hbv_sask/mc_gpce_p4_ct07_100000_lhc_nse02_2004_2007_oldman')

    recompute_gpce = False
    recompute_statistics = False
    reevaluate_surrogate = False
    recompute_sobol_indices = False
    recompute_generalized_sobol_indices = True

    compute_other_stat_besides_pce_surrogate = True
    compute_Sobol_t = True
    compute_Sobol_m = True
    compute_Sobol_m2 = False
    dict_stat_to_compute = {
         "Var": True, "StdDev": True, "P10": True, "P90": True,
         "E_minus_std": False, "E_plus_std": False,
         "Skew": False, "Kurt": False, "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True
         }

    compute_generalized_sobol_indices_from_kl_expansion = False
    compute_generalized_sobol_indices_over_time = True
    look_back_winodw_size = 'whole'

    main(
        workingDir=workingDir,
        inputModelDir=inputModelDir,
        directory_for_saving_plots=directory_for_saving_plots,
        single_timestamp_single_file=False,
        add_measured_data=True,
        add_forcing_data=True,
        read_saved_simulations=True,
        printing=True, plotting=True,
        recompute_gpce = recompute_gpce,
        recompute_statistics = recompute_statistics,
        reevaluate_surrogate = reevaluate_surrogate,
        recompute_sobol_indices = recompute_sobol_indices,
        recompute_generalized_sobol_indices = recompute_generalized_sobol_indices,
        compute_generalized_sobol_indices_from_kl_expansion = compute_generalized_sobol_indices_from_kl_expansion,
        compute_generalized_sobol_indices_over_time = compute_generalized_sobol_indices_over_time,
        compute_other_stat_besides_pce_surrogate = compute_other_stat_besides_pce_surrogate,
        look_back_winodw_size=look_back_winodw_size,
        compute_Sobol_t = compute_Sobol_t,
        compute_Sobol_m = compute_Sobol_m,
        compute_Sobol_m2 = compute_Sobol_m2,
        dict_stat_to_compute = dict_stat_to_compute,
    )