"""
This file is used to compute the statistics for gPCE surrogate model; 
it is mainly used for the parallel computing for the hydrological model HBV-SASK / LARSIM
@author: Ivana Jovanovic Buha
"""
import inspect
import json
import os
import subprocess
from distutils.util import strtobool
import dill
import numpy as np
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

import chaospy as cp
import uqef

linux_cluster_run = True
# sys.path.insert(0, os.getcwd())
# if linux_cluster_run:
#     sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')
# else:
#     sys.path.insert(0, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')
sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')

from uqef_dynamic.utils import parallelStatistics
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


def main(workingDir=None):
    """
    Main function for computing the statistics for gPCE surrogate model
    The default is that the gPCE surrogate model and computed coefficients are saved in the corresponding files in the workingDir;  
    If that is not the case, then the function tries to recreate a Statistics Object and read them from the saved statistics dictionary
    """
    if rank == 0:
        print(f"Number of MPI processes: {size}")

        if workingDir is None:
            workingDir = pathlib.Path(os.getcwd())

        read_all_saved_simulations_file = False

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

        # Load the UQSim args dictionary
        uqsim_args_dict = utility.load_uqsim_args_dict(args_file)
        print("INFO: uqsim_args_dict: ", uqsim_args_dict)
        model = uqsim_args_dict["model"]
        inputModelDir = uqsim_args_dict["inputModelDir"]

        # Load the configuration object
        configurationObject = utility.load_configuration_object(configuration_object_file)
        print("configurationObject: ", configurationObject)
        simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)

        if model == "hbvsask":
            inputModelDir = uqsim_args_dict["inputModelDir"]
            basis = configurationObject['model_settings']['basis']
        
        # Reading Nodes and Parameters
        with open(nodes_file, 'rb') as f:
            simulationNodes = pickle.load(f)
        print("INFO: simulationNodes: ", simulationNodes)
        dim = simulationNodes.nodes.shape[0]
        model_runs = simulationNodes.nodes.shape[1]
        distStandard = simulationNodes.joinedStandardDists
        dist = simulationNodes.joinedDists
        print(f"INFO: model-{model}; dim - {dim}; model_runs - {model_runs}")
        df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
        df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)

        with open(time_info_file, 'r') as f:
            time_info = f.read()
        print("INFO: time_info: ", time_info)

        # Reading Prameters and GoF Computed Data
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
        else:
            gof_list = None

        # or in case of a big simulation, skip reading df_simulation_result
        df_simulation_result = None
        if read_all_saved_simulations_file and df_simulations_file.is_file():
            # Reading Saved Simulations - Note: This migh be a huge file,
            # especially for MC/Saltelli kind of simulations
            if df_simulations_file.is_file():
                df_simulation_result = pd.read_pickle(df_simulations_file, compression="gzip")
                print(f"INFO: df_simulation_result - {df_simulation_result}")

        # Re-create Statistics Object and DataFrame Object That contains all the Statistics Data
        statisticsObject = create_stat_object.create_statistics_object(
            configuration_object=configurationObject, uqsim_args_dict=uqsim_args_dict, \
            workingDir=workingDir, model=model)

        # Recreate statisticsObject.result_dict
        statistics_dictionary = uqef_dynamic_utils.read_all_saved_statistics_dict(\
            workingDir=workingDir, list_qoi_column=statisticsObject.list_qoi_column, 
            single_timestamp_single_file=uqsim_args_dict.get("instantly_save_results_for_each_time_step", False), 
            throw_error=False
            )

        # print(f"DEBUGGING - statistics_dictionary.keys()={statistics_dictionary.keys()}")  # should be a list of list_qoi_column
        for single_qoi in statisticsObject.list_qoi_column:
            print(f"DEBUGGING - single_qoi={single_qoi}; statistics_dictionary[single_qoi].keys()={statistics_dictionary[single_qoi].keys()}")

        # Once you have satistics_dictionary extend StatisticsObject...
        uqef_dynamic_utils.extend_statistics_object(
            statisticsObject=statisticsObject, 
            statistics_dictionary=statistics_dictionary, 
            df_simulation_result=df_simulation_result,
            get_measured_data=False, 
            get_unaltered_data=False
        )

        # Add measured Data
        if model == "larsim":
            raise NotImplementedError
        elif model == "hbvsask":
            # This is hard-coded for HBV
            statisticsObject.inputModelDir_basis = inputModelDir / basis
            statisticsObject.get_measured_data(
                timestepRange=(statisticsObject.timesteps_min, statisticsObject.timesteps_max),
                transforme_mesured_data_as_original_model="False")
        else:
            raise NotImplementedError

        # Create a Pandas.DataFrame
        statisticsObject.create_df_from_statistics_data()

        # Add forcing Data
        statisticsObject.get_forcing_data(time_column_name=utility.TIME_COLUMN_NAME)

        # Merge Everything into a single DataFrame
        df_statistics_and_measured = pd.merge(
            statisticsObject.df_statistics, statisticsObject.forcing_df, 
            left_on=statisticsObject.time_column_name, right_index=True)
        df_statistics_and_measured[utility.TIME_COLUMN_NAME] = pd.to_datetime(df_statistics_and_measured[utility.TIME_COLUMN_NAME])
        df_statistics_and_measured = df_statistics_and_measured.sort_values(by=utility.TIME_COLUMN_NAME)

        print(df_statistics_and_measured)

        # ========================================================
        # Read the gPCE surrogate model and its coefficients
        # ========================================================

        # In case gPCE surrogate and the coefficeints are not saved in the stat_dictionary but as a separate files
        try_reading_gPCE_from_statisticsObject = False
        try_reading_gPCE_coeff_from_statisticsObject = False
        gpce_surrogate_dictionary = uqef_dynamic_utils.read_all_saved_gpce_surrogate_models(workingDir, statisticsObject.list_qoi_column, throw_error=False)
        if gpce_surrogate_dictionary is None:
            try_reading_gPCE_from_statisticsObject = True
        gpce_coeff_dictionary = uqef_dynamic_utils.read_all_saved_gpce_coeffs(workingDir, statisticsObject.list_qoi_column, throw_error=False)
        if gpce_coeff_dictionary is None:
            try_reading_gPCE_coeff_from_statisticsObject = True

        extended_result_dict = defaultdict(dict)
        for single_qoi in statisticsObject.list_qoi_column:
            extended_result_dict[single_qoi] = {}
            print(f"Computation for single_qoi={single_qoi} is just starting!")
            df_statistics_and_measured_subset = df_statistics_and_measured[df_statistics_and_measured['qoi']==single_qoi]
            # print(f"DEBUGGING - single_qoi={single_qoi}; df_statistics_and_measured_subset--{df_statistics_and_measured_subset}")
            # print(f"DEBUGGING - single_qoi={single_qoi}; df_statistics_and_measured_subset.columns--{df_statistics_and_measured_subset.columns}")

            # gpce_surrogate_dictionary_subset = gpce_surrogate_dictionary[single_qoi]
            # gpce_coeff_dictionary_subset = gpce_coeff_dictionary[single_qoi]
            # print(f"DEBUGGING - single_qoi={single_qoi}; gpce_surrogate_dictionary_subset--{gpce_surrogate_dictionary_subset}; gpce_coeff_dictionary_subset--{gpce_coeff_dictionary_subset}")

            statistics_pdTimesteps_to_process = []
            # for single_timestamp in statisticsObject.pdTimesteps:
            # for single_timestamp in [pd.Timestamp('2007-04-24 00:00:00'), pd.Timestamp('2007-04-25 00:00:00'),\
            # pd.Timestamp('2007-04-26 00:00:00'), pd.Timestamp('2007-04-27 00:00:00'), pd.Timestamp('2007-04-28 00:00:00'), \
            # pd.Timestamp('2007-04-29 00:00:00'), pd.Timestamp('2007-04-30 00:00:00')]:
            for single_timestamp in [pd.Timestamp('2005-10-06 00:00:00'), ]: # 2007-04-24
                print(f"{single_qoi} - Started computation for date {single_timestamp}")
            #     exists = df_statistics_and_measured['TimeStamp'].isin([single_timestamp]).any()
                if not single_timestamp in df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME].values:
                    print(f"{single_qoi}-Sorry there is no {single_timestamp} in the statistics dataframe!")
                    continue
                
                statistics_pdTimesteps_to_process.append(single_timestamp)

                # ========================================================
                # Fatch the gPCE surrogate model and its coefficients; either from the saved files or from the statisticsObject
                # ========================================================

                if not try_reading_gPCE_from_statisticsObject:
                    try:
                        if gpce_surrogate_dictionary[single_qoi] is None or gpce_surrogate_dictionary[single_qoi][single_timestamp] is None:
                            try_reading_gPCE_from_statisticsObject = True
                        else:
                            temp_gpce_model = gpce_surrogate_dictionary[single_qoi][single_timestamp]  
                            # print(f"DEBUGGING - gPCE surrogate model was read from saved file")
                            # print(f"DEBUGGING - {type(temp_gpce_model)}")     
                    except:
                        try_reading_gPCE_from_statisticsObject = True

                if try_reading_gPCE_from_statisticsObject:
                    if 'gPCE' in statisticsObject.result_dict[single_qoi][single_timestamp].keys():
                        temp_gpce_model = statisticsObject.result_dict[single_qoi][single_timestamp]['gPCE']
                    elif 'gPCE' in df_statistics_and_measured_subset.columns:
                        temp_gpce_model = df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==single_timestamp]['gPCE'].values[0]
                    else:
                        raise ValueError(f"{single_qoi}-{single_timestamp}-gPCE surrogate model is not found in files in the working directory, nor in the statistics dictionary or in the statistics dataframe")
                # print(f"DEBUGGING - gPCE surrogate model for date {single_timestamp} - {temp_gpce_model}")

                if not try_reading_gPCE_coeff_from_statisticsObject:
                    try:
                        if gpce_coeff_dictionary[single_qoi] is None or gpce_coeff_dictionary[single_qoi][single_timestamp] is None:
                            try_reading_gPCE_coeff_from_statisticsObject = True
                        else:
                            temp_gpce_coeff = gpce_coeff_dictionary[single_qoi][single_timestamp]  
                            # print(f"DEBUGGING - gPCE surrogate model was read from saved file")
                            # print(f"DEBUGGING - {type(temp_gpce_model)}")     
                    except:
                        try_reading_gPCE_coeff_from_statisticsObject = True

                if try_reading_gPCE_coeff_from_statisticsObject:
                    if 'gpce_coeff' in statisticsObject.result_dict[single_qoi][single_timestamp].keys():
                        temp_gpce_coeff = statisticsObject.result_dict[single_qoi][single_timestamp]['gpce_coeff']
                    elif 'gPCE' in df_statistics_and_measured_subset.columns:
                        temp_gpce_coeff = df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==single_timestamp]['gpce_coeff'].values[0]
                    else:
                        raise ValueError(f"{single_qoi}-{single_timestamp}-Coeff of the gPCE surrogate model were not found in files in the working directory, nor in the statistics dictionary or in the statistics dataframe")
                # print(f"DEBUGGING - gPCE coefficients model for date {single_timestamp} - {temp_gpce_coeff}")

                # ========================================================

                # Check if the mean value is computed and saved in the statistics dictionary
                if 'E' in df_statistics_and_measured_subset.columns:
                    temp_E = df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==single_timestamp]['E'].values[0]
                    print(f"{single_qoi}-{single_timestamp}-Reading mean from saved statistics dictionary E={temp_E} \n")
                
                # Start the computation of the additional statistics
                start = time.time()
                local_result_dict = {}
                qoi_gPCE = temp_gpce_model
                start = time.time()
                parallelStatistics.calculate_stats_gpce(
                    local_result_dict, qoi_gPCE, distStandard, compute_other_stat_besides_pce_surrogate=True,
                    compute_Sobol_t=True, compute_Sobol_m=True)
                local_result_dict
                extended_result_dict[single_qoi][single_timestamp] = local_result_dict
                end = time.time()
                duration = end - start
                print(f"{single_qoi}-{single_timestamp}-Time needed for statistics computation for single date {single_timestamp} in {dim}D space, with in total {model_runs} executions of model runs is {duration}")
                print(f"{single_qoi}-{single_timestamp}-local_result_dict={local_result_dict} \n")

                # Check if the 'autoregressive mode' mode is activated - then (re)compute the mean value to correspond to the mean of final QoI
                if strtobool(configurationObject["simulation_settings"]["autoregressive_model_first_order"]):
                    previous_timestamp  = utility.compute_previous_timestamp(single_timestamp, resolution="daily")
                    temp_E = local_result_dict["E"]  # float(cp.E(temp_gpce_model, distStandard))
                    temp_E += 0.8*df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==previous_timestamp]['measured'].values[0]
                        # print(f"E original={local_result_dict['E']}")
                        # print(f"Measured={df_statistics_and_measured_subset[df_statistics_and_measured_subset[utility.TIME_COLUMN_NAME]==single_timestamp]['measured'].values[0]}")
                    print(f"{single_qoi}-{single_timestamp}-E recomputed={temp_E}\n")

        # TODO Extend statisticsObject.result_dict, i.e., merge statisticsObject.result_dict and extended_result_dict[single_qoi][single_timestamp]
        # TODO Rely on statisticsObject plotting and re-computing stat dataframe subroutines


if __name__ == '__main__':

    # 7D Sparse-gPCE l=7, p=2 2007 deltaQ_cms - 203
    workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/paper_hydro_uq_sim/hbv_uq_cm2.0220')
    # 3D Sparse-gPCE l=7, p=3 2005-2007 deltaQ_cms
    workingDir = pathlib.Path('/gpfs/scratch/pr63so/ga45met2/hbvsask_runs/pce_deltq_3d_short_oldman')
    workingDir = pathlib.Path('/gpfs/scratch/pr63so/ga45met2/hbvsask_runs/pce_deltq_3d_longer_oldman')

    main(workindDir=workingDir)