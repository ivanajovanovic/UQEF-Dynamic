"""
@author: Ivana Jovanovic
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

from uqef_dynamic.utils import parallelStatistics
from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqPostprocessing

#####################################
### MPI infos:
#####################################
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Get the number of threads
# num_threads = threading.active_count()


def main():
    if rank == 0:
        print(f"Number of MPI processes: {size}")

        model = "hbvsask"  # "larsim"
        inputModelDir = None
        if model == "larsim":
            if linux_cluster_run:
                inputModelDir = os.path.abspath(
                    os.path.join('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2', 'Larsim-data'))
            else:
                pass
        elif model == "hbvsask":
            if linux_cluster_run:
                inputModelDir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data")
            else:
                pass
            basis = "Oldman_Basin"  # 'Banff_Basin'
        else:
            raise NotImplementedError

        # 7D Sparse-gPCE l=7, p=2 2007 deltaQ_cms - 203
        workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/paper_hydro_uq_sim/hbv_uq_cm2.0220')

        nodes_file, parameters_file, args_file, configuration_object_file, \
        df_all_simulations_file, df_all_index_parameter_gof_file, df_all_index_parameter_file, \
        df_time_varying_grad_analysis_file, df_time_aggregated_grad_analysis_file, \
        statistics_dictionary_file, dict_of_approx_matrix_c_file, dict_of_matrix_c_eigen_decomposition_file = \
            utility.update_output_file_paths_based_on_workingDir(workingDir)

        with open(configuration_object_file, 'rb') as f:
            configurationObject = dill.load(f)
        
        if model == "hbvsask":
            basis = configurationObject['model_settings']['basis']

        simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)
        
        with open(args_file, 'rb') as f:
            uqsim_args = pickle.load(f)
        uqsim_args_dict = vars(uqsim_args)

        with open(nodes_file, 'rb') as f:
        #     simulationNodes = dill.load(f)
            simulationNodes = pickle.load(f)

        dim = simulationNodes.nodes.shape[0]
        model_runs = simulationNodes.nodes.shape[1]
        dist = simulationNodes.joinedStandardDists
        print(f"dim - {dim}; model_runs - {model_runs}")

        df_index_parameter = pd.read_pickle(df_all_index_parameter_file, compression="gzip")
        params_list = utility._get_parameter_columns_df_index_parameter_gof(
            df_index_parameter)

        if df_all_index_parameter_gof_file.is_file():
            df_index_parameter_gof = pd.read_pickle(df_all_index_parameter_gof_file, compression="gzip")
            gof_list = utility._get_gof_columns_df_index_parameter_gof(
                df_index_parameter_gof)

        df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
        df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)

        # or in case of a big simulation, skip reading df_simulation_result
        df_simulation_result = None
        # df_simulation_result = pd.read_pickle(df_all_simulations_file, compression="gzip")

        statisticsObject = uqPostprocessing.create_statistics_object(
            configurationObject, uqsim_args_dict, workingDir, model=model)

        # Way of doing thinks when instantly_save_results_for_each_time_step is True...
        statistics_dictionary = uqPostprocessing.read_all_saved_statistics_dict(
            workingDir, [statisticsObject.list_qoi_column[0],], single_timestamp_single_file=True)
        # statistics_dictionary = uqPostprocessing.read_all_saved_statistics_dict(
        #     workingDir, statisticsObject.list_qoi_column)

        list_qoi_column = statisticsObject.list_qoi_column

        uqPostprocessing.extend_statistics_object(
            statisticsObject=statisticsObject, 
            statistics_dictionary=statistics_dictionary, 
            df_simulation_result=df_simulation_result,  # df_simulation_result=None,
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
        df_statistics = statisticsObject.create_df_from_statistics_data()

        # Add forcing Data
        statisticsObject.get_forcing_data(time_column_name=utility.TIME_COLUMN_NAME)

        # Merge Everything
        df_statistics_and_measured = pd.merge(
            statisticsObject.df_statistics, statisticsObject.forcing_df, 
            left_on=statisticsObject.time_column_name, right_index=True)
        df_statistics_and_measured[utility.TIME_COLUMN_NAME] = pd.to_datetime(df_statistics_and_measured[utility.TIME_COLUMN_NAME])
        df_statistics_and_measured = df_statistics_and_measured.sort_values(by=utility.TIME_COLUMN_NAME)

        print(df_statistics_and_measured)

        # In case gPCE surrogate and the coefficeints are not saved in the stat_dictionary but as a separate files
        gpce_surrogate_dictionary = uqPostprocessing.read_all_saved_gpce_surrogate_models(workingDir, statisticsObject.list_qoi_column)
        gpce_coeff_dictionary = uqPostprocessing.read_all_saved_gpce_coeffs(workingDir, statisticsObject.list_qoi_column)

        for single_qoi in statisticsObject.list_qoi_column:
            df_statistics_and_measured_subset = df_statistics_and_measured[df_statistics_and_measured['qoi']==single_qoi]
            gpce_surrogate_dictionary_subset = gpce_surrogate_dictionary[single_qoi]
            gpce_coeff_dictionary_subset = gpce_coeff_dictionary[single_qoi]

        statistics_pdTimesteps_to_process = []
        stat_result_dict = {}
        # for single_timestamp in statisticsObject.pdTimesteps:
        # for single_timestamp in [pd.Timestamp('2007-04-24 00:00:00'), pd.Timestamp('2007-04-25 00:00:00'),\
        # pd.Timestamp('2007-04-26 00:00:00'), pd.Timestamp('2007-04-27 00:00:00'), pd.Timestamp('2007-04-28 00:00:00'), \
        # pd.Timestamp('2007-04-29 00:00:00'), pd.Timestamp('2007-04-30 00:00:00')]:
        for single_timestamp in [pd.Timestamp('2007-04-24 00:00:00'), ]:
        #     exists = df_statistics_and_measured['TimeStamp'].isin([single_timestamp]).any()
            if not single_timestamp in df_statistics_and_measured[utility.TIME_COLUMN_NAME].values:
                print(f"False for {single_timestamp}")
                continue
            
            statistics_pdTimesteps_to_process.append(single_timestamp)

            if 'gPCE' in df_statistics_and_measured.columns:
                temp_gpce_model = df_statistics_and_measured[df_statistics_and_measured[utility.TIME_COLUMN_NAME]==single_timestamp]['gPCE'].values[0]
            else:
                temp_gpce_model = gpce_surrogate_dictionary[single_timestamp]

            if 'gpce_coeff' in df_statistics_and_measured.columns:
                temp_gpce_coeff = df_statistics_and_measured[df_statistics_and_measured[utility.TIME_COLUMN_NAME]==single_timestamp]['gpce_coeff'].values[0]
            else:
                temp_gpce_coeff = gpce_coeff_dictionary[single_timestamp]

            if 'E' in df_statistics_and_measured.columns:
                temp_E = df_statistics_and_measured[df_statistics_and_measured[utility.TIME_COLUMN_NAME]==single_timestamp]['E'].values[0]
                
                # previous_timestamp  = utility.compute_previous_timestamp(single_timestamp, resolution="daily"):
                # temp_E_recomputed = float(cp.E(temp_gpce_model, simulationNodes.joinedStandardDists))
                # temp_E_recomputed += 0.8*df_statistics_and_measured[df_statistics_and_measured[utility.TIME_COLUMN_NAME]==previous_timestamp]['measured'].values[0]
                # print(f"E={temp_E}; E recomputed={temp_E_recomputed}")

            start = time.time()
            local_result_dict = {}
            qoi_gPCE = temp_gpce_model
            start = time.time()
            parallelStatistics.calculate_stats_gpce(
                local_result_dict, qoi_gPCE, dist, compute_other_stat_besides_pce_surrogate=True,
                compute_Sobol_t=True, compute_Sobol_m=True)
            stat_result_dict[single_timestamp] = local_result_dict
            end = time.time()
            duration = end - start
            print(f"Time needed for statistics computation for single date {single_timestamp} in {dim}D space, \
            with in total {model_runs} executions of model runs is {duration}")
            print(f"local_result_dict={local_result_dict}")


if __name__ == '__main__':
    main()