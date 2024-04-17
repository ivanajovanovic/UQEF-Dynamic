"""
@author: Ivana Jovanovic
"""
import inspect
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
import threading
import psutil

# importing modules/libs for plotting
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import plotly.express as px
# import plotly.offline as pyo
# import matplotlib.pyplot as plt
# # pd.options.plotting.backend = "plotly"

import chaospy as cp
import uqef

# # additionally added for the debugging of the nodes
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# pd.options.mode.chained_assignment = None

linux_cluster_run = True
# sys.path.insert(0, os.getcwd())
if linux_cluster_run:
    sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')
else:
    sys.path.insert(0, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')

from common import utility
from common import uqPostprocessing

from larsim import LarsimModelUQ
from larsim import LarsimStatistics

from hbv_sask import HBVSASKModelUQ
from hbv_sask import HBVSASKStatisticsMultipleQoI as HBVSASKStatistics

#####################################
### MPI infos:
#####################################
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Get the number of threads
num_threads = threading.active_count()

# Defining paths
def update_output_file_paths_based_on_workingDir(workingDir):
    nodes_file = workingDir / "nodes.simnodes.zip"
    parameters_file = workingDir / "parameters.pkl"
    args_file = workingDir / 'uqsim_args.pkl'
    configuration_object_file = workingDir / "configurationObject"

    # Files produced by Samples class
    df_all_simulations_file = workingDir / "df_all_simulations.pkl"
    df_all_index_parameter_gof_file = workingDir / "df_all_index_parameter_gof_values.pkl"
    df_all_index_parameter_file = workingDir / "df_all_index_parameter_values.pkl"
    df_time_varying_grad_analysis_file = workingDir / "df_time_varying_grad_analysis.pkl"
    df_time_aggregated_grad_analysis_file = workingDir / "df_time_aggregated_grad_analysis.pkl"

    # Files produced by UQEF.Statistics and tatistics
    statistics_dictionary_file = workingDir / "statistics_dictionary_qoi_Value.pkl"

    # Active Subspaces related files
    dict_of_approx_matrix_c_file = workingDir / "dict_of_approx_matrix_c.pkl"
    dict_of_matrix_c_eigen_decomposition_file = workingDir / "dict_of_matrix_c_eigen_decomposition.pkl"

    return nodes_file, parameters_file, args_file, configuration_object_file, \
           df_all_simulations_file, df_all_index_parameter_gof_file, df_all_index_parameter_file, \
           df_time_varying_grad_analysis_file, df_time_aggregated_grad_analysis_file, \
           statistics_dictionary_file, dict_of_approx_matrix_c_file, dict_of_matrix_c_eigen_decomposition_file

# Function to parallelize
def evaluate_gPCE_model_single_date(single_date, result_dict, single_qoi, nodes):
    gPCE_model = result_dict[single_qoi][single_date]['gPCE']
    # # result = np.empty([nodes.shape[1], ])
    # result = []
    # for idx, sample in np.ndenumerate(nodes.T):
    #     result.append(gPCE_model(sample))
    result = gPCE_model(*nodes)
    return result

if __name__ == '__main__':
    # Number of threads per process (you can adjust this)
    num_threads_per_process = 4
    num_threads = threading.active_count()

    if rank == 0:
        print(f"Number of MPI processes: {size}")
        print(f"Number of threads: {num_threads}")

        model = "hbvsask"  # "larsim"
        single_qoi = "Q_cms"
        sampling_rule = "halton"  # 'sobol' 'random'
        number_of_samples = 1000
        sampleFromStandardDist = True

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

        # 8D gPCE l=7, p=2 Q_cms 2006 - 155
        # workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/paper_hydro_uq_sim/hbv_uq_cm2.0155')
        # 6D gPCE l=, p= Q_cms 2006 - 173
        workingDir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/paper_hydro_uq_sim/hbv_uq_cm2.0173")
        nodes_file, parameters_file, args_file, configuration_object_file, \
        df_all_simulations_file, df_all_index_parameter_gof_file, df_all_index_parameter_file, \
        df_time_varying_grad_analysis_file, df_time_aggregated_grad_analysis_file, \
        statistics_dictionary_file, dict_of_approx_matrix_c_file, dict_of_matrix_c_eigen_decomposition_file = \
            update_output_file_paths_based_on_workingDir(workingDir)

        # Reading Saved - modified Files
        with open(configuration_object_file, 'rb') as f:
            configurationObject = dill.load(f)
        simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)

        with open(args_file, 'rb') as f:
            uqsim_args = pickle.load(f)
        uqsim_args_dict = vars(uqsim_args)

        # Reading Nodes and Parameters
        with open(nodes_file, 'rb') as f:
            simulationNodes = pickle.load(f)

        # re-create simulationNodes from configuration file
        node_names = []
        for parameter_config in configurationObject["parameters"]:
            node_names.append(parameter_config["name"])
        uqef_simulationNodes = uqef.nodes.Nodes(node_names)
        if sampleFromStandardDist:
            uqef_simulationNodes.setTransformation()
        for parameter_config in configurationObject["parameters"]:
            if parameter_config["distribution"] == "None":
                uqef_simulationNodes.setValue(parameter_config["name"], parameter_config["default"])
            else:
                cp_dist_signature = inspect.signature(getattr(cp, parameter_config["distribution"]))
                dist_parameters_values = []
                for p in cp_dist_signature.parameters:
                    dist_parameters_values.append(parameter_config[p])

                uqef_simulationNodes.setDist(parameter_config["name"],
                                             getattr(cp, parameter_config["distribution"])(
                                                 *dist_parameters_values))
                if sampleFromStandardDist:
                    if parameter_config["distribution"] == "Uniform":
                        uqef_simulationNodes.setStandardDist(parameter_config["name"],
                                                        getattr(cp, parameter_config["distribution"])(
                                                            lower=-1, upper=1
                                                        ))
                    else:
                        uqef_simulationNodes.setStandardDist(parameter_config["name"],
                                                             getattr(cp, parameter_config["distribution"])())
        nodes, parameters = uqef_simulationNodes.generateNodesForMC(
                    number_of_samples, rule=sampling_rule)

        # Reading Parameters and GoF Computed Data
        df_index_parameter = pd.read_pickle(df_all_index_parameter_file, compression="gzip")
        df_index_parameter_gof = pd.read_pickle(df_all_index_parameter_gof_file, compression="gzip")
        params_list = utility._get_parameter_columns_df_index_parameter_gof(
            df_index_parameter_gof)
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)

        df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
        df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)

        # Reading Saved Simulations - Note: This migh be a huge file,
        # especially for MC/Saltelli kind of simulations
        df_simulation_result = pd.read_pickle(df_all_simulations_file, compression="gzip")
        # df_simulation_result = None

        # Re-create Statistics Object and DataFrame Object That contains all the Statistics Data
        statisticsObject = uqPostprocessing.create_statistics_object(
            configurationObject, uqsim_args_dict, workingDir, model=model)

        # Way of doing thinks when instantly_save_results_for_each_time_step is True...
        # Recreate statisticsObject.result_dict
        statistics_dictionary = uqPostprocessing.read_all_saved_statistics_dict(
            workingDir, [single_qoi, ], single_timestamp_single_file=True)

        # Way of doing thing when instantly_save_results_for_each_time_step is False...
        # statistics_dictionary = uqPostprocessing.read_all_saved_statistics_dict(
        #     workingDir, statisticsObject.list_qoi_column, single_timestamp_single_file=False)

        # Once you have satistics_dictionary extend StatisticsObject...
        uqPostprocessing.extend_statistics_object(
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
        df_statistics = statisticsObject.create_df_from_statistics_data()

        # Add forcing Data
        statisticsObject.get_forcing_data(time_column_name="TimeStamp")

        # Merge Everything
        df_statistics_and_measured = pd.merge(
            statisticsObject.df_statistics, statisticsObject.forcing_df, left_on=statisticsObject.time_column_name, right_index=True)

        print(df_statistics_and_measured)
        print(f"nodes.shape = {nodes.shape}")
        single_date = statisticsObject.pdTimesteps[-1]
        temp = statisticsObject.result_dict[single_qoi][single_date]['gPCE']
        print(f"gPCe for date {single_date} for qoi {single_qoi} - {temp} ")
        # Sensitivity Analysis - Computing DataFrame with SI
        # si_m_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_m")
        # si_t_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_t")

        # Read the assumed prior distribution over parameters
        # list_of_single_distr = []
        # for param in configurationObject["parameters"]:
        #     # for now this is hard-coded
        #     if param["distribution"] == "Uniform":
        #         list_of_single_distr.append(cp.Uniform(param["lower"], param["upper"]))
        #     else:
        #         raise NotImplementedError
        # joint = cp.J(*list_of_single_distr)
        # joint_standard = cp.J(cp.Uniform(), cp.Uniform(), cp.Uniform(), cp.Uniform(), cp.Uniform())
        # joint_standard_min_1_1 = cp.J(
        #     cp.Uniform(lower=-1, upper=1),cp.Uniform(lower=-1, upper=1), cp.Uniform(lower=-1, upper=1),
        #     cp.Uniform(lower=-1, upper=1), cp.Uniform(lower=-1, upper=1), cp.Uniform(lower=-1, upper=1)
        # )
        # samples_to_evaluate_gPCE = joint.sample(number_of_samples, rule=sampling_rule)
        # sample_to_evaluate_gPCE_min_1_1 = joint_standard_min_1_1.sample(number_of_samples, rule=sampling_rule)
        # samples_to_evaluate_gPCE_transformed = utility.transformation_of_parameters_var1(
        #     samples_to_evaluate_gPCE, joint, joint_standard_min_1_1)
        # # sample_to_evaluate_gPCE_min_1_1_transformed = utility.transformation_of_parameters_var1(
        # #     sample_to_evaluate_gPCE_min_1_1, joint_standard_min_1_1, joint)

        # gPCE_model = defaultdict()
        # for single_date in statisticsObject.pdTimesteps:
        #     gPCE_model[single_date] = statisticsObject.result_dict["Q_cms"][single_date]['gPCE']
        #
        statistics_result_dict = statisticsObject.result_dict
        statistics_pdtimesteps = statisticsObject.pdTimesteps
    else:
        single_qoi = None
        nodes = None
        statistics_pdtimesteps = None
        statistics_result_dict = None

    single_qoi = comm.bcast(single_qoi, root=0)
    nodes = comm.bcast(nodes, root=0)
    statistics_pdtimesteps = comm.bcast(statistics_pdtimesteps, root=0)
    statistics_result_dict = comm.bcast(statistics_result_dict, root=0)
    # List of dates to process (assuming statisticsObject.pdTimesteps is a list)
    dates_to_process = statistics_pdtimesteps

    if rank == 0:
        start = time.time()

    # Initialize a dictionary to store the results
    gPCE_model_evaluated = {}

    # Monitor memory usage at regular intervals
    memory_usage_history = []

    # Distribute work among processes
    dates_per_process = len(dates_to_process) // size
    remainder = len(dates_to_process) % size
    start_index = rank * dates_per_process
    end_index = start_index + dates_per_process if rank < size - 1 else len(dates_to_process)
    # end_index = (rank + 1) * dates_per_process + (1 if rank < remainder else 0)

    # Create a list to hold thread objects
    threads = []

    # Function for each thread to execute
    def thread_worker(start, end):
        for i in range(start, end):
            date = dates_to_process[i]
            result = evaluate_gPCE_model_single_date(
                date, statistics_result_dict, single_qoi, nodes)
            gPCE_model_evaluated[date] = np.array(result)

    # Divide the work for each thread
    dates_per_thread = (end_index - start_index) // num_threads_per_process

    # Create and start threads
    for i in range(num_threads_per_process):
        thread_start = start_index + i * dates_per_thread
        thread_end = thread_start + dates_per_thread
        if i == num_threads_per_process - 1:
            # Include any remaining dates in the last thread
            thread_end = end_index
        thread = threading.Thread(target=thread_worker, args=(thread_start, thread_end))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Query memory usage and record it
    memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # Memory usage in MB
    memory_usage_history.append(memory_usage)

    # Gather results from all processes into one dictionary
    all_results = comm.gather(gPCE_model_evaluated, root=0)

    # Print memory usage history
    print(f"Memory Usage History (Rank {rank}): {memory_usage_history}")

    # The root process (rank 0) will have all the results in one dictionary
    if rank == 0:
        combined_results = {}
        for results in all_results:
            combined_results.update(results)

        # gPCE_model_evaluated now contains the combined results for all dates processed in parallel on rank 0
        gPCE_model_evaluated = combined_results

    if rank == 0:
        end = time.time()
        runtime = end - start
        print(f"MPI+Threading: Time needed for evaluating {number_of_samples} \
        gPCE model (qoi is {single_qoi}) for {len(statisticsObject.pdTimesteps)} time steps is: {runtime}")
        print(f"gPCE_model_evaluated at times - {gPCE_model_evaluated.keys()} \n")

        print(f"len(statisticsObject.pdTimesteps) - {len(statisticsObject.pdTimesteps)}")
        print(f"len(dates_to_process) - {len(dates_to_process)}")
        print(f"len(gPCE_model_evaluated.keys()) - {len(gPCE_model_evaluated.keys())}")

        temp_date = statisticsObject.pdTimesteps[-1]  # dates_to_process[-1]
        # print(f"temp_date - {temp_date}, {type(temp_date)}")
        temp_evaluation_of_surrogate = gPCE_model_evaluated[temp_date]
        print(f"gPCE_model_evaluated for date {temp_date} - {temp_evaluation_of_surrogate}")
        # # print(f"{type(gPCE_model_evaluated[gPCE_model_evaluated.keys()[0]])} \n {gPCE_model_evaluated[gPCE_model_evaluated.keys()[0]]}")

    MPI.Finalize()