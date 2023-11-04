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

# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
import matplotlib.pyplot as plt
# pd.options.plotting.backend = "plotly"

import chaospy as cp
import uqef

# additionally added for the debugging of the nodes
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

linux_cluster_run = True
# sys.path.insert(0, os.getcwd())
if linux_cluster_run:
    sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Hydro')
else:
    sys.path.insert(0, '/work/ga45met/mnt/linux_cluster_2/UQEF-Hydro')

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

# # Function to parallelize
# def evaluate_gPCE_model_single_date(single_date, result_dict, single_uq, uqef_simulationNodes):
#     gPCE_model = result_dict[single_uq][single_date]['gPCE']
#     result = gPCE_model(uqef_simulationNodes.nodes.T)
#     return single_date, result

# Define a function to process a single date
def evaluate_gPCE_model_single_date(single_date, single_qoi, result_dict):
    gPCE_model = result_dict[single_qoi][single_date]['gPCE']
    gPCE_model_evaluated[single_date] = gPCE_model(uqef_simulationNodes.nodes.T)

# def evaluate_gPCE_model_single_date(single_date, single_qoi):
#     gPCE_model = statisticsObject.result_dict[single_qoi][single_date]['gPCE']
#     result = gPCE_model(uqef_simulationNodes.nodes.T)
#     return (single_date, result)

if __name__ == '__main__':
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
    statistics_dictionary = defaultdict(dict)
    if df_simulation_result is not None:
        list_TimeStamp = [pd.Timestamp(timestep) for timestep in list(df_simulation_result.TimeStamp.unique())]
    else:
        # in this case get the list of all files with the format "statistics_dictionary_{single_qoi}_{single_timestep}.pkl"
        all_files = os.listdir(workingDir)
        list_TimeStamp = []
        for filename in all_files:
            parts = filename.split('_')
            if parts[0] == "statistics" and parts[-1].endswith(".pkl"):
                single_timestep = parts[-1].split('.')[0]
                list_TimeStamp.append(single_timestep)
    # hard-coded for now, otherwise: for single_qoi in statisticsObject.list_qoi_column:
    # temp = [statisticsObject.list_qoi_column[0], ]
    # for single_qoi in temp:
    statistics_dictionary[single_qoi] = dict()
    for single_timestep in list_TimeStamp:
        statistics_dictionary_file_temp = workingDir / f"statistics_dictionary_{single_qoi}_{single_timestep}.pkl"
        assert statistics_dictionary_file_temp.is_file()
        with open(statistics_dictionary_file_temp, 'rb') as f:
            statistics_dictionary_temp = pickle.load(f)
        statistics_dictionary[single_qoi][single_timestep] = statistics_dictionary_temp

    # Way of doing thing when instantly_save_results_for_each_time_step is False...
    # statistics_dictionary = uqPostprocessing.read_all_saved_statistics_dict(
    #     workingDir, statisticsObject.list_qoi_column)

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

    # List of dates to process (assuming statisticsObject.pdTimesteps is a list)
    dates_to_process = statisticsObject.pdTimesteps

    start = time.time()

    # Initialize a dictionary to store the results
    gPCE_model_evaluated = {}

    # Split the dates among processes
    chunk_size = len(dates_to_process) // size
    remainder = len(dates_to_process) % size
    start_index = rank * chunk_size
    end_index = start_index + chunk_size if rank < size - 1 else len(dates_to_process)
    # end_index = (rank + 1) * chunk_size + (1 if rank < remainder else 0)

    # Distribute dates to processes
    my_dates = dates_to_process[start_index:end_index]

    # Process dates
    for date in my_dates:
        evaluate_gPCE_model_single_date(date, single_qoi, statisticsObject.result_dict)
    # # Iterate over the assigned date range
    # for i in range(start_index, end_index):
    #     single_date = dates_to_process[i]
    #     gPCE_model = statisticsObject.result_dict[single_uq][single_date]['gPCE']
    #     gPCE_model_evaluated[single_date] = gPCE_model(uqef_simulationNodes.nodes.T)

    # Gather results from all processes
    all_results = comm.gather(gPCE_model_evaluated, root=0)

    # Combine results on rank 0
    if rank == 0:
        combined_results = {}
        for results in all_results:
            combined_results.update(results)

        # gPCE_model_evaluated now contains the combined results for all dates processed in parallel on rank 0
        gPCE_model_evaluated = combined_results

    # # Parallelize the processing of dates
    # pool.map(evaluate_gPCE_model_single_date, dates_to_process)
    # # Close the pool of processes
    # pool.close()
    # pool.join()

    # # for single_date in statisticsObject.pdTimesteps:
    # #     gPCE_model = statisticsObject.result_dict["Q_cms"][single_date]['gPCE']
    # #     # gPCE_model_evaluated[single_date] = gPCE_model(samples_to_evaluate_gPCE_transformed.T)
    # #     # gPCE_model_evaluated[single_date] = gPCE_model(sample_to_evaluate_gPCE_min_1_1.T)
    # #     gPCE_model_evaluated[single_date] = gPCE_model(uqef_simulationNodes.nodes.T)
    # print(statisticsObject.pdTimesteps)
    # with multiprocessing.Pool() as pool:
    #     for result in pool.imap(evaluate_gPCE_model_single_date, statisticsObject.pdTimesteps):
    #         gPCE_model_evaluated[result[0]] = result[1]

    end = time.time()
    runtime = end - start
    print(f"Time needed for evaluating {number_of_samples} \
    gPCE model (qoi is {single_qoi}) for {len(statisticsObject.pdTimesteps)} time steps is: {runtime}")

    print(gPCE_model_evaluated)

######################
# from mpi4py import MPI
# import numpy as np
#
# # Initialize MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
#
# # Distribute work among processes
# all_dates = statisticsObject.pdTimesteps
# dates_per_process = len(all_dates) // size
# remainder = len(all_dates) % size
#
# # Calculate the range of dates for this process
# start_index = rank * dates_per_process
# end_index = (rank + 1) * dates_per_process + (1 if rank < remainder else 0)
#
# # Initialize a dictionary to store the results for this process
# gPCE_model_evaluated = {}
#
# # Iterate over the assigned date range
# for i in range(start_index, end_index):
#     single_date = all_dates[i]
#     gPCE_model = statisticsObject.result_dict[single_uq][single_date]['gPCE']
#     gPCE_model_evaluated[single_date] = gPCE_model(uqef_simulationNodes.nodes.T)
#
# # Gather results from all processes into one dictionary
# all_results = comm.gather(gPCE_model_evaluated, root=0)
#
# # The root process (rank 0) will have all the results in one dictionary
# if rank == 0:
#     final_result = {}
#     for result_dict in all_results:
#         final_result.update(result_dict)
#     print("Final result:", final_result)
######################

# Use this one when running on your machine...
# Distribute work among processes
all_dates = statisticsObject.pdTimesteps
dates_per_process = len(all_dates) // size
remainder = len(all_dates) % size

# Calculate the range of dates for this process
start_index = rank * dates_per_process
end_index = (rank + 1) * dates_per_process + (1 if rank < remainder else 0)

# Initialize a dictionary to store the results for this process
gPCE_model_evaluated = {}

# Create a list to hold thread objects
threads = []

# Function for each thread to execute
def thread_worker(start, end):
    for i in range(start, end):
        date = all_dates[i]
        result = evaluate_gPCE_model_single_date(date, statisticsObject.result_dict, single_uq, uqef_simulationNodes)
        gPCE_model_evaluated[result[0]] = result[1]

# Number of threads per process (you can adjust this)
num_threads_per_process = 4

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

# Gather results from all processes into one dictionary
all_results = comm.gather(gPCE_model_evaluated, root=0)

# The root process (rank 0) will have all the results in one dictionary
if rank == 0:
    final_result = {}
    for result_dict in all_results:
        final_result.update(result_dict)
    print("Final result:", final_result)