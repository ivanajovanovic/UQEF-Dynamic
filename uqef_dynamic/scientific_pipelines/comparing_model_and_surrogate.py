"""
@author: Ivana Jovanovic Buha
This script is used to evaluate the surrogate model (gPCE) for a single QoI at all time steps.
It is parallelized using the multiprocessing module and is ment to be run on local machine.
"""
import inspect
import dill
import numpy as np
import sys
import pathlib
import pandas as pd
import pickle
import time

# for parallel computing
import multiprocessing
# import concurrent.futures
import psutil
# for message passing
# from mpi4py import MPI

# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo
# pd.options.plotting.backend = "plotly"

import chaospy as cp
import uqef

# additionally added for the debugging of the nodes
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

linux_cluster_run = False
# sys.path.insert(0, os.getcwd())
if linux_cluster_run:
    sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')
else:
    sys.path.insert(0, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')

from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqPostprocessing

#####################################
### MPI infos:
#####################################

# size = MPI.COMM_WORLD.Get_size()
# rank = MPI.COMM_WORLD.Get_rank()
# name = MPI.Get_processor_name()
# version = MPI.Get_library_version()
# version2 = MPI.Get_version()

# Define a function to process a single date
# def evaluate_gPCE_model_single_date(single_date, single_qoi, result_dict, gPCE_model_evaluated, uqef_simulationNodes):
#     gPCE_model = result_dict[single_qoi][single_date]['gPCE']
#     gPCE_model_evaluated[single_date] = gPCE_model(uqef_simulationNodes.nodes.T)

# # for concurrent.futures
# # Define a function to process a single date
# def evaluate_gPCE_model_single_date(single_date, single_qoi, result_dict, uqef_simulationNodes):
#     gPCE_model = result_dict[single_qoi][single_date]['gPCE']
#     gPCE_result = gPCE_model(uqef_simulationNodes.nodes.T)
#     return single_date, gPCE_result


def evaluate_gPCE_model_single_date(single_date, single_qoi, result_dict, nodes, workingDir):
    if 'gPCE' in result_dict[single_qoi][single_date]:
        gPCE_model = result_dict[single_qoi][single_date]['gPCE']
    else:
        gPCE_model = uqPostprocessing.read_single_gpce_surrogate_models(
            workingDir, single_qoi, single_date, throw_error=True)
    # result = gPCE_model(uqef_simulationNodes.nodes.T)
    # return (single_date, result)
    # result = []
    # for idx, sample in np.ndenumerate(nodes.T):
    #     result.append(gPCE_model(sample))
    # gPCE_model_evaluated[single_date] = np.array(gPCE_model(*nodes))
    return single_date, np.array(gPCE_model(*nodes))


if __name__ == '__main__':
    single_qoi = "Q_cms"
    sampling_rule = "random"  # 'sobol' 'random' 'halton' 'latin_hypercube'
    number_of_samples = 100
    sampleFromStandardDist = True

    # 8D gPCE l=7, p=2 Q_cms 2006 - 155
    # workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/paper_hydro_uq_sim/hbv_uq_cm2.0155')
    # 6D gPCE l=, p= Q_cms 2006 - 173
    if linux_cluster_run:
        workingDir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/paper_hydro_uq_sim/hbv_uq_cm2.0173")
    else:
        workingDir = pathlib.Path("/work/ga45met/mnt/linux_cluster_scratch_hbv_2/hbv_uq_cm2.0173")
    
    # Define the paths to the output files
    dict_output_file_paths = utility.get_dict_with_output_file_paths_based_on_workingDir(workingDir)
    args_file = dict_output_file_paths.get("args_file")
    configuration_object_file = dict_output_file_paths.get("configuration_object_file")
    nodes_file = dict_output_file_paths.get("nodes_file")
    df_all_index_parameter_file = dict_output_file_paths.get("df_all_index_parameter_file")
    df_all_index_parameter_gof_file = dict_output_file_paths.get("df_all_index_parameter_gof_file")
    df_all_simulations_file = dict_output_file_paths.get("df_all_simulations_file")

    # ====================================================================================
    # Reading save files
    # ====================================================================================

    # Load the UQSim args dictionary
    with open(args_file, 'rb') as f:
        uqsim_args = pickle.load(f)
    uqsim_args_dict = vars(uqsim_args)
    model = uqsim_args_dict["model"]
    inputModelDir = uqsim_args_dict["inputModelDir"]

    # Load the configuration object
    with open(configuration_object_file, 'rb') as f:
        configurationObject = dill.load(f)
    simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)
    if model == "hbvsask":
        basis = configurationObject['model_settings']['basis']

    # Reading Nodes and Parameters
    with open(nodes_file, 'rb') as f:
        simulationNodes = pickle.load(f)

    # Re-create simulationNodes from configuration file
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
    if df_all_index_parameter_file.is_file():
        df_index_parameter = pd.read_pickle(df_all_index_parameter_file, compression="gzip")
        params_list = utility._get_parameter_columns_df_index_parameter_gof(
            df_index_parameter)
    else:
        raise FileNotFoundError(f"File {df_all_index_parameter_file} not found; it is needed to get the list of uncertain parameters atm")

    if df_all_index_parameter_gof_file.is_file():
        df_index_parameter_gof = pd.read_pickle(df_all_index_parameter_gof_file, compression="gzip")
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)
    df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
    df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)

    # ====================================================================================
    # Re-creating Statstics Object
    # ====================================================================================

    # Reading Saved Simulations - Note: This migh be a huge file,
    # especially for MC/Saltelli kind of simulations
    if df_all_simulations_file.is_file():
        df_simulation_result = pd.read_pickle(df_all_simulations_file, compression="gzip")
    else:
        df_simulation_result = None

    # Re-create Statistics Object and DataFrame Object That contains all the Statistics Data
    statisticsObject = uqPostprocessing.create_statistics_object(
        configurationObject, uqsim_args_dict, workingDir, model=model)
    
    if single_qoi not in statisticsObject.list_qoi_column:
        raise ValueError(f"single_qoi - {single_qoi} not in statisticsObject.list_qoi_column - {statisticsObject.list_qoi_column}")

    # Recreate statisticsObject.result_dict
    statistics_dictionary = uqPostprocessing.read_all_saved_statistics_dict(\
        workingDir, [single_qoi, ], uqsim_args_dict.get("instantly_save_results_for_each_time_step", False), throw_error=True)

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

    # ====================================================================================

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

    ###########################
    # # for concurrent.futures
    # # Create a generator function to process dates concurrently
    # def process_dates_concurrently(dates):
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #         for date, result in executor.map(
    #                 evaluate_gPCE_model_single_date, dates, [single_qoi] * len(dates),
    #                 [statisticsObject.result_dict] * len(dates), [uqef_simulationNodes] * len(dates)
    #         ):
    #             yield date, result
    #
    # # Process dates concurrently using the generator
    # for date, gPCE_result in process_dates_concurrently(dates_to_process):
    #     gPCE_model_evaluated[date] = gPCE_result

    # ====================================================================================
    # Create a pool of processes / Paralle part . Evaluationg gPCE model in parallel
    # ====================================================================================

    # Number of parallel processes
    num_processes = multiprocessing.cpu_count()

    # List of dates to process (assuming statisticsObject.pdTimesteps is a list)
    dates_to_process = statisticsObject.pdTimesteps

    start = time.time()

    # Initialize a dictionary to store the results
    gPCE_model_evaluated = {}

    # Monitor memory usage at regular intervals
    memory_usage_history = []

    def process_dates_concurrently(dates):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for date, result in pool.starmap(evaluate_gPCE_model_single_date, \
                                       [(date, single_qoi, statisticsObject.result_dict, nodes, workingDir) for date in dates]):
                yield date, result
    for date, gPCE_result in process_dates_concurrently(dates_to_process):
        gPCE_model_evaluated[date] = gPCE_result

    # pool = multiprocessing.Pool(processes=num_processes)
    # # Parallelize the processing of dates
    # pool.map(evaluate_gPCE_model_single_date, dates_to_process)
    # # pool.starmap(evaluate_gPCE_model_single_date, [(date, single_qoi, statisticsObject.result_dict, gPCE_model_evaluated, uqef_simulationNodes) for date in dates_to_process])

    # Query memory usage and record it
    memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # Memory usage in MB
    memory_usage_history.append(memory_usage)
    print(f"Memory Usage History: {memory_usage_history}")

    # # Close the pool of processes
    # pool.close()
    # pool.join()
    ###########################
    # How I initially wanted to do it
    # # for single_date in statisticsObject.pdTimesteps:
    # #     gPCE_model = statisticsObject.result_dict["Q_cms"][single_date]['gPCE']
    # #     # gPCE_model_evaluated[single_date] = gPCE_model(samples_to_evaluate_gPCE_transformed.T)
    # #     # gPCE_model_evaluated[single_date] = gPCE_model(sample_to_evaluate_gPCE_min_1_1.T)
    # #     gPCE_model_evaluated[single_date] = gPCE_model(uqef_simulationNodes.nodes.T)
    # print(statisticsObject.pdTimesteps)
    # with multiprocessing.Pool() as pool:
    #     for date, gPCE_result in pool.imap(evaluate_gPCE_model_single_date, statisticsObject.pdTimesteps):
    #         gPCE_model_evaluated[date] = gPCE_result

    # ====================================================================================
    # Printing
    # ====================================================================================

    end = time.time()
    runtime = end - start
    print(f"multiprocessing.Pool: Time needed for evaluating {number_of_samples} \
    gPCE model (qoi is {single_qoi}) for {len(statisticsObject.pdTimesteps)} time steps is: {runtime}")
    print(f"gPCE_model_evaluated at times - {gPCE_model_evaluated.keys()} \n")

    print(f"len(statisticsObject.pdTimesteps) - {len(statisticsObject.pdTimesteps)}")
    print(f"len(dates_to_process) - {len(dates_to_process)}")
    print(f"len(gPCE_model_evaluated.keys()) - {len(gPCE_model_evaluated.keys())}")

    # Printing
    # print(gPCE_model_evaluated)
    temp_date = statisticsObject.pdTimesteps[-1]
    temp_evaluation_of_surrogate = gPCE_model_evaluated[temp_date]
    print(f"{temp_evaluation_of_surrogate.shape}")
    print(f"gPCE_model_evaluated for date {temp_date} - {temp_evaluation_of_surrogate}")

    # ====================================================================================
    # Plotting; Note: Mainly focused to HBV-SASK model
    # ====================================================================================

    directory_for_saving_plots = workingDir
    if not str(directory_for_saving_plots).endswith("/"):
        directory_for_saving_plots = str(directory_for_saving_plots) + "/"

    # Extract the lists from the dictionary
    lists = list(gPCE_model_evaluated.values())
    # Use the zip function to transpose the lists into columns
    gPCE_model_evaluated_matrix = list(zip(*lists))

    assert len(gPCE_model_evaluated_matrix[0]) == len(statisticsObject.pdTimesteps)

    # fig_1, max_n_row = uqPostprocessing.plot_forcing_mean_predicted_and_observed_all_qoi(
    #     statisticsObject, directory=directory_for_saving_plots, fileName=None)
    # fig = make_subplots(rows=max_n_row+1, cols=1, shared_xaxes=True)
    # n_row = 1
    # for trace in fig_1.data:
    #     fig.add_trace(trace, row=n_row, col=1)
    #     n_row += 1

    df_statistics_single_qoi = statisticsObject.df_statistics.loc[
        statisticsObject.df_statistics['qoi'] == single_qoi]
    corresponding_measured_column = statisticsObject.dict_corresponding_original_qoi_column[single_qoi]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True
    )
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
            name="Temperature", mode='lines+markers'
        ),
        row=2, col=1
    )
    lines = [
        go.Scatter(
            x=statisticsObject.pdTimesteps,
            y=single_row,
            showlegend=False,
            # legendgroup=colours[i],
            mode="lines",
            line=dict(
                color='LightSkyBlue'),
            opacity=0.1
        )
        for single_row in gPCE_model_evaluated_matrix
    ]
    # fig = go.Figure(
    #     data=lines,
    # )
    for trace in lines:
        fig.add_trace(trace, row=3, col=1)
    if 'E' in df_statistics_single_qoi.columns:
        fig.add_trace(
            go.Scatter(
                x=statisticsObject.pdTimesteps,
                y=df_statistics_single_qoi['E'],
                text=df_statistics_single_qoi['E'],
                name=f"Mean predicted {single_qoi}", mode='lines'),
            row=3, col=1
        )
    if 'measured' in df_statistics_single_qoi.columns:
        fig.add_trace(
            go.Scatter(
                x=statisticsObject.pdTimesteps,
                y=df_statistics_single_qoi['measured'],
                name=f"Observed {corresponding_measured_column}", mode='lines',
                line=dict(color='Yellow'),
            ),
            row=3, col=1
        )
    fig.update_traces(hovertemplate=None, hoverinfo='none')
    fig.update_xaxes(fixedrange=True, showspikes=True, spikemode='across', spikesnap="cursor", spikedash='solid', spikethickness=2, spikecolor='grey')
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(title_text="Detailed plot of most important time-series plus ensemble of surrogate (gPCE) evaluations")
    fig.update_layout(xaxis=dict(type="date"))
    fig.update_layout(xaxis_range=[min(statisticsObject.pdTimesteps),
                                   max(statisticsObject.pdTimesteps)])
    # fig.update_layout(yaxis_type=scale, hovermode="x", spikedistance=-1)
    fileName = "datailed_plot_all_qois_plus_gpce_ensemble.html"
    fileName = directory_for_saving_plots + fileName
    pyo.plot(fig, filename=fileName)

    # sys.stdout.write(json.dumps(gPCE_model_evaluated))

