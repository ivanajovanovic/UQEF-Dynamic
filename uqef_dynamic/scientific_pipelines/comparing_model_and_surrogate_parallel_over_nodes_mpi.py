"""
@author: Ivana Jovanovic Buha
This script is used to evaluate the surrogate model (gPCE) for a single QoI at all time steps.
It is parallelized using MPI.
"""
from distutils.util import strtobool
import dill
import inspect
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import os
import pathlib
import pandas as pd
import pickle
import subprocess
import sys
import time

# for message passing
from mpi4py import MPI
import psutil

# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo

import chaospy as cp
import uqef

# # additionally added for the debugging of the nodes
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# pd.options.mode.chained_assignment = None

linux_cluster_run = False
# sys.path.insert(0, os.getcwd())
if linux_cluster_run:
    sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')
else:
    sys.path.insert(0, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')

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
# num_threads = threading.active_count()

# TODO Add option to evaluate surrogate in paralle over nodes
# TODO Add option to evaluate over time Generalized S.S.I from the gPCE
# TODO Add option to evaluate / run KL surrogate
# TODO Extant such that it is not only plotted, but compared, via the sutable metric, with a ground-truth data / observed model runs...

# Define a function to process a single date
def evaluate_gPCE_model_single_date_old(single_date, single_qoi, nodes, gPCE_model_evaluated, result_dict=None, workingDir=None):
    if 'gPCE' in result_dict[single_qoi][single_date]:
        gPCE_model = result_dict[single_qoi][single_date]['gPCE']
    else:
        gPCE_model = uqef_dynamic_utils.read_single_gpce_surrogate_models(
            workingDir, single_qoi, single_date, throw_error=True)
    gPCE_model_evaluated[single_date] = np.array(gPCE_model(*nodes))


def evaluate_gPCE_model_single_date(single_date, gPCE_model, nodes, gPCE_model_evaluated):
    gPCE_model_evaluated[single_date] = np.array(gPCE_model(*nodes))


def evaluate_gPCE_model_multiple_nodes(gPCE_model, nodes, indices, gPCE_model_evaluated):
    my_nodes = nodes[:,indices]
    my_model_evaluated = np.array(gPCE_model(*my_nodes))
    for i, index in enumerate(indices):
        gPCE_model_evaluated[index] = my_model_evaluated[i]


def evaluate_gPCE_model_single_node(gPCE_model, single_node, single_index, gPCE_model_evaluated):
    pass

def evaluate_gPCE_surrogate_model_over_time_single_qoi(
    workingDir,  single_qoi="Value", number_of_samples=1000, sampling_rule="halton",
    sample_new_nodes_from_standard_dist=True, read_new_nodes_from_file=False,
    rounding=False, round_dec=4, inputModelDir=None, directory_for_saving_plots=None,  
    single_timestamp_single_file=False, printing=False, plotting=True, **kwargs):
    if rank == 0:
        print(f"Number of MPI processes: {size}")

        add_measured_data=kwargs.get('add_measured_data', False)
        add_forcing_data=kwargs.get('add_forcing_data', False)
        read_saved_simulations = kwargs.get('read_saved_simulations', False)

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

        jointDists = simulationNodes.joinedDists
        jointStandard = simulationNodes.joinedStandardDists

        # This variable marks if the surrogate model is meant to be evalauted in the parameters
        # originating from a 'standard' parameters space / distribution
        evaluateSurrogateAtStandardDist = statisticsObject.sampleFromStandardDist  # uqsim_args_dict['sampleFromStandardDist']

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

        print(f"DEBUGGING - nodes.shape={nodes.shape}")

        # =============================================================

        # statistics_result_dict = statisticsObject.result_dict  # statistics_dictionary
        statistics_pdTimesteps = statisticsObject.pdTimesteps
        statistics_single_qois = statisticsObject.result_dict[single_qoi].keys()

        gpce_surrogate = None 
        gpce_coeffs = None

        statisticsObject.prepareForScStatistics(
            simulationNodes, order=uqsim_args_dict['sc_p_order'], 
            poly_normed=uqsim_args_dict['sc_poly_normed'], 
            poly_rule=uqsim_args_dict['sc_poly_rule'], 
            regression=uqsim_args_dict['regression'], 
            cross_truncation=uqsim_args_dict['cross_truncation']
        )
        polynomial_expansion = statisticsObject.polynomial_expansion
        polynomial_norms = statisticsObject.polynomial_norms

        gpce_surrogate = uqef_dynamic_utils.fetch_gpce_surrogate_single_qoi(
            qoi_column_name=single_qoi, workingDir=workingDir,
            statistics_dictionary=statisticsObject.result_dict, 
            throw_error=False, single_timestamp_single_file=single_timestamp_single_file)

        if gpce_surrogate is None:
            gpce_coeffs = uqef_dynamic_utils.fetch_gpce_coeff_single_qoi(
                qoi_column_name=single_qoi, workingDir=workingDir,
                statistics_dictionary=statisticsObject.result_dict, 
                throw_error=False, single_timestamp_single_file=single_timestamp_single_file)
            #convert_to_pd_timestamp
            # If you have only saved gPCE coefficints, then one must build the polynomial basis as well..
            # polynomial_basis

            # check what is the structure of gpce_coeffs
            # gpce_surrogate = defaultdict()
            gpce_surrogate = utility.build_gpce_surrogate_from_coefficients(
                gpce_coeffs, polynomial_expansion, polynomial_norms)

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
            print(f"gpce_surrogate - {gpce_surrogate}")
            temp = gpce_coeffs[statistics_pdTimesteps[0]]
            print(f"gpce_coeffs - {temp}")

        # =============================================================

        # Add measured Data and/or forcing
        # This might be relevant for plotting in the end
        if printing:
            print(f"inputModelDir from model - {inputModelDir}; inputModelDir_function_input_argument-{inputModelDir_function_input_argument}")
            print(f"workingDir from model - {workingDir}; workingDir_function_input_argument-{workingDir_function_input_argument}")

        df_statistics_and_measured = statisticsObject.merge_df_statistics_data_with_measured_and_forcing_data(
            add_measured_data=add_measured_data, add_forcing_data=add_forcing_data, transform_measured_data_as_original_model=True)

        # filter only relevant qoi
        df_statistics_and_measured_single_qoi = df_statistics_and_measured[df_statistics_and_measured["qoi"]==single_qoi]

        if printing:
            print(f"statisticsObject.df_statistics-{statisticsObject.df_statistics}")
            print(f"statisticsObject.forcing_df-{statisticsObject.forcing_df}")
            print(f"statisticsObject.df_measured-{statisticsObject.df_measured}")
            print(f"df_statistics_and_measured-{df_statistics_and_measured}")
            print(f"df_simulation_result-{df_simulation_result}")

    else:
        gpce_surrogate = None
        gpce_coeffs = None
        single_qoi = None
        nodes = None
        statistics_pdTimesteps = None
        # statistics_result_dict = None
        workingDir = None
        polynomial_expansion = None
        single_timestamp_single_file = None

    single_qoi = comm.bcast(single_qoi, root=0)
    nodes = comm.bcast(nodes, root=0)
    statistics_pdTimesteps = comm.bcast(statistics_pdTimesteps, root=0)
    # statistics_result_dict = comm.bcast(statistics_result_dict, root=0)
    workingDir = comm.bcast(workingDir, root=0)
    gpce_surrogate = comm.bcast(gpce_surrogate, root=0)
    gpce_coeffs = comm.bcast(gpce_coeffs, root=0)
    polynomial_expansion = comm.bcast(polynomial_expansion, root=0)
    # List of dates to process (assuming statisticsObject.pdTimesteps is a list)
    dates_to_process = statistics_pdTimesteps
    single_timestamp_single_file = single_timestamp_single_file

    if rank == 0:
        start = time.time()

    # Initialize a dictionary to store the results
    gPCE_model_evaluated = {}

    if nodes is None:
        raise Exception("Nodes are not defined!")
    dim  = nodes.shape[0]
    number_of_samples = nodes.shape[1]

    # Things should be changed from this point on...
    # Split the dates among processes
    chunk_size = len(number_of_samples) // size
    remainder = len(number_of_samples) % size
    start_index = rank * chunk_size
    end_index = start_index + chunk_size if rank < size - 1 else number_of_samples

    # Distribute dates to processes
    my_nodes = nodes[:,start_index:end_index]
    my_indices = list(range(start_index, end_index))
    print(f"INFO-{rank} - my_nodes.shape={my_nodes.shape}")

    # Monitor memory usage at regular intervals
    memory_usage_history = []

    # =============================================================

    evaluate_gPCE_model_multiple_nodes(
        gPCE_model=gpce_surrogate, nodes=nodes, indices=my_indices, gPCE_model_evaluated=gPCE_model_evaluated)
    
    # # Process dates
    # for single_index in my_indices:
    #     node = nodes[:, single_index]
    #     evaluate_gPCE_model_single_node(
    #         gPCE_model=gpce_surrogate, single_node=node, single_index=single_index,, gPCE_model_evaluated=gPCE_model_evaluated)

    #     # Query memory usage and record it
    #     memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # Memory usage in MB
    #     memory_usage_history.append(memory_usage)

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

    # Print memory usage history
    print(f"Memory Usage History (Rank {rank}): {memory_usage_history}")

    if rank == 0:
        # TODO One gPCE_model_evaluated is populated, extra work has to be done to get the results for all QoIs,
        # especially if strtobool(configurationObject["simulation_settings"]["autoregressive_model_first_order"])
        end = time.time()
        runtime = end - start
        print(f"MPI: Time needed for evaluating {number_of_samples} \
         gPCE model (qoi is {single_qoi}) for {len(statistics_pdTimesteps)} time steps is: {runtime}")
        print(f"gPCE_model_evaluated at times - {gPCE_model_evaluated.keys()} \n")

        print(f"len(statisticsObject.pdTimesteps) - {len(statisticsObject.pdTimesteps)}")
        print(f"len(dates_to_process) - {len(dates_to_process)}")
        print(f"len(gPCE_model_evaluated.keys()) - {len(gPCE_model_evaluated.keys())}")
        print(f"len(statistics_pdTimesteps) - {len(statistics_pdTimesteps)}")
 
        print(f"gPCE_model_evaluated={gPCE_model_evaluated}")
        
        # # Some debugging
        # temp_date = statisticsObject.pdTimesteps[-1]  # dates_to_process[-1]
        # # print(f"temp_date - {temp_date}, {type(temp_date)}")
        # temp_evaluation_of_surrogate = gPCE_model_evaluated[temp_date]
        # print(f"DEBUGGING type(temp_evaluation_of_surrogate)  - {type(temp_evaluation_of_surrogate)}")
        # print(f"DEBUGGING len(temp_evaluation_of_surrogate)  - {temp_evaluation_of_surrogate.shape}")
        # print(f"gPCE_model_evaluated for date {temp_date} - {temp_evaluation_of_surrogate}")
        # # # print(f"{type(gPCE_model_evaluated[gPCE_model_evaluated.keys()[0]])} \n {gPCE_model_evaluated[gPCE_model_evaluated.keys()[0]]}")

        # # ====================================================================================
        # # Plotting
        # # ====================================================================================

        # if directory_for_saving_plots is None:
        #     directory_for_saving_plots = workingDir
        # if not str(directory_for_saving_plots).endswith("/"):
        #     directory_for_saving_plots = str(directory_for_saving_plots) + "/"
        # if not os.path.isdir(directory_for_saving_plots):
        #     subprocess.run(["mkdir", directory_for_saving_plots])

        # # Extract the lists from the dictionary
        # lists = list(gPCE_model_evaluated.values())
        # print(f"DEBUGGING len(lists)={len(lists)}")
        # # Use the zip function to transpose the lists into columns
        # gPCE_model_evaluated_matrix = list(zip(*lists))

        # assert len(gPCE_model_evaluated_matrix[0]) == len(statisticsObject.pdTimesteps)

        # df_statistics_single_qoi = statisticsObject.df_statistics.loc[
        #     statisticsObject.df_statistics['qoi'] == single_qoi]
        # corresponding_measured_column = statisticsObject.dict_corresponding_original_qoi_column[single_qoi]
        
        # fig = make_subplots(
        #     rows=3, cols=1, shared_xaxes=True
        # )
        # fig.add_trace(
        #     go.Bar(
        #         x=statisticsObject.forcing_df.index, y=statisticsObject.forcing_df['precipitation'],
        #         text=statisticsObject.forcing_df['precipitation'],
        #         name="Precipitation"
        #     ),
        #     row=1, col=1
        # )
        # fig.add_trace(
        #     go.Scatter(
        #         x=statisticsObject.forcing_df.index, y=statisticsObject.forcing_df['temperature'],
        #         text=statisticsObject.forcing_df['temperature'],
        #         name="Temperature", mode='lines+markers'
        #     ),
        #     row=2, col=1
        # )
        # lines = [
        #     go.Scatter(
        #         x=statisticsObject.pdTimesteps,
        #         y=single_row,
        #         showlegend=False,
        #         # legendgroup=colours[i],
        #         mode="lines",
        #         line=dict(
        #             color='LightSkyBlue'),
        #         opacity=0.1
        #     )
        #     for single_row in gPCE_model_evaluated_matrix
        # ]
        # # fig = go.Figure(
        # #     data=lines,
        # # )
        # for trace in lines:
        #     fig.add_trace(trace, row=3, col=1)
        
        # if 'E' in df_statistics_single_qoi.columns:
        #     fig.add_trace(
        #         go.Scatter(
        #             x=statisticsObject.pdTimesteps,
        #             y=df_statistics_single_qoi['E'],
        #             text=df_statistics_single_qoi['E'],
        #             name=f"Mean predicted {single_qoi}", mode='lines'),
        #         row=3, col=1
        #     )
        # if 'measured' in df_statistics_single_qoi.columns:
        #     fig.add_trace(
        #         go.Scatter(
        #             x=statisticsObject.pdTimesteps,
        #             y=df_statistics_single_qoi['measured'],
        #             name=f"Observed {corresponding_measured_column}", mode='lines',
        #             line=dict(color='Yellow'),
        #         ),
        #         row=3, col=1
        #     )
        # fig.update_traces(hovertemplate=None, hoverinfo='none')
        # fig.update_xaxes(fixedrange=True, showspikes=True, spikemode='across', spikesnap="cursor", spikedash='solid', spikethickness=2, spikecolor='grey')
        # fig.update_yaxes(autorange="reversed", row=1, col=1)
        # fig.update_yaxes(fixedrange=True)
        # fig.update_layout(title_text="Detailed plot of most important time-series plus ensemble of surrogate (gPCE) evaluations")
        # fig.update_layout(
        #     xaxis=dict(
        #     rangemode='normal',
        #     range=[min(statisticsObject.pdTimesteps), max(statisticsObject.pdTimesteps)],
        #     type="date")
        # )
        # fig.update_xaxes(
        #     tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
        #     dtick="M2"                     # Set tick interval to 1 day for denser ticks
        # )
        # fig.update_layout(height=1100, width=1100)
        # fig.update_layout(title=None)
        # fig.update_layout(
        #     margin=dict(
        #         t=10,  # Top margin
        #         b=10,  # Bottom margin
        #         l=20,  # Left margin
        #         r=20   # Right margin
        #     )
        # )
        # # fig.update_layout(xaxis_range=[min(statisticsObject.pdTimesteps),
        # #                             max(statisticsObject.pdTimesteps)])
        # # fig.update_layout(yaxis_type=scale, hovermode="x", spikedistance=-1)
        # fileName = "datailed_plot_all_qois_plus_gpce_ensemble.html"
        # fileName =str(directory_for_saving_plots) + fileName
        # pyo.plot(fig, filename=fileName)
        # fileName = "datailed_plot_all_qois_plus_gpce_ensemble.pdf"
        # fileName =str(directory_for_saving_plots) + fileName
        # fig.write_image(str(fileName), format="pdf", width=1100,)

    # else:
    #     comm.Abort(0)  # 0 is the error code to indicate a successful shutdown

    MPI.Finalize()


if __name__ == '__main__':
    basis_workingDir = pathlib.Path('/work/ga45met/paper_uqef_dynamic_sim/hbvsask_runs_lxc_autumn_24')

    # 8D gPCE l=7, p=2 Q_cms 2006 - 155
    # workingDir = pathlib.Path('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/paper_hydro_uq_sim/hbv_uq_cm2.0155')
    # 6D gPCE l=, p= Q_cms 2006 - 173
    workingDir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/paper_hydro_uq_sim/hbv_uq_cm2.0173")
    # 3D Sparse-gPCE l=7, p=3 2005-2007 deltaQ_cms
    workingDir = pathlib.Path('/gpfs/scratch/pr63so/ga45met2/hbvsask_runs/pce_deltq_3d_longer_oldman')
    # 10D Sparse-gPCE l=6, p=4, ct=0.7 2006 Q_cms
    workingDir = pathlib.Path('/work/ga45met/mnt/hbv_uq_mpp3.0035')
    # 10D Sparse-gPCE l=6, p=5, ct=0.7 2004-2005 Q_cms + generalized S.S.I 
    workingDir =basis_workingDir / 'hbv_uq_mpp3.0053'

    # Parameters relevant for generating MC like samples to evaluate the surrogate gPCE model
    sampling_rule = "latin_hypercube"  # 'sobol' 'random'
    number_of_samples = 100
    sample_new_nodes_from_standard_dist = False

    single_qoi="Q_cms"  # e.g., "Q_cms" 'delta_Q_cms'

    inputModelDir = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
    directory_for_saving_plots = pathlib.Path('/work/ga45met/paper_uqef_dynamic_sim/hbv_sask/gpce_p4_sgl6_ct07_generalized_2006_oldman')
    directory_for_saving_plots = pathlib.Path('/work/ga45met/paper_uqef_dynamic_sim/hbv_sask/gpce_p5_sgl6_ct07_generalized_2006_oldman')
    directory_for_saving_plots = pathlib.Path('/work/ga45met/paper_uqef_dynamic_sim/hbv_sask/gpce_p5_sgl6_ct07_generalized_2004_2007_oldmans')
    evaluate_gPCE_surrogate_model_over_time_single_qoi(
        workingDir, single_qoi=single_qoi, 
        sample_new_nodes_from_standard_dist=sample_new_nodes_from_standard_dist, 
        sampling_rule=sampling_rule, 
        number_of_samples=number_of_samples,
        inputModelDir=inputModelDir,
        directory_for_saving_plots=directory_for_saving_plots,
        single_timestamp_single_file=False,
        add_measured_data=True,
        add_forcing_data=True,
        read_saved_simulations=True,
        printing=True, plotting=True,
    )