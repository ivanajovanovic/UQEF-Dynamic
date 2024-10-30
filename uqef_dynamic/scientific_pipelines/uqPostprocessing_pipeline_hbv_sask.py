import dill
import pathlib
import pandas as pd
import pickle
from collections import defaultdict
import plotly.offline as pyo

pd.options.plotting.backend = "plotly"

from uqef_dynamic.utils import utility
from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv
from uqef_dynamic.models.hbv_sask import HBVSASKModel as hbvmodel
from uqef_dynamic.models.hbv_sask import HBVSASKStatistics
from uqef_dynamic.utils import uqef_dynamic_utils
from uqef_dynamic.utils import create_stat_object

# Defining Paths
# TODO - change these paths accordingly
hbv_model_data_path = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
inputModelDir = hbv_model_data_path
basis = "Oldman_Basin"  # 'Banff_Basin'
# TODO - change these paths accordingly
configurationObject = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/configurations/configuration_hbv_5D.json')
# 8D Saltelli 10 000 Q_cms, AET - 141
workingDir = pathlib.Path('/work/ga45met/mnt/linux_cluster_scratch_hbv_2/hbv_uq_cm2.0141/')
# 5D Sparse-gPCE l=6, p=3 Q_cms May_2005- gpce_d5_l6_p3
workingDir = pathlib.Path('/work/ga45met/mnt/linux_cluster_scratch_hbv_2/gpce_d5_l6_p3_summer_2006/')
# 8D gPCE l=7, p=2 Q_cms 2006 - 155
workingDir = pathlib.Path('/work/ga45met/mnt/linux_cluster_scratch_hbv_2/hbv_uq_cm2.0155/')
# 10D Saltelli GoF sliding window Q_cms 2006 - 171
workingDir = pathlib.Path('/work/ga45met/mnt/linux_cluster_scratch_hbv_2/hbv_uq_cm2.0171/')
# 6D gPCE l=, p=3 Q_cms 2006- 173
workingDir = pathlib.Path('/work/ga45met/mnt/linux_cluster_scratch_hbv_2/hbv_uq_cm2.0173/')

#
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
# Files produced by Samples class
df_time_varying_grad_analysis_file = dict_output_file_paths.get("df_time_varying_grad_analysis_file")
df_time_aggregated_grad_analysis_file = dict_output_file_paths.get("df_time_aggregated_grad_analysis_file")
# Files produced by UQEF.Statistics and tatistics

# Active Subspaces related files
dict_approx_matrix_c_file = dict_output_file_paths.get("dict_approx_matrix_c_file")
dict_matrix_c_eigen_decomposition_file = dict_output_file_paths.get("dict_matrix_c_eigen_decomposition_file")

# statistics_dictionary_file = utility.get_dict_with_qoi_name_specific_output_file_paths_based_on_workingDir(\
    # workingDir, qoi_string=qoi_string)
# flux_df_path = None
# gof_df_path = None
# parameters_dict_path = None
# def update_output_file_paths_for_specific_model_run(workingDir, index_run=0):
#     # Sepcific run related files, might not exist, depending on the configuration
#     global flux_df_path, gof_df_path, parameters_dict_path
#     flux_df_path = workingDir / f'flux_df_{index_run}.pkl'
#     gof_df_path = workingDir / f'gof_{index_run}.pkl'
#     parameters_dict_path = workingDir / f'parameters_HBVSASK_run_{index_run}.pkl'
# update_output_file_paths_for_specific_model_run(workingDir, index_run=0)

# Reading Save Data
# Load the configuration object
configurationObject = utility.load_configuration_object(workingDir)
print("configurationObject: ", configurationObject)
simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)

# Load the UQSim args dictionary
uqsim_args_dict = utility.load_uqsim_args_dict(args_file)
print("INFO: uqsim_args_dict: ", uqsim_args_dict)
model = uqsim_args_dict["model"]

# Reading Nodes and Parameters
with open(nodes_file, 'rb') as f:
    simulationNodes = pickle.load(f)
print("INFO: simulationNodes: ", simulationNodes)
dim = simulationNodes.nodes.shape[0]
model_runs = simulationNodes.nodes.shape[1]
distStandard = simulationNodes.joinedStandardDists
dist = simulationNodes.joinedDists
print(f"INFO: model-{model}; dim - {dim}; model_runs - {model_runs}")
print(simulationNodes.nodes.shape)
print(simulationNodes.parameters.shape)

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

## Nodes & Parameter  in a DataFrame -  after transformation
df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)

# Reading Saved Simulations
# or in case of a big simulation, skip reading df_simulation_result
read_all_saved_simulations_file = False
if read_all_saved_simulations_file and df_simulations_file.is_file():
    # Reading Saved Simulations - Note: This migh be a huge file,
    # especially for MC/Saltelli kind of simulations
    df_simulation_result = pd.read_pickle(df_simulations_file, compression="gzip")
else:
    df_simulation_result = None

# Re-create Statistics Object and DataFrame Object That contains all the Statistics Data
statisticsObject = create_stat_object.create_statistics_object(
    configuration_object=configurationObject, uqsim_args_dict=uqsim_args_dict, \
    workingDir=workingDir, model=model)

statistics_dictionary = uqef_dynamic_utils.read_all_saved_statistics_dict(\
    workingDir=workingDir, list_qoi_column=statisticsObject.list_qoi_column, 
    single_timestamp_single_file=uqsim_args_dict.get("instantly_save_results_for_each_time_step", False), 
    throw_error=False
    )

uqef_dynamic_utils.extend_statistics_object(
    statisticsObject=statisticsObject,
    statistics_dictionary=statistics_dictionary,
    df_simulation_result=df_simulation_result,
    get_measured_data=False,
    get_unaltered_data=False
)

# Add measured Data - This is hardcoded for HBV
statisticsObject.inputModelDir_basis = hbv_model_data_path / basis
statisticsObject.get_measured_data(
    timestepRange=(statisticsObject.timesteps_min, statisticsObject.timesteps_max),
    transforme_mesured_data_as_original_model="False")

# Create a Pandas.DataFrame
df_statistics = statisticsObject.create_df_from_statistics_data()

# Add forcing Data
statisticsObject.get_forcing_data(time_column_name="TimeStamp")

# Merge Everything
df_statistics_and_measured = pd.merge(
    statisticsObject.df_statistics, statisticsObject.forcing_df, left_on=statisticsObject.time_column_name, right_index=True)
print(df_statistics_and_measured)

# Sensitivity Analysis - Recomputing DataFrames
si_m_df = statisticsObject.create_df_from_sensitivity_indices(
    si_type="Sobol_m",compute_measured_normalized_data=True)
si_t_df = statisticsObject.create_df_from_sensitivity_indices(
    si_type="Sobol_t",compute_measured_normalized_data=True)

# Describe - Statistics of Statistics DataFrame :-)
print(statisticsObject.describe_df_statistics())
# or, step-by-step
# for single_qoi in statisticsObject.list_qoi_column:
#     df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured.loc[
#         df_statistics_and_measured['qoi'] == single_qoi]
#     print(f"{single_qoi}\n\n")
#     print(df_statistics_and_measured_single_qoi_subset.describe(include=np.number))

# Plotting Producing Details Statistics Plots
dict_what_to_plot = {
            "E_minus_std": True, "E_plus_std": True, "P10": True, "P90": True,
            "StdDev": True, "Skew": True, "Kurt": True, "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True
        }

directory_for_saving_plots = workingDir
if not str(directory_for_saving_plots).endswith("/"):
    directory_for_saving_plots = str(directory_for_saving_plots) + "/"

# Calling function(s) from uqef_dynamic_utils module
for single_qoi in statisticsObject.list_qoi_column:
    df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured.loc[
        df_statistics_and_measured['qoi'] == single_qoi]
    fig = uqef_dynamic_utils.plotting_function_single_qoi(
        df_statistics_and_measured_single_qoi_subset,
        single_qoi=single_qoi,
        qoi=statisticsObject.qoi,  # this is relevan only when qoi is == "GoF"/"gof"
        dict_what_to_plot=dict_what_to_plot,
        directory=statisticsObject.workingDir,
        fileName=f"simulation_big_plot_{single_qoi}.html"
    )
    fig.show()

fig, _ = uqef_dynamic_utils.plot_forcing_mean_predicted_and_observed_all_qoi(
    statisticsObject, directory=directory_for_saving_plots, fileName="Datailed_plot_all_qois.html")
fig.show()

# Calling plotting method from Statistics class
dict_what_to_plot = {
            "E_minus_std": False, "E_plus_std": False, "P10": True, "P90": True,
            "StdDev": True, "Skew": False, "Kurt": False, "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True
        }
statisticsObject.prepare_for_plotting(
    plot_measured_timeseries=True,
    plot_forcing_timeseries=True,
    time_column_name="TimeStamp"
)
# single_qoi = statisticsObject.list_qoi_column[0]
for single_qoi in statisticsObject.list_qoi_column:
    statisticsObject.plotResults_single_qoi(
        directory = directory_for_saving_plots,
        fileName = f"{single_qoi}_STAT_SI_TimeSignals",
        display=True, dict_time_vs_qoi_stat=None,
        single_qoi_column=single_qoi,
        precipitation_df_timestamp_column="index",
        temperature_df_timestamp_column="index",
        streamflow_df_timestamp_column="index",
        dict_what_to_plot=dict_what_to_plot
    )

# Plotting - Sensitivity Analysis
for single_qoi in statisticsObject.list_qoi_column:
    fig = statisticsObject.plot_heatmap_si_single_qoi(
        qoi_column=single_qoi, si_df=None, si_type="Sobol_m")
    fig.update_layout(title_text=f"Sobol First-order SI w.r.t. QoI - {single_qoi}")
    fileName = directory_for_saving_plots + f"Sobol_First_HeatMap_{single_qoi}.html"
    pyo.plot(fig, filename=fileName)
    fig.show()
for single_qoi in statisticsObject.list_qoi_column:
    fig = statisticsObject.plot_heatmap_si_single_qoi(
        qoi_column=single_qoi, si_type="Sobol_t")
    fig.update_layout(title_text=f"Sobol Total SI w.r.t. QoI - {single_qoi}")
    fileName = directory_for_saving_plots + f"Sobol_Total_HeatMap_{single_qoi}.html"
    pyo.plot(fig, filename=fileName)
    fig.show()
for single_qoi in statisticsObject.list_qoi_column:
    fig = statisticsObject.plot_si_and_normalized_measured_time_signal_single_qoi(
        qoi_column=single_qoi, si_df=None, si_type="Sobol_t")
    fig.update_layout(title_text=f"Sobol Total SI w.r.t. QoI - {single_qoi}")
    fileName = directory_for_saving_plots + f"Sobol_Total_Time_Signal_{single_qoi}.html"
    pyo.plot(fig, filename=fileName)
    fig.show()

# GPCE surrogate if computed:
gPCE_model = defaultdict()
for single_date in statisticsObject.pdTimesteps:
    gPCE_model[single_date] = statisticsObject.result_dict["Q_cms"][single_date]['gPCE']

