#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[161]:


import dill
import numpy as np
import sys
import pathlib
import pandas as pd
import pickle
import time


# In[162]:


# TODO - change these paths accordingly
# sys.path.insert(1, '/work/ga45met/Hydro_Models/HBV-SASK-py-tool')
sys.path.insert(1, '/work/ga45met/mnt/linux_cluster_2/UQEFPP')


# In[163]:


from common import utility
from hbv_sask import hbvsask_utility as hbv
from hbv_sask import HBVSASKModel as hbvmodel
from hbv_sask import HBVSASKStatisticsMultipleQoI as HBVSASKStatistics
from common import utility


# In[164]:


from common import uqPostprocessing


# In[165]:


# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# from plotly.offline import plot
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()

import matplotlib.pyplot as plt

pd.options.plotting.backend = "plotly"


# In[166]:


import scipy.special
scipy.special.binom(8, 5)


# In[167]:


import chaospy as cp


# # Defining paths

# In[168]:


# TODO - change these paths accordingly
hbv_model_data_path = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
inputModelDir = hbv_model_data_path
basis = "Oldman_Basin"  # 'Banff_Basin'


# In[169]:


# TODO - change these paths accordingly
configurationObject = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/UQEFPP/configurations/configuration_hbv_5D.json')
# 8D Saltelli 10 000 Q_cms, AET - 141
workingDir = pathlib.Path('/work/ga45met/mnt/linux_cluster_scratch_hbv_2/hbv_uq_cm2.0141/')

# 5D Sparse-gPCE l=6, p=3 Q_cms May_2005- gpce_d5_l6_p3
workingDir = pathlib.Path('/work/ga45met/mnt/linux_cluster_scratch_hbv_2/gpce_d5_l6_p3_summer_2006/')

# 8D gPCE l=7, p=2 Q_cms 2006 - 155
workingDir = pathlib.Path('/work/ga45met/mnt/linux_cluster_scratch_hbv_2/hbv_uq_cm2.0155/')

# workingDir = pathlib.Path('/work/ga45met/mnt/linux_cluster_scratch_hbv_2/grad_analysisi_in_statistics')


# In[170]:


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

output_stat_graph_filename = workingDir / "sim-plotly.html"
output_stat_graph_filename = str(output_stat_graph_filename)

# Sepcific run related files, might not exist, depending on the configuration
flux_df_path = workingDir / 'flux_df_0.pkl'
gof_df_path = workingDir / 'gof_0.pkl'
parameters_dict_path = workingDir / 'parameters_HBVSASK_run_0.pkl'


def update_output_file_paths_based_on_workingDir(workingDir):
    global nodes_file, parameters_file, args_file, configuration_object_file,     df_all_simulations_file, df_all_index_parameter_gof_file, df_all_index_parameter_file,    df_time_varying_grad_analysis_file, df_time_aggregated_grad_analysis_file,    statistics_dictionary_file, dict_of_approx_matrix_c_file, dict_of_matrix_c_eigen_decomposition_file,    output_stat_graph_filename
    
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

    output_stat_graph_filename = workingDir / "sim-plotly.html"
    output_stat_graph_filename = str(output_stat_graph_filename)

    # Sepcific run related files, might not exist, depending on the configuration
    flux_df_path = workingDir / 'flux_df_0.pkl'
    gof_df_path = workingDir / 'gof_0.pkl'
    parameters_dict_path = workingDir / 'parameters_HBVSASK_run_0.pkl'

    
def update_output_file_paths_for_specific_model_run(workingDir, index_run=0):
    global flux_df_path, gof_df_path, parameters_dict_path
    flux_df_path = workingDir / f'flux_df_{index_run}.pkl'
    gof_df_path = workingDir / f'gof_{index_run}.pkl'
    parameters_dict_path = workingDir / f'parameters_HBVSASK_run_{index_run}.pkl'
  

update_output_file_paths_based_on_workingDir(workingDir)
update_output_file_paths_for_specific_model_run(workingDir, index_run=0)


# # Reading Saved Files

# In[171]:


with open(configuration_object_file, 'rb') as f:
    configurationObject = dill.load(f)
configurationObject


# In[172]:


simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)
simulation_settings_dict


# In[173]:


with open(args_file, 'rb') as f:
    uqsim_args = pickle.load(f)
uqsim_args_dict = vars(uqsim_args)
uqsim_args_dict


# #### extra anlyzing

# In[174]:


# Number of uncertain parameters
len(configurationObject["parameters"])


# In[175]:


# Quantity of Interes
print(configurationObject["simulation_settings"]["qoi"])
print(configurationObject["simulation_settings"]["qoi_column"])


# # Reading Nodes and Parameters

# In[176]:


with open(nodes_file, 'rb') as f:
#     simulationNodes = dill.load(f)
    simulationNodes = pickle.load(f)
simulationNodes

print(simulationNodes.nodes.shape)
print(simulationNodes.parameters.shape)


# #### extra anlyzing

# In[177]:


simulationNodes.nodes


# In[178]:


simulationNodes.nodes.shape


# In[179]:


simulationNodes.parameters


# In[180]:


simulationNodes.parameters.shape


# In[181]:


simulationNodes.joinedDists


# In[182]:


simulationNodes.joinedStandardDists


# # Reading Prameters and GoF Computed Data

# In[183]:


df_index_parameter = pd.read_pickle(df_all_index_parameter_file, compression="gzip")
# params_list = LarsimUQPostprocessing._get_parameter_columns_df_index_parameter_gof(
#     df_index_parameter_gof)
df_index_parameter


# In[184]:


df_index_parameter_gof = pd.read_pickle(df_all_index_parameter_gof_file, compression="gzip")
df_index_parameter_gof


# In[185]:


params_list = utility._get_parameter_columns_df_index_parameter_gof(
    df_index_parameter_gof)
params_list


# In[186]:


gof_list = utility._get_gof_columns_df_index_parameter_gof(
    df_index_parameter_gof)
gof_list


# #### extra anlyzing

# In[187]:


df_index_parameter_gof['RMSE'].min()


# In[188]:


fig = utility.plot_subplot_params_hist_from_df(df_index_parameter_gof)
fig.update_layout(title="Prior Distribution of the Parameters",)
fig.show()


# In[31]:


df_index_parameter_gof["TT"].nunique()


# In[32]:


df_index_parameter_gof["TT"].unique()


# In[33]:


print(df_index_parameter_gof['RMSE'].min())
print(df_index_parameter_gof['RMSE'].max())
print(df_index_parameter_gof['LogNSE'].min())
print(df_index_parameter_gof['LogNSE'].max())
print(df_index_parameter_gof['KGE'].min())
print(df_index_parameter_gof['KGE'].max())


# In[34]:


fig = utility.plot_subplot_params_hist_from_df_conditioned(
    df_index_parameter_gof, name_of_gof_column="LogNSE", 
    threshold_gof_value = 0.6, comparison="greater")
fig.update_layout(title="Prior Distribution of the Parameters Conditioned on 'good' values of LogNSE",)
fig.show()


# In[37]:


fig = utility.plot_subplot_params_hist_from_df_conditioned(
    df_index_parameter_gof, name_of_gof_column="RMSE", 
    threshold_gof_value = 10.0, comparison="smaller")
fig.update_layout(title="Prior Distribution of the Parameters Conditioned on 'good' values of RMSE",)
fig.show()


# In[38]:


fig = utility.plot_subplot_params_hist_from_df_conditioned(
    df_index_parameter_gof, name_of_gof_column="KGE", 
    threshold_gof_value = 0.6, comparison="greater")
fig.update_layout(title="Prior Distribution of the Parameters Conditioned on 'good' values of KGE",)
fig.show()


# In[39]:


fig = utility.plot_scatter_matrix_params_vs_gof(
    df_index_parameter_gof, name_of_gof_column="RMSE",
    hover_name="index_run"
)
fig.show()


# In[40]:


fig = utility.plot_parallel_params_vs_gof(
    df_index_parameter_gof, name_of_gof_column="NSE"
)
fig.show()


# In[41]:


fig = utility.plot_parallel_params_vs_gof(
    df_index_parameter_gof, name_of_gof_column="LogNSE"
)
fig.show()


# In[42]:


fig = utility.plot_parallel_params_vs_gof(
    df_index_parameter_gof, name_of_gof_column="KGE"
)
fig.show()


# ## Nodes & Parameter Values after transformation

# In[189]:


df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)


# #### extra anlyzing

# In[45]:


df_nodes


# In[46]:


df_nodes_params


# # Reading Saved Simulations
# Note: This migh be a huge file

# In[190]:


df_simulation_result = pd.read_pickle(df_all_simulations_file, compression="gzip")
df_simulation_result


# # Reading Statistics Dict for a Specific QoI

# In[191]:


simulation_settings_dict["list_qoi_column"]


# In[192]:


qoi = simulation_settings_dict["list_qoi_column"][0]


# In[193]:


qoi


# In[194]:


statistics_dictionary_file = workingDir / f"statistics_dictionary_qoi_{qoi}.pkl"


# In[195]:


statistics_dictionary_file.is_file()


# In[49]:


with open(statistics_dictionary_file, 'rb') as f:
    statistics_dictionary = pickle.load(f)


# In[50]:


len(statistics_dictionary)


# In[51]:


simulation_time_steps = list(statistics_dictionary.keys())


# In[52]:


statistics_dictionary[simulation_time_steps[29]]


# ## Re-create Statistics Object and DataFrame Object

# In[196]:


statisticsObject = uqPostprocessing.create_statistics_object(
    configurationObject, uqsim_args_dict, workingDir, model="hbvsask")


# In[197]:


statisticsObject


# In[198]:


# Make one big statistics_dictionary
from collections import defaultdict
new_statistics_dictionary = defaultdict(dict)
for single_qoi in statisticsObject.list_qoi_column:
    statistics_dictionary_file_temp = workingDir / f"statistics_dictionary_qoi_{single_qoi}.pkl"
    assert statistics_dictionary_file_temp.is_file()
    with open(statistics_dictionary_file_temp, 'rb') as f:
        statistics_dictionary_temp = pickle.load(f)
    new_statistics_dictionary[single_qoi] = statistics_dictionary_temp
statistics_dictionary = new_statistics_dictionary
statistics_dictionary

# statistics_dictionary = uqPostprocessing.read_all_save_statistics_dict(
#     workingDir, statisticsObject.list_qoi_column)


# In[199]:


statisticsObject.list_qoi_column


# In[200]:


statistics_dictionary['Q_cms'][pd.Timestamp('2006-04-30 00:00:00')].keys()


# In[201]:


df_simulation_result["qoi"] = 'Q_cms'
df_simulation_result


# In[202]:


uqPostprocessing.extend_statistics_object(
    statisticsObject=statisticsObject, 
    statistics_dictionary=statistics_dictionary, 
    df_simulation_result=df_simulation_result,
    get_measured_data=False, 
    get_unaltered_data=False
)


# In[203]:


print(statisticsObject.workingDir)
print(statisticsObject.inputModelDir)
# print(statisticsObject.nodeNames)


# In[204]:


statisticsObject.result_dict


# In[205]:


# If you want to add measured/observed data
# TODO - This is hardcoded for HBV
statisticsObject.inputModelDir_basis = hbv_model_data_path / basis
statisticsObject.inputModelDir_basis

statisticsObject.get_measured_data(
    timestepRange=(statisticsObject.timesteps_min, statisticsObject.timesteps_max), 
    transforme_mesured_data_as_original_model="False")


# In[206]:


statisticsObject.df_measured


# In[207]:


statisticsObject.df_measured.columns


# In[208]:


df_measured_subset = statisticsObject.df_measured.loc[statisticsObject.df_measured["qoi"] == statisticsObject.list_qoi_column[0]][
                    [statisticsObject.time_column_name, "measured"]]
df_measured_subset


# In[209]:


statisticsObject.measured_fetched


# In[210]:


statisticsObject.pdTimesteps


# In[211]:


df_statistics_single_qoi = statisticsObject.create_df_from_statistics_data_single_qoi(qoi_column="Q_cms")
# statisticsObject.df_statistics


# In[212]:


df_statistics = statisticsObject.create_df_from_statistics_data()
df_statistics


# In[213]:


statisticsObject.df_statistics


# In[ ]:


# larsimStatisticsObject._plotStatisticsDict_plotly(unalatered=True, measured=True, station="MARI",
#                         recalculateTimesteps=True, window_title='Larsim Forward UQ & SA',
#                         filename=output_stat_graph_filename, display=True)


# In[214]:


# again hard-coded for HBV model, fatching forcing data
# statisticsObject.inputModelDir_basis
# statisticsObject.timesteps_min
# statisticsObject.timesteps_max

statisticsObject.get_forcing_data(time_column_name="TimeStamp")


# In[215]:


print(statisticsObject.forcing_data_fetched)
print(statisticsObject.forcing_df)


# In[216]:


df_statistics_and_measured = pd.merge(
    statisticsObject.df_statistics, statisticsObject.forcing_df, left_on=statisticsObject.time_column_name, right_index=True)
df_statistics_and_measured


# In[217]:


df_statistics_and_measured.columns


# In[219]:


df_statistics_and_measured[df_statistics_and_measured["precipitation"]>0]


# In[87]:


df_statistics_and_measured[df_statistics_and_measured["E"]<0]


# In[220]:


# or more detailed plotting of precipitation and temperature as main input data
# predicted streamflow and measured one
# and state data...

fig = make_subplots(
    rows=5, cols=1,
    subplot_titles=("Precipitation", "Temperature", "Streamflow", "Skew", "Kurt")
)

fig.add_trace(
    go.Bar(
        x=df_statistics_and_measured['TimeStamp'], y=df_statistics_and_measured['precipitation'],
        text=df_statistics_and_measured['precipitation'], 
        name="Precipitation"
    ), 
    row=1, col=1
)


fig.add_trace(
    go.Scatter(
        x=df_statistics_and_measured['TimeStamp'], y=df_statistics_and_measured['temperature'],
        text=df_statistics_and_measured['temperature'], 
        name="Temperature", mode='lines+markers'
    ), 
    row=2, col=1
)


fig.add_trace(
    go.Scatter(
        x=df_statistics_and_measured['TimeStamp'], y=df_statistics_and_measured['measured'],
        name="Observed Streamflow", mode='lines'
    ),
    row=3, col=1
)


fig.add_trace(
    go.Scatter(
        x=df_statistics_and_measured['TimeStamp'], y=df_statistics_and_measured['E'],
        text=df_statistics_and_measured['E'], 
        name="Mean Predicted Streamflow", mode='lines'
    ), 
    row=3, col=1
)
fig.add_trace(
    go.Scatter(
        x=df_statistics_and_measured['TimeStamp'], y=df_statistics_and_measured['E_minus_std'],
        text=df_statistics_and_measured['E_minus_std'], mode='lines', line_color="grey",
    ), 
    row=3, col=1
)
fig.add_trace(
    go.Scatter(
        x=df_statistics_and_measured['TimeStamp'], y=df_statistics_and_measured['E_plus_std'],
        text=df_statistics_and_measured['E_plus_std'], line_color="grey",
        mode='lines', fill='tonexty'
    ), 
    row=3, col=1
)
fig.add_trace(
    go.Scatter(
        x=df_statistics_and_measured['TimeStamp'], y=df_statistics_and_measured['Skew'],
        text=df_statistics_and_measured['Skew'], name="Skew", mode='markers'
    ), 
    row=4, col=1
)
fig.add_trace(
    go.Scatter(
        x=df_statistics_and_measured['TimeStamp'], y=df_statistics_and_measured['Kurt'],
        text=df_statistics_and_measured['Kurt'], name="Skew", mode='markers'
    ), 
    row=5, col=1
)

fig.update_layout(height=600, width=800, title_text="Detailed plot of most important time-series")
fig.update_layout(xaxis=dict(type="date"))
fig.show()


# In[92]:


df_statistics_and_measured['Skew'].plot(kind='hist')


# In[97]:


print(df_statistics_and_measured['Skew'].mean())
print(df_statistics_and_measured['Kurt'].mean())


# In[91]:


df_statistics_and_measured['Kurt'].plot(kind='hist') #'kde'


# In[99]:


df_statistics_and_measured.describe(include=np.number)


# In[ ]:


# statisticsObject.plotResults(
#     display=True, fileName="new_plot", plot_measured_timeseries=True, plot_forcing_timeseries=True, time_column_name="TimeStamp",
#     precipitation_df_timestamp_column="index", temperature_df_timestamp_column="index", streamflow_df_timestamp_column="index"
# )


# In[221]:


fig = statisticsObject._plotStatisticsDict_plotly(measured=True, forcing=True, window_title='Forward UQ & SA',
                                                  filename="sim-plotly.html", display=False,
                                                  precipitation_df_timestamp_column="index",
                                                  temperature_df_timestamp_column="index",
                                                  streamflow_df_timestamp_column="index")
fig.show()


# ### Examing the learned distribution of QoI from gPCE

# In[222]:


date_QoI = pd.Timestamp('2006-05-27 00:00:00')


# In[223]:


learned_expansion = statisticsObject.result_dict["Q_cms"][date_QoI]['gPCE']
learned_expansion


# In[107]:


df_statistics_and_measured.dtypes
# df_statistics_and_measured['qoi_dist'].dtypes


# In[122]:


learned_distribution = df_statistics_and_measured[
    df_statistics_and_measured["TimeStamp"]==date_QoI]["qoi_dist"].values[0]


# In[135]:


simulationNodes.nodes.T[1000].shape


# In[136]:


simulationNodes.joinedStandardDists


# In[142]:


samples = simulationNodes.joinedStandardDists.sample(100, rule="sobol")


# In[143]:


samples.shape


# In[141]:


learned_distribution.pdf(samples).shape


# In[ ]:


checked_expansion = cp.generate_expansion(1, learned_distribution, rule="cholesky")


# # GoFs / Metrices / P&R Factors

# In[224]:


qoi_column="Q_cms"


# In[225]:


# df_statistics_and_measured
uqPostprocessing.compute_gof_over_different_time_series(
    statisticsObject.df_statistics, 
    objective_function=["MAE", "NSE", "LogNSE", "RMSE", "NRMSE", "KGE"], 
    qoi=qoi_column, measuredDF_column_names=["measured"])


# In[228]:


p=statisticsObject.calculate_p_factor_single_qoi(
    qoi_column=qoi_column, df_statistics=df_statistics_and_measured,
    column_lower_uncertainty_bound="P10", column_upper_uncertainty_bound="P90",
    observed_column="measured")


# In[229]:


mean_uncertainty_band, std_uncertainty_band, mean_observed, std_observed = statisticsObject.compute_stat_of_uncertainty_band(
    qoi_column=qoi_column, df_statistics=df_statistics_and_measured,
    column_lower_uncertainty_bound="P10", column_upper_uncertainty_bound="P90",
    observed_column="measured"
)


# # Sensitivity Analysis - Plotting and printing SI

# In[76]:


si_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_m")
si_df


# In[77]:


# si_df.set_index(statisticsObject.time_column_name, inplace=True)
fig = statisticsObject.plot_heatmap_si_single_qoi(qoi_column="Q_cms", si_df=None, si_type="Sobol_m", uq_method="sc")
fig.show()


# In[78]:


si_t_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_t")
si_t_df


# In[79]:


fig = statisticsObject.plot_heatmap_si_single_qoi(qoi_column="Q_cms", si_type="Sobol_t", uq_method="sc")
fig.show()


# # MIC
# 

# Here important DataFrames computed so far are:
# * df_statistics_and_measured
# * statisticsObject.forcing_df 
# * statisticsObject.df_measured

# In[306]:


# df_statistics_and_measured.describe(include=np.number)
list_of_columns_with_sobol_indices = ["measured","precipitation","temperature"]
for param in configurationObject["parameters"]:
    param_name = param["name"]
    list_of_columns_with_sobol_indices.append(f"sobol_m_{param_name}")
    list_of_columns_with_sobol_indices.append(f"sobol_t_{param_name}")


# In[314]:


# list_of_columns_with_sobol_indices
df_statistics_and_measured[list_of_columns_with_sobol_indices].describe(include=np.number)


# In[315]:


df_statistics_and_measured_subset = df_statistics_and_measured[list_of_columns_with_sobol_indices]


# In[313]:


df_sobol_p_zero = df_statistics_and_measured_subset[df_statistics_and_measured_subset["precipitation"]==0].copy()
df_sobol_p_zero.describe(include=np.number)


# In[312]:


df_sobol_p_greater_than_zero = df_statistics_and_measured_subset[
    df_statistics_and_measured_subset["precipitation"]>0].copy()
df_sobol_p_greater_than_zero.describe(include=np.number)


# In[316]:


corr_df_statistics_and_measured_subset = df_statistics_and_measured_subset.corr()
corr_df_statistics_and_measured_subset


# In[317]:


import seaborn as sns
sns.set(style="darkgrid")
mask = np.triu(np.ones_like(corr_df_statistics_and_measured_subset, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_df_statistics_and_measured_subset, mask=mask, square=True, annot=True, linewidths=.5)


# In[241]:


fig, axs = plt.subplots(1, 3)
axs[0,].scatter(*df_statistics_and_measured[["measured","sobol_m_TT",]].values.T)
axs[1,].scatter(*df_statistics_and_measured[["precipitation","sobol_m_TT",]].values.T)
axs[2,].scatter(*df_statistics_and_measured[["temperature","sobol_m_TT",]].values.T)
plt.show()


# In[252]:


num_param = len(configurationObject["parameters"])
fig, axs = plt.subplots(3, num_param, figsize=(15, 8))
# param = configurationObject["parameters"][0]["name"]
# param_num = 0
param_counter = 0
for i, ax in enumerate(axs.flat):
    param_num = param_counter % num_param
    param = configurationObject["parameters"][param_num]["name"]
    if i%3==0:
        ax.scatter(*df_statistics_and_measured[[f"sobol_m_{param}","measured",]].values.T)
    elif i%3==1:
        ax.scatter(*df_statistics_and_measured[[f"sobol_m_{param}", "precipitation"]].values.T)
    elif i%3==2:
        ax.scatter(*df_statistics_and_measured[[f"sobol_m_{param}", "temperature"]].values.T)
    param_counter+=1
#     ax.set_title(f'{param}')
    
# set labels
for i in range(len(configurationObject["parameters"])):
    param = configurationObject["parameters"][i]["name"]
    plt.setp(axs[-1, i], xlabel=f'{param}')
plt.setp(axs[0, 0], ylabel='measured')
plt.setp(axs[1, 0], ylabel='precipitation')
plt.setp(axs[2, 0], ylabel='temperature')


# In[257]:


num_param = len(configurationObject["parameters"])
fig, axs = plt.subplots(3, num_param, figsize=(20, 10))
# param = configurationObject["parameters"][0]["name"]
# param_num = 0
param_counter = 0
for i, ax in enumerate(axs.flat):
    param_num = param_counter % num_param
    param = configurationObject["parameters"][param_num]["name"]
    if i%3==0:
        ax.scatter(*df_statistics_and_measured[["measured", f"sobol_t_{param}",]].values.T)
    elif i%3==1:
        ax.scatter(*df_statistics_and_measured[["precipitation", f"sobol_t_{param}"]].values.T)
    elif i%3==2:
        ax.scatter(*df_statistics_and_measured[["temperature", f"sobol_t_{param}"]].values.T)
    param_counter+=1
#     ax.set_title(f'{param}')
    
# set labels
for i in range(len(configurationObject["parameters"])):
    param = configurationObject["parameters"][i]["name"]
    plt.setp(axs[:, i], ylabel=f'{param}')
plt.setp(axs[0, :], xlabel='measured')
plt.setp(axs[1, :], xlabel='precipitation')
plt.setp(axs[2, :], xlabel='temperature')


# In[237]:


plt.scatter(*df_statistics_and_measured[["precipitation","sobol_m_TT",]].values.T)
plt.show()
plt.scatter(*df_statistics_and_measured[["measured","sobol_m_TT"]].values.T)
plt.show()
plt.scatter(*df_statistics_and_measured[["temperature","sobol_m_TT"]].values.T)
plt.show()


# In[291]:


samples_m_tt = df_statistics_and_measured["sobol_m_TT"].values
plt.hist(samples_m_tt, bins=100, density=True, alpha=0.5)
t = np.linspace(0, 1.0, 1000)
distribution = cp.GaussianKDE(samples_m_tt, h_mat=0.02**2)
plt.plot(t, distribution.pdf(t), label="0.05")
plt.legend()
plt.show()


# In[286]:


samples_m_tt = df_statistics_and_measured["sobol_m_beta"].values
plt.hist(samples_m_tt, bins=100, density=True, alpha=0.5)
t = np.linspace(0, 1.0, 1000)
distribution = cp.GaussianKDE(samples_m_tt, h_mat=0.02**2)
plt.plot(t, distribution.pdf(t), label="0.05")
plt.legend()
plt.show()


# In[289]:


t = np.linspace(0, 1.0, 1000)
distribution = cp.GaussianKDE(samples_m_tt, h_mat=0.02**2)
plt.plot(t, distribution.cdf(t), label="0.05")
plt.grid()
plt.legend()
plt.show()


# In[294]:


num_param = len(configurationObject["parameters"])

t = np.linspace(0, 1.0, 1000)
fig, axs = plt.subplots(1, num_param, figsize=(20, 10))
for i in range(num_param):
    param_name = configurationObject["parameters"][i]["name"]
    param_eval = df_statistics_and_measured[f"sobol_m_{param_name}"].values
    distribution = cp.GaussianKDE(param_eval, h_mat=0.005**2)
    axs[i,].plot(t, distribution.cdf(t), label=f"KDE m {param_name}")
    plt.setp(axs[i,], xlabel=f'{param_name}')
    axs[i,].grid()
# plt.legend()
plt.setp(axs[0], ylabel='CDF')
plt.show()


# # gPCE Surrogate
# If computed...

# In[318]:


# read the assumed prior distribution over parameters
import inspect
list_of_single_distr = []
for param in configurationObject["parameters"]:
    # for now this is hard-coded
    if param["distribution"]=="Uniform":
        list_of_single_distr.append(cp.Uniform(param["lower"], param["upper"]))
joint = cp.J(*list_of_single_distr)
joint_standard = cp.J(cp.Uniform(), cp.Uniform(), cp.Uniform(), cp.Uniform(), cp.Uniform())
joint_standard_min_1_1 = cp.J(
    cp.Uniform(lower=-1, upper=1),cp.Uniform(lower=-1, upper=1), cp.Uniform(lower=-1, upper=1), 
    cp.Uniform(lower=-1, upper=1), cp.Uniform(lower=-1, upper=1)
)


# In[319]:


samples_to_evaluate_gPCE = joint.sample(100, rule="halton") # 'sobol' 'random'
samples_to_evaluate_gPCE_transformed = utility.transformation_of_parameters_var1(
    samples_to_evaluate_gPCE, joint, joint_standard_min_1_1)


# In[320]:


samples_to_evaluate_gPCE_transformed.shape


# In[151]:


gPCE_model = defaultdict()
for single_date in statisticsObject.pdTimesteps:
    gPCE_model[single_date] = statisticsObject.result_dict["Q_cms"][single_date]['gPCE']


# In[153]:


type(gPCE_model[list(gPCE_model.keys())[0]])
gPCE_model[list(gPCE_model.keys())[0]].values


# In[321]:


start = time.time()

gPCE_model_evaluated = defaultdict()
for single_date in statisticsObject.pdTimesteps:
    gPCE_model = statisticsObject.result_dict["Q_cms"][single_date]['gPCE']
    gPCE_model_evaluated[single_date] = gPCE_model(samples_to_evaluate_gPCE_transformed.T)

end = time.time()
runtime = end - start
print(f"Time needed for evaluating {samples_to_evaluate_gPCE_transformed.shape[1]} gPCE model for {len(statisticsObject.pdTimesteps)} days is: {runtime}")


# In[160]:


gPCE_model_evaluated


# In[ ]:


# Looking closer for a particulat time-step
# learn the distribution of gPCE evaluations for a particular time step
date_QoI = pd.Timestamp('2006-05-27 00:00:00')
samples_of_gPCE_evals = gPCE_model_evaluated[date_QoI]
distribution = cp.GaussianKDE(samples_of_gPCE_evals, h_mat=0.05)


plt.hist(samples_1d, bins=50, density=True, alpha=0.5)
t = numpy.linspace(-3, 3, 400)
distribution = chaospy.GaussianKDE(samples_1d, h_mat=0.05**2)
plt.plot(t, distribution.pdf(t), label="0.05")
plt.legend()
plt.show()


# In[322]:


gpc_eval_df = pd.DataFrame.from_dict(gPCE_model_evaluated, orient="index", columns=range(1000))
gpc_eval_df


# In[ ]:


gpc_eval_df['new_E'] = gpc_eval_df.mean(numeric_only=True, axis=1)
gpc_eval_df = gpc_eval_df.loc[:, ['new_E',]]
gpc_eval_df


# # Specific Run Output Files

# In[ ]:


flux_df = pd.read_pickle(flux_df_path, compression="gzip")


# In[ ]:


index_parameter_gof_DF = pd.read_pickle(gof_df_path, compression="gzip")
index_parameter_gof_DF


# In[ ]:


import dill
with open(parameters_dict_path, 'rb') as f:
    index_run_and_parameters_dict = dill.load(f)
index_run_and_parameters_dict


# # Plotting

# plotting input, predicted/simulated and measured time-series

# In[ ]:


fig = hbv._plot_streamflow_and_precipitation(
    input_data_df=hbvsaskModelObject.time_series_measured_data_df, 
    simulated_data_df=results_array[0][0]['result_time_series'], 
    input_data_time_column=hbvsaskModelObject.time_column_name,
    simulated_time_column=hbvsaskModelObject.time_column_name, 
    observed_streamflow_column=hbvsaskModelObject.streamflow_column_name,
    simulated_streamflow_column="Q_cms", 
    precipitation_columns=hbvsaskModelObject.precipitation_column_name)
fig.show()


# column 'stramflow' contains the measured data, column "Q_cms" contains predicted data (i.e., streamflow expressed in cubic meters per second) by the model defined with the current values for the uncertain parameters

# as a QoI, one can take Q_cms time-series (extract it from above dataframe), AET (Actual EvapoTranspiration), or some likelihood (i.e., goodness-of-fit (GoF)) function value

# In[ ]:


# or more detailed plotting of precipitation and temperature as main input data
# predicted streamflow and measured one
# and state data...
result_df = results_array_changed_param[0][0]['result_time_series']
state_df = results_array_changed_param[0][0]['state_df']
parsed_input_data_df = hbvsaskModelObject.time_series_measured_data_df.loc[
    result_df["TimeStamp"].min():result_df["TimeStamp"].max()]

fig = make_subplots(
    rows=5, cols=1,
    subplot_titles=("Temperature", "Precipitation", "Streamflow", "EvapoTranspiration","SWE")
)

fig.add_trace(
    go.Scatter(
        x=parsed_input_data_df.index, y=parsed_input_data_df['temperature'],
        text=parsed_input_data_df['temperature'], 
        name="Temperature"
    ), 
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=parsed_input_data_df.index, y=parsed_input_data_df['precipitation'],
        text=parsed_input_data_df['precipitation'], 
        name="Precipitation"
    ), 
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=parsed_input_data_df.index, y=parsed_input_data_df['streamflow'],
        name="Observed Streamflow"
    ),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(
        x=result_df['TimeStamp'], y=result_df['Q_cms'],
        text=result_df['Q_cms'], 
        name="Predicted Streamflow"
    ), 
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=result_df['TimeStamp'], y=result_df['AET'],
        text=result_df['AET'], 
        name="AET"
    ), 
    row=4, col=1
)
fig.add_trace(
    go.Scatter(a
        x=result_df['TimeStamp'], y=result_df['PET'],
        text=result_df['PET'], 
        name="PET"
    ), 
    row=4, col=1
)

fig.add_trace(
    go.Scatter(
        x=state_df['TimeStamp'], y=state_df['initial_SWE'],
        text=state_df['initial_SWE'], 
        name="initial_SWE"
    ), 
    row=5, col=1
)

fig.update_layout(height=600, width=800, title_text="Detailed plot of most important time-series")
fig.show()

