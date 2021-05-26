# Set of functions to call if you want to execute some functionality of LarsimStatistics "afterward"
# e.g., to compute statistics, to draw statistics
# use this script to prototype the parallelization if statistics calculations
# Assumes that nodes object (dictionary) and raw results dataframe are stored

import chaospy as cp
import dill
import json
import numpy as np
import pickle
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import pathlib
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plotly.offline import iplot, plot #as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import itertools
import os
import seaborn as sns
import string as str
from distutils.util import strtobool

from LarsimUtilityFunctions import larsimDataPostProcessing
from LarsimUtilityFunctions import larsimInputOutputUtilities
from LarsimUtilityFunctions import larsimPaths as paths

from common import saltelliSobolIndicesHelpingFunctions

from larsim import LarsimModel
from larsim import LarsimStatistics
###################################################################################################################
# Reading saved data
###################################################################################################################

def read_configuration_object_json(configurationObject, element=None):
    """
    element can be, e.g., "Output", "parameters", "Timeframe", "parameters_settings"
    """
    with open(configurationObject) as f:
        configuration_object = json.load(f)
    if element is None:
        return configuration_object
    else:
        return configuration_object[element]


def read_configuration_object_dill(configurationObject, element=None):
    """
    element can be, e.g., "Output", "parameters", "Timeframe", "parameters_settings"
    """
    with open(configurationObject, 'rb') as f:
        configuration_object = dill.load(f)
    if element is None:
        return configuration_object
    else:
        return configuration_object[element]


def read_and_print_uqsim_args_file(file):
    with open(file, 'rb') as f:
        uqsim_args = pickle.load(f)
    uqsim_args_temp_dict = vars(uqsim_args)
    print(f"UQSIM.ARGS")
    for key, value in uqsim_args_temp_dict.items():
        print(f"{key}: {value}")


def _get_df_simulation_from_file(working_folder):
    df_all_simulations = os.path.abspath(os.path.join(working_folder, "df_all_simulations.pkl"))
    df_all_simulations = pd.read_pickle(df_all_simulations, compression='gzip')
    return df_all_simulations


def _get_nodes_from_file(working_folder):
    working_folder = paths.pathlib_to_from_str(working_folder, transfrom_to='Path')
    nodes_dict = working_folder / 'nodes.simnodes'
    with open(nodes_dict, 'rb') as f:
        simulationNodes = dill.load(f)
    return simulationNodes


def _get_df_index_parameter_gof_from_file(working_folder):
    df_all_index_parameter_gof_values_file = pathlib.Path(working_folder/"df_all_index_parameter_gof_values.pkl")
    df_all_index_parameter_gof_values = pd.read_pickle(df_all_index_parameter_gof_values_file, compression='gzip')
    return df_all_index_parameter_gof_values


def _get_statistics_dict(working_folder):
    statistics_dictionary_file = pathlib.Path(working_folder/"statistics_dictionary.pkl")
    statistics_dictionary = None
    if statistics_dictionary_file.is_file():
        with open(statistics_dictionary_file, 'rb') as f:
            statistics_dictionary = pickle.load(f)
    return statistics_dictionary

###################################################################################################################
# Plotting params and GoF values - mostly from df_index_parameter_gof filtered for a single station
###################################################################################################################


def _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof):
    return [x for x in df_index_parameter_gof.columns.tolist()
            if not x.startswith("calculate") and x not in ["index_run","station"]]


def _get_gof_columns_df_index_parameter_gof(df_index_parameter_gof):
    return [x for x in df_index_parameter_gof.columns.tolist()
            if x.startswith("calculate") and x not in ["index_run","station"]]


def plot_hist_of_gof_values_from_df(df_index_parameter_gof, name_of_gof_column="calculateNSE"):
    return df_index_parameter_gof.hist(name_of_gof_column)


def plot_subplot_params_hist_from_df(df_index_parameter_gof):
    columns_with_parameters = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    fig = make_subplots(rows=1, cols=len(columns_with_parameters))
    for i in range(len(columns_with_parameters)):
        fig.append_trace(go.Histogram(x=df_index_parameter_gof[columns_with_parameters[i]]), row=1, col=i + 1)
    return fig


def plot_subplot_params_hist_from_df_conditioned(df_index_parameter_gof, name_of_gof_column="calculateNSE",
                                                 threshold_gof_value = 0, comparison="smaller"):
    """
    comparison should be: "smaller", "greater", "equal"
    """
    if comparison == "smaller":
        mask = df_index_parameter_gof[name_of_gof_column] < threshold_gof_value
    elif comparison == "greater":
        mask = df_index_parameter_gof[name_of_gof_column] > threshold_gof_value
    elif comparison == "equal":
        mask = df_index_parameter_gof[name_of_gof_column] == threshold_gof_value
    else:
        raise Exception("Comparison should be: \"smaller\", \"greater\", \"equal\" ")
    columns_with_parameters = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    fig = make_subplots(rows=1, cols=len(columns_with_parameters))
    for i in range(len(columns_with_parameters)):
        fig.append_trace(
            go.Histogram(
                x=df_index_parameter_gof[mask][columns_with_parameters[i]],
                name=columns_with_parameters[i]
            ), row=1, col=i + 1
        )
    return fig


def plot_scatter_matrix_params_vs_gof(df_index_parameter_gof, name_of_gof_column="calculateNSE"):
    columns_with_parameters = [x for x in df_index_parameter_gof.columns.tolist()
                               if not x.startswith("calculate") and x not in ["index_run", "station"]]
    fig = px.scatter_matrix(df_index_parameter_gof,
                            dimensions=columns_with_parameters,
                            color=name_of_gof_column)
    fig.update_traces(diagonal_visible=False)
    return fig


def plot_2d_contour_params_vs_gof(df_index_parameter_gof, param1, param2,
                                  num_of_points_in_1d=8, name_of_gof_column="calculateNSE"):
    x = df_index_parameter_gof[param1].values
    y = df_index_parameter_gof[param2].values
    X = np.reshape(x, (-1, num_of_points_in_1d))
    Y = np.reshape(y, (-1, num_of_points_in_1d))
    fig = go.Figure(data=go.Contour(x=X[:, 0],
                                    y=Y[0, :],
                                    z=np.reshape(df_index_parameter_gof[name_of_gof_column].values,
                                                 (-1, num_of_points_in_1d)).T))
    fig.update_layout(title="calculateNSE",
                      xaxis_title=param1,
                      yaxis_title=param2
                      )
    return fig


def scatter_3d_params_vs_gof(df_index_parameter_gof, param1, param2, param3, name_of_gof_column="calculateNSE"):
    fig = px.scatter_3d(df_index_parameter_gof, x=param1, y=param2, z=param3,
                        color=name_of_gof_column, opacity=0.7, hover_data=[name_of_gof_column])
    return fig


def plot_surface_2d_params_vs_gof(df_index_parameter_gof, param1, param2, num_of_points_in_1d=8,
                                  name_of_gof_column="calculateNSE"):
    # from mpl_toolkits.mplot3d import Axes3D
    # from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    # from matplotlib import cm
    # from matplotlib.ticker import LinearLocator, FormatStrFormatter
    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.gca(projection='3d')

    # x = samples_df_index_parameter_gof['A2'].to_numpy()
    # y = samples_df_index_parameter_gof['BSF'].to_numpy()
    # z = samples_df_index_parameter_gof['calculateRMSE'].to_numpy()

    # X = np.reshape(x, (-1, 8))
    # Y = np.reshape(y, (-1, 8))
    # Z = np.reshape(z, (-1, 8))

    # surf_Rosen = ax.plot_surface(X, Y, Z, cmap=cm.rainbow)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('RMSE')
    # plt.tight_layout()
    x = df_index_parameter_gof[param1].values
    y = df_index_parameter_gof[param2].values
    z = df_index_parameter_gof[name_of_gof_column].values
    X = np.reshape(x, (-1, num_of_points_in_1d))
    Y = np.reshape(y, (-1, num_of_points_in_1d))
    Z = np.reshape(z, (-1, num_of_points_in_1d))
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    fig.update_layout(scene=dict(xaxis_title=param1, yaxis_title=param2, zaxis_title=name_of_gof_column),
                      autosize=False,
                      title='GoF Plot',
                      scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                      width=900, height=500,
                      margin=dict(r=20, b=10, l=10, t=10)
                      )
    return fig


def plot_parallel_params_vs_gof(df_index_parameter_gof, name_of_gof_column="calculateNSE", list_of_params=None):
    if list_of_params is None:
        list_of_params = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    dimensions = list_of_params + [name_of_gof_column, ]
    fig = px.parallel_coordinates(df_index_parameter_gof, color=name_of_gof_column, dimensions=dimensions)
    return fig


def plot_scatter_matrix_params_vs_gof_seaborn(df_index_parameter_gof, name_of_gof_column="calculateNSE"):
    columns_with_parameters = [x for x in df_index_parameter_gof.columns.tolist()
                               if not x.startswith("calculate") and x not in ["index_run", "station"]]
    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(df_index_parameter_gof,
                     vars=columns_with_parameters,
                     corner=True, hue=name_of_gof_column)
    plt.show()


def scatter_3d_params_static(df_index_parameter_gof, param1, param2, param3):
    fig = plt.figure()
    axs = fig.gca(projection='3d')
    axs.scatter(list(df_index_parameter_gof[param1]),
                list(df_index_parameter_gof[param2]),
                list(df_index_parameter_gof[param3]), marker="o")
    plt.title('Parameters\n')
    axs.set_xlabel(param1)
    axs.set_ylabel(param2)
    axs.set_zlabel(param3)
    plt.show()
###################################################################################################################


def pivot_df_all_simulations(df_all_simulations):
    return df_all_simulations.pivot(index='TimeStamp', columns='Index_run', values='Value')


def plot_time_series_of_all_simulations(df_all_simulations):
    fig = px.line(df_all_simulations, x="TimeStamp", y="Value", color="Index_run",
                  line_group="Index_run", hover_name="Index_run")
    return fig


###################################################################################################################


def extracting_statistics_df_for_one_station(station="MARI"):
    pass


def create_larsimStatistics_object(configuration_object, uqsim_args_dict, workingDir):
    larsimStatisticsObject = LarsimStatistics.LarsimStatistics(configuration_object,
                                                               workingDir=workingDir,
                                                               parallel_statistics=uqsim_args_dict["parallel_statistics"],
                                                               mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
                                                               unordered=False,
                                                               uq_method=uqsim_args_dict["uq_method"],
                                                               compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
                                                               compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"]
                                                               )
    return larsimStatisticsObject


###################################################################################################################

# TODO Add _calcStatisticsForMC and calcStatisticsForMc
def _calcStatisticsForSc(df_all_simulations, simulationNodes, order=2, regression=False):
    timesteps = df_all_simulations.TimeStamp.unique()
    numbTimesteps = len(timesteps)
    station_names = df_all_simulations.Stationskennung.unique()

    nodes = simulationNodes.distNodes
    dist = simulationNodes.joinedDists
    weights = simulationNodes.weights
    polynomial_expansion = cp.orth_ttr(order, dist)

    Abfluss = {}

    grouped = df_all_simulations.groupby(['Stationskennung','TimeStamp'])
    groups = grouped.groups
    for key,val_indices in groups.items():
        discharge_values = df_all_simulations.loc[val_indices.values].Value.values
        Abfluss[key] = {}
        Abfluss[key]["Q"] = discharge_values
        if regression:
            qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, discharge_values)
        else:
            qoi_gPCE = cp.fit_quadrature(polynomial_expansion, nodes, weights, discharge_values)
        numPercSamples = 10 ** 5
        Abfluss[key]["gPCE"] = qoi_gPCE
        Abfluss[key]["E"] = float((cp.E(qoi_gPCE, dist)))
        Abfluss[key]["Var"] = float((cp.Var(qoi_gPCE, dist)))
        Abfluss[key]["StdDev"] = float((cp.Std(qoi_gPCE, dist)))
        Abfluss[key]["Sobol_m"] = cp.Sens_m(qoi_gPCE, dist)
        Abfluss[key]["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist)
        Abfluss[key]["Sobol_t"] = cp.Sens_t(qoi_gPCE, dist)
        Abfluss[key]["P10"] = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
        Abfluss[key]["P90"] = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
        if isinstance(Abfluss[key]["P10"], (list)) and len(Abfluss[key]["P10"]) == 1:
            Abfluss[key]["P10"]= Abfluss[key]["P10"][0]
            Abfluss[key]["P90"] = Abfluss[key]["P90"][0]
    return Abfluss


def _calcStatisticsForSaltelli(df_all_simulations, simulationNodes, numEvaluations, order, regression):
    timesteps = df_all_simulations.TimeStamp.unique()
    numbTimesteps = len(timesteps)
    station_names = df_all_simulations.Stationskennung.unique()

    nodes = simulationNodes.distNodes

    Abfluss = {}

    grouped = df_all_simulations.groupby(['Stationskennung','TimeStamp'])
    groups = grouped.groups

    dim = len(simulationNodes.nodeNames)

    for key,val_indices in groups.items():
        Abfluss[key] = {}

        discharge_values = df_all_simulations.loc[val_indices.values].Value.values #numpy array - for sartelli it should be n(2+d)x1
        #extended_standard_discharge_values = discharge_values[:(2*numEvaluations)]
        discharge_values_saltelli = discharge_values[:, np.newaxis]
        standard_discharge_values = discharge_values_saltelli[:numEvaluations,:] #values based on which we calculate standard statistics
        extended_standard_discharge_values = discharge_values_saltelli[:(2*numEvaluations),:]

        Abfluss[key]["Q"] = standard_discharge_values

        #result_dict[key]["min_q"] = np.amin(discharge_values) #standard_discharge_values.min()
        #result_dict[key]["max_q"] = np.amax(discharge_values) #standard_discharge_values.max()

        #result_dict[key]["E"] = np.sum(extended_standard_discharge_values, axis=0, dtype=np.float64) / (2*numEvaluations)
        Abfluss[key]["E"] = np.mean(discharge_values[:(2*numEvaluations)], 0)
        #result_dict[key]["E"] = np.mean(extended_standard_discharge_values, 0)
        #result_dict[key]["Var"] = float(np.sum(power(standard_discharge_values)) / numEvaluations - result_dict[key]["E"] ** 2)
        #result_dict[key]["Var"] = np.sum((extended_standard_discharge_values - result_dict[key]["E"]) ** 2, axis=0, dtype=np.float64) / (2*numEvaluations - 1)
        #result_dict[key]["StdDev"] = np.sqrt(result_dict[key]["Var"], dtype=np.float64)
        Abfluss[key]["StdDev"] = np.std(discharge_values[:(2*numEvaluations)], 0, ddof=1)
        #result_dict[key]["StdDev"] = np.std(extended_standard_discharge_values, 0, ddof=1)

        #result_dict[key]["P10"] = np.percentile(discharge_values[:numEvaluations], 10, axis=0)
        #result_dict[key]["P90"] = np.percentile(discharge_values[:numEvaluations], 90, axis=0)
        Abfluss[key]["P10"] = np.percentile(discharge_values[:(2*numEvaluations)], 10, axis=0)
        Abfluss[key]["P90"] = np.percentile(discharge_values[:(2*numEvaluations)], 90, axis=0)

        Abfluss[key]["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_4(discharge_values_saltelli, dim, numEvaluations)
        #result_dict[key]["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_3(discharge_values_saltelli, dim, numEvaluations)
        Abfluss[key]["Sobol_t"] = saltelliSobolIndicesHelpingFunctions._Sens_t_sample_4(discharge_values_saltelli, dim, numEvaluations)

        if isinstance(Abfluss[key]["P10"], (list)) and len(Abfluss[key]["P10"]) == 1:
            Abfluss[key]["P10"]=Abfluss[key]["P10"][0]
            Abfluss[key]["P90"]=Abfluss[key]["P90"][0]


# TODO Add sub-options for calcStatisticsForSaltelli
def _replot_statistics(working_folder):
    df_all_simulations = _get_df_simulation_from_file(working_folder)
    simulationNodes = _get_nodes_from_file(working_folder)
    Abfluss = _calcStatisticsForSc(df_all_simulations, simulationNodes, order=2, regression=False)

    timesteps = df_all_simulations.TimeStamp.unique()
    pdTimesteps = [pd.Timestamp(timestep) for timestep in timesteps]
    keyIter = list(itertools.product(["MARI",],pdTimesteps))
    colors = ['darkred', 'midnightblue', 'mediumseagreen', 'darkorange']
    labels = [nodeName.strip() for nodeName in simulationNodes.nodeNames]
    n_rows = 4

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False)
    fig.add_trace(go.Scatter(x=pdTimesteps, y=[Abfluss[key]["E"] for key in keyIter], name='E[Q]',line_color='green', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdTimesteps, y=[(Abfluss[key]["E"] - Abfluss[key]["StdDev"]) for key in keyIter], name='mean - std. dev', line_color='darkviolet', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdTimesteps, y=[(Abfluss[key]["E"] + Abfluss[key]["StdDev"]) for key in keyIter], name='mean + std. dev', line_color='darkviolet', mode='lines', fill='tonexty'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdTimesteps, y=[Abfluss[key]["P10"] for key in keyIter], name='10th percentile',line_color='yellow', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdTimesteps, y=[Abfluss[key]["P90"] for key in keyIter], name='90th percentile',line_color='yellow', mode='lines',fill='tonexty'), row=1, col=1)

    fig.add_trace(go.Scatter(x=pdTimesteps, y=[Abfluss[key]["StdDev"] for key in keyIter], name='std. dev', line_color='darkviolet'), row=2, col=1)

    for i in range(len(labels)):
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[Abfluss[key]["Sobol_m"][i] for key in keyIter], name=labels[i], legendgroup=labels[i], line_color=colors[i]), row=3, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[Abfluss[key]["Sobol_t"][i] for key in keyIter], legendgroup=labels[i], showlegend = False, line_color=colors[i]), row=4, col=1)

    fig.update_traces(mode='lines')
    fig.update_yaxes(title_text="Q [m^3/s]", side='left', showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="Std. Dev. [m^3/s]", side='left', showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Sobol_m", side='left', showgrid=True, range=[0, 1], row=3, col=1)
    fig.update_yaxes(title_text="Sobol_t", side='left', showgrid=True, range=[0, 1], row=4, col=1)
    fig.update_layout(height=800, width=1200, title_text="Lai Mai UQ")

    display = False
    plot(fig, filename=os.path.abspath(os.path.join(working_folder,"lai_mai_plotly.html")), auto_open=display)
    #fig.write_image(os.path.abspath(os.path.join(working_folder,"lai_mai.png")))
    fig.show()
