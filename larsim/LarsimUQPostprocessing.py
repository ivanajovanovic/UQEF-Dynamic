# Set of functions to call if you want to execute some functionality of LarsimStatistics "afterward"
# e.g., to compute statistics, to draw statistics

import chaospy as cp
from collections import defaultdict
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
from common import utility

# from larsim import LarsimModel
from larsim import LarsimStatistics
###################################################################################################################
# Reading saved data
###################################################################################################################

MAPPING_STATION_TO_TGB = {"MARI": 3085}

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
    return uqsim_args


def _get_df_simulation_from_file(working_folder):
    df_all_simulations = os.path.abspath(os.path.join(working_folder, "df_all_simulations.pkl"))
    df_all_simulations = pd.read_pickle(df_all_simulations, compression='gzip')
    return df_all_simulations


def _get_nodes_from_file(working_folder, dill_or_pickle="dill"):
    working_folder = paths.pathlib_to_from_str(working_folder, transfrom_to='Path')
    nodes_dict = working_folder / 'nodes.simnodes'
    with open(nodes_dict, 'rb') as f:
        if dill_or_pickle == "dill":
            simulationNodes = dill.load(f)
        elif dill_or_pickle == "pickle":
            simulationNodes = pickle.load(f)
        else:
            simulationNodes = None
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
    def check_if_column_stores_parameter(x):
        if isinstance(x, tuple):
            x = x[0]
        return not x.startswith("calculate") and not x.startswith("d_calculate") and \
               x not in ["index_run", "station", "successful_run"]
    return [x for x in df_index_parameter_gof.columns.tolist() if check_if_column_stores_parameter(x)]


def _get_gof_columns_df_index_parameter_gof(df_index_parameter_gof):
    def check_if_column_stores_gof(x):
        if isinstance(x, tuple):
            x = x[0]
        return x.startswith("calculate") and x not in ["index_run", "station"]
    return [x for x in df_index_parameter_gof.columns.tolist() if check_if_column_stores_gof(x)]


def _get_grad_columns_df_index_parameter_gof(df_index_parameter_gof):
    return [x for x in df_index_parameter_gof.columns.tolist() if x.startswith("d_calculate")]


def plot_hist_of_gof_values_from_df(df_index_parameter_gof, name_of_gof_column="calculateNSE"):
    return df_index_parameter_gof.hist(name_of_gof_column)


def plot_subplot_params_hist_from_df(df_index_parameter_gof):
    columns_with_parameters = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    fig = make_subplots(rows=1, cols=len(columns_with_parameters))
    for i in range(len(columns_with_parameters)):
        if isinstance(columns_with_parameters[i], tuple):
            fig.append_trace(
                go.Histogram(
                    x=df_index_parameter_gof[columns_with_parameters[i]],
                    name=columns_with_parameters[i][0]
                ), row=1, col=i + 1)
        else:
            fig.append_trace(
                go.Histogram(
                    x=df_index_parameter_gof[columns_with_parameters[i]],
                    name=columns_with_parameters[i]
                ), row=1, col=i + 1)
    return fig


def plot_subplot_params_hist_from_df_conditioned(df_index_parameter_gof, name_of_gof_column="calculateNSE",
                                                 threshold_gof_value=0, comparison="smaller"):
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
        if isinstance(columns_with_parameters[i], tuple):
            fig.append_trace(
                go.Histogram(
                    x=df_index_parameter_gof[mask][columns_with_parameters[i]],
                    name=columns_with_parameters[i][0]
                ), row=1, col=i + 1)
        else:
            fig.append_trace(
                go.Histogram(
                    x=df_index_parameter_gof[mask][columns_with_parameters[i]],
                    name=columns_with_parameters[i]
                ), row=1, col=i + 1)
    return fig


def plot_scatter_matrix_params_vs_gof(df_index_parameter_gof, name_of_gof_column="calculateNSE",
                                      hover_name="index_run"):
    columns_with_parameters = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    fig = px.scatter_matrix(df_index_parameter_gof,
                            dimensions=columns_with_parameters,
                            color=name_of_gof_column,
                            hover_name=hover_name)
    fig.update_traces(diagonal_visible=False)
    return fig


# TODO  - This function does not work properly
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
    fig.update_layout(title=name_of_gof_column,
                      xaxis_title=param1,
                      yaxis_title=param2
                      )
    return fig


def scatter_3d_params_vs_gof(df_index_parameter_gof, param1, param2, param3, name_of_gof_column="calculateNSE",
                             name_of_index_run_column="index_run"):
    fig = px.scatter_3d(
        df_index_parameter_gof, x=param1, y=param2, z=param3, color=name_of_gof_column, opacity=0.7,
        hover_data=[name_of_gof_column, name_of_index_run_column]
    )
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
    columns_with_parameters = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
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


def create_larsimStatistics_object(configuration_object, uqsim_args_dict, workingDir):
    larsimStatisticsObject = LarsimStatistics.LarsimStatistics(configuration_object, workingDir=workingDir,
                                                               parallel_statistics=uqsim_args_dict[
                                                                   "parallel_statistics"],
                                                               mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
                                                               unordered=False,
                                                               uq_method=uqsim_args_dict["uq_method"],
                                                               compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
                                                               compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"])
    return larsimStatisticsObject


def extend_larsimStatistics_object(larsimStatisticsObject, statistics_dictionary, df_simulation_result,
                                   get_measured_data=False, get_unaltered_data=False
                                   ):
    larsimStatisticsObject.result_dict = statistics_dictionary
    larsimStatisticsObject.timesteps = list(df_simulation_result.TimeStamp.unique())
    larsimStatisticsObject.timesteps_min = df_simulation_result.TimeStamp.min()
    larsimStatisticsObject.timesteps_max = df_simulation_result.TimeStamp.max()

    timestepRange = (pd.Timestamp(larsimStatisticsObject.timesteps_min),
                     pd.Timestamp(larsimStatisticsObject.timesteps_max))

    larsimStatisticsObject.pdTimesteps = [pd.Timestamp(timestep) for timestep in larsimStatisticsObject.timesteps]

    larsimStatisticsObject.number_of_unique_index_runs = get_number_of_unique_runs(
        df_simulation_result, index_run_column_name="Index_run")
    larsimStatisticsObject.numEvaluations = larsimStatisticsObject.number_of_unique_index_runs

    larsimStatisticsObject.numbTimesteps = len(larsimStatisticsObject.timesteps)

    larsimStatisticsObject.samples_station_names = list(df_simulation_result["Stationskennung"].unique())
    larsimStatisticsObject.station_of_Interest = list(set(larsimStatisticsObject.samples_station_names).intersection(
        larsimStatisticsObject.station_of_Interest))
    if not larsimStatisticsObject.station_of_Interest:
        larsimStatisticsObject.station_of_Interest = larsimStatisticsObject.samples_station_names

    larsimStatisticsObject._check_if_Sobol_t_computed(list(larsimStatisticsObject.result_dict.keys()))
    larsimStatisticsObject._check_if_Sobol_m_computed(list(larsimStatisticsObject.result_dict.keys()))

    if get_measured_data:
        larsimStatisticsObject.get_measured_data(timestepRange=timestepRange)

    if get_unaltered_data:
        larsimStatisticsObject.get_unaltered_run_data(timestepRange=timestepRange)


def get_number_of_unique_runs(df, index_run_column_name="Index_run"):
    return df[index_run_column_name].nunique()


def extracting_statistics_df_for_one_station(larsimStatisticsObject, station="MARI"):
    pass


def compute_gof_over_different_time_series(objective_function, station, df_statistics_station):
    """
    This function will run only for a single station
    """
    if isinstance(station, list):
        station = station[0]

    if not callable(
            objective_function) and objective_function in larsimDataPostProcessing.mapping_gof_names_to_functions:
        objective_function = larsimDataPostProcessing.mapping_gof_names_to_functions[objective_function]
    elif not callable(
            objective_function) and objective_function not in larsimDataPostProcessing.mapping_gof_names_to_functions \
            or callable(objective_function) and objective_function not in larsimDataPostProcessing._all_functions:
        raise ValueError("Not proper specification of Goodness of Fit function name")

    gof_meas_unalt = None
    gof_meas_mean = None
    gof_meas_mean_m_std = None
    gof_meas_mean_p_std = None
    gof_meas_p10 = None
    gof_meas_p90 = None

    if 'unaltered' in df_statistics_station.columns:
        gof_meas_unalt = objective_function(df_statistics_station, df_statistics_station,
                                            measuredDF_column_name='measured', simulatedDF_column_name='unaltered')
    if 'E' in df_statistics_station.columns:
        gof_meas_mean = objective_function(df_statistics_station, df_statistics_station,
                                           measuredDF_column_name='measured', simulatedDF_column_name='E')
    if 'E_minus_std' in df_statistics_station.columns:
        gof_meas_mean_m_std = objective_function(df_statistics_station, df_statistics_station,
                                                 measuredDF_column_name='measured',
                                                 simulatedDF_column_name='E_minus_std')
    if 'E_plus_std' in df_statistics_station.columns:
        gof_meas_mean_p_std = objective_function(df_statistics_station, df_statistics_station,
                                                 measuredDF_column_name='measured',
                                                 simulatedDF_column_name='E_plus_std')
    if 'P10' in df_statistics_station.columns:
        gof_meas_p10 = objective_function(df_statistics_station, df_statistics_station,
                                          measuredDF_column_name='measured', simulatedDF_column_name='P10')
    if 'P90' in df_statistics_station.columns:
        gof_meas_p90 = objective_function(df_statistics_station, df_statistics_station,
                                          measuredDF_column_name='measured', simulatedDF_column_name='P90')

    print(f"gof_meas_unalt:{gof_meas_unalt} \ngof_meas_mean:{gof_meas_mean} \n"
          f"gof_meas_mean_m_std:{gof_meas_mean_m_std} \ngof_meas_mean_p_std:{gof_meas_mean_p_std} \n"
          f"gof_meas_p10:{gof_meas_p10} \ngof_meas_p90:{gof_meas_p90} \n")


def redo_all_statistics(
        workingDir, get_measured_data=False, get_unaltered_data=False, station="MARI", uq_method="sc", plotting=False):

    gPCE_model_evaluated_at_calib_sample_list = None

    uq_output_paths_obj = utility.UQOutputPaths(workingDir)

    with open(uq_output_paths_obj.configuration_object_file, 'rb') as f:
        configuration_object = dill.load(f)

    with open(uq_output_paths_obj.args_file, 'rb') as f:
        uqsim_args = pickle.load(f)
    uqsim_args_dict = vars(uqsim_args)

    #####################
    # extra
    #####################
    samples_df_index_parameter_gof = pd.read_pickle(uq_output_paths_obj.df_all_index_parameter_gof_file, compression="gzip")
    params_list = _get_parameter_columns_df_index_parameter_gof(
        samples_df_index_parameter_gof)
    samples_df_index_parameter_gof_station = samples_df_index_parameter_gof.loc[
        samples_df_index_parameter_gof["station"] == station]

    with open(uq_output_paths_obj.nodes_file, 'rb') as f:
        simulationNodes = pickle.load(f)
    df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
    df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",
                                                          params_list=params_list)

    # df_measured = pd.read_pickle(uq_output_paths_obj.df_measured_file, compression="gzip")
    #####################

    samples_df_simulation_result = pd.read_pickle(uq_output_paths_obj.df_all_simulations_file, compression="gzip")
    # samples_df_simulation_result = samples_df_simulation_result.query(f"Stationskennung=={station}")

    if uq_output_paths_obj.statistics_dictionary_file.is_file():
        with open(uq_output_paths_obj.statistics_dictionary_file, 'rb') as f:
            statistics_dictionary = pickle.load(f)

        # larsimStatisticsObject = create_larsimStatistics_object(
        #     configurationObject, uqsim_args_dict, workingDir)
        larsimStatisticsObject = create_larsimStatistics_object(
            configuration_object, uqsim_args_dict, uq_output_paths_obj.model_runs_folder)

        extend_larsimStatistics_object(
            larsimStatisticsObject=larsimStatisticsObject,
            statistics_dictionary=statistics_dictionary,
            df_simulation_result=samples_df_simulation_result,
            get_measured_data=get_measured_data,
            get_unaltered_data=get_unaltered_data
        )

        if plotting:
            larsimStatisticsObject._plotStatisticsDict_plotly(
                unalatered=get_unaltered_data, measured=get_measured_data, station=station,
                recalculateTimesteps=True, window_title='Larsim Forward UQ & SA',
                filename=uq_output_paths_obj.output_stat_graph_filename, display=True
            )
        # TODO Add only time series model output plotting

        df_statistics_station = larsimStatisticsObject.create_df_from_statistics_data_single_station(
            station=station, uq_method=uq_method)

        # larsimStatisticsObject.compute_gof_over_different_time_series(
        #     objective_function="KGE", df_statistics_station=df_statistics_station)

        si_m_df = larsimStatisticsObject.create_df_from_sensitivity_indices_for_singe_station(station=station,
                                                                                              si_type="Sobol_m")
        si_t_df = larsimStatisticsObject.create_df_from_sensitivity_indices_for_singe_station(station=station,
                                                                                              si_type="Sobol_t")
        # P factor
        p = larsimStatisticsObject.calculate_p_factor(df_statistics_station=df_statistics_station, station=station)
        print(f"P factor is: {p * 100}%")

        # mean width and std of of 'uncertainty band'
        mean_10_90 = np.mean(
            df_statistics_station["P90"] - df_statistics_station["P10"])
        std_10_90 = np.std(
            df_statistics_station["P90"] - df_statistics_station["P10"])
        print(f"mean_10_90={mean_10_90}; std_10_90={std_10_90}")
        mean_obs = df_statistics_station["measured"].mean()
        std_obs = df_statistics_station['measured'].std(ddof=1)
        print(f"mean_obs={mean_obs}; std_obs={std_obs}")

        if plotting:
            fig = larsimStatisticsObject.plot_heatmap_si_for_single_station(station=station, si_type="Sobol_m")
            fig.update_layout(title="Time Varying First Order Sobol Indices")
            fig.show()

            fig = larsimStatisticsObject.plot_heatmap_si_for_single_station(station=station, si_type="Sobol_t")
            fig.update_layout(title="Time Varying Total Order Sobol Indices")
            fig.show()

            fig = larsimStatisticsObject.plot_si_and_normalized_measured_time_signal(si_df=si_m_df, station=station)
            fig.update_yaxes(title_text='First Order Sobol Indices')
            fig = _add_precipitation_to_graph(
                fig, uq_output_paths_obj.master_configuration_folder, larsimStatisticsObject)
            fig.show()

            fig = larsimStatisticsObject.plot_si_and_normalized_measured_time_signal(si_df=si_t_df, station=station)
            fig.update_yaxes(title_text='Total Order Sobol Indices')
            fig = _add_precipitation_to_graph(
                fig, uq_output_paths_obj.master_configuration_folder, larsimStatisticsObject)
            fig.show()

        # Extracting gPCE surrogate model and evaluating it at the calibrated point
        tape35_file = uq_output_paths_obj.master_configuration_folder / "tape35"
        tape35 = pd.read_csv(tape35_file, index_col=False, delimiter=";")
        dict_temp = {x: x.strip() for x in list(tape35.columns)}
        tape35.rename(columns=dict_temp, inplace=True)
        tape35.set_index("TGB", inplace=True)
        default_values_dict = {}
        for param in params_list:
            default_values_dict[param] = float(tape35.loc[MAPPING_STATION_TO_TGB[station]][param])
        list_of_single_distr = []
        for param in configuration_object["parameters"]:  # larsimStatisticsObject.larsimConfObject.configurationObject
            if param["distribution"] == "Uniform":
                list_of_single_distr.append(cp.Uniform(param["lower_limit"], param["upper_limit"]))
        joint = cp.J(*list_of_single_distr)
        # TODO - make this general!!!
        joint_standard = cp.J(cp.Uniform(), cp.Uniform(), cp.Uniform(), cp.Uniform(), cp.Uniform())
        # number_of_parameters = len(configurationObject["parameters"])
        # temp = [cp.Uniform()]*number_of_parameters
        # joint_standard = cp.J(*temp)
        sample = list(default_values_dict.values())
        transformed_sample = utility.transformation_of_parameters_var1(sample, joint, joint_standard)
        start = samples_df_simulation_result.TimeStamp.min()
        end = samples_df_simulation_result.TimeStamp.max()
        data_range = pd.date_range(start=start, end=end, freq="1H")
        # gPCE_model = defaultdict()
        # for single_date in data_range:
        #     gPCE_model[single_date] = statistics_dictionary[(station, single_date)]["gPCE"]
        gPCE_model_evaluated_at_calib_sample_list = defaultdict()  # []
        for single_date in data_range:
            gPCE_model_for_single_timestep = statistics_dictionary[(station, single_date)]["gPCE"]
            model_output = gPCE_model_for_single_timestep(transformed_sample)
            gPCE_model_evaluated_at_calib_sample_list[single_date] = model_output
            # gPCE_model_evaluated_at_calib_sample_list.append(model_output)

    return gPCE_model_evaluated_at_calib_sample_list


def _add_precipitation_to_graph(fig, master_configuration_folder, larsimStatisticsObject):
    n_lila_local_file = pathlib.Path(master_configuration_folder/"station-n.lila")
    df_n = larsimInputOutputUtilities.any_lila_parser_toPandas(n_lila_local_file)
    df_n = larsimDataPostProcessing.get_time_vs_station_values_df(df_n)
    df_n = larsimDataPostProcessing.parse_df_based_on_time(df_n, (larsimStatisticsObject.timesteps_min,
                                                                  larsimStatisticsObject.timesteps_max
                                                                  ))
    df_n.set_index("TimeStamp", inplace=True)
    mean_n = df_n.mean(axis=1)
    max_n = df_n.max(axis=1)
    max_N = max(list(df_n.max()))

    fig.add_trace(go.Scatter(x=df_n.index, \
                             y=max_n, \
                             text=max_n, \
                             name="N_Max",
                             yaxis="y2", ))
    fig.update_layout(
        xaxis=dict(
            autorange=True,
            range=[larsimStatisticsObject.timesteps_min, larsimStatisticsObject.timesteps_max],
            type="date"
        ),
        yaxis=dict(
            side="left",
            domain=[0, 0.7],
            mirror=True,
            tickfont={"color": "#d62728"},
            tickmode="auto",
            ticks="inside",
            titlefont={"color": "#d62728"},
        ),
        yaxis2=dict(
            anchor="x",
            domain=[0.7, 1],
            mirror=True,
            range=[max_N, 0],
            side="right",
            tickfont={"color": '#1f77b4'},
            nticks=3,
            tickmode="auto",
            ticks="inside",
            titlefont={"color": '#1f77b4'},
            title="Precipitation [mm/h]",
            type="linear",
        )
    )
    return fig

###################################################################################################################
# Refactor this
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

    result_dict = {}

    grouped = df_all_simulations.groupby(['Stationskennung','TimeStamp'])
    groups = grouped.groups
    for key,val_indices in groups.items():
        discharge_values = df_all_simulations.loc[val_indices.values].Value.values
        result_dict[key] = {}
        result_dict[key]["Q"] = discharge_values
        if regression:
            qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, discharge_values)
        else:
            qoi_gPCE = cp.fit_quadrature(polynomial_expansion, nodes, weights, discharge_values)
        numPercSamples = 10 ** 5
        result_dict[key]["gPCE"] = qoi_gPCE
        result_dict[key]["E"] = float((cp.E(qoi_gPCE, dist)))
        result_dict[key]["Var"] = float((cp.Var(qoi_gPCE, dist)))
        result_dict[key]["StdDev"] = float((cp.Std(qoi_gPCE, dist)))
        result_dict[key]["Sobol_m"] = cp.Sens_m(qoi_gPCE, dist)
        result_dict[key]["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist)
        result_dict[key]["Sobol_t"] = cp.Sens_t(qoi_gPCE, dist)
        result_dict[key]["P10"] = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
        result_dict[key]["P90"] = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
        if isinstance(result_dict[key]["P10"], (list)) and len(result_dict[key]["P10"]) == 1:
            result_dict[key]["P10"] = result_dict[key]["P10"][0]
            result_dict[key]["P90"] = result_dict[key]["P90"][0]
    return result_dict


def _calcStatisticsForSaltelli(df_all_simulations, simulationNodes, numEvaluations, order, regression):
    timesteps = df_all_simulations.TimeStamp.unique()
    numbTimesteps = len(timesteps)
    station_names = df_all_simulations.Stationskennung.unique()

    nodes = simulationNodes.distNodes

    result_dict = {}

    grouped = df_all_simulations.groupby(['Stationskennung','TimeStamp'])
    groups = grouped.groups

    dim = len(simulationNodes.nodeNames)

    for key,val_indices in groups.items():
        result_dict[key] = {}

        discharge_values = df_all_simulations.loc[val_indices.values].Value.values #numpy array - for sartelli it should be n(2+d)x1
        #extended_standard_discharge_values = discharge_values[:(2*numEvaluations)]
        discharge_values_saltelli = discharge_values[:, np.newaxis]
        standard_discharge_values = discharge_values_saltelli[:numEvaluations,:] #values based on which we calculate standard statistics
        extended_standard_discharge_values = discharge_values_saltelli[:(2*numEvaluations),:]

        result_dict[key]["Q"] = standard_discharge_values

        #result_dict[key]["min_q"] = np.amin(discharge_values) #standard_discharge_values.min()
        #result_dict[key]["max_q"] = np.amax(discharge_values) #standard_discharge_values.max()

        #result_dict[key]["E"] = np.sum(extended_standard_discharge_values, axis=0, dtype=np.float64) / (2*numEvaluations)
        result_dict[key]["E"] = np.mean(discharge_values[:(2*numEvaluations)], 0)
        #result_dict[key]["E"] = np.mean(extended_standard_discharge_values, 0)
        #result_dict[key]["Var"] = float(np.sum(power(standard_discharge_values)) / numEvaluations - result_dict[key]["E"] ** 2)
        #result_dict[key]["Var"] = np.sum((extended_standard_discharge_values - result_dict[key]["E"]) ** 2, axis=0, dtype=np.float64) / (2*numEvaluations - 1)
        #result_dict[key]["StdDev"] = np.sqrt(result_dict[key]["Var"], dtype=np.float64)
        result_dict[key]["StdDev"] = np.std(discharge_values[:(2*numEvaluations)], 0, ddof=1)
        #result_dict[key]["StdDev"] = np.std(extended_standard_discharge_values, 0, ddof=1)

        #result_dict[key]["P10"] = np.percentile(discharge_values[:numEvaluations], 10, axis=0)
        #result_dict[key]["P90"] = np.percentile(discharge_values[:numEvaluations], 90, axis=0)
        result_dict[key]["P10"] = np.percentile(discharge_values[:(2*numEvaluations)], 10, axis=0)
        result_dict[key]["P90"] = np.percentile(discharge_values[:(2*numEvaluations)], 90, axis=0)

        result_dict[key]["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_4(discharge_values_saltelli, dim, numEvaluations)
        #result_dict[key]["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_3(discharge_values_saltelli, dim, numEvaluations)
        result_dict[key]["Sobol_t"] = saltelliSobolIndicesHelpingFunctions._Sens_t_sample_4(discharge_values_saltelli, dim, numEvaluations)

        if isinstance(result_dict[key]["P10"], (list)) and len(result_dict[key]["P10"]) == 1:
            result_dict[key]["P10"]=result_dict[key]["P10"][0]
            result_dict[key]["P90"]=result_dict[key]["P90"][0]


def _replot_statistics(working_folder):
    df_all_simulations = _get_df_simulation_from_file(working_folder)
    simulationNodes = _get_nodes_from_file(working_folder)
    result_dict = _calcStatisticsForSc(df_all_simulations, simulationNodes, order=2, regression=False)

    timesteps = df_all_simulations.TimeStamp.unique()
    pdTimesteps = [pd.Timestamp(timestep) for timestep in timesteps]
    keyIter = list(itertools.product(["MARI",],pdTimesteps))
    colors = ['darkred', 'midnightblue', 'mediumseagreen', 'darkorange']
    labels = [nodeName.strip() for nodeName in simulationNodes.nodeNames]
    n_rows = 4

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False)
    fig.add_trace(go.Scatter(x=pdTimesteps, y=[result_dict[key]["E"] for key in keyIter], name='E[Q]',line_color='green', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdTimesteps, y=[(result_dict[key]["E"] - result_dict[key]["StdDev"]) for key in keyIter], name='mean - std. dev', line_color='darkviolet', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdTimesteps, y=[(result_dict[key]["E"] + result_dict[key]["StdDev"]) for key in keyIter], name='mean + std. dev', line_color='darkviolet', mode='lines', fill='tonexty'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdTimesteps, y=[result_dict[key]["P10"] for key in keyIter], name='10th percentile',line_color='yellow', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdTimesteps, y=[result_dict[key]["P90"] for key in keyIter], name='90th percentile',line_color='yellow', mode='lines',fill='tonexty'), row=1, col=1)

    fig.add_trace(go.Scatter(x=pdTimesteps, y=[result_dict[key]["StdDev"] for key in keyIter], name='std. dev', line_color='darkviolet'), row=2, col=1)

    for i in range(len(labels)):
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[result_dict[key]["Sobol_m"][i] for key in keyIter], name=labels[i], legendgroup=labels[i], line_color=colors[i]), row=3, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[result_dict[key]["Sobol_t"][i] for key in keyIter], legendgroup=labels[i], showlegend = False, line_color=colors[i]), row=4, col=1)

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
