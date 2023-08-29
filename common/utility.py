"""
Set of utility functions for preparing and/or postprocessing data for UQ runs of different models

@author: Ivana Jovanovic Buha
"""

import chaospy as cp
from collections import defaultdict
import datetime
from distutils.util import strtobool
import dill
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas as pd
import pathlib
import pickle
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from tabulate import tabulate


# TODO Update this class with new changes
class UQOutputPaths(object):
    def __init__(self, workingDir):
        self.workingDir = workingDir
        self._update_other_paths_based_on_workingDir(workingDir)

    def _update_other_paths_based_on_workingDir(self, workingDir):
        # 'global' files
        self.nodes_file = workingDir / "nodes.simnodes.zip"
        self.parameters_file = workingDir / "parameters.pkl"
        self.args_file = workingDir / 'uqsim_args.pkl'
        self.configuration_object_file = workingDir / "configurationObject"

        # master_configuration_folder = workingDir / "master_configuration"
        self.model_runs_folder = workingDir / "model_runs"
        self.master_configuration_folder = self.model_runs_folder / "master_configuration"

        # Files produced by LarsimModelSetUp - __init__
        # TODO this files are either in workingDir or workingDir/"model_runs" - model_runs_folder
        self.df_measured_file = self.model_runs_folder / "df_measured.pkl"  # model_runs_folder/"df_measured.pkl"
        self.df_past_simulated_file = self.model_runs_folder / "df_past_simulated.pkl"  # "df_simulated.pkl"
        self.df_unaltered_file = self.model_runs_folder / "df_unaltered.pkl"  # "df_unaltered_ergebnis.pkl"
        self.gof_past_sim_meas_file = self.model_runs_folder / "gof_past_sim_meas.pkl"
        self.gof_unaltered_meas_file = self.model_runs_folder / "gof_unaltered_meas.pkl"
        self.gpce_file = self.model_runs_folder / "gpce.pkl"

        # Files produced by LarsimSamples
        self.df_all_simulations_file = self.model_runs_folder / "df_all_simulations.pkl"
        self.df_all_index_parameter_gof_file = self.model_runs_folder / "df_all_index_parameter_gof_values.pkl"
        self.df_all_index_parameter_file = self.model_runs_folder / "df_all_index_parameter_values.pkl"

        # Files produced by UQEF.Statistics and LarsimStatistics
        self.statistics_dictionary_file = self.model_runs_folder / "statistics_dictionary_qoi_Value.pkl"
        # self.statistics_dictionary_file = self.model_runs_folder / "statistics_dictionary_qoi_calculateNSE.pkl"

        # self.df_all_simulations_file = workingDir / "df_all_simulations.pkl"
        # self.df_all_index_parameter_gof_file = workingDir / "df_all_index_parameter_gof_values.pkl"
        # self.df_all_index_parameter_file = workingDir / "df_all_index_parameter_values.pkl"

        self.dict_of_approx_matrix_c_file = self.model_runs_folder / "dict_of_approx_matrix_c.pkl"
        self.dict_of_matrix_c_eigen_decomposition_file = self.model_runs_folder / "dict_of_matrix_c_eigen_decomposition.pkl"

        self.output_stat_graph_filename = workingDir / "sim-plotly.html"
        self.output_stat_graph_filename = str(self.output_stat_graph_filename)

    def update_specifi_model_run_output_file_paths(self, model_runs_folder, i):
        """
        These files are outputed in case when
        run_and_save_simulations and always_save_original_model_runs options are set to True
        model_runs_folder = workingDir / "model_runs"
        """
        parameters_Larsim_run_file = model_runs_folder / f"parameters_Larsim_run_{i}.pkl"
        parameters_Larsim_run_failed_file = model_runs_folder / f"parameters_Larsim_run_{i}_failed.pkl"
        df_Larsim_run_file = model_runs_folder / f"df_Larsim_run_{i}.pkl"
        gof_file = model_runs_folder / f"gof_{i}.pkl"
        gradients_matrices_file = model_runs_folder / f"gradients_matrices_{i}.npy"
        df_Larsim_raw_run_fileh = model_runs_folder / f"df_Larsim_raw_run_{i}.pkl"
        return parameters_Larsim_run_file, df_Larsim_run_file, gof_file, gradients_matrices_file, df_Larsim_raw_run_fileh


def get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=None):
    numDim = simulationNodes.nodes.shape[0]
    numSamples = simulationNodes.nodes.shape[1]
    if nodes_or_paramters == "nodes":
        if params_list is None:
            my_ditc = {f"x{i}": simulationNodes.nodes[i, :] for i in range(numDim)}
        else:
            # important assumption about the order of the nodes and their corresponding names
            my_ditc = {f"{params_list[i]}": simulationNodes.nodes[i, :] for i in range(numDim)}
    else:
        if params_list is None:
            my_ditc = {f"x{i}": simulationNodes.parameters[i, :] for i in range(numDim)}
        else:
            # important assumption about the order of the nodes and their corresponding names
            my_ditc = {f"{params_list[i]}": simulationNodes.parameters[i, :] for i in range(numDim)}
    df_nodes = pd.DataFrame(my_ditc)
    return df_nodes


def get_df_from_simulationNodes_list(simulationNodes_list):
    """
    simulationNodes_list.shape = (d, N)
    """
    numDim = simulationNodes_list.shape[0]
    numSamples = simulationNodes_list.shape[1]
    my_ditc = {f"x{i}": simulationNodes_list[i, :] for i in range(numDim)}
    df_nodes = pd.DataFrame(my_ditc)
    return df_nodes


def generate_df_with_nodes_and_weights_from_file(file_path, params_list=None):
    nodes_and_weights_array = np.loadtxt(file_path, delimiter=',')
    numDim = nodes_and_weights_array.shape[1] - 1
    numSamples = nodes_and_weights_array.shape[0]

    if params_list is None:
        my_ditc = {f"x{i}": nodes_and_weights_array[:, i] for i in range(numDim)}
    else:
        # important assumption about the order of the nodes and their corresponding names
        my_ditc = {f"{params_list[i]}": nodes_and_weights_array[:, i] for i in range(numDim)}
    my_ditc["w"] = nodes_and_weights_array[:, numDim]
    df_nodes_and_weights = pd.DataFrame(my_ditc)
    return df_nodes_and_weights


def plot_2d_matrix_static_from_list(simulationNodes_list, title="Plot nodes"):
    dfsimulationNodes = get_df_from_simulationNodes_list(simulationNodes_list)

    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(dfsimulationNodes, vars=list(dfsimulationNodes.columns), corner=True)
    plt.title(title, loc='left')
    plt.show()


def plot_3d_dynamic(dfsimulationNodes):
    my_columns = list(dfsimulationNodes.columns)
    fig = px.scatter_3d(dfsimulationNodes, x=my_columns[0], y=my_columns[1], z=my_columns[2])
    return fig


def plot_3d_statis(simulationNodes, nodes_or_paramters="nodes"):
    fig = plt.figure()
    axs = fig.gca(projection='3d')
    if nodes_or_paramters == "nodes":
        axs.scatter(list(simulationNodes.nodes[0, :]),
                    list(simulationNodes.nodes[1, :]),
                    list(simulationNodes.nodes[2, :]), marker="o")
        plt.title('Nodes\n')
    else:
        axs.scatter(list(simulationNodes.parameters[0, :]),
                    list(simulationNodes.parameters[1, :]),
                    list(simulationNodes.parameters[2, :]), marker="o")
        plt.title('Parameters\n')
    axs.set_xlabel("x1")
    axs.set_ylabel("x2")
    axs.set_zlabel("x3")
    plt.show()


def plot_2d_matrix_static(simulationNodes, nodes_or_paramters="nodes"):
    dfsimulationNodes = get_df_from_simulationNodes(simulationNodes, nodes_or_paramters)

    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(dfsimulationNodes, vars=list(dfsimulationNodes.columns), corner=True)
    plt.title(f"Plot: {nodes_or_paramters}", loc='left')
    plt.show()

#####################################
# Playing with transformations
#####################################


def transformation_of_parameters_var1(samples, distribution_r, distribution_q):
    """
    :param samples: array of samples from distribution_r
    :param distribution_r: 'standard' distribution
    :param distribution_q: 'user-defined' distribution
    :return: array of samples from distribution_q
    """
    # var 1
    return distribution_q.inv(distribution_r.fwd(samples))


def transformation_of_parameters_var1_1(samples, distribution_q):
    """
    :param samples: array of samples from distribution_r - when distribution_r is U[0,1]
    :param distribution_r: 'standard' distribution
    :param distribution_q: 'user-defined' distribution
    :return: array of samples from distribution_q
    """
    # var 1
    return distribution_q.inv(samples)


def transformation_of_parameters_var2(samples, distribution_r, distribution_q):
    """
    :param samples: array of samples from distribution_r, when distribution_r is U[-1,1] or U[0,1]
    :param distribution_r: 'standard' distribution either U[-1,1] or U[0,1]
    :param distribution_q: 'user-defined' distribution
    :return: array of samples from distribution_q
    """
    #distinqush between distribution_r is U[-1,1] or U[0,1]
    dim = len(distribution_r)
    assert len(distribution_r) == len(distribution_q)
    _a = np.empty([dim, 1])
    _b = np.empty([dim, 1])

    for i in range(dim):
        r_lower = distribution_r[i].lower
        r_upper = distribution_r[i].upper
        q_lower = distribution_q[i].lower
        q_upper = distribution_q[i].upper

        if r_lower == -1:
            _a[i] = (q_lower + q_upper) / 2
            _b[i] = (q_upper - q_lower) / 2
        elif r_lower == 0:
            _a[i] = q_lower
            _b[i] = (q_upper - q_lower)

    return _a + _b * samples

#####################################
# Utility for SG analysis
#####################################


def generate_table_single_rule_over_dim_and_orders_sparse_and_nonsparse(rule, dists, dim, q_orders, growth=None):
    num_nodes = np.zeros((len(dists), len(q_orders), 2), dtype=np.int32)

    if growth is None:
        growth = True if rule == "c" else False

    for dist_i, dist in enumerate(dists):
        for q_i, order in enumerate(q_orders):
            abscissas, weights = cp.generate_quadrature(order, dist, rule=rule, sparse=True, growth=growth)
            num_nodes[dist_i][q_i][0] = len(abscissas.T)
            abscissas, weights = cp.generate_quadrature(order, dist, rule=rule, sparse=False, growth=growth)
            num_nodes[dist_i][q_i][1] = len(abscissas.T)

    table = []
    for dist_i, dist in enumerate(dists):
        table.append(["*", "*", "*", "*"])
        for q_i, order in enumerate(q_orders):
            table.append([rule, dim[dist_i], order, num_nodes[dist_i][q_i][0], num_nodes[dist_i][q_i][1]])
    print(tabulate(table, headers=["rule", "dim", "q", "#nodes sparse tensor", "#nodes full tensor"], numalign="right"))


def generate_table_single_rule_over_dim_and_orders(rule, dists, dim, q_orders, sparse=True, growth=None):
    num_nodes = np.zeros((len(dists), len(q_orders), 1), dtype=np.int32)

    table_column_name = "#nodes sparse tensor" if sparse else "#nodes full tensor"

    if growth is None:
        growth = True if rule == "c" else False

    for dist_i, dist in enumerate(dists):
        for q_i, order in enumerate(q_orders):
            abscissas, weights = cp.generate_quadrature(order, dist, rule=rule, sparse=sparse, growth=growth)
            num_nodes[dist_i][q_i][0] = len(abscissas.T)

    table = []
    for dist_i, dist in enumerate(dists):
        table.append(["*", "*", "*", "*"])
        for q_i, order in enumerate(q_orders):
            table.append([rule, dim[dist_i], order, num_nodes[dist_i][q_i][0]])
    print(tabulate(table, headers=["rule", "dim", "q", table_column_name], numalign="right"))


def plot_2d_matrix_of_nodes_over_orders(rule, dist, orders, sparse=False, growth=None):
    if growth is None:
        growth = True if rule == "c" else False

    for order in orders:
        abscissas, weights = cp.generate_quadrature(order, dist, rule=rule, growth=growth, sparse=sparse)
        # print(order, abscissas.round(3), weights.round(3))

        dimensionality = len(abscissas)

        abscissas = abscissas.T

        dict_for_plotting = {f"x{i}": abscissas[:, i] for i in range(dimensionality)}

        df_nodes_weights = pd.DataFrame(dict_for_plotting)

        sns.set(style="ticks", color_codes=True)
        g = sns.pairplot(df_nodes_weights, vars=list(dict_for_plotting.keys()), corner=True)
        if growth:
            title = f"{rule} points chaospy; order = {order}; sparse={str(sparse)}; #nodes={abscissas.shape[0]}; growth=True"
        else:
            title = f"{rule} points chaospy; order = {order}; sparse={str(sparse)}; #nodes={abscissas.shape[0]}"
        plt.title(title, loc='left')
        plt.show()


def generate_table_over_rules_orders_for_single_dim(rules, dist, dim, q_orders, growth=None):
    num_nodes = np.zeros((len(rules), len(q_orders), 2), dtype=np.int32)

    # produce num_nodes matrix
    for r_i, r in enumerate(rules):
        for q_i, q in enumerate(q_orders):

            if growth is None:
                growth = True if r == "c" else False

            nodes, weights = cp.generate_quadrature(q, dist, rule=r, growth=growth)
            num_nodes[r_i][q_i][0] = len(nodes.T)

            nodes, weights = cp.generate_quadrature(q, dist, rule=r, sparse=True)
            num_nodes[r_i][q_i][1] = len(nodes.T)

    # create table
    table = []
    for r_i, r in enumerate(rules):
        for q_i, q in enumerate(q_orders):
            ok = num_nodes[r_i][q_i][1] < num_nodes[r_i][q_i][0]
            table.append([r, q, num_nodes[r_i][q_i][0], num_nodes[r_i][q_i][1], "ok" if ok else "nok"])

    print(tabulate(table,
                   headers=["rule", "q", "#nodes full tensor", "#nodes sparse", "#nodes sparse < #nodes full tensor"],
                   numalign="right"))

#####################################
# Utility functions for processing/manipulating pandas.DataFrame (i.e., main data structure)
#####################################


def transform_column_in_df(df, transformation_function_str, column_name, new_column_name=None):
    if new_column_name is None:
        new_column_name = column_name
    if transformation_function_str == "log":
        df[column_name] = df[column_name].apply(lambda x: x if x > 1e-4 else 1e-4)
        df[new_column_name] = np.log(df[column_name], out=np.zeros_like(df[column_name].values), where=(df[column_name].values!=0))
    elif transformation_function_str == "log2":
        df[column_name] = df[column_name].apply(lambda x: x if x > 1e-4 else 1e-4)
        df[new_column_name] = np.log2(df[column_name], out=np.zeros_like(df[column_name].values), where=(df[column_name].values!=0))
    elif transformation_function_str == "log10":
        df[column_name] = df[column_name].apply(lambda x: x if x > 1e-4 else 1e-4)
        df[new_column_name] = np.log10(df[column_name], out=np.zeros_like(df[column_name].values), where=(df[column_name].values!=0))
    elif transformation_function_str == "exp":
        # df[column_name] = df[column_name].apply(lambda x: x if x > 1e-4 else 1e-4)
        df[new_column_name] = np.exp(df[column_name])
    else:
        # flux_df[new_column_name] = flux_df[single_qoi_column].apply(single_transformation)
        raise NotImplementedError("For know only log transformation is supported")

#####################################
# Utility for calculating different GoF/Objective/Likelihood funtions/metrices
#####################################
from . import objectivefunctions

_all_functions = [objectivefunctions.MAE, objectivefunctions.MSE,
                  objectivefunctions.RMSE, objectivefunctions.NRMSE, objectivefunctions.RSR,
                  objectivefunctions.BIAS, objectivefunctions.PBIAS, objectivefunctions.ROCE,
                  objectivefunctions.NSE, objectivefunctions.LogNSE,
                  objectivefunctions.LogGaussian, objectivefunctions.CorrelationCoefficient,
                  objectivefunctions.KGE]


mapping_gof_names_to_functions = {
    "MAE": objectivefunctions.MAE,
    "MSE": objectivefunctions.MSE,
    "RMSE": objectivefunctions.RMSE,
    "NRMSE": objectivefunctions.NRMSE,
    "RSR": objectivefunctions.RSR,
    "BIAS": objectivefunctions.BIAS,
    "PBIAS": objectivefunctions.PBIAS,
    "ROCE": objectivefunctions.ROCE,
    "NSE": objectivefunctions.NSE,
    "LogNSE": objectivefunctions.LogNSE,
    "LogGaussian": objectivefunctions.LogGaussian,
    "CorrelationCoefficient": objectivefunctions.CorrelationCoefficient,
    "KGE": objectivefunctions.KGE
}


def gof_list_to_function_names(gof_list):
    if gof_list == "all" or gof_list is None:
        gof_list = _all_functions
    else:
        if not isinstance(gof_list, list):
            gof_list = [gof_list,]
        for idx, f in enumerate(gof_list):
            if not callable(f) and f in mapping_gof_names_to_functions:
                gof_list[idx] = mapping_gof_names_to_functions[f]
            elif callable(f) and f in _all_functions:
                continue
            else:
                raise ValueError("Not proper specification of Goodness of Fit function name")
    return gof_list


def _check_if_measured_or_predicted_are_empty(measuredDF, predictedDF, measuredDF_column_name, simulatedDF_column_name, gof_list):
    if measuredDF.empty or predictedDF.empty \
            or measuredDF_column_name not in measuredDF.columns \
            or simulatedDF_column_name not in predictedDF.columns:
        result_dict = {}
        for f in gof_list:
            result_dict[f.__name__] = np.nan
        return True, result_dict
    else:
        return False, None


def dataframe_difference(df1, df2, which=None):
    """Find rows which are different between two DataFrames."""
    comparison_df = df1.merge(df2,
                              indicator=True,
                              how='outer')
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df


def filter_two_DF_on_common_timesteps(DF1, DF2, saveToFile=None, column_name_df1="TimeStamp", column_name_df2=None):
    # really important to check if there are any missing time steps compared to measured array
    # in other words measured and observed values have to be of the same length
    # TODO Should we make a copy of DF1 and DF1 or change the origins?
    # TODO Maybe there is more elegant way to 
    if column_name_df2 is None:
        column_name_df2 = column_name_df1
    df1_column_is_index_column = False
    df2_column_is_index_column = False
    if DF1.index.name == column_name_df1:
        df1_column_is_index_column = True
        DF1 = DF1.reset_index()
        DF1.rename(columns={DF1.index.name: column_name_df1}, inplace=True)
    if DF2.index.name == column_name_df2:
        df2_column_is_index_column = True
        DF2 = DF2.reset_index()
        DF2.rename(columns={DF2.index.name: column_name_df2}, inplace=True)

    list_of_columns_to_drop = [x for x in list(DF1.columns) if x != column_name_df1]
    DF1_timeSteps = DF1.drop(list_of_columns_to_drop, axis=1, errors='ignore')
    list_of_columns_to_drop = [x for x in list(DF2.columns) if x != column_name_df2]
    DF2_timeSteps = DF2.drop(list_of_columns_to_drop, axis=1, errors='ignore')

    diff_df = dataframe_difference(DF1_timeSteps, DF2_timeSteps, which=None)
    if saveToFile is not None:
        diff_df.to_pickle(saveToFile, compression="gzip")

    left_only = diff_df[diff_df['_merge'] == 'left_only']
    right_only = diff_df[diff_df['_merge'] == 'right_only']

    DF1 = DF1[~DF1[column_name_df1].isin(left_only[column_name_df1].values)]
    DF2 = DF2[~DF2[column_name_df2].isin(right_only[column_name_df2].values)]

    if df1_column_is_index_column:
        DF1.set_index(column_name_df1, inplace=True)
    if df2_column_is_index_column:
        DF2.set_index(column_name_df2, inplace=True)

    return DF1, DF2


def calculateGoodnessofFit_simple(measuredDF, predictedDF, gof_list,
                                  measuredDF_time_column_name="TimeStamp",
                                  measuredDF_column_name='Value',
                                  simulatedDF_time_column_name="TimeStamp",
                                  simulatedDF_column_name='Value',
                                  return_dict=True,
                                  **kwargs):
    """
    Assumption - two columns of interest are aligned with respect to time
    :param measuredDF:
    :param predictedDF:
    :param gof_list:
    :param measuredDF_column_name:
    :param simulatedDF_column_name:
    :return:
    """
    # calculate mean of the observed - measured discharge
    # mean_gt_discharge = np.mean(measuredDF[measuredDF_column_name].values)

    # TODO Duplicated code
    gof_list = gof_list_to_function_names(gof_list)
    is_empty, result_dict = _check_if_measured_or_predicted_are_empty(measuredDF, predictedDF,
                                                                      measuredDF_column_name, simulatedDF_column_name,
                                                                      gof_list)
    if is_empty:
        return result_dict

    # DataFrames containing measurements might be longer than the one containing model predictions - alignment is needed
    # It might be as well that one of DataFrames does not contain all the timesteps the other one does
    # therefore, apply one of these two filtering functions
    # assert measuredDF_time_column_name == simulatedDF_time_column_name, "Assertion failed in utility.calculateGoodnessofFit_simple"
    predictedDF, measuredDF = filter_two_DF_on_common_timesteps(predictedDF, measuredDF,
                                                                column_name_df1=measuredDF_time_column_name,
                                                                column_name_df2=simulatedDF_time_column_name)
    #predictedDF, measuredDF = align_dataFrames_timewise_2(predictedDF, measuredDF)

    if return_dict:
        result = dict()
    else:
        result = []

    for f in gof_list:
        try:
            temp_result = f(measuredDF, predictedDF, measuredDF_column_name, simulatedDF_column_name, **kwargs)
        except:
            temp_result = np.nan

        if return_dict:
            result[f.__name__] = temp_result
        else:
            result.append(temp_result)

    return result

###################################################################################################################
# Paths related functions
###################################################################################################################


class FileError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        return repr(self.message)


def _check_if_file_exists_pathlib(file: pathlib.PosixPath):
    return file.exists()


def _check_is_file_pathlib(file: pathlib.PosixPath):
    return file.is_file()


def check_if_file_exists_pathlib(file_path, message=None):
    if not _check_is_file_pathlib(file_path) or not _check_if_file_exists_pathlib(file_path):
        raise FileError(message)
    else:
        return True


def _check_if_file_exists_str(file_path, message=None):
    if not os.path.isfile(file_path):
        raise FileError(message)
    else:
        return True


def check_if_file_exists(file_path, message=None):
    if message is None:
        message = f'{file_path} is not a file, or it does not exists!'
    if isinstance(file_path, str):
        return _check_if_file_exists_str(file_path, message)
    elif isinstance(file_path, pathlib.PosixPath):
        return check_if_file_exists_pathlib(file_path, message)
    else:
        raise FileError(f"{file_path} is neither a pathlib object nor string!")


def pathlib_to_from_str(path, transfrom_to='str'):
    """
    Take the input path and transform it to transfrom_to type
    transfrom_to: 'str' or 'Path'
    """
    if transfrom_to == 'Path' and isinstance(path, str):
        return pathlib.Path(path)
    elif transfrom_to == 'str' and isinstance(path, pathlib.PosixPath):
        return str(path)
    elif (transfrom_to == 'str' and isinstance(path, str)) or (transfrom_to == 'Path' and
                                                               isinstance(path, pathlib.PosixPath)):
        return path
    else:
        raise ValueError(f"Error in pathlib_to_from_str fucnction got"
                         f"unexpected argument; First argument should be of type string or pathlib.PosixPath")


def _path_to_str(path):
    return str(path)


def _path_to_pathlib(path):
    return pathlib.Path(path)


def get_current_directory():
    #return os.getcwd()
    return pathlib.Path().cwd()


def get_full_path_of_file(file: pathlib.PosixPath):
    return file.resolve()


def get_home_directory():
    return pathlib.Path.home()

###################################################################################################################
# Time configurations
###################################################################################################################


def parse_datetime_configuration(time_settings_config):
    """
    Reads configuration dictionary and determines the start end end date of the simulation
    """
    if time_settings_config is None:
        time_settings_config = dict()

    if isinstance(time_settings_config, dict):
        if "time_settings" in time_settings_config:
            data = time_settings_config["time_settings"]
        else:
            data = time_settings_config
        start_year = data.get("start_year", None)
        start_month = data.get("start_month", None)
        start_day = data.get("start_day", None)
        start_hour = data.get("start_hour", None)
        start_minute = data.get("start_minute", None)
        end_year = data.get("end_year", None)
        end_month = data.get("end_month", None)
        end_day = data.get("end_day", None)
        end_hour = data.get("end_hour", None)
        end_minute = data.get("end_minute", None)
    elif isinstance(time_settings_config, list) or isinstance(time_settings_config, tuple):
        start_year, start_month, start_day, start_hour, start_minute, \
        end_year, end_month, end_day, end_hour, end_minute = time_settings_config
    else:
        raise Exception(f"[Error] in parse_datetime_configuration: time_settings_config should be "
                        f"dict, list or tuple storing start_year, start_month, etc.")

    start_dt = datetime.datetime(year=start_year, month=start_month, day=start_day,
                                 hour=start_hour, minute=start_minute)

    end_dt = datetime.datetime(year=end_year, month=end_month, day=end_day,
                               hour=end_hour, minute=end_minute)
    return [start_dt, end_dt]


###################################################################################################################
# Reading, extract data and manipulate configuration file/object
###################################################################################################################


def return_configuration_object(configurationObject):
    if isinstance(configurationObject, str) or isinstance(configurationObject, pathlib.PosixPath):
        try:
            with open(configurationObject) as f:
                return json.load(f)
        except ValueError:
            raise ValueError(
                f"[Error] The .json file {configurationObject} "
                f"which should be Configuration Object does not exist!\n")
    elif isinstance(configurationObject, dict):
        return configurationObject
    else:
        raise ValueError(
            "[Error] You have to specify the Configuration Object or provide the path to the .json file!\n")


def check_if_configurationObject_is_in_right_format_and_return(configurationObject, raise_error=True):
    if isinstance(configurationObject, dict) or isinstance(configurationObject, pd.DataFrame):
        return configurationObject
    elif isinstance(configurationObject, str) or isinstance(configurationObject, pathlib.PosixPath):
        configurationObject = read_configuration_dict_from_json_file(configurationObject)
        return configurationObject
    else:
        if raise_error:
            raise Exception("Error in configure_parameters_value function - configurationObject is not in the expected form!")
        else:
            return None


def read_configuration_object_json(configurationObject, element=None):
    """
    :param configurationObject: posixPath to the json file
    element can be, e.g., "Output", "parameters", "Timeframe", "parameters_settings"
    """
    # with configurationObject.open(encoding="UTF-8") as source:
    with open(configurationObject) as f:
        configuration_object = json.load(f)
    if element is None:
        return configuration_object
    else:
        return configuration_object[element]


def read_configuration_dict_from_json_file(configurationFile):
    """

    :param configurationFile: posixPath to the json file
    :return:
    """
    return read_configuration_object_json(configurationObject=configurationFile)


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


def read_simulation_settings_from_configuration_object(configurationObject, **kwargs):
    # TODO Delete unnecessary copy() below...
    configurationObject = check_if_configurationObject_is_in_right_format_and_return(configurationObject,
                                                                                     raise_error=False)

    result_dict = dict()

    if configurationObject is None:
        return result_dict

    # dict_config_simulation_settings = configurationObject["simulation_settings"]
    dict_config_simulation_settings = configurationObject.get("simulation_settings", dict())

    if "qoi" in kwargs:
        qoi = kwargs['qoi']
    else:
        qoi = dict_config_simulation_settings.get("qoi", "Q")
    result_dict["qoi"] = qoi

    if "qoi_column" in kwargs:
        qoi_column = kwargs['qoi_column']
    else:
        qoi_column = dict_config_simulation_settings.get("qoi_column", "Value")
    result_dict["qoi_column"] = qoi_column

    temp = result_dict["qoi_column"]

    multiple_qoi = False
    number_of_qois = 1
    if (isinstance(qoi, list) and isinstance(qoi_column, list)) or (qoi == "GoF" and isinstance(qoi_column, list)):
        multiple_qoi = True
        number_of_qois = len(qoi_column)
    result_dict["multiple_qoi"] = multiple_qoi
    result_dict["number_of_qois"] = number_of_qois

    if "transform_model_output" in kwargs:
        transform_model_output = kwargs['transform_model_output']
    else:
        if multiple_qoi:
            try:
                transform_model_output = dict_config_simulation_settings["transform_model_output"]
                for idx, single_transform_model_output in enumerate(transform_model_output):
                    if single_transform_model_output == "None":
                        transform_model_output[idx] = None
            except KeyError:
                transform_model_output = [None] * number_of_qois
        else:
            transform_model_output = dict_config_simulation_settings.get("transform_model_output", "None")
            if transform_model_output == "None":
                transform_model_output = None
    result_dict["transform_model_output"] = transform_model_output

    if "read_measured_data" in kwargs:
        read_measured_data = kwargs['read_measured_data']
    else:
        if multiple_qoi:
            read_measured_data = []
            try:
                temp = dict_config_simulation_settings["read_measured_data"]
            except KeyError:
                temp = ["False"] * number_of_qois
            for i in range(number_of_qois):
                if isinstance(temp[i], str):
                    read_measured_data.append(strtobool(temp[i]))
                else:
                    read_measured_data.append(temp[i])
        else:
            read_measured_data = dict_config_simulation_settings.get("read_measured_data", "False")
            if isinstance(read_measured_data, str):
                read_measured_data = strtobool(read_measured_data)

    if "qoi_column_measured" in kwargs:
        qoi_column_measured = kwargs['qoi_column_measured']
    else:
        if multiple_qoi:
            try:
                qoi_column_measured = dict_config_simulation_settings["qoi_column_measured"]
                for idx, single_qoi_column_measured in enumerate(qoi_column_measured):
                    if single_qoi_column_measured == "None":
                        qoi_column_measured[idx] = None
            except KeyError:
                qoi_column_measured = [None] * number_of_qois
        else:
            qoi_column_measured = dict_config_simulation_settings.get("qoi_column_measured", "streamflow")
            if qoi_column_measured == "None":
                qoi_column_measured = None
    result_dict["qoi_column_measured"] = qoi_column_measured

    if multiple_qoi:
        for idx, single_read_measured_data in enumerate(read_measured_data):
            if single_read_measured_data and qoi_column_measured[idx] is None:
                # raise ValueError
                read_measured_data[idx] = False
    else:
        if read_measured_data and qoi_column_measured is None:
            # raise ValueError
            read_measured_data = False
    result_dict["read_measured_data"] = read_measured_data

    if multiple_qoi:
        assert len(read_measured_data) == len(qoi_column)
        assert len(read_measured_data) == len(qoi_column_measured)

    calculate_GoF = strtobool(dict_config_simulation_settings.get("calculate_GoF", "False"))
    # self.calculate_GoF has to follow the self.read_measured_data which tells if ground truth data for that qoi is available
    list_calculate_GoF = [False, ]
    if calculate_GoF:
        if multiple_qoi:
            list_calculate_GoF = [False] * len(read_measured_data)
            for idx, single_read_measured_data in enumerate(read_measured_data):
                list_calculate_GoF[idx] = single_read_measured_data
        else:
            list_calculate_GoF = [read_measured_data, ]
    result_dict["calculate_GoF"] = calculate_GoF
    result_dict["list_calculate_GoF"] = list_calculate_GoF

    objective_function = dict_config_simulation_settings.get("objective_function", [])
    result_dict["objective_function"] = objective_function

    objective_function_qoi = None
    objective_function_names_qoi = None
    list_objective_function_qoi = None
    list_objective_function_names_qoi = None
    if qoi == "GoF":
        # take only those Outputs of Interest that have measured data
        if multiple_qoi:
            updated_qoi_column = []
            updated_qoi_column_measured = []
            updated_read_measured_data = []
            for idx, single_qoi_column in enumerate(qoi_column):
                if read_measured_data[idx]:
                    updated_qoi_column.append(single_qoi_column)
                    updated_qoi_column_measured.append(qoi_column_measured[idx])
                    updated_read_measured_data.append(True)
            # here, we overwrite qoi_column such that it only contains columns which do have a corresponding measured data
            qoi_column = updated_qoi_column
            qoi_column_measured = updated_qoi_column_measured
            read_measured_data = updated_read_measured_data
            result_dict["qoi_column"] = qoi_column
            result_dict["qoi_column_measured"] = qoi_column_measured
        else:
            if not read_measured_data:
                raise ValueError
        objective_function_qoi = dict_config_simulation_settings.get("objective_function_qoi", "all")
        objective_function_qoi = gof_list_to_function_names(objective_function_qoi)
        if isinstance(objective_function_qoi, list):
            objective_function_names_qoi = [
                single_gof.__name__ if callable(single_gof) else single_gof \
                for single_gof in objective_function_qoi]
            list_objective_function_qoi = objective_function_qoi.copy()
            list_objective_function_names_qoi = objective_function_names_qoi.copy()
        else:
            list_objective_function_qoi = [objective_function_qoi, ]
            if callable(objective_function_qoi):
                objective_function_names_qoi = objective_function_qoi.__name__
            else:
                objective_function_names_qoi = objective_function_qoi
            list_objective_function_names_qoi = [objective_function_names_qoi,]
    result_dict["objective_function_qoi"] = objective_function_qoi
    result_dict["objective_function_names_qoi"] = objective_function_names_qoi
    result_dict["list_objective_function_qoi"] = list_objective_function_qoi
    result_dict["list_objective_function_names_qoi"] = list_objective_function_names_qoi

    # Create a list version of some configuration parameters which might be needed when computing GoF
    # if self.qoi.lower() == "gof" or self.calculate_GoF:
    if not isinstance(qoi_column, list):
        list_qoi_column = [qoi_column, ]
    else:
        list_qoi_column = qoi_column.copy()
    if not isinstance(qoi_column_measured, list):
        list_qoi_column_measured = [qoi_column_measured, ]
    else:
        list_qoi_column_measured = qoi_column_measured.copy()
    if not isinstance(read_measured_data, list):
        list_read_measured_data = [read_measured_data, ]
    else:
        list_read_measured_data = read_measured_data.copy()
    if not isinstance(transform_model_output, list):
        list_transform_model_output = [transform_model_output,]
    else:
        list_transform_model_output = transform_model_output.copy()
    result_dict["list_qoi_column"] = list_qoi_column
    result_dict["list_qoi_column_measured"] = list_qoi_column_measured
    result_dict["list_read_measured_data"] = list_read_measured_data
    result_dict["list_transform_model_output"] = list_transform_model_output

    assert len(list_qoi_column) == len(list_qoi_column_measured)
    assert len(list_qoi_column_measured) == len(list_read_measured_data)

    dict_qoi_column_and_measured_info = defaultdict(tuple)
    for idx, single_qoi_column in enumerate(list_qoi_column):
        dict_qoi_column_and_measured_info[single_qoi_column] = \
            (list_read_measured_data[idx], list_qoi_column_measured[idx], list_transform_model_output[idx])
    result_dict["dict_qoi_column_and_measured_info"] = dict_qoi_column_and_measured_info

    mode = dict_config_simulation_settings.get("mode", "continuous")
    if mode != "continuous" and mode != "sliding_window" and mode != "resampling":
        raise Exception(f"[ERROR] mode should have one of the following values:"
                        f" \"continuous\" or \"sliding_window\" or \"resampling\"")
    interval = dict_config_simulation_settings.get("interval", 24)
    min_periods = dict_config_simulation_settings.get("min_periods", 1)
    method = dict_config_simulation_settings.get("method", "avrg")
    center = dict_config_simulation_settings.get("center", "center")
    if center != "center" and center != "left" and center != "right":
        raise Exception(f"[ERROR:] center should be either \"center\" or \"left\" or \"right\"")
    if method != "avrg" and method != "max" and method != "min":
        raise Exception(f"[ERROR:] method should be either \"avrg\" or \"max\" or \"min\"")
    result_dict["mode"] = mode
    result_dict["method"] = method
    result_dict["interval"] = interval
    result_dict["min_periods"] = min_periods
    result_dict["center"] = center

    compute_gradients = strtobool(dict_config_simulation_settings.get("compute_gradients", "False"))
    result_dict["compute_gradients"] = compute_gradients

    CD = None
    eps_val_global = None
    compute_active_subspaces = False
    save_gradient_related_runs = False
    gradient_analysis = False
    if compute_gradients:
        gradient_method = dict_config_simulation_settings.get("gradient_method", "Forward Difference")
        if gradient_method == "Central Difference":
            CD = 1  # flag for using Central Differences (with 2 * num_evaluations)
        elif gradient_method == "Forward Difference":
            CD = 0  # flag for using Forward Differences (with num_evaluations)
        else:
            raise Exception(f"NUMERICAL GRADIENT EVALUATION ERROR: "
                            f"Only \"Central Difference\" and \"Forward Difference\" supported")
        eps_val_global = dict_config_simulation_settings.get("eps_gradients", 1e-4)

        compute_active_subspaces = strtobool(
            dict_config_simulation_settings.get("compute_active_subspaces", "False"))
        save_gradient_related_runs = strtobool(
            dict_config_simulation_settings.get("save_gradient_related_runs", "False"))
        gradient_analysis = strtobool(
            dict_config_simulation_settings.get("gradient_analysis", "False"))

    result_dict["CD"] = CD
    result_dict["eps_val_global"] = eps_val_global
    result_dict["compute_active_subspaces"] = compute_active_subspaces
    result_dict["save_gradient_related_runs"] = save_gradient_related_runs
    result_dict["gradient_analysis"] = gradient_analysis

    return result_dict

# def infer_qoi_column_names(dict_processed_simulation_settings_from_config_file, **kwargs):
#     always_process_original_model_output = kwargs.get("always_process_original_model_output", False)
#     list_qoi_column_processed = []
#
#     if dict_processed_simulation_settings_from_config_file["mode"] == "continuous":
#         if dict_processed_simulation_settings_from_config_file["qoi"] == "GoF":
#             for idx, single_qoi_column in enumerate(
#                     dict_processed_simulation_settings_from_config_file["list_qoi_column"]):
#                 if dict_processed_simulation_settings_from_config_file["list_read_measured_data"][idx]:
#                     for single_objective_function_name_qoi in \
#                             dict_processed_simulation_settings_from_config_file["list_objective_function_names_qoi"]:
#                         new_column_name = single_objective_function_name_qoi + "_" + single_qoi_column
#                         list_qoi_column_processed.append(new_column_name)
#                         # self.additional_qoi_columns_besides_original_model_output = True
#                         # self.qoi_is_a_single_number = True
#                         # TODO in this case QoI is a single number, not a time-series!!!
#         else:
#             # here, model output itself is regarded as a QoI
#             pass
#             # list_qoi_column_processed.append(self.list_qoi_column)
#     elif self.mode == "sliding_window":
#         if self.qoi == "GoF":
#             for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                 if self.list_read_measured_data[idx]:
#                     for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
#                         new_column_name = single_objective_function_name_qoi + "_" + single_qoi_column + \
#                                           "_sliding_window"
#                         list_qoi_column_processed.append(new_column_name)
#                         # self.additional_qoi_columns_besides_original_model_output = True
#         else:
#             for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                 new_column_name = single_qoi_column + "_" + self.method + "_sliding_window"
#                 list_qoi_column_processed.append(new_column_name)
#                 # self.additional_qoi_columns_besides_original_model_output = True
#
#     if self.compute_gradients:
#         if self.gradient_analysis:
#             always_process_original_model_output = True
#             for single_param_name in self.nodeNames:
#                 if self.mode == "continuous":
#                     if self.qoi == "GoF":
#                         for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                             if self.list_read_measured_data[idx]:
#                                 for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
#                                     new_column_name = "d_" + single_objective_function_name_qoi + "_" + \
#                                                       single_qoi_column + "_" + "_d_" + single_param_name
#                                     # list_qoi_column_processed.append(new_column_name)
#                                     self.list_grad_columns.append(new_column_name)
#                                     # self.additional_qoi_columns_besides_original_model_output = True
#                                     # self.qoi_is_a_single_number = True
#                                     # TODO in this case QoI is a single number, not a time-series!!!
#                     else:
#                         for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                             new_column_name = "d_" + single_qoi_column + "_d_" + single_param_name
#                             # list_qoi_column_processed.append(new_column_name)
#                             self.list_grad_columns.append(new_column_name)
#                             # self.additional_qoi_columns_besides_original_model_output = True
#                 elif self.mode == "sliding_window":
#                     if self.qoi == "GoF":
#                         for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                             if self.list_read_measured_data[idx]:
#                                 for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
#                                     new_column_name = "d_" + single_objective_function_name_qoi + "_" + \
#                                                       single_qoi_column + "_sliding_window" \
#                                                       + "_d_" + single_param_name
#                                     # list_qoi_column_processed.append(new_column_name)
#                                     self.list_grad_columns.append(new_column_name)
#                                     # self.additional_qoi_columns_besides_original_model_output = True
#                     else:
#                         for idx, single_qoi_column in enumerate(self.list_qoi_column):
#                             new_column_name = "d_" + single_qoi_column + "_" + self.method + "_sliding_window" + \
#                                               "_d_" + single_param_name
#                             # list_qoi_column_processed.append(new_column_name)
#                             self.list_grad_columns.append(new_column_name)
#                             # self.additional_qoi_columns_besides_original_model_output = True
#
#         elif self.compute_active_subspaces:
#             always_process_original_model_output = True
#             pass
#
#     if self.corrupt_forcing_data:
#         always_process_original_model_output = True
#         pass  # 'precipitation' column is in the results df
#
#     wrong_computation_of_new_qoi_columns = self.additional_qoi_columns_besides_original_model_output and \
#                           len(list_qoi_column_processed) == 0
#     assert not wrong_computation_of_new_qoi_columns
#
#     if self.additional_qoi_columns_besides_original_model_output and len(list_qoi_column_processed) != 0:
#         if always_process_original_model_output:
#             self.list_qoi_column = self.list_original_model_output_columns + list_qoi_column_processed
#         else:
#             self.list_qoi_column = list_qoi_column_processed


def get_param_info_dict_from_configurationObject(configurationObject):
    configurationObject = check_if_configurationObject_is_in_right_format_and_return(configurationObject,
                                                                                     raise_error=False)
    result_dict = dict()
    if configurationObject is None:
        return result_dict
    list_of_parameters_from_json = configurationObject["parameters"]
    for _, param_entry_dict in enumerate(list_of_parameters_from_json):
        param_name = param_entry_dict.get("name")
        distribution = param_entry_dict.get("distribution", None)
        if "lower_limit" in param_entry_dict:
            lower_limit = param_entry_dict["lower_limit"]
        elif "lower" in param_entry_dict:
            lower_limit = param_entry_dict["lower"]
        else:
            lower_limit = None
        if "upper_limit" in param_entry_dict:
            upper_limit = param_entry_dict["upper_limit"]
        elif "upper" in param_entry_dict:
            upper_limit = param_entry_dict["upper"]
        else:
            upper_limit = None
        # lower_limit = param_entry_dict.get("lower_limit", None)
        # upper_limit = param_entry_dict.get("upper_limit", None)
        default_value = param_entry_dict.get("default", None)
        # parameter_value = param_entry_dict.get("value", None)

        result_dict[param_name] = {
            'distribution': distribution, 'default_value': default_value,
            'lower_limit': lower_limit, 'upper_limit': upper_limit
        }
    return result_dict

###################################################################################################################
# Reading saved data - produced by UQEF related run/simulation
###################################################################################################################


def read_and_print_uqsim_args_file(file):
    with open(file, 'rb') as f:
        uqsim_args = pickle.load(f)
    uqsim_args_temp_dict = vars(uqsim_args)
    print(f"UQSIM.ARGS")
    for key, value in uqsim_args_temp_dict.items():
        print(f"{key}: {value}")
    return uqsim_args


def _get_df_simulation_from_file(working_folder, df_all_simulations_name="df_all_simulations.pkl"):
    df_all_simulations = pathlib.Path(working_folder) / df_all_simulations_name
    df_all_simulations = pd.read_pickle(df_all_simulations, compression='gzip')
    return df_all_simulations


def _get_nodes_from_file(working_folder, dill_or_pickle="dill"):
    working_folder = pathlib_to_from_str(working_folder, transfrom_to='Path')
    nodes_dict = working_folder / 'nodes.simnodes'
    with open(nodes_dict, 'rb') as f:
        if dill_or_pickle == "dill":
            simulationNodes = dill.load(f)
        elif dill_or_pickle == "pickle":
            simulationNodes = pickle.load(f)
        else:
            simulationNodes = None
    return simulationNodes


def _get_df_index_parameter_gof_from_file(
        working_folder, df_all_index_parameter_gof_values_file_name="df_all_index_parameter_gof_values.pkl"):
    df_all_index_parameter_gof_values_file = pathlib.Path(working_folder) / df_all_index_parameter_gof_values_file_name
    df_all_index_parameter_gof_values = pd.read_pickle(df_all_index_parameter_gof_values_file, compression='gzip')
    return df_all_index_parameter_gof_values


def _get_statistics_dict(working_folder, statistics_dictionary_file_name="statistics_dictionary.pkl"):
    statistics_dictionary_file = pathlib.Path(working_folder) / statistics_dictionary_file_name
    statistics_dictionary = None
    if statistics_dictionary_file.is_file():
        with open(statistics_dictionary_file, 'rb') as f:
            statistics_dictionary = pickle.load(f)
    return statistics_dictionary


def create_statistics_dictionary_from_saved_single_qoi_statistics_dictionary(workingDir, list_qoi_column):
    statistics_dictionary = defaultdict(dict)
    for single_qoi in list_qoi_column:
        statistics_dictionary_file_temp = workingDir / f"statistics_dictionary_qoi_{single_qoi}.pkl"
        assert statistics_dictionary_file_temp.is_file()
        with open(statistics_dictionary_file_temp, 'rb') as f:
            statistics_dictionary_temp = pickle.load(f)
        statistics_dictionary[single_qoi] = statistics_dictionary_temp
    return statistics_dictionary

###################################################################################################################
# Plotting params and GoF values - mostly from df_index_parameter_gof filtered for a single station
###################################################################################################################


def _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof):
    def check_if_column_stores_parameter(x):
        if isinstance(x, tuple):
            x = x[0]
        return x not in list(mapping_gof_names_to_functions.keys()) and not x.startswith("d_") and \
               x not in ["index_run", "station", "successful_run", "qoi"]
    return [x for x in df_index_parameter_gof.columns.tolist() if check_if_column_stores_parameter(x)]


def _get_gof_columns_df_index_parameter_gof(df_index_parameter_gof):
    def check_if_column_stores_gof(x):
        if isinstance(x, tuple):
            x = x[0]
        return x in list(mapping_gof_names_to_functions.keys()) and x not in ["index_run", "station", "successful_run", "qoi"]
    return [x for x in df_index_parameter_gof.columns.tolist() if check_if_column_stores_gof(x)]


def _get_grad_columns_df_index_parameter_gof(df_index_parameter_gof):
    return [x for x in df_index_parameter_gof.columns.tolist() if x.startswith("d_")]


def plot_hist_of_gof_values_from_df(df_index_parameter_gof, name_of_gof_column="NSE"):
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


def plot_subplot_params_hist_from_df_conditioned(df_index_parameter_gof, name_of_gof_column="NSE",
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


def plot_scatter_matrix_params_vs_gof(df_index_parameter_gof, name_of_gof_column="NSE",
                                      hover_name="index_run"):
    columns_with_parameters = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    fig = px.scatter_matrix(df_index_parameter_gof,
                            dimensions=columns_with_parameters,
                            color=name_of_gof_column,
                            hover_name=hover_name)
    fig.update_traces(diagonal_visible=False)
    return fig

###################################################################################################################
# Plotting UQ & SA Output
###################################################################################################################


def scatter_3d_params_vs_gof(df_index_parameter_gof, param1, param2, param3, name_of_gof_column="NSE",
                             name_of_index_run_column="index_run"):
    fig = px.scatter_3d(
        df_index_parameter_gof, x=param1, y=param2, z=param3, color=name_of_gof_column, opacity=0.7,
        hover_data=[name_of_gof_column, name_of_index_run_column]
    )
    return fig


def plot_surface_2d_params_vs_gof(df_index_parameter_gof, param1, param2, num_of_points_in_1d=8,
                                  name_of_gof_column="NSE"):
    # from mpl_toolkits.mplot3d import Axes3D
    # from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    # from matplotlib import cm
    # from matplotlib.ticker import LinearLocator, FormatStrFormatter
    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.gca(projection='3d')

    # x = samples_df_index_parameter_gof['A2'].to_numpy()
    # y = samples_df_index_parameter_gof['BSF'].to_numpy()
    # z = samples_df_index_parameter_gof['RMSE'].to_numpy()

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


def plot_parallel_params_vs_gof(df_index_parameter_gof, name_of_gof_column="NSE", list_of_params=None):
    if list_of_params is None:
        list_of_params = _get_parameter_columns_df_index_parameter_gof(df_index_parameter_gof)
    dimensions = list_of_params + [name_of_gof_column, ]
    fig = px.parallel_coordinates(df_index_parameter_gof, color=name_of_gof_column, dimensions=dimensions)
    return fig


def plot_scatter_matrix_params_vs_gof_seaborn(df_index_parameter_gof, name_of_gof_column="NSE"):
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