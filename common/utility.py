"""
Set of utility functions for preparing and/or postprocessing data for UQ runs of different models

@author: Ivana Jovanovic Buha
"""

import chaospy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from tabulate import tabulate


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
    numSamples = simulationNodes.nodes.shape[0]
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
# Utility for calculating different GoF/Objective/Likelihood funtions/metrices
#####################################
from . import objectivefunctions

_all_functions = [objectivefunctions.calculateMAE, objectivefunctions.calculateMSE,
                  objectivefunctions.calculateRMSE, objectivefunctions.calculateNRMSE, objectivefunctions.calculateRSR,
                  objectivefunctions.calculateBIAS, objectivefunctions.calculatePBIAS, objectivefunctions.calculateROCE,
                  objectivefunctions.calculateNSE, objectivefunctions.calculateLogNSE,
                  objectivefunctions.calculateLogGaussian, objectivefunctions.calculateCorrelationCoefficient,
                  objectivefunctions.calculateKGE]


mapping_gof_names_to_functions = {
    "MAE": objectivefunctions.calculateMAE,
    "calculateMAE": objectivefunctions.calculateMAE,
    "MSE": objectivefunctions.calculateMSE,
    "calculateMSE": objectivefunctions.calculateMSE,
    "RMSE": objectivefunctions.calculateRMSE,
    "calculateRMSE": objectivefunctions.calculateRMSE,
    "NRMSE": objectivefunctions.calculateNRMSE,
    "calculateNRMSE": objectivefunctions.calculateNRMSE,
    "RSR": objectivefunctions.calculateRSR,
    "calculateRSR": objectivefunctions.calculateRSR,
    "BIAS": objectivefunctions.calculateBIAS,
    "calculateBIAS": objectivefunctions.calculateBIAS,
    "PBIAS": objectivefunctions.calculatePBIAS,
    "calculatePBIAS": objectivefunctions.calculatePBIAS,
    "ROCE": objectivefunctions.calculateROCE,
    "calculateROCE": objectivefunctions.calculateROCE,
    "NSE": objectivefunctions.calculateNSE,
    "calculateNSE": objectivefunctions.calculateNSE,
    "LogNSE": objectivefunctions.calculateLogNSE,
    "calculateLogNSE": objectivefunctions.calculateLogNSE,
    "LogGaussian": objectivefunctions.calculateLogGaussian,
    "calculateLogGaussian": objectivefunctions.calculateLogGaussian,
    "CorrelationCoefficient": objectivefunctions.calculateCorrelationCoefficient,
    "calculateCorrelationCoefficient": objectivefunctions.calculateCorrelationCoefficient,
    "KGE": objectivefunctions.calculateKGE,
    "calculateKGE": objectivefunctions.calculateKGE
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


def filter_two_DF_on_common_timesteps(DF1, DF2, saveToFile=None, column_name="TimeStamp"):
    # really important to check if there are any missing time steps compared to measured array
    # in other words measured and observed values have to be of the same length
    # TODO Should we make a copy of DF1 and DF1 or change the origins?
    df1_column_is_index_column = False
    df2_column_is_index_column = False
    if DF1.index.name == column_name:
        df1_column_is_index_column = True
        DF1 = DF1.reset_index()
        DF1.rename(columns={DF1.index.name: column_name}, inplace=True)
    if DF2.index.name == column_name:
        df2_column_is_index_column = True
        DF2 = DF2.reset_index()
        DF2.rename(columns={DF2.index.name: column_name}, inplace=True)

    list_of_columns_to_drop = [x for x in list(DF1.columns) if x != column_name]
    DF1_timeSteps = DF1.drop(list_of_columns_to_drop, axis=1, errors='ignore')
    list_of_columns_to_drop = [x for x in list(DF2.columns) if x != column_name]
    DF2_timeSteps = DF2.drop(list_of_columns_to_drop, axis=1, errors='ignore')

    diff_df = dataframe_difference(DF1_timeSteps, DF2_timeSteps, which=None)
    if saveToFile is not None:
        diff_df.to_pickle(saveToFile, compression="gzip")

    left_only = diff_df[diff_df['_merge'] == 'left_only']
    right_only = diff_df[diff_df['_merge'] == 'right_only']

    DF1 = DF1[~DF1[column_name].isin(left_only[column_name].values)]
    DF2 = DF2[~DF2[column_name].isin(right_only[column_name].values)]

    if df1_column_is_index_column:
        DF1.set_index(column_name, inplace=True)
    if df2_column_is_index_column:
        DF2.set_index(column_name, inplace=True)

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
    assert measuredDF_time_column_name == simulatedDF_time_column_name, "Assertion failed in utility.calculateGoodnessofFit_simple"
    predictedDF, measuredDF = filter_two_DF_on_common_timesteps(predictedDF, measuredDF, column_name=measuredDF_time_column_name)
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
    return uqsim_args


def _get_df_simulation_from_file(working_folder, df_all_simulations_name="df_all_simulations.pkl"):
    df_all_simulations = pathlib.Path(working_folder) / df_all_simulations_name
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

###################################################################################################################
# Plotting UQ & SA Output
###################################################################################################################

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