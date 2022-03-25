import pathlib
import pickle
import numpy as np
import sys
import math
from tabulate import tabulate

# plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

import pandas as pd

import chaospy as cp
import uqef
import sparseSpACE

#####################################
dist_1d = cp.Uniform(0, 1)

dist_2d = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1))

# GQ points order vs number of points in 3D - chaospy
dist_3d = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1))

# GQ points order vs number of points in 4D - chaospy
dist_4d = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1))

# GQ points order vs number of points in 5D - chaospy
dist_5d = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1))

d6_uniform = [cp.Uniform(0, 1)] * 6

dist_6d = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1),
               cp.Uniform(0, 1))

dists = [dist_3d, dist_4d, dist_5d]
dim = ["3d", "4d", "5d"]

x1 = cp.Uniform(-math.pi, math.pi)
x2 = cp.Uniform(-math.pi, math.pi)
x3 = cp.Uniform(-math.pi, math.pi)
joint = cp.J(x1, x2, x3)
joint_standard = cp.J(cp.Uniform(), cp.Uniform(), cp.Uniform())
joint_standard_min_1_1 = cp.J(cp.Uniform(lower=-1, upper=1), cp.Uniform(lower=-1, upper=1), cp.Uniform(lower=-1, upper=1))
samples = joint.sample(100)

#####################################
# Set of Utility Functions
# originally in he notebook - SG_SparseSpace/Experimenting_with_Sparse_Quadrature_Points.ipynb
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
    print(tabulate(table, headers=["rule", "dim", "q", "#nodes sparse_utility tensor", "#nodes full tensor"], numalign="right"))


def generate_table_single_rule_over_dim_and_orders(rule, dists, dim, q_orders, sparse=True, growth=None):
    num_nodes = np.zeros((len(dists), len(q_orders), 1), dtype=np.int32)

    table_column_name = "#nodes sparse_utility tensor" if sparse else "#nodes full tensor"

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
                   headers=["rule", "q", "#nodes full tensor", "#nodes sparse_utility", "#nodes sparse_utility < #nodes full tensor"],
                   numalign="right"))


def generate_df_with_nodes_and_weights(l, file_path, type="KPU"):
    nodes_and_weights_array = np.loadtxt(file_path, delimiter=',')
    numDim = nodes_and_weights_array.shape[1] - 1
    numSamples = nodes_and_weights_array.shape[0]

    my_ditc = {f"x{i}": nodes_and_weights_array[:, i] for i in range(numDim)}
    my_ditc["w"] = nodes_and_weights_array[:, numDim]
    df_nodes_and_weights = pd.DataFrame(my_ditc)
    return df_nodes_and_weights


def get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes"):
    """
    simulationNodes: UQEF.Nodes object
    """
    numDim = simulationNodes.nodes.shape[0]
    numSamples = simulationNodes.nodes.shape[1]
    if nodes_or_paramters=="nodes":
        my_ditc = {f"x{i}": simulationNodes.nodes[i, :] for i in range(numDim)}
    else:
        my_ditc = {f"x{i}": simulationNodes.parameters[i, :] for i in range(numDim)}
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


def plot_2d_matrix_static_from_list(simulationNodes_list, title="Plot nodes"):
    dfsimulationNodes = get_df_from_simulationNodes_list(simulationNodes_list)

    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(dfsimulationNodes, vars=list(dfsimulationNodes.columns), corner=True)
    plt.title(title, loc='left')
    plt.show()


def plot_points_for_level(l, file_path, type_of_sg="KPU"):
    nodes_and_weights_array = np.loadtxt(file_path, delimiter=',')
    numDim = nodes_and_weights_array.shape[1] - 1
    numSamples = nodes_and_weights_array.shape[0]

    my_ditc = {f"x{i}": nodes_and_weights_array[:, i] for i in range(numDim)}
    df_nodes = pd.DataFrame(my_ditc)

    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(df_nodes, vars=list(my_ditc.keys()), corner=True)
    plt.title(f"{type_of_sg} - d={numDim}; L={l}; num={numSamples}")
    plt.show()


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
            title = f"{rule} points chaospy; order = {order}; sparse_utility={str(sparse)}; #nodes={abscissas.shape[0]}; growth=True"
        else:
            title = f"{rule} points chaospy; order = {order}; sparse_utility={str(sparse)}; #nodes={abscissas.shape[0]}"
        plt.title(title, loc='left')
        plt.show()


def plot_2d_matrix_of_nodes_over_orders_with_weights(rule, dist, orders, sparse=False, growth=None):
    if growth is None:
        growth = True if rule == "c" else False

    for order in orders:
        abscissas, weights = cp.generate_quadrature(order, dist, rule=rule, growth=growth, sparse=sparse)

        #         print(order, abscissas.round(3), weights.round(3))
        #         print(abscissas.shape)
        #         print(weights.shape)

        dimensionality = len(abscissas)

        abscissas = abscissas.T

        dict_for_plotting = {f"x{i}": abscissas[:, i] for i in range(dimensionality)}
        vars = list(dict_for_plotting.keys())
        dict_for_plotting["weights"] = weights[:]

        #         idx = weights > 0
        #         pyplot.scatter(*abscissas[idx, :], s=weights[idx]*2e3)
        #         pyplot.scatter(*abscissas[~idx, :], s=-weights[~idx]*2e3, color="grey")

        df_nodes_weights = pd.DataFrame(dict_for_plotting)

        sns.set(style="ticks", color_codes=True)

        g = sns.pairplot(data=df_nodes_weights, vars=vars, corner=True, diag_kws={"fill": True},
                         plot_kws={"s": abs(weights) * 2e3})

        if growth:
            title = f"{rule} points chaospy; order = {order}; sparse_utility={str(sparse)}; #nodes={abscissas.shape[0]}; growth=True"
        else:
            title = f"{rule} points chaospy; order = {order}; sparse_utility={str(sparse)}; #nodes={abscissas.shape[0]}"
        plt.title(title, loc='left')
        plt.show()


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
