"""
Set of utility functions for postprocessing data of the UQ runs produced for the Ishigami model

Here is a set of functions and procedures that can be used to postprocess the data produced by the UQEF-Dynamic; and that are 
also presented in the Jupyter notebooks Ishigami_UQ_SA_UQEFDynamic.ipynb and Convergence_plots_ishigami.ipynb

@author: Ivana Jovanovic Buha
"""
import argparse
from collections import defaultdict
import dill
import pickle
import os
import numpy as np
import math
import pathlib
import pandas as pd
import pickle
import time

# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
import matplotlib.cm as cm
import seaborn as sns

import matplotlib.pyplot as plt
pd.options.plotting.backend = "plotly"

import sys

import chaospy as cp

# TODO Change this!!!
# sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')
sys.path.insert(0, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')

from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqef_dynamic_utils

from uqef_dynamic.models.ishigami import IshigamiModel
from uqef_dynamic.models.ishigami import IshigamiStatistics

# ============================================================================================
# Whole pipeline for reading the output saved by UQEF-Dynamic simulation and producing dict of interes
# ============================================================================================


if __name__ == "__main__":
    # ============================================================================================
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Example script for argument propagation.")
    
    # Add arguments to the parser
    parser.add_argument('--workingDir', type=str, help='workingDir path', required=True)
    
    # Parse the arguments
    args = parser.parse_args()

    workingDir = pathlib.Path(args.workingDir)

    #workingDir = pathlib.Path("/work/ga45met/ishigami_runs/simulations_sep_2024") / "sc_full_p7_q14"

    timestamp = 0.0
    qoi_column_name = qoi_string = "Value"

    ###################################
    a = 7
    b = 0.1
    # Creating the Ishigami Model (type of time-dependent model form the UQEF-Dynamic framework)
    ishigamiModelObject = IshigamiModel.IshigamiModel(configurationObject=None, a=a, b=b)

    x1 = cp.Uniform(-math.pi, math.pi)
    x2 = cp.Uniform(-math.pi, math.pi)
    x3 = cp.Uniform(-math.pi, math.pi)

    joint_isghigami = cp.J(x1, x2, x3)
    joint_isghigami_standard = cp.J(cp.Uniform(-1,1), cp.Uniform(-1,1), cp.Uniform(-1,1))

    joint_dist_standard = joint_isghigami_standard
    joint_dist = joint_isghigami

    ###################################
    # Examples of reading some saved Runs/Files and printing the dictionary with all the relevant information
    # Relevant for producing convergence graphs...
    dict_with_results_of_interest = uqef_dynamic_utils.read_all_saved_uqef_dynamic_results_and_produce_dict_of_interest_single_qoi_single_timestamp(
        workingDir=workingDir, 
        timestamp=timestamp, 
        qoi_column_name=qoi_column_name,
        plotting=False, 
        model=ishigamiModelObject,
        analytical_E=3.48227783540168,
        analytical_Var=13.887058470972093,
        analytical_Sobol_t=np.array([0.5574, 0.4424, 0.2436], dtype=np.float64),
        analytical_Sobol_m=np.array([0.3138, 0.4424, 0.0], dtype=np.float64),
        compare_surrogate_and_original_model=True
    )
    print(f"dict_with_results_of_interest - {dict_with_results_of_interest}")

    ###################################
    analytical_mean = 3.48227783540168 #None
    analytical_var = 13.887058470972093 #None
    if analytical_mean is None or analytical_var is None:
        numSamples = 100000
        rule = "R"
        analytical_mean, analytical_var = utility.compute_mc_quantity(
        model=IshigamiModel.ishigami_func, jointDists=joint_dist, numSamples=numSamples, rule=rule, 
        compute_mean=True, compute_var=True)
    print(f"Analytical mean - {analytical_mean}; Analytical var - {analytical_var}")

    ###################################
    Sobol_m_analytical = np.array([0.3138, 0.4424, 0.0], dtype=np.float64)
    Sobol_m_analytical_2 = np.array([0.3139, 0.4424, 0.0000], dtype=np.float64)

    Sobol_t_analytical = np.array([0.5574, 0.4424, 0.2436], dtype=np.float64)
    Sobol_t_analytical_2 = np.array([0.5576, 0.4424, 0.2437], dtype=np.float64)

    # Additional files being saved
    sobol_m_error_file = workingDir / "sobol_m_error.npy"
    sobol_m_qoi_file = workingDir / "sobol_m_qoi_file.npy"
    sobol_t_error_file = workingDir / "sobol_t_error.npy"
    sobol_t_qoi_file = workingDir / "sobol_t_qoi_file.npy"

    if sobol_m_qoi_file.is_file():
        Sobol_m = np.load(sobol_m_qoi_file)
        print(f"Sobol_m - {Sobol_m}")
        Sobol_m_error = abs(Sobol_m - Sobol_m_analytical)
        print(f"Sobol_m_error - {Sobol_m_error}")
        if sobol_m_error_file.is_file():
            Sobol_m_error_precomputed = abs(np.load(sobol_m_error_file))
            print(f"Sobol_m_error_precomputed - {Sobol_m_error_precomputed}")
    if sobol_t_qoi_file.is_file():
        Sobol_t = np.load(sobol_t_qoi_file)
        print(f"Sobol_t - {Sobol_t}")
        Sobol_t_error = abs(Sobol_t - Sobol_t_analytical)
        print(f"Sobol_t_error - {Sobol_t_error}")
        if sobol_t_error_file.is_file():
            Sobol_t_error_precomputed = abs(np.load(sobol_t_error_file))
            print(f"Sobol_t_error_precomputed - {Sobol_t_error_precomputed}")

    ###################################
    # Visualizing the Ishigami function
    ###################################
    x1_array = [0.0, math.pi/4, math.pi/2, math.pi]
    x2_array = [0.0, math.pi/4, math.pi/2, math.pi]
    x3_array = [0.0, math.pi/4, math.pi/2, math.pi]
    l = len(x3_array)
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.25))
    x = y = np.arange(-math.pi, math.pi, 0.05)
    X, Y = np.meshgrid(x, y)
    for i in range(l):
        ax = fig.add_subplot(1, l, i+1, projection='3d')
        zs = np.array(
            IshigamiModel.ishigami_func_vec((x1_array[i], np.ravel(X), np.ravel(Y)), a_model_param=7, b_model_param=0.1)
        )
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        ax.set_xlabel('X2')
        ax.set_ylabel('X3')
        ax.set_zlabel(f'Ishigami(x2,x3)')
        ax.set_title(f'x1={x1_array[i]}')

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.show()

    x = y = np.arange(-math.pi, math.pi, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array(
            IshigamiModel.ishigami_func_vec((np.ravel(X), np.ravel(Y), math.pi), a_model_param=7, b_model_param=0.1)
    )
    Z = zs.reshape(X.shape)
    Z_0 = np.zeros(X.shape)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.add_trace(go.Surface(z=Z_0, x=X, y=Y))
    fig.update_layout(scene = dict(
                        xaxis_title='x1',
                        yaxis_title='x2',
                        zaxis_title='ishigami(x1,x2)'),
                        width=650,
                        margin=dict(r=20, b=10, l=10, t=10))
    #fig.show()

    ###################################
    # (Simple) Running the Ishigami function for multiple samples
    ###################################

    coordinates = joint_isghigami.sample(10)
    print(f"Runnign the simple Ishigamin function in samples of size - {coordinates.shape}")
    results = ishigamiModelObject(i_s=range(coordinates.shape[1]), parameters=coordinates.T)
    print(f"Results - {results}")
    print(f"Checking - ishigamiModelObject.list_qoi_column - {ishigamiModelObject.list_qoi_column}")
    print(f"Checking - ishigamiModelObject.time_column_name - {ishigamiModelObject.time_column_name}")
    print(f"Checking - ishigamiModelObject.index_column_name - {ishigamiModelObject.index_column_name}")

    # processing the collected result
    df_simulation_result, df_index_parameter_values, _, _, _, _ =  uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe(results, 
    extract_only_qoi_columns=False, qoi_columns=ishigamiModelObject.list_qoi_column, 
    time_column_name= ishigamiModelObject.time_column_name, index_column_name= ishigamiModelObject.index_column_name)
    print(f"Final df_index_parameter_values - {df_index_parameter_values}")
    print(f"Final DF - {df_simulation_result}")

    ###################################
    # GPCE Surrogate of the Ishigami Model TODO add show-case how to compute statistics based on the gPCE surrogate
    ###################################

    sampleFromStandardDist = True
    gPCE_over_time, polynomial_expansion, norms, coeff = uqef_dynamic_utils.compute_PSP_for_uqef_dynamic_model(
    ishigamiModelObject, joint_dist, \
    quadrature_order=14, expansion_order=7, 
    sampleFromStandardDist=sampleFromStandardDist,
    joint_dist_standard=joint_dist_standard,
    rule_quadrature='g', \
    poly_rule='three_terms_recurrence', poly_normed=True, \
    qoi_column_name=ishigamiModelObject.qoi_column, 
    time_column_name=ishigamiModelObject.time_column_name, 
    index_column_name=ishigamiModelObject.index_column_name,
    return_dict_over_timestamps=True
    )

    gpce_surrogate = gPCE_over_time[timestamp]
    print(f"gpce_surrogate - {gpce_surrogate}")
    coeff = coeff[timestamp]
    print(f"coeff - {coeff}")

    # visualizing gPCE surrogate
    indices = gpce_surrogate.exponents
    dimensionality = indices.shape[1]
    number_of_terms = indices.shape[0]
    dict_for_plotting = {f"q_{i+1}":indices[:, i] for i in range(dimensionality)}
    df_nodes_weights = pd.DataFrame(dict_for_plotting)
    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(df_nodes_weights, vars = list(dict_for_plotting.keys()), corner=True)
    # plt.title(title, loc='left')
    #plt.show()

    # Generating new samples to compare the gpce-based surrogate and the original model
    numSamples = 1000 #5**dim #10**dim  # Note: Big Memory problem when more than 10**4 points?
    rule = 'r'

    new_set_of_parameters = joint_dist.sample(size=numSamples, rule=rule)
    new_set_of_nodes_transformed = utility.transformation_of_parameters(
        new_set_of_parameters, joint_dist, joint_dist_standard)

    gPCE_model_evaluated_original = gpce_surrogate(*new_set_of_nodes_transformed)

    original_model_evaluated = np.empty([new_set_of_parameters.shape[1],])
    i=0
    for single_sample in new_set_of_parameters.T:
        original_model_evaluated[i] = IshigamiModel.ishigami_func(
            single_sample, a_model_param=a, b_model_param=b)
        i+=1

    error_linf = np.max(np.abs(original_model_evaluated - gPCE_model_evaluated_original))
    error_l2_scaled = np.sqrt(np.sum((original_model_evaluated - gPCE_model_evaluated_original)**2)) / math.sqrt(numSamples)
    print(f"gPCE_model_evaluated_original - Linf Error = {error_linf};")
    print(f"gPCE_model_evaluated_original - L2 Error scaled = {error_l2_scaled}")

    ###################################
    # MC Analysis of the Ishigami Model
    ###################################

    read_nodes_from_file = False
    sampleFromStandardDist = True
    E_over_time, Var_over_time, StdDev_over_time, \
    Skew_over_time, Kurt_over_time, P10_over_time, P90_over_time, sobol_m_over_time = \
    uqef_dynamic_utils.run_uq_mc_sim_and_compute_mc_stat_for_uqef_dynamic_model(
        model=ishigamiModelObject,
        jointDists=joint_dist, jointStandard=joint_dist_standard, numSamples=1000, rule="R",
        sampleFromStandardDist=sampleFromStandardDist,
        read_nodes_from_file=False, 
        rounding=False, round_dec=4,
        qoi_column_name=ishigamiModelObject.qoi_column, 
        time_column_name=ishigamiModelObject.time_column_name, 
        index_column_name=ishigamiModelObject.index_column_name,
        return_dict_over_timestamps=False,
        compute_mean=True, compute_var=True, compute_std=True,
        compute_skew=True,
        compute_kurt=True,
        compute_p10=True,
        compute_p90=True,
        compute_Sobol_m=True,
    )
    print(f"MC - E_over_time - {E_over_time}; sobol_m_over_time - {sobol_m_over_time}")
