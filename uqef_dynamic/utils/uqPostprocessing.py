"""
Set of utility functions for postprocessing data for UQ runs of different models.
Many of these functions exist as well as part of time_dependent_statistics.TimeDependentStatistics or in utilty module
but here we are trying to provide a more general set of functions that can be used for postprocessing data from different UQ and SA runs


@author: Ivana Jovanovic Buha
"""
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import pickle

# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo

import matplotlib.pyplot as plt
pd.options.plotting.backend = "plotly"

import sys
sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')

import chaospy as cp

from uqef_dynamic.utils import colors
from uqef_dynamic.utils import utility


def extend_statistics_object(statisticsObject, statistics_dictionary, df_simulation_result=None,
                             time_stamp_column="TimeStamp", get_measured_data=False, get_unaltered_data=False):
    statisticsObject.set_result_dict(statistics_dictionary)
    # Update statisticsObject.list_qoi_column based on columns inside df_simulation_result
    statisticsObject.list_qoi_column = list(statistics_dictionary.keys())
    # assert set(statisticsObject.list_qoi_column) == ...
    # assert set(statisticsObject.list_qoi_column) == ...

    if df_simulation_result is not None:
        statisticsObject.set_timesteps(list(df_simulation_result[time_stamp_column].unique()))
        statisticsObject.set_timesteps_min(df_simulation_result[time_stamp_column].min())
        statisticsObject.set_timesteps_max(df_simulation_result[time_stamp_column].max())
        statisticsObject.set_number_of_unique_index_runs(
            get_number_of_unique_runs(
                df_simulation_result, index_run_column_name="Index_run")
        )
    else:
        # Read all this data from statistics_dictionary = statisticsObject.result_dict
        statisticsObject.set_timesteps()
        statisticsObject.set_timesteps_min()
        statisticsObject.set_timesteps_max()

    timestepRange = (pd.Timestamp(statisticsObject.timesteps_min),
                     pd.Timestamp(statisticsObject.timesteps_max))


    statisticsObject.set_numTimesteps()

    statisticsObject._check_if_Sobol_t_computed()
    statisticsObject._check_if_Sobol_m_computed()
    statisticsObject._check_if_Sobol_m2_computed()

    if get_measured_data:
        statisticsObject.get_measured_data(
            timestepRange=timestepRange, transforme_mesured_data_as_original_model="False")

    if get_unaltered_data:
        statisticsObject.get_unaltered_run_data(timestepRange=timestepRange)


def get_number_of_unique_runs(df, index_run_column_name="Index_run"):
    return df[index_run_column_name].nunique()


def extracting_statistics_df_for_single_qoi(statisticsObject, qoi="Q_cms"):
    pass


def get_all_timesteps_from_saved_files(workingDir, first_part_of_the_file = "statistics"):
    all_files = os.listdir(workingDir)
    list_TimeStamp = set() # []
    for filename in all_files:
        parts = filename.split('_')
        if parts[0] == first_part_of_the_file and parts[-1].endswith(".pkl"):
            single_timestep = parts[-1].split('.')[0]
            list_TimeStamp.add(single_timestep)  # pd.Timestamp(single_timestep)
    return list_TimeStamp

    
def read_all_saved_statistics_dict(workingDir, list_qoi_column, single_timestamp_single_file=False):
    if single_timestamp_single_file:
        list_TimeStamp = get_all_timesteps_from_saved_files(workingDir, first_part_of_the_file = "statistics")
    statistics_dictionary = defaultdict(dict)
    for single_qoi in list_qoi_column:
        if single_timestamp_single_file:
            statistics_dictionary[single_qoi] = dict()
            for single_timestep in list_TimeStamp:
                statistics_dictionary_file_temp = workingDir / f"statistics_dictionary_{single_qoi}_{single_timestep}.pkl"
                assert statistics_dictionary_file_temp.is_file(), \
                    f"The statistics file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist"
                with open(statistics_dictionary_file_temp, 'rb') as f:
                    statistics_dictionary_temp = pickle.load(f)
                statistics_dictionary[single_qoi][pd.Timestamp(single_timestep)] = statistics_dictionary_temp
        else:
            statistics_dictionary_file_temp = workingDir / f"statistics_dictionary_qoi_{single_qoi}.pkl"
            assert statistics_dictionary_file_temp.is_file(), \
                                f"The statistics file for qoi-{single_qoi} does not exist"
            with open(statistics_dictionary_file_temp, 'rb') as f:
                statistics_dictionary_temp = pickle.load(f)
            statistics_dictionary[single_qoi] = statistics_dictionary_temp
    return statistics_dictionary


def save_gpce_surrogate_model(workingDir, gpce, qoi, timestamp):
    # timestamp = pd.Timestamp(timestamp).strftime('%Y-%m-%d %X')
    fileName = f"gpce_surrogate_{qoi}_{timestamp}.pkl"
    fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
    with open(fullFileName, 'wb') as handle:
        pickle.dump(gpce, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def save_all_gpce_surrogate_model(workingDir, gpce_surrogate_dictionary, list_qoi_column=None, timestamps=None):
    if list_qoi_column is None:
        list_qoi_column = list(gpce_surrogate_dictionary.keys())
    if timestamps is None:
        timestamps = gpce_surrogate_dictionary[list_qoi_column[0]].keys()
    if not isinstance(list_qoi_column, list):
        list_qoi_column = [list_qoi_column, ]
    if not isinstance(timestamps, list):
        timestamps = [timestamps, ]
    for single_qoi in list_qoi_column:
        for single_timestep in timestamps:
            save_gpce_surrogate_model(workingDir, gpce_surrogate_dictionary[single_qoi][single_timestep], single_qoi, single_timestep)


def save_gpce_coeffs(workingDir, coeff, qoi, timestamp):
    fileName = f"gpce_coeffs_{qoi}_{timestamp}.npy"
    fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
    np.save(fullFileName, coeff)


def save_all_gpce_coeffs(workingDir, gpce_coeff_dictionary, list_qoi_column=None, timestamps=None):
    if list_qoi_column is None:
        list_qoi_column = list(gpce_coeff_dictionary.keys())
    if timestamps is None:
        timestamps = gpce_coeff_dictionary[list_qoi_column[0]].keys()
    if not isinstance(list_qoi_column, list):
        list_qoi_column = [list_qoi_column, ]
    if not isinstance(timestamps, list):
        timestamps = [timestamps, ]
    for single_qoi in list_qoi_column:
        for single_timestep in timestamps:
            save_gpce_coeffs(workingDir, gpce_coeff_dictionary[single_qoi][single_timestep], single_qoi, single_timestep)


def read_all_saved_gpce_surrogate_models(workingDir, list_qoi_column, single_timestamp_single_file=False):
    list_TimeStamp = get_all_timesteps_from_saved_files(workingDir, first_part_of_the_file = "gpce")
    gpce_surrogate_dictionary = defaultdict(dict)
    for single_qoi in list_qoi_column:
        gpce_surrogate_dictionary[single_qoi] = dict()
        for single_timestep in list_TimeStamp:
            gpce_surrogate_file_temp = workingDir / f"gpce_surrogate_{single_qoi}_{single_timestep}.pkl"
            assert gpce_surrogate_file_temp.is_file(), \
            f"The gpce surrogate file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist"
            with open(gpce_surrogate_file_temp, 'rb') as f:
                gpce_surrogate_temp = pickle.load(f)
            gpce_surrogate_dictionary[single_qoi][pd.Timestamp(single_timestep)] = gpce_surrogate_temp
    return gpce_surrogate_dictionary


def read_all_saved_gpce_coeffs(workingDir, list_qoi_column, single_timestamp_single_file=False):
    list_TimeStamp = get_all_timesteps_from_saved_files(workingDir, first_part_of_the_file = "gpce")
    gpce_coeff_dictionary = defaultdict(dict)
    for single_qoi in list_qoi_column:
        gpce_coeff_dictionary[single_qoi] = dict()
        for single_timestep in list_TimeStamp:
            gpce_coeffs_file_temp = workingDir / f"gpce_coeffs_{single_qoi}_{single_timestep}.npy"
            assert gpce_coeffs_file_temp.is_file(), \
            f"The gpce coefficients file for qoi-{single_qoi} and time-stamp-{single_timestep} does not exist"
            gpce_coeff_dictionary[single_qoi][pd.Timestamp(single_timestep)] = np.load(gpce_coeffs_file_temp)
    return gpce_coeff_dictionary

# ==============================================================================================================
# Functions for computing goodness-of-fit functions for different statistics time-signals produced by UQ and SA simulations
# ==============================================================================================================

def compute_gof_over_different_time_series(df_statistics, objective_function="MAE", qoi_column="Q",
                                           measuredDF_column_names="measured"):
    """
    This function will run only for a single qoi
    """
    if not isinstance(qoi_column, list):
        qoi_column = [qoi_column, ]

    if not isinstance(measuredDF_column_names, list):
        measuredDF_column_names = [measuredDF_column_names, ]

    if not isinstance(objective_function, list):
        objective_function = [objective_function, ]

    result_dict = defaultdict(dict)
    for idx, single_qoi in enumerate(qoi_column):
        result_dict[single_qoi] = defaultdict(dict)
        df_statistics_single_qoi = df_statistics.loc[df_statistics['qoi'] == single_qoi]

        measuredDF_column_name = measuredDF_column_names[idx]

        if measuredDF_column_name not in df_statistics_single_qoi.columns:
            continue

        for single_objective_function in objective_function:
            if not callable(single_objective_function) and single_objective_function in utility.mapping_gof_names_to_functions:
                single_objective_function = utility.mapping_gof_names_to_functions[single_objective_function]
            elif not callable(
                    single_objective_function) and single_objective_function not in utility.mapping_gof_names_to_functions \
                    or callable(single_objective_function) and single_objective_function not in utility._all_functions:
                raise ValueError("Not proper specification of Goodness of Fit function name")

            gof_means_unalt = None
            gof_means_mean = None
            gof_means_mean_m_std = None
            gof_means_mean_p_std = None
            gof_means_p10 = None
            gof_means_p90 = None
            if 'unaltered' in df_statistics_single_qoi.columns:
                gof_means_unalt = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='unaltered')
            if 'E' in df_statistics_single_qoi.columns:
                gof_means_mean = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='E')
            if 'E_minus_std' in df_statistics_single_qoi.columns:
                gof_means_mean_m_std = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='E_minus_std')
            if 'E_plus_std' in df_statistics_single_qoi.columns:
                gof_means_mean_p_std = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='E_plus_std')
            if 'P10' in df_statistics_single_qoi.columns:
                gof_means_p10 = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='P10')
            if 'P90' in df_statistics_single_qoi.columns:
                gof_means_p90 = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='P90')

            result_dict[single_qoi][single_objective_function]["gof_means_unalt"] = gof_means_unalt
            result_dict[single_qoi][single_objective_function]["gof_means_mean"] = gof_means_mean
            result_dict[single_qoi][single_objective_function]["gof_means_mean_m_std"] = gof_means_mean_m_std
            result_dict[single_qoi][single_objective_function]["gof_means_mean_p_std"] = gof_means_mean_p_std
            result_dict[single_qoi][single_objective_function]["gof_means_p10"] = gof_means_p10
            result_dict[single_qoi][single_objective_function]["gof_means_p90"] = gof_means_p90

            print(f"{single_objective_function} - Comparing measured time series and different computed time series"
                  f"\n gof_means_unalt:{gof_means_unalt} \ngof_means_mean:{gof_means_mean} \n"
                  f"gof_means_mean_m_std:{gof_means_mean_m_std} \ngof_means_mean_p_std:{gof_means_mean_p_std} \n"
                  f"gof_means_p10:{gof_means_p10} \ngof_means_p90:{gof_means_p90} \n")
            

def redo_all_statistics(
        workingDir, get_measured_data=False, get_unaltered_data=False, station="MARI", uq_method="sc", plotting=False):
    raise NotImplementedError

###################################################################################################################
# Set of different functions for analyzing df_statistics DataFrame
    # produced as part of UQ simulation - these function may go to utility as well?
###################################################################################################################


def describe_df_statistics(df_statistics_and_measured, single_qoi=None):
    if single_qoi is None:
        result = df_statistics_and_measured.describe(include=np.number)
        print(result)
    else:
        df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured.loc[
            df_statistics_and_measured['qoi'] == single_qoi]
        result = df_statistics_and_measured_single_qoi_subset.describe(include=np.number)
        print(f"{single_qoi}\n\n")
        print(result)


def _get_sensitivity_indices_subset_of_big_df_statistics(df_statistics, single_qoi, param_names,
        si_type="Sobol_t"):
    df_statistics_and_measured_single_qoi = df_statistics.loc[df_statistics['qoi'] == single_qoi]
    # TODO allow user to specify which column names
    list_of_columns_to_keep = ["measured", "precipitation", "temperature"]
    if not isinstance(param_names, list):
        param_names = [param_names,]
    for param_name in param_names:
        if si_type == "Sobol_t":
            list_of_columns_to_keep.append(f"Sobol_t_{param_name}")
        elif si_type == "Sobol_m":
            list_of_columns_to_keep.append(f"Sobol_m_{param_name}")
        elif si_type == "Sobol_m2":
            list_of_columns_to_keep.append(f"Sobol_m2_{param_name}")
    df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured_single_qoi[list_of_columns_to_keep]
    return df_statistics_and_measured_single_qoi_subset


def describe_sensitivity_indices_single_qoi_under_some_condition(
        df_statistics, single_qoi, param_names,
        si_type="Sobol_t", condition_columns=None,condition_value=None, condition_sign="equal"):
    df_statistics_and_measured_single_qoi_subset = _get_sensitivity_indices_subset_of_big_df_statistics(
        df_statistics, single_qoi, param_names, si_type)
    if condition_columns is None or condition_value is None:
        result_describe = df_statistics_and_measured_single_qoi_subset.describe(include=np.number)
        print(f"{result_describe}")
    else:
        if condition_sign=="equal":
            df_subset = df_statistics_and_measured_single_qoi_subset[
                df_statistics_and_measured_single_qoi_subset[condition_columns] == condition_value].copy()
            result_describe = df_subset.describe(include=np.number)
            print(f"{result_describe}")
        elif condition_sign=="greater than":
            df_subset = df_statistics_and_measured_single_qoi_subset[
                df_statistics_and_measured_single_qoi_subset[condition_columns] > condition_value].copy()
            result_describe = df_subset.describe(include=np.number)
            print(f"{result_describe}")
        elif condition_sign=="greater than or equal":
            df_subset = df_statistics_and_measured_single_qoi_subset[
                df_statistics_and_measured_single_qoi_subset[condition_columns] >= condition_value].copy()
            result_describe = df_subset.describe(include=np.number)
            print(f"{result_describe}")
        elif condition_sign=="less than":
            df_subset = df_statistics_and_measured_single_qoi_subset[
                df_statistics_and_measured_single_qoi_subset[condition_columns] < condition_value].copy()
            result_describe = df_subset.describe(include=np.number)
            print(f"{result_describe}")
        elif condition_sign=="less than or equal":
            df_subset = df_statistics_and_measured_single_qoi_subset[
                df_statistics_and_measured_single_qoi_subset[condition_columns] <= condition_value].copy()
            result_describe = df_subset.describe(include=np.number)
            print(f"{result_describe}")
        else:
            raise Exception(f"Error in describe_sensitivity_indices_single_qoi_under_some_condition "
                            f"method - condition_sign should be one of the following strings: equal"
                            f"/greater than/greater than or equal/less than/less than or equal")


def compute_df_statistics_columns_correlation(
        df_statistics, single_qoi, param_names,
        only_sensitivity_indices_columns=True, si_type="Sobol_t", plot=True):
    if only_sensitivity_indices_columns:
        df_statistics_and_measured_single_qoi_subset = _get_sensitivity_indices_subset_of_big_df_statistics(
            df_statistics, single_qoi, param_names, si_type)
    else:
        df_statistics_and_measured_single_qoi = df_statistics.loc[df_statistics['qoi'] == single_qoi]
        df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured_single_qoi

    corr_df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured_single_qoi_subset.corr()
    print(f"Correlation matrix: {corr_df_statistics_and_measured_single_qoi_subset} \n")
    if plot:
        import seaborn as sns
        sns.set(style="darkgrid")
        mask = np.triu(np.ones_like(corr_df_statistics_and_measured_single_qoi_subset, dtype=bool))
        fig, axs = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr_df_statistics_and_measured_single_qoi_subset, mask=mask, square=True, annot=True,
                    linewidths=.5)
        plt.show()


###################################################################################################################
# Set of functions for plotting and or computing (KDE-based) CDFs and PDFs - these function may go to utility as well?
###################################################################################################################

def plot_parameters_sensitivity_indices_vs_temp_prec_measured(
        df_statistics, single_qoi, param_names, si_type="Sobol_t"):
    """
    :param df_statistics:
    :param single_qoi:
    :param param_names:
    :param si_type:
    :return:
    """
    df_statistics_and_measured_single_qoi_subset = _get_sensitivity_indices_subset_of_big_df_statistics(
        df_statistics, single_qoi, param_names, si_type)
    num_param = len(param_names)
    fig, axs = plt.subplots(3, num_param, figsize=(15, 8))
    # param = configurationObject["parameters"][0]["name"]
    # param_num = 0
    param_counter = 0
    for i, ax in enumerate(axs.flat):
        param_num = param_counter % num_param
        param = param_names[param_num]
        if i % 3 == 0:
            if si_type == "Sobol_t":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_t_{param}", "measured"]].values.T)
                # ax.scatter(*df_statistics_and_measured_single_qoi_subset[["measured", f"sobol_t_{param}", ]].values.T)
            elif si_type == "Sobol_m":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_m_{param}", "measured"]].values.T)
            elif si_type == "Sobol_m2":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_m2_{param}", "measured"]].values.T)
        elif i % 3 == 1:
            if si_type == "Sobol_t":
                ax.scatter(
                    *df_statistics_and_measured_single_qoi_subset[[f"Sobol_t_{param}", "precipitation"]].values.T)
            elif si_type == "Sobol_m":
                ax.scatter(
                    *df_statistics_and_measured_single_qoi_subset[[f"Sobol_m_{param}", "precipitation"]].values.T)
            elif si_type == "Sobol_m2":
                ax.scatter(
                    *df_statistics_and_measured_single_qoi_subset[[f"Sobol_m2_{param}", "precipitation"]].values.T)
        elif i % 3 == 2:
            if si_type == "Sobol_t":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_t_{param}", "temperature"]].values.T)
            elif si_type == "Sobol_m":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_m_{param}", "temperature"]].values.T)
            elif si_type == "Sobol_m2":
                ax.scatter(*df_statistics_and_measured_single_qoi_subset[[f"Sobol_m2_{param}", "temperature"]].values.T)
        param_counter += 1
    #     ax.set_title(f'{param}')
    # set labels
    for i in range(num_param):
        param = param_names[i]
        plt.setp(axs[-1, i], xlabel=f'{si_type} {param}')
        # plt.setp(axs[:, i], ylabel=f'{param}')
    plt.setp(axs[0, 0], ylabel='measured')
    plt.setp(axs[1, 0], ylabel='precipitation')
    plt.setp(axs[2, 0], ylabel='temperature')
    # plt.setp(axs[0, :], xlabel='measured')
    # plt.setp(axs[1, :], xlabel='precipitation')
    # plt.setp(axs[2, :], xlabel='temperature')


def plot_cdfs_of_parameters_sensitivity_indices(df_statistics, single_qoi, param_names, si_type="Sobol_t"):
    df_statistics_and_measured_single_qoi_subset = _get_sensitivity_indices_subset_of_big_df_statistics(
        df_statistics, single_qoi, param_names, si_type)
    num_param = len(param_names)

    t = np.linspace(0, 1.0, 1000)
    fig, axs = plt.subplots(1, num_param, figsize=(20, 10))
    for i in range(num_param):
        param_name = param_names[i]
        if si_type == "Sobol_t":
            param_eval = df_statistics_and_measured_single_qoi_subset[f"Sobol_t_{param_name}"].values
        elif si_type == "Sobol_m":
            param_eval = df_statistics_and_measured_single_qoi_subset[f"Sobol_m_{param_name}"].values
        elif si_type == "Sobol_m2":
            param_eval = df_statistics_and_measured_single_qoi_subset[f"Sobol_m2_{param_name}"].values
        distribution = cp.GaussianKDE(param_eval, h_mat=0.005 ** 2)
        axs[i,].plot(t, distribution.cdf(t), label=f"KDE CDF {si_type} {param_name}")
        plt.setp(axs[i,], xlabel=f'{si_type}_{param_name}')
        axs[i,].grid()
    # plt.legend()
    plt.setp(axs[0], ylabel='CDF')
    plt.show()


def single_param_single_qoi_sensitivity_indices_GaussianKDE(
        df_statistics, single_qoi, param_name, si_type="Sobol_t", plot=True, plot_pdf_or_cdf="pdf",
        h_mat=0.02, alpha=0.5):
    df_statistics_and_measured_single_qoi_subset = _get_sensitivity_indices_subset_of_big_df_statistics(
        df_statistics, single_qoi, param_name, si_type)
    column_to_keep = None
    if si_type == "Sobol_t":
        column_to_keep = f"Sobol_t_{param_name}"
    elif si_type == "Sobol_m":
        column_to_keep = f"Sobol_m_{param_name}"
    elif si_type == "Sobol_m2":
        column_to_keep = f"Sobol_m2_{param_name}"
    samples = df_statistics_and_measured_single_qoi_subset[column_to_keep].values

    t = np.linspace(0, 1.0, 1000)
    distribution = cp.GaussianKDE(samples, h_mat=h_mat ** 2)

    if plot:
        if plot_pdf_or_cdf == "pdf":
            plt.hist(samples, bins=100, density=True, alpha=alpha)
            plt.plot(t, distribution.pdf(t), label=f"PDF - {si_type} {single_qoi} {param_name}")
        elif plot_pdf_or_cdf == "cdf":
            plt.plot(t, distribution.cdf(t), label=f"CDF - {si_type} {single_qoi} {param_name}")
        else:
            raise Exception()
        plt.legend()
        plt.show()

    return distribution


def gof_values_GaussianKDE(df_index_parameter_gof, gof_list=None, plot=True, plot_pdf_or_cdf="pdf"):
    result_dic_of_distributions = defaultdict()
    if gof_list is None:
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)

    if plot:
        fig, axs = plt.subplots(1, len(gof_list), figsize=(20, 10))

    for i in range(len(gof_list)):
        single_gof = gof_list[i]
        min_single_gof = df_index_parameter_gof[single_gof].min() - abs(df_index_parameter_gof[single_gof].min())*0.001
        max_single_gof = df_index_parameter_gof[single_gof].max() + abs(df_index_parameter_gof[single_gof].max())*0.001
        t = np.linspace(min_single_gof, max_single_gof, 1000)
        gof_eval = df_index_parameter_gof[single_gof].values
        distribution = cp.GaussianKDE(gof_eval, h_mat=0.005 ** 2)
        result_dic_of_distributions[single_gof] = distribution
        if plot:
            if plot_pdf_or_cdf == "pdf":
                axs[i,].hist(gof_eval, bins=100, density=True, alpha=0.5)
                axs[i,].plot(t, distribution.pdf(t), label=f"KDE PDF {single_gof} {gof_list}")
            elif plot_pdf_or_cdf == "cdf":
                axs[i,].plot(t, distribution.cdf(t), label=f"KDE CDF {single_gof} {gof_list}")
            else:
                raise Exception()
            plt.setp(axs[i,], xlabel=f'{single_gof}')
            axs[i,].grid()
    if plot:
        plt.setp(axs[0], ylabel=f'{plot_pdf_or_cdf}')
        plt.show()

    return result_dic_of_distributions

###################################################################################################################
# Set of functions for plotting - these function may go to utility as well?
###################################################################################################################


def plot_heatmap_si_single_qoi(qoi_column, si_df, si_type="Sobol_t", uq_method="sc", time_column_name="TimeStamp"):
    reset_index_at_the_end = False
    if si_df.index.name != time_column_name:
        si_df.set_index(time_column_name, inplace=True)
        reset_index_at_the_end = True

    si_columns_to_plot = [x for x in si_df.columns.tolist() if x != 'measured' and x != 'measured_norm' and x != 'qoi']
    # fig = px.imshow(si_df[si_columns_to_plot].T, labels=dict(y='Parameters', x='Dates'))

    if 'qoi' in si_df.columns.tolist():
        fig = px.imshow(si_df.loc[si_df['qoi'] == qoi_column][si_columns_to_plot].T,
                        labels=dict(y='Parameters', x='Dates'))
    else:
        fig = px.imshow(si_df[si_columns_to_plot].T,
                        labels=dict(y='Parameters', x='Dates'))

    if reset_index_at_the_end:
        si_df.reset_index(inplace=True)
        si_df.rename(columns={si_df.index.name: time_column_name}, inplace=True)

    return fig


def plot_si_and_normalized_measured_time_signal_single_qoi(
        qoi_column, si_df, si_type="Sobol_t",
        observed_column_normalized="measured_norm", uq_method="sc", plot_forcing_data=False):

    si_columns = [x for x in si_df.columns.tolist() if x != 'measured' and x != 'measured_norm']

    fig = px.line(si_df, x=si_df.index, y=si_columns)

    if observed_column_normalized in si_df.columns.tolist() and not si_df[observed_column_normalized].isnull().all():
        fig.add_trace(go.Scatter(x=si_df.index,
                                 y=si_df[observed_column_normalized],
                                 fill='tozeroy', name=f"Normalized {qoi_column}"))
    # if plot_forcing_data:
    #     pass
    return fig


def plot_forcing_mean_predicted_and_observed_all_qoi(statisticsObject, directory="./", fileName="simulation_big_plot.html"):
    measured_column_names_set = set()
    for single_qoi in statisticsObject.list_qoi_column:
        measured_column_names_set.add(statisticsObject.dict_corresponding_original_qoi_column[single_qoi])

    total_number_of_rows = 2 + len(statisticsObject.list_qoi_column) + len(measured_column_names_set)
    fig = make_subplots(
        rows=total_number_of_rows, cols=1, shared_xaxes=True
        #     subplot_titles=tuple(statisticsObject.list_qoi_column)
    )
    n_row = 3

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

    if statisticsObject.df_statistics is None or statisticsObject.df_statistics.empty:
        raise Exception(f"You are trying to call a plotting utiltiy function whereas "
                        f"statisticsObject.df_statistics object is still not computed - make sure to first call"
                        f"statisticsObject.create_df_from_statistics_data")

    measured_column_names_set = set()
    for single_qoi in statisticsObject.list_qoi_column:
        df_statistics_single_qoi = statisticsObject.df_statistics.loc[
            statisticsObject.df_statistics['qoi'] == single_qoi]
        corresponding_measured_column = statisticsObject.dict_corresponding_original_qoi_column[single_qoi]
        if corresponding_measured_column not in measured_column_names_set:
            measured_column_names_set.add(corresponding_measured_column)
            fig.add_trace(
                go.Scatter(
                    x=df_statistics_single_qoi['TimeStamp'],
                    y=df_statistics_single_qoi['measured'],
                    name=f"Observed {corresponding_measured_column}", mode='lines'
                ),
                row=n_row, col=1
            )
            n_row += 1

        fig.add_trace(
            go.Scatter(
                x=df_statistics_single_qoi['TimeStamp'],
                y=df_statistics_single_qoi['E'],
                text=df_statistics_single_qoi['E'],
                name=f"Mean predicted {single_qoi}", mode='lines'),
            row=n_row, col=1
        )
        n_row += 1

    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_layout(height=600, width=800, title_text="Detailed plot of most important time-series")
    fig.update_layout(xaxis=dict(type="date"))
    if not str(directory).endswith("/"):
        directory = str(directory) + "/"
    if fileName is None:
        fileName = "datailed_plot_all_qois.html"
    fileName = directory + fileName
    pyo.plot(fig, filename=fileName)
    return fig, n_row


def plotting_function_single_qoi(
        df, single_qoi, qoi="Q", subplot_titles=None, dict_what_to_plot=None, directory="./", fileName="simulation_big_plot.html"
):
    """
    At the moment this is tailored for HBV model
    :param df:
    :return:
    """
    n_rows = 3
    starting_row_for_predicted_data = 3

    if dict_what_to_plot is None:
        dict_what_to_plot = {
            "E_minus_std": False, "E_plus_std": False, "P10": False, "P90": False,
            "StdDev": False, "Skew": False, "Kurt": False
        }

    if not isinstance(qoi, list) and qoi.lower() == "gof":
        n_rows += 1
        starting_row_for_predicted_data = 4
        # subplot_titles = ("Precipitation", "Temperature", "Measured Streamflow", "Mean", "Skew", "Kurt")

    if dict_what_to_plot.get("StdDev", False):
        n_rows += 1
    if dict_what_to_plot.get("Skew", False):
        n_rows += 1
    if dict_what_to_plot.get("Kurt", False):
        n_rows += 1

    if subplot_titles is None:
        subplot_titles = ("Precipitation", "Temperature")
        if not isinstance(qoi, list) and qoi.lower() == "gof":
            subplot_titles = subplot_titles + ("Measured Streamflow",)
            subplot_titles = subplot_titles + ("Predicted",)
        else:
            subplot_titles= subplot_titles + ("Measured and Predicted Data",)
        if dict_what_to_plot.get("StdDev", False):
            subplot_titles = subplot_titles + ("Standard Deviation",)
        if dict_what_to_plot.get("Skew", False):
            subplot_titles = subplot_titles + ("Skew",)
        if dict_what_to_plot.get("Kurt", False):
            subplot_titles = subplot_titles + ("Kurt",)

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
    )

    fig.add_trace(
        go.Bar(
            x=df['TimeStamp'], y=df['precipitation'],
            text=df['precipitation'],
            name="Precipitation",
            # marker_color='red'
            # mode="lines",
            #         line=dict(
            #             color='LightSkyBlue')
        ),
        row=1, col=1
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=df['TimeStamp'], y=df['temperature'],
            text=df['temperature'],
            name="Temperature", mode='lines+markers',
            # marker_color='blue'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['TimeStamp'], y=df['measured'],
            name="Observed Streamflow", mode='lines',
            # line=dict(color='green'),
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['TimeStamp'], y=df['E'],
            text=df['E'],
            name=f"Mean predicted {single_qoi}", mode='lines'
        ),
        row=starting_row_for_predicted_data, col=1
    )

    if dict_what_to_plot.get("E_minus_std", False):
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E_minus_std'],
                name=f'E_minus_std',
                text=df['E_minus_std'], mode='lines', line_color="grey",
            ),
            row=starting_row_for_predicted_data, col=1
        )
    if dict_what_to_plot.get("E_plus_std", False):
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E_plus_std'],
                name=f'E_plus_std',
                text=df['E_plus_std'], line_color="grey",
                mode='lines', fill='tonexty'
            ),
            row=starting_row_for_predicted_data, col=1
        )
    if dict_what_to_plot.get("P10", False):
        fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["P10"],
                                 name=f'P10',
                                 line_color='yellow', mode='lines'),
                      row=starting_row_for_predicted_data, col=1)
    if dict_what_to_plot.get("P90", False):
        fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["P90"],
                                 name=f'P90',
                                 line_color='yellow', mode='lines', fill='tonexty'),
                      row=starting_row_for_predicted_data, col=1)

    starting_row_for_predicted_data += 1

    if dict_what_to_plot.get("StdDev", False):
        fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["StdDev"],
                                 name=f'std. dev of {single_qoi}', line_color='darkviolet', mode='lines'),
                      row=starting_row_for_predicted_data, col=1)
        starting_row_for_predicted_data += 1

    if dict_what_to_plot.get("Skew", False):
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['Skew'],
                text=df['Skew'], name=f"Skewness of {single_qoi}", mode='markers'
            ),
            row=starting_row_for_predicted_data, col=1
        )
        starting_row_for_predicted_data += 1

    if dict_what_to_plot.get("Kurt", False):
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['Kurt'],
                text=df['Kurt'], name=f"Kurtosis of {single_qoi}", mode='markers'
            ),
            row=starting_row_for_predicted_data, col=1
        )
        starting_row_for_predicted_data += 1

    fig.update_layout(height=600, width=800,
                      title_text=f"Detailed plot of most important time-series - QoI {single_qoi}")
    fig.update_layout(xaxis=dict(type="date"))

    if not str(directory).endswith("/"):
        directory = str(directory) + "/"
    fileName = directory + fileName
    pyo.plot(fig, filename=fileName, auto_open=False)
    return fig