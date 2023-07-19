"""
Set of utility functions for postprocessing data for UQ runs of different models

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

# sys.path.insert(0, os.getcwd())
import sys
sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP')

from common import colors
#
from common import utility
from collections import defaultdict

from hydro_model import HydroStatistics

from larsim import LarsimStatistics

from linearDampedOscillator import LinearDampedOscillatorStatistics

from ishigami import IshigamiStatistics

from productFunction import ProductFunctionStatistics

from hbv_sask import HBVSASKStatisticsMultipleQoI as HBVSASKStatistics

def create_statistics_object(configuration_object, uqsim_args_dict, workingDir, model="hbvsask"):
    """

    :param configuration_object:
    :param uqsim_args_dict:
    :param workingDir:
    :param model: "larsim" | "hbvsask"
    :return:
    """
    if model == "larsim":
        statisticsObject = LarsimStatistics.LarsimStatistics(configuration_object, workingDir=workingDir,
                                                                   parallel_statistics=uqsim_args_dict[
                                                                       "parallel_statistics"],
                                                                   mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
                                                                   unordered=False,
                                                                   uq_method=uqsim_args_dict["uq_method"],
                                                                   compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
                                                                   compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"])
    elif model == "hbvsask":
        statisticsObject = HBVSASKStatistics.HBVSASKStatistics(
            configurationObject=configuration_object,
            workingDir=workingDir,
            sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
            parallel_statistics=uqsim_args_dict["parallel_statistics"],
            mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
            uq_method=uqsim_args_dict["uq_method"],
            compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
            compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"],
            inputModelDir=uqsim_args_dict["inputModelDir"]
        )
    else:
        statisticsObject = HydroStatistics.HydroStatistics(
            configurationObject=configuration_object,
            workingDir=workingDir,
            sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
            parallel_statistics=uqsim_args_dict["parallel_statistics"],
            mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
            uq_method=uqsim_args_dict["uq_method"],
            compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
            compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"],
            inputModelDir=uqsim_args_dict["inputModelDir"]
        )

    return statisticsObject


def extend_statistics_object(statisticsObject, statistics_dictionary, df_simulation_result,
                             get_measured_data=False, get_unaltered_data=False):
    statisticsObject.set_result_dict(statistics_dictionary)
    statisticsObject.set_timesteps(list(df_simulation_result.TimeStamp.unique()))
    statisticsObject.timesteps_min = df_simulation_result.TimeStamp.min()
    statisticsObject.timesteps_max = df_simulation_result.TimeStamp.max()

    timestepRange = (pd.Timestamp(statisticsObject.timesteps_min),
                     pd.Timestamp(statisticsObject.timesteps_max))

    statisticsObject.set_number_of_unique_index_runs(
        get_number_of_unique_runs(
            df_simulation_result, index_run_column_name="Index_run")
    )

    statisticsObject.set_numbTimesteps(len(statisticsObject.timesteps))

    # TODO Update statisticsObject.list_qoi_column based on columns inside df_simulation_result?
    # assert set(statisticsObject.list_qoi_column) == ...
    # assert set(statisticsObject.list_qoi_column) == ...

    statisticsObject._check_if_Sobol_t_computed()
    statisticsObject._check_if_Sobol_m_computed()

    if get_measured_data:
        statisticsObject.get_measured_data(
            timestepRange=timestepRange, transforme_mesured_data_as_original_model="False")

    if get_unaltered_data:
        statisticsObject.get_unaltered_run_data(timestepRange=timestepRange)


def get_number_of_unique_runs(df, index_run_column_name="Index_run"):
    return df[index_run_column_name].nunique()


def extracting_statistics_df_for_single_qoi(statisticsObject, qoi="Q_cms"):
    pass


def read_all_save_statistics_dict(workingDir, list_qoi_column):
    statistics_dictionary = defaultdict(dict)
    for single_qoi in list_qoi_column:
        statistics_dictionary_file_temp = workingDir / f"statistics_dictionary_qoi_{single_qoi}.pkl"
        assert statistics_dictionary_file_temp.is_file()
        with open(statistics_dictionary_file_temp, 'rb') as f:
            statistics_dictionary_temp = pickle.load(f)
        statistics_dictionary[single_qoi] = statistics_dictionary_temp
    return statistics_dictionary


def compute_gof_over_different_time_series(df_statistics,
                                           objective_function="MAE", qoi="Q", measuredDF_column_names=["measured"]):
    """
    This function will run only for a single station
    """
    if not isinstance(qoi, list):
        qoi = [qoi, ]

    if not isinstance(measuredDF_column_names, list):
        measuredDF_column_names = [measuredDF_column_names, ]

    if not isinstance(objective_function, list):
        objective_function = [objective_function, ]

    result_dict = defaultdict(dict)
    for idx, single_qoi in enumerate(qoi):
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

            if 'unaltered' in df_statistics_single_qoi.columns:
                gof_meas_unalt = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='unaltered')
            if 'E' in df_statistics_single_qoi.columns:
                gof_meas_mean = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='E')
            if 'E_minus_std' in df_statistics_single_qoi.columns:
                gof_meas_mean_m_std = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='E_minus_std')
            if 'E_plus_std' in df_statistics_single_qoi.columns:
                gof_meas_mean_p_std = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='E_plus_std')
            if 'P10' in df_statistics_single_qoi.columns:
                gof_meas_p10 = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='P10')
            if 'P90' in df_statistics_single_qoi.columns:
                gof_meas_p90 = single_objective_function(
                    df_statistics_single_qoi, df_statistics_single_qoi,
                    measuredDF_column_name=measuredDF_column_name, simulatedDF_column_name='P90')

            result_dict[single_qoi][single_objective_function]["gof_meas_unalt"] = gof_meas_unalt
            result_dict[single_qoi][single_objective_function]["gof_meas_mean"] = gof_meas_mean
            result_dict[single_qoi][single_objective_function]["gof_meas_mean_m_std"] = gof_meas_mean_m_std
            result_dict[single_qoi][single_objective_function]["gof_meas_mean_p_std"] = gof_meas_mean_p_std
            result_dict[single_qoi][single_objective_function]["gof_meas_p10"] = gof_meas_p10
            result_dict[single_qoi][single_objective_function]["gof_meas_p90"] = gof_meas_p90

            print(f"gof_meas_unalt:{gof_meas_unalt} \ngof_meas_mean:{gof_meas_mean} \n"
                  f"gof_meas_mean_m_std:{gof_meas_mean_m_std} \ngof_meas_mean_p_std:{gof_meas_mean_p_std} \n"
                  f"gof_meas_p10:{gof_meas_p10} \ngof_meas_p90:{gof_meas_p90} \n")
            

def redo_all_statistics(
        workingDir, get_measured_data=False, get_unaltered_data=False, station="MARI", uq_method="sc", plotting=False):
    raise NotImplementedError