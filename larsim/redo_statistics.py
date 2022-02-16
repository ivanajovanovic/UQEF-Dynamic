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

import sys
# sys.path.insert(1, '/work/ga45met/mnt/linux_cluster_2/UQEFPP')
sys.path.insert(1, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP')

from common import saltelliSobolIndicesHelpingFunctions
from common import utility

# from larsim import LarsimModel
from larsim import LarsimStatistics
from larsim import LarsimUQPostprocessing

###################################################################################################################
# Reading saved data
###################################################################################################################

MAPPING_STATION_TO_TGB = {"MARI": 3085}

def redo_all_statistics(
        workingDir, get_measured_data=False, get_unaltered_data=False, station="MARI", uq_method="sc", plotting=False
):
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
    samples_df_index_parameter_gof = pd.read_pickle(uq_output_paths_obj.df_all_index_parameter_gof_file,
                                                    compression="gzip")
    params_list = LarsimUQPostprocessing._get_parameter_columns_df_index_parameter_gof(
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
        larsimStatisticsObject = LarsimUQPostprocessing.create_larsimStatistics_object(
            configuration_object, uqsim_args_dict, uq_output_paths_obj.model_runs_folder)

        LarsimUQPostprocessing.extend_larsimStatistics_object(
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
            fig = LarsimUQPostprocessing._add_precipitation_to_graph(
                fig, uq_output_paths_obj.master_configuration_folder, larsimStatisticsObject)
            fig.show()

            fig = larsimStatisticsObject.plot_si_and_normalized_measured_time_signal(si_df=si_t_df, station=station)
            fig.update_yaxes(title_text='Total Order Sobol Indices')
            fig = LarsimUQPostprocessing._add_precipitation_to_graph(
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


if __name__ == "__main__":
    workingDir = pathlib.Path("/gpfs/scratch/pr63so/ga45met2/Larsim_runs/larsim_uq_cm2.0088")
    # workingDir = pathlib.Path("/work/ga45met/mnt/linux_cluster_scratch/larsim_uq_cm2.0088")

    gPCE_model_evaluated_at_calib_sample_list = redo_all_statistics(
        workingDir,
        get_measured_data=True,
        get_unaltered_data=True,
        station="MARI",
        uq_method="sc",
        plotting=False
    )
    gPCE_model_file = workingDir / "gpce_model_calib.pkl"
    with open(gPCE_model_file, 'wb') as handle:
        pickle.dump(gPCE_model_evaluated_at_calib_sample_list, handle, protocol=pickle.HIGHEST_PROTOCOL)