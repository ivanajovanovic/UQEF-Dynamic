"""
Set of utility functions for postprocessing data for UQ runs of different models

@author: Ivana Jovanovic Buha
"""
import pandas as pd
import pickle

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


def read_all_saved_statistics_dict(workingDir, list_qoi_column):
    statistics_dictionary = defaultdict(dict)
    for single_qoi in list_qoi_column:
        statistics_dictionary_file_temp = workingDir / f"statistics_dictionary_qoi_{single_qoi}.pkl"
        assert statistics_dictionary_file_temp.is_file()
        with open(statistics_dictionary_file_temp, 'rb') as f:
            statistics_dictionary_temp = pickle.load(f)
        statistics_dictionary[single_qoi] = statistics_dictionary_temp
    return statistics_dictionary


def compute_gof_over_different_time_series(df_statistics,
                                           objective_function="MAE", qoi="Q", measuredDF_column_names="measured"):
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

            print(f"gof_means_unalt:{gof_means_unalt} \ngof_means_mean:{gof_means_mean} \n"
                  f"gof_means_mean_m_std:{gof_means_mean_m_std} \ngof_means_mean_p_std:{gof_means_mean_p_std} \n"
                  f"gof_means_p10:{gof_means_p10} \ngof_means_p90:{gof_means_p90} \n")
            

def redo_all_statistics(
        workingDir, get_measured_data=False, get_unaltered_data=False, station="MARI", uq_method="sc", plotting=False):
    raise NotImplementedError

###################################################################################################################
# Set of functions for plotting - these function may go to utility as well?
###################################################################################################################


def plot_heatmap_si_single_qoi(self, qoi_column, si_df, si_type="Sobol_t", uq_method="sc"):
    si_columns = [x for x in si_df.columns.tolist() if x != 'measured' and x != 'measured_norm']
    fig = px.imshow(si_df[si_columns].T, labels=dict(y='Parameters', x='Dates'))
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

    if qoi.lower() == "gof":
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
        if qoi.lower() == "gof":
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
        subplot_titles=subplot_titles
    )

    fig.add_trace(
        go.Bar(
            x=df['TimeStamp'], y=df['precipitation'],
            text=df['precipitation'],
            name="Precipitation"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['TimeStamp'], y=df['temperature'],
            text=df['temperature'],
            name="Temperature", mode='lines+markers'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['TimeStamp'], y=df['measured'],
            name="Observed Streamflow", mode='lines'
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