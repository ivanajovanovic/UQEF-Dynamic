import chaospy as cp
from collections import defaultdict
from distutils.util import strtobool
from functools import reduce
import json
import itertools
import matplotlib.pyplot as plotter
import more_itertools
from mpi4py import MPI
import mpi4py.futures as futures
import numpy as np
import os
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import pathlib
import pickle
from plotly.offline import iplot, plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import sys
import time

# from uqef.stat import Statistics
#
# from common import saltelliSobolIndicesHelpingFunctions
# from common import parallelStatistics
from common import colors
#
from common import utility
from hydro_model import HydroStatistics
from hbv_sask import hbvsask_utility as hbv

class HBVSASKStatistics(HydroStatistics.HydroStatistics):

    def __init__(self, configurationObject, workingDir=None, *args, **kwargs):
        super(HBVSASKStatistics, self).__init__(configurationObject, workingDir, *args, **kwargs)
        # Statistics.__init__(self)

        if "basis" in kwargs:
            self.basis = kwargs['basis']
        else:
            self.basis = self.configurationObject["model_settings"].get("basis", 'Oldman_Basin')

        inputModelDir = kwargs.get('inputModelDir', self.workingDir)
        self.inputModelDir = pathlib.Path(inputModelDir)
        self.inputModelDir_basis = self.inputModelDir / self.basis

        #####################################
        # Set of configuration variables propagated via **kwargs or read from configurationObject
        # These are mainly model related configurations
        #####################################
        # This is actually index name in the propageted results DataFrame
        self.time_column_name = kwargs.get("time_column_name", "TimeStamp")
        self.precipitation_column_name = kwargs.get("precipitation_column_name", "precipitation")
        self.temperature_column_name = kwargs.get("temperature_column_name", "temperature")
        self.forcing_data_column_names = [self.precipitation_column_name, self.temperature_column_name]

        if "run_full_timespan" in kwargs:
            self.run_full_timespan = kwargs['run_full_timespan']
        else:
            self.run_full_timespan = strtobool(
                self.configurationObject["time_settings"].get("run_full_timespan", 'False'))

        if "corrupt_forcing_data" in kwargs:
            self.corrupt_forcing_data = kwargs['corrupt_forcing_data']
        else:
            self.corrupt_forcing_data = strtobool(self.configurationObject["model_settings"].get(
                "corrupt_forcing_data", False))

        #####################################
        # streamflow is of special importance here, since we have saved/measured/ground truth that for it and it is inside input data
        # self.streamflow_column_name = kwargs.get("streamflow_column_name", "streamflow")
        #####################################

        self.read_measured_streamflow = False
        if self.multiple_qoi:
            for idx, single_qoi_column in enumerate(self.qoi_column):
                if single_qoi_column == "Q_cms" or single_qoi_column == "Q" or single_qoi_column == "streamflow":
                    self.read_measured_streamflow = self.read_measured_data[idx]
                    self.streamflow_column_name = self.qoi_column_measured[idx]
        else:
            if self.qoi_column == "Q_cms" or self.qoi_column == "Q" or self.qoi_column == "streamflow":
                self.read_measured_streamflow = self.read_measured_data
                self.streamflow_column_name = self.qoi_column_measured

    ###################################################################################################################

    def prepare(self, rawSamples, **kwargs):
        super(HBVSASKStatistics, self).prepare(rawSamples, **kwargs)

    ###################################################################################################################

    def preparePolyExpanForMc(self, simulationNodes, numEvaluations, regression=None, order=None,
                              poly_normed=None, poly_rule=None, *args, **kwargs):
        super(HBVSASKStatistics, self).preparePolyExpanForMc(simulationNodes, numEvaluations, regression, order,
                              poly_normed, poly_rule, *args, **kwargs)

    def preparePolyExpanForSc(self, simulationNodes, order, poly_normed, poly_rule, *args, **kwargs):
        super(HBVSASKStatistics, self).preparePolyExpanForSc(simulationNodes, order, poly_normed, poly_rule,
                                                             *args, **kwargs)

    def preparePolyExpanForSaltelli(self, simulationNodes, numEvaluations=None, regression=None, order=None,
                                    poly_normed=None, poly_rule=None, *args, **kwargs):
        super(HBVSASKStatistics, self).preparePolyExpanForSaltelli(simulationNodes, numEvaluations, regression, order,
                                                             poly_normed, poly_rule, *args, **kwargs)

    ###################################################################################################################

    def calcStatisticsForMcParallel(self, chunksize=1, regression=False, *args, **kwargs):
        super(HBVSASKStatistics, self).calcStatisticsForMcParallel(chunksize, regression, *args, **kwargs)

    def calcStatisticsForEnsembleParallel(self, chunksize=1, regression=False, *args, **kwargs):
        super(HBVSASKStatistics, self).calcStatisticsForEnsembleParallel(chunksize, regression, *args, **kwargs)

    def calcStatisticsForScParallel(self, chunksize=1, regression=False, *args, **kwargs):
        super(HBVSASKStatistics, self).calcStatisticsForScParallel(chunksize, regression, *args, **kwargs)

    def calcStatisticsForSaltelliParallel(self, chunksize=1, regression=False, *args, **kwargs):
        super(HBVSASKStatistics, self).calcStatisticsForSaltelliParallel(chunksize, regression, *args, **kwargs)

    ###################################################################################################################

    def param_grad_analysis(self):
        super(HBVSASKStatistics, self).param_grad_analysis()

    ###################################################################################################################

    def _check_if_Sobol_t_computed(self, keyIter, qoi_column=None):
        super(HBVSASKStatistics, self)._check_if_Sobol_t_computed(keyIter, qoi_column)

    def _check_if_Sobol_m_computed(self, keyIter, qoi_column=None):
        super(HBVSASKStatistics, self)._check_if_Sobol_m_computed(keyIter, qoi_column)

    def _check_if_Sobol_m2_computed(self, keyIter, qoi_column=None):
        super(HBVSASKStatistics, self)._check_if_Sobol_m2_computed(keyIter, qoi_column)

    ###################################################################################################################

    def saveToFile(self, fileName="statistics_dict", fileNameIdent="", directory="./",
                   fileNameIdentIsFullName=False, **kwargs):
        super(HBVSASKStatistics, self).saveToFile(fileName, fileNameIdent, directory,
                   fileNameIdentIsFullName, **kwargs)

    ###################################################################################################################
    # TODO What about AET or other QoI/Model output, make this more general
    # TODO Make below functions more general

    def get_measured_data(self, timestepRange=None, time_column_name="TimeStamp", qoi_column_name="streamflow",
                          **kwargs):
        # In this particular set-up, we only have access to the measured streamflow
        self.df_measured = self._get_measured_streamflow(time_column_name=time_column_name,
                                                         streamflow_column_name=qoi_column_name, **kwargs)

        transforme_mesured_data_as_original_model = kwargs.get(
            "transforme_mesured_data_as_original_model", True)

        # This data will be used for plotting or comparing with approximated data
        # Perform the same transformation as on original model output
        if transforme_mesured_data_as_original_model:
            for idx, single_qoi_column in enumerate(self.list_original_model_output_columns):
                single_transformation = self.list_transform_model_output[idx]
                if single_transformation is not None and single_transformation != "None":
                    if self.list_read_measured_data[idx]:
                        utility.transform_column_in_df(
                            self.df_measured,
                            transformation_function_str=single_transformation,
                            column_name=self.list_qoi_column_measured[idx],
                            new_column_name=self.list_qoi_column_measured[idx])
        self.measured_fetched = True

    def get_unaltered_run_data(self):
        self.df_unaltered = None
        self.unaltered_computed = False

    def get_forcing_data(self, timestepRange=None, time_column_name="TimeStamp", forcing_column_names="precipitation",
                          **kwargs):
        self.forcing_df = self._get_precipitation_temperature_input_data(time_column_name=time_column_name, **kwargs)
        self.forcing_data_fetched = True

    ##########################
    # HBV Specific functions for fetching saved data
    ##########################

    def _get_measured_streamflow(self, time_column_name="TimeStamp", streamflow_column_name="streamflow", **kwargs):
        streamflow_inp = kwargs.get("streamflow_inp", "streamflow.inp")
        streamflow_inp = self.inputModelDir_basis / streamflow_inp

        if streamflow_column_name is None:
            streamflow_column_name = self.streamflow_column_name
        streamflow_df = hbv.read_streamflow(streamflow_inp,
                                            time_column_name=time_column_name,
                                            streamflow_column_name=streamflow_column_name)
        # Parse input based on some timeframe
        if time_column_name in streamflow_df.columns:
            streamflow_df = streamflow_df.loc[
                (streamflow_df[time_column_name] >= self.timesteps_min) & (
                        streamflow_df[time_column_name] <= self.timesteps_max)]
        else:
            streamflow_df = streamflow_df[self.timesteps_min:self.timesteps_max]
        return streamflow_df

    def _get_precipitation_temperature_input_data(self, time_column_name="TimeStamp", **kwargs):
        precipitation_temperature_inp = kwargs.get("precipitation_temperature_inp", "Precipitation_Temperature.inp")
        precipitation_column_name = kwargs.get("precipitation_column_name", "precipitation")
        temperature_column_name = kwargs.get("temperature_column_name", "temperature")
        precipitation_temperature_inp = self.inputModelDir_basis / precipitation_temperature_inp

        precipitation_temperature_df = hbv.read_precipitation_temperature(
            precipitation_temperature_inp, time_column_name=time_column_name,
            precipitation_column_name=precipitation_column_name, temperature_column_name=temperature_column_name
        )

        # Parse input based on some timeframe
        if time_column_name in precipitation_temperature_df.columns:
            precipitation_temperature_df = precipitation_temperature_df.loc[
                (precipitation_temperature_df[time_column_name] >= self.timesteps_min) \
                & (precipitation_temperature_df[time_column_name] <= self.timesteps_max)]
        else:
            precipitation_temperature_df = precipitation_temperature_df[self.timesteps_min:self.timesteps_max]
        return precipitation_temperature_df

    def input_and_measured_data_setup(self, time_column_name="TimeStamp", precipitation_column_name="precipitation",
                                       temperature_column_name="temperature",
                                       read_measured_streamflow=None, streamflow_column_name="streamflow",
                                       **kwargs):
        # % ********  Forcing (Precipitation and Temperature)  *********
        precipitation_temperature_df = self._get_precipitation_temperature_input_data(
            time_column_name=time_column_name,
            precipitation_column_name=precipitation_column_name,
            temperature_column_name=temperature_column_name, **kwargs)

        # % ********  Observed Streamflow  *********
        # if read_measured_streamflow is None:
        #     read_measured_streamflow = self.read_measured_streamflow

        if read_measured_streamflow:
            streamflow_df = self._get_measured_streamflow(time_column_name=time_column_name,
                                                          streamflow_column_name=streamflow_column_name, **kwargs)

            time_series_measured_data_df = pd.merge(
                streamflow_df, precipitation_temperature_df,  left_index=True, right_index=True
            )
        else:
            time_series_measured_data_df = precipitation_temperature_df

        return time_series_measured_data_df

    ###################################################################################################################

    def plotResults(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):

        plot_measured_timeseries = kwargs.get('plot_measured_timeseries', False)
        plot_unaltered_timeseries = kwargs.get('plot_unaltered_timeseries', False)
        plot_forcing_timeseries = kwargs.get('plot_forcing_timeseries', False)
        time_column_name = kwargs.get('time_column_name', self.time_column_name)

        if plot_measured_timeseries:
            self.get_measured_data(time_column_name=time_column_name,
                                   qoi_column_name=kwargs.get('measured_df_column_to_draw', "streamflow"))

        if plot_unaltered_timeseries:
            self.get_unaltered_run_data()

        if plot_forcing_timeseries:
            precipitation_column_name = kwargs.get('precipitation_df_column_to_draw', "precipitation")
            temperature_column_name = kwargs.get('temperature_df_column_to_draw', "temperature")
            self.get_forcing_data(time_column_name=time_column_name,
                                  precipitation_column_name=precipitation_column_name,
                                  temperature_column_name=temperature_column_name)

        single_fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html",
                                                directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)
        fig = self._plotStatisticsDict_plotly(
            unalatered=self.unaltered_computed, measured=self.measured_fetched, forcing=self.forcing_data_fetched,
            recalculateTimesteps=False, filename=single_fileName, display=display, **kwargs)

        if display:
            fig.show()

        print(f"[STAT INFO] plotResults function is done!")

    def _plotStatisticsDict_plotly(self, unalatered=False, measured=False, forcing=False, recalculateTimesteps=False,
                                   window_title='Forward UQ & SA', filename="sim-plotly.html",
                                   display=False, **kwargs):
        pdTimesteps = self.pdTimesteps
        keyIter = list(pdTimesteps)
        self._check_if_Sobol_t_computed(keyIter)
        self._check_if_Sobol_m_computed(keyIter)

        n_rows = 0
        starting_row = 1
        if forcing and self.forcing_data_fetched:
            n_rows += 3
            starting_row = 4
        n_rows += len(self.list_qoi_column)
        if self._is_Sobol_m_computed:
            n_rows += len(self.list_qoi_column)
        if self._is_Sobol_t_computed:
            n_rows += len(self.list_qoi_column)

        fig = make_subplots(rows=n_rows, cols=1,
                            print_grid=True, shared_xaxes=False,
                            vertical_spacing=0.1)

        if forcing and self.forcing_data_fetched:
            # Precipitation
            column_to_draw = kwargs.get('precipitation_df_column_to_draw', 'precipitation')
            timestamp_column = kwargs.get('precipitation_df_timestamp_column', 'TimeStamp')
            N_max = self.forcing_df[column_to_draw].max()
            if timestamp_column == "index":
                fig.add_trace(go.Bar(x=self.forcing_df.index,
                                     y=self.forcing_df[column_to_draw],
                                     name="Precipitation", marker_color='magenta'),
                              row=1, col=1)
            else:
                fig.add_trace(go.Bar(x=self.forcing_df[timestamp_column],
                                     y=self.forcing_df[column_to_draw],
                                     name="Precipitation", marker_color='magenta'),
                              row=1, col=1)

            # Temperature
            column_to_draw = kwargs.get('temperature_df_column_to_draw', 'temperature')
            timestamp_column = kwargs.get('temperature_df_timestamp_column', 'TimeStamp')
            if timestamp_column == "index":
                fig.add_trace(go.Scatter(x=self.forcing_df.index,
                                         y=self.forcing_df[column_to_draw],
                                         name="Temperature", line_color='blue', mode='lines+markers'),
                              row=2, col=1)
            else:
                fig.add_trace(go.Scatter(x=self.forcing_df[timestamp_column],
                                         y=self.forcing_df[column_to_draw],
                                         name="Temperature", line_color='blue', mode='lines+markers'),
                              row=2, col=1)

            # Streamflow - hard-coded for HBV
            column_to_draw = kwargs.get('streamflow_df_column_to_draw', 'temperature')
            timestamp_column = kwargs.get('streamflow_df_timestamp_column', 'TimeStamp')
            streamflow_df = self._get_measured_streamflow(
                time_column_name="TimeStamp", streamflow_column_name=column_to_draw)
            if timestamp_column == "index":
                fig.add_trace(go.Scatter(x=streamflow_df.index,
                                         y=streamflow_df[column_to_draw],
                                         name="Obs Streamflow", line_color='blue', mode='lines'),
                              row=3, col=1)
            else:
                fig.add_trace(go.Scatter(x=streamflow_df[timestamp_column],
                                         y=streamflow_df[column_to_draw],
                                         name="Obs Streamflow", line_color='blue', mode='lines'),
                              row=3, col=1)

        dict_qoi_vs_plot_rows = defaultdict(dict, {single_qoi_column: {} for single_qoi_column in self.list_qoi_column})
        # One big Figure for each QoI; Note: self.list_qoi_column contain first original model output
        for idx, single_qoi_column in enumerate(self.list_qoi_column):
            if single_qoi_column in self.list_original_model_output_columns:
                if measured and self.measured_fetched and self.list_read_measured_data[idx]:
                    column_to_draw = self.list_qoi_column_measured[idx]
                    timestamp_column = kwargs.get('measured_df_timestamp_column', 'TimeStamp')
                    if timestamp_column == "index":
                        fig.add_trace(go.Scatter(x=self.df_measured.index, y=self.df_measured[column_to_draw],
                                                 name=f"{single_qoi_column} (measured)", line_color='red', mode='lines'),
                                      row=starting_row, col=1)
                    else:
                        fig.add_trace(go.Scatter(x=self.df_measured[timestamp_column], y=self.df_measured[column_to_draw],
                                                 name=f"{single_qoi_column} (measured)", line_color='red', mode='lines'),
                                      row=starting_row, col=1)

            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[self.result_dict[single_qoi_column][key]["E"] for key in keyIter],
                                     name=f'E[{single_qoi_column}]',
                                     line_color='green', mode='lines'),
                          row=starting_row, col=1)
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[(self.result_dict[single_qoi_column][key]["E"] \
                                         - self.result_dict[single_qoi_column][key]["StdDev"]) for key in keyIter],
                                     name='mean - std. dev', line_color='darkviolet', mode='lines'),
                          row=starting_row, col=1)
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[(self.result_dict[single_qoi_column][key]["E"] +\
                                         self.result_dict[single_qoi_column][key]["StdDev"]) for key in keyIter],
                                     name='mean + std. dev', line_color='darkviolet', mode='lines', fill='tonexty'),
                          row=starting_row, col=1)
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[self.result_dict[single_qoi_column][key]["P10"] for key in keyIter],
                                     name='10th percentile', line_color='yellow', mode='lines'),
                          row=starting_row, col=1)
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[self.result_dict[single_qoi_column][key]["P90"] for key in keyIter],
                                     name='90th percentile', line_color='yellow', mode='lines', fill='tonexty'),
                          row=starting_row, col=1)
            dict_qoi_vs_plot_rows[single_qoi_column]["qoi"] = starting_row
            starting_row += 1

            # fig.add_trace(go.Scatter(x=pdTimesteps,
            #                          y=[self.result_dict[single_qoi_column][key]["StdDev"] for key in keyIter],
            #                          name='std. dev', line_color='darkviolet', mode='lines'),
            #               row=starting_row+1, col=1)

        if self._is_Sobol_m_computed:
            for single_qoi_column in self.list_qoi_column:
                for i in range(len(self.labels)):
                    name = self.labels[i] + "_" + single_qoi_column + "_S_m"
                    fig.add_trace(go.Scatter(
                        x=pdTimesteps,
                        y=[self.result_dict[single_qoi_column][key]["Sobol_m"][i] for key in keyIter],
                        name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                        row=starting_row, col=1)
                dict_qoi_vs_plot_rows[single_qoi_column]["sobol_m"] = starting_row
                starting_row += 1

        if self._is_Sobol_t_computed:
            for single_qoi_column in self.list_qoi_column:
                for i in range(len(self.labels)):
                    name = self.labels[i] + "_" + single_qoi_column + "_S_t"
                    fig.add_trace(go.Scatter(
                        x=pdTimesteps,
                        y=[self.result_dict[single_qoi_column][key]["Sobol_t"][i] for key in keyIter],
                        name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                        row=starting_row, col=1)
                dict_qoi_vs_plot_rows[single_qoi_column]["sobol_t"] = starting_row
                starting_row += 1

        # fig.update_traces(mode='lines')
        if forcing and self.forcing_data_fetched:
            fig.update_yaxes(title_text="N [mm/h]", showgrid=True, row=1, col=1)
            fig.update_yaxes(autorange="reversed", row=1, col=1)
            fig.update_yaxes(title_text="T [c]", showgrid=True, row=2, col=1)

        for single_qoi_column in self.list_qoi_column:
            fig.update_yaxes(title_text=single_qoi_column, showgrid=True,
                             row=dict_qoi_vs_plot_rows[single_qoi_column]["qoi"], col=1)
            # fig.update_yaxes(title_text=f"Std. Dev. [{single_qoi_column}]", side='left', showgrid=True,
            #                  row=starting_row+1, col=1)
            if self._is_Sobol_m_computed:
                fig.update_yaxes(title_text=f"{single_qoi_column}_m", showgrid=True, range=[0, 1],
                                 row=dict_qoi_vs_plot_rows[single_qoi_column]["sobol_m"], col=1)
            if self._is_Sobol_t_computed:
                fig.update_yaxes(title_text=f"{single_qoi_column}_t", showgrid=True, range=[0, 1],
                                 row=dict_qoi_vs_plot_rows[single_qoi_column]["sobol_t"], col=1)

        fig.update_layout(width=1000)
        fig.update_layout(title_text=window_title)
        fig.update_layout(xaxis=dict(type="date"))

        print(f"[HVB STAT INFO] _plotStatisticsDict_plotly function is almost over, just to save the plot!")

        # filename = pathlib.Path(filename)
        plot(fig, filename=filename, auto_open=display)
        return fig

    def _compute_number_of_rows_for_plotting(self, starting_row=1):
        super(HBVSASKStatistics, self)._compute_number_of_rows_for_plotting(starting_row)
    ###################################################################################################################

    def extract_mean_time_series(self):
        if self.result_dict is None:
            raise Exception('[STAT INFO] extract_mean_time_series - self.result_dict is None. '
                            'Calculate the statistics first!')
        list_of_single_qoi_mean_df = []
        for single_qoi_column in self.list_qoi_column:
            keyIter = list(self.pdTimesteps)
            mean_time_series = [self.result_dict[single_qoi_column][key]["E"] for key in keyIter]
            qoi_column = [single_qoi_column] * len(keyIter)
            mean_df_single_qoi = pd.DataFrame(list(zip(qoi_column, mean_time_series, self.pdTimesteps)),
                                              columns=['QoI', 'Mean_QoI', 'TimeStamp'])
            list_of_single_qoi_mean_df.append(mean_df_single_qoi)
        self.qoi_mean_df = pd.concat(list_of_single_qoi_mean_df, ignore_index=True, sort=False, axis=0)

    def create_df_from_statistics_data_single_qoi(self, qoi_column, uq_method="sc"):
        keyIter = list(self.pdTimesteps)
        mean_time_series = [self.result_dict[qoi_column][key]["E"] for key in keyIter]
        std_time_series = [self.result_dict[qoi_column][key]["StdDev"] for key in keyIter]
        p10_time_series = [self.result_dict[qoi_column][key]["P10"] for key in keyIter]
        p90_time_series = [self.result_dict[qoi_column][key]["P90"] for key in keyIter]
        list_of_columns = [self.pdTimesteps, mean_time_series, std_time_series,
                           p10_time_series, p90_time_series]
        list_of_columns_names = ['TimeStamp', "E", "Std", "P10", "P90"]

        self._check_if_Sobol_t_computed(keyIter, qoi_column=qoi_column)
        self._check_if_Sobol_m_computed(keyIter, qoi_column=qoi_column)
        if self._is_Sobol_m_computed:
            for i in range(len(self.labels)):
                sobol_m_time_series = [self.result_dict[qoi_column][key]["Sobol_m"][i] for key in keyIter]
                list_of_columns.append(sobol_m_time_series)
                temp = "sobol_m_" + self.labels[i]
                list_of_columns_names.append(temp)
        if self._is_Sobol_t_computed:
            for i in range(len(self.labels)):
                sobol_t_time_series = [self.result_dict[qoi_column][key]["Sobol_t"][i] for key in keyIter]
                list_of_columns.append(sobol_t_time_series)
                temp = "sobol_t_" + self.labels[i]
                list_of_columns_names.append(temp)

        df_statistics = pd.DataFrame(list(zip(*list_of_columns)), columns=list_of_columns_names)

        if self.measured_fetched:
            pass  # TODO

        if self.unaltered_computed:
            pass  # TODO

        df_statistics["E_minus_std"] = df_statistics['E'] - df_statistics['Std']
        df_statistics["E_plus_std"] = df_statistics['E'] + df_statistics['Std']
        return df_statistics

