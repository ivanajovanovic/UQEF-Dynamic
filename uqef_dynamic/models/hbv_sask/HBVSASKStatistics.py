from collections import defaultdict
from distutils.util import strtobool
import pandas as pd
import pathlib
from plotly.offline import iplot, plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from uqef_dynamic.utils import colors
from uqef_dynamic.models.time_dependent_baseclass import time_dependent_statistics
from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv


class HBVSASKStatistics(time_dependent_statistics.TimeDependentStatistics):

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

    def get_forcing_data(self, timestepRange=None, time_column_name=None, **kwargs):
        if timestepRange is None:
            timestepRange = (self.timesteps_min, self.timesteps_max)
        if time_column_name is None:
            time_column_name = self.time_column_name
        self.forcing_df = self._get_precipitation_temperature_input_data(
            timestepRange=timestepRange, time_column_name=time_column_name, **kwargs)
        self.forcing_data_fetched = True

    ##########################
    # HBV Specific functions for fetching saved data
    ##########################
    def _get_measured_single_qoi(self, timestepRange=None, time_column_name="TimeStamp",
        qoi_column_measured="measured", **kwargs):
        if timestepRange is None:
            timestepRange = (self.timesteps_min, self.timesteps_max)
        if time_column_name is None:
            time_column_name = self.time_column_name
        if qoi_column_measured == "streamflow":
            return self._get_measured_streamflow(
                timestepRange=timestepRange, time_column_name=time_column_name,
                streamflow_column_name=qoi_column_measured, **kwargs)
        else:
            print(f"[HBV-SASK Stat - Sorry no other measured data ({qoi_column_measured}) is available besides streamflow]")
            raise NotImplementedError

    def _get_measured_streamflow(self, timestepRange=None, time_column_name="TimeStamp",
                                 streamflow_column_name="streamflow", **kwargs):
        streamflow_inp = kwargs.get("streamflow_inp", "streamflow.inp")
        streamflow_inp = self.inputModelDir_basis / streamflow_inp

        if streamflow_column_name is None:
            streamflow_column_name = self.streamflow_column_name
        streamflow_df = hbv.read_streamflow(streamflow_inp,
                                            time_column_name=time_column_name,
                                            streamflow_column_name=streamflow_column_name)
        if timestepRange is None:
            timestepRange = (self.timesteps_min, self.timesteps_max)

        # Parse input based on some timeframe
        if time_column_name in streamflow_df.columns:
            streamflow_df = streamflow_df.loc[
                (streamflow_df[time_column_name] >= timestepRange[0]) & (
                        streamflow_df[time_column_name] <= timestepRange[1])]
        else:
            streamflow_df = streamflow_df[timestepRange[0]:timestepRange[1]]
        return streamflow_df

    def _get_precipitation_temperature_input_data(self, timestepRange=None, time_column_name="TimeStamp", **kwargs):
        precipitation_temperature_inp = kwargs.get("precipitation_temperature_inp", "Precipitation_Temperature.inp")
        precipitation_column_name = kwargs.get("precipitation_column_name", "precipitation")
        temperature_column_name = kwargs.get("temperature_column_name", "temperature")
        precipitation_temperature_inp = self.inputModelDir_basis / precipitation_temperature_inp

        precipitation_temperature_df = hbv.read_precipitation_temperature(
            precipitation_temperature_inp, time_column_name=time_column_name,
            precipitation_column_name=precipitation_column_name, temperature_column_name=temperature_column_name
        )

        if timestepRange is None:
            timestepRange = (self.timesteps_min, self.timesteps_max)

        # Parse input based on some timeframe
        if time_column_name in precipitation_temperature_df.columns:
            precipitation_temperature_df = precipitation_temperature_df.loc[
                (precipitation_temperature_df[time_column_name] >= timestepRange[0]) \
                & (precipitation_temperature_df[time_column_name] <= timestepRange[1])]
        else:
            precipitation_temperature_df = precipitation_temperature_df[timestepRange[0]:timestepRange[1]]
        return precipitation_temperature_df

    def input_and_measured_data_setup(self, timestepRange=None, time_column_name="TimeStamp",
                                      precipitation_column_name="precipitation", temperature_column_name="temperature",
                                      read_measured_streamflow=None, streamflow_column_name="streamflow", **kwargs):
        # % ********  Forcing (Precipitation and Temperature)  *********
        precipitation_temperature_df = self._get_precipitation_temperature_input_data(
            timestepRange=timestepRange, time_column_name=time_column_name,
            precipitation_column_name=precipitation_column_name,
            temperature_column_name=temperature_column_name,
            **kwargs)

        # % ********  Observed Streamflow  *********
        # if read_measured_streamflow is None:
        #     read_measured_streamflow = self.read_measured_streamflow

        if read_measured_streamflow:
            streamflow_df = self._get_measured_streamflow(
                timestepRange=timestepRange, time_column_name=time_column_name,
                streamflow_column_name=streamflow_column_name, **kwargs)

            time_series_measured_data_df = pd.merge(
                streamflow_df, precipitation_temperature_df,  left_index=True, right_index=True
            )
        else:
            time_series_measured_data_df = precipitation_temperature_df

        return time_series_measured_data_df

    ###################################################################################################################

    def plotResults(self, timestep=-1, display=False, fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, dict_what_to_plot=None, **kwargs):

        timestepRange = (min(self.pdTimesteps), max(self.pdTimesteps))

        plot_measured_timeseries = kwargs.get('plot_measured_timeseries', True)
        plot_unaltered_timeseries = kwargs.get('plot_unaltered_timeseries', False)
        plot_forcing_timeseries = kwargs.get('plot_forcing_timeseries', True)
        time_column_name = kwargs.get('time_column_name', self.time_column_name)

        if plot_measured_timeseries and (not self.measured_fetched or self.df_measured is None or self.df_measured.empty):
            self.get_measured_data(
                time_column_name=time_column_name,
                qoi_column_name=self.list_original_model_output_columns)

        if plot_unaltered_timeseries:
            self.get_unaltered_run_data()

        if plot_forcing_timeseries and (not self.forcing_data_fetched or self.forcing_df is None or self.forcing_df.empty):
            precipitation_column_name = kwargs.get('precipitation_df_column_to_draw', "precipitation")
            temperature_column_name = kwargs.get('temperature_df_column_to_draw', "temperature")
            self.get_forcing_data(time_column_name=time_column_name,
                                  precipitation_column_name=precipitation_column_name,
                                  temperature_column_name=temperature_column_name)

        single_fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html",
                                                directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)
        fig = self._plotStatisticsDict_plotly(unalatered=self.unaltered_computed, measured=self.measured_fetched,
                                              forcing=self.forcing_data_fetched, recalculateTimesteps=False,
                                              filename=single_fileName, display=display,
                                              dict_what_to_plot=dict_what_to_plot,**kwargs)

        if display:
            fig.show()

        print(f"[STAT INFO] plotResults function is done!")

    def _plotStatisticsDict_plotly(self, unalatered=False, measured=False, forcing=False, recalculateTimesteps=False,
                                   window_title='Forward UQ & SA', filename="sim-plotly.html", display=False,
                                   dict_what_to_plot=None, **kwargs):
        pdTimesteps = self.pdTimesteps
        keyIter = list(pdTimesteps)
        timesteps_min = min(pdTimesteps)
        timesteps_max = max(pdTimesteps)

        if dict_what_to_plot is None:
            if self.dict_what_to_plot is not None:
                dict_what_to_plot = self.dict_what_to_plot
            else:
                dict_what_to_plot = {
                    "E_minus_std": False, "E_plus_std": False, "P10": False, "P90": False,
                    "StdDev": False, "Skew": False, "Kurt": False, "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False
                }
                self.dict_what_to_plot = dict_what_to_plot

        n_rows, starting_row = self._compute_number_of_rows_for_plotting(dict_what_to_plot, forcing)

        fig = make_subplots(
            rows=n_rows, cols=1, print_grid=True,
            shared_xaxes=True, vertical_spacing=0.1
        )

        # HBV - Specific plotting of observed data, i.e., forcing data and measured streamflow
        if forcing and self.forcing_data_fetched:
            reset_index_at_the_end = False
            if self.forcing_df.index.name != self.time_column_name:
                self.forcing_df.set_index(self.time_column_name, inplace=True)
                reset_index_at_the_end = True

            # Precipitation
            column_to_draw = kwargs.get('precipitation_df_column_to_draw', 'precipitation')
            N_max = self.forcing_df[column_to_draw].max()
            fig.add_trace(go.Bar(x=self.forcing_df.index,
                                 y=self.forcing_df[column_to_draw],
                                 name="Precipitation", marker_color='blue'),
                          row=1, col=1)

            # Temperature
            column_to_draw = kwargs.get('temperature_df_column_to_draw', 'temperature')
            fig.add_trace(go.Scatter(x=self.forcing_df.index,
                                     y=self.forcing_df[column_to_draw],
                                     name="Temperature", line_color='blue', mode='lines'),
                          row=2, col=1)

            if reset_index_at_the_end:
                self.forcing_df.reset_index(inplace=True)
                self.forcing_df.rename(columns={self.forcing_df.index.name: self.time_column_name}, inplace=True)

            # Streamflow - hard-coded for HBV
            streamflow_df = self._get_measured_streamflow(time_column_name="TimeStamp",
                                                          streamflow_column_name="streamflow")
            column_to_draw = kwargs.get('streamflow_df_column_to_draw', 'streamflow')
            reset_index_at_the_end = False
            if streamflow_df.index.name != self.time_column_name:
                streamflow_df.set_index(self.time_column_name, inplace=True)
                reset_index_at_the_end = True
            fig.add_trace(go.Scatter(x=streamflow_df.index,
                                     y=streamflow_df[column_to_draw],
                                     name="Obs Streamflow", line_color='red', mode='lines'),
                          row=3, col=1)
            if reset_index_at_the_end:
                streamflow_df.reset_index(inplace=True)
                streamflow_df.rename(columns={streamflow_df.index.name: self.time_column_name}, inplace=True)

        dict_qoi_vs_plot_rows = defaultdict(dict, {single_qoi_column: {} for single_qoi_column in self.list_qoi_column})

        # One big Figure for each QoI; Note: self.list_qoi_column contain first original model output
        for idx, single_qoi_column in enumerate(self.list_qoi_column):
            if single_qoi_column in self.list_original_model_output_columns:
                if measured and self.measured_fetched and self.df_measured is not None \
                        and not self.df_measured.empty and self.list_read_measured_data[idx]:
                    # column_to_draw = self.list_qoi_column_measured[idx]
                    # timestamp_column = kwargs.get('measured_df_timestamp_column', 'TimeStamp')
                    df_measured_subset = self.df_measured.loc[self.df_measured["qoi"] == single_qoi_column]
                    if not df_measured_subset.empty:
                        column_to_draw = "measured"
                        timestamp_column = self.time_column_name
                        fig.add_trace(
                            go.Scatter(x=df_measured_subset[timestamp_column], y=df_measured_subset[column_to_draw],
                                       name=f"{single_qoi_column} (measured)", line_color='red', mode='lines'),
                            row=starting_row, col=1)

            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[self.result_dict[single_qoi_column][key]["E"] for key in keyIter],
                                     name=f'E[{single_qoi_column}]',
                                     line_color='green', mode='lines'),
                          row=starting_row, col=1)
            
            if dict_what_to_plot.get("E_minus_std", False) and "StdDev" in self.result_dict[single_qoi_column][keyIter[0]]:
                fig.add_trace(go.Scatter(x=pdTimesteps,
                                         y=[(self.result_dict[single_qoi_column][key]["E"] \
                                             - self.result_dict[single_qoi_column][key]["StdDev"]) for key in keyIter],
                                             name='mean - std. dev', mode='lines', showlegend=False, line_color='rgba(200, 200, 200, 0.4)',),
                              row=starting_row, col=1)
            if dict_what_to_plot.get("E_plus_std", False) and "StdDev" in self.result_dict[single_qoi_column][keyIter[0]]:
                fig.add_trace(go.Scatter(x=pdTimesteps,
                                         y=[(self.result_dict[single_qoi_column][key]["E"] +\
                                             self.result_dict[single_qoi_column][key]["StdDev"]) for key in keyIter],
                                             name='mean +- std. dev', mode='lines', fill='tonexty', showlegend=True, line_color='rgba(200, 200, 200, 0.4)',),
                              row=starting_row, col=1)
            if dict_what_to_plot.get("E_minus_2std", False) and "StdDev" in self.result_dict[single_qoi_column][keyIter[0]]:
                fig.add_trace(go.Scatter(x=pdTimesteps,
                                         y=[(self.result_dict[single_qoi_column][key]["E"] \
                                             - 2*self.result_dict[single_qoi_column][key]["StdDev"]) for key in keyIter],
                                             name='mean - 2*std. dev', mode='lines', showlegend=False, line_color='rgba(200, 200, 200, 0.4)',),
                              row=starting_row, col=1)
            if dict_what_to_plot.get("E_plus_2std", False) and "StdDev" in self.result_dict[single_qoi_column][keyIter[0]]:
                fig.add_trace(go.Scatter(x=pdTimesteps,
                                         y=[(self.result_dict[single_qoi_column][key]["E"] +\
                                             2*self.result_dict[single_qoi_column][key]["StdDev"]) for key in keyIter],
                                             name='mean +- 2*std. dev', mode='lines', fill='tonexty', showlegend=True, line_color='rgba(200, 200, 200, 0.4)',),
                              row=starting_row, col=1)

            if "P10" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("P10", False):
                fig.add_trace(go.Scatter(x=pdTimesteps,
                                         y=[self.result_dict[single_qoi_column][key]["P10"] for key in keyIter],
                                         name='10th percentile', line_color='rgba(128,128,128, 0.3)', mode='lines', showlegend=False,),
                              row=starting_row, col=1)
            if "P90" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("P90", False):
                fig.add_trace(go.Scatter(x=pdTimesteps,
                                         y=[self.result_dict[single_qoi_column][key]["P90"] for key in keyIter],
                                         name='10th percentile-90th percentile', mode='lines', fill='tonexty', showlegend=True, line=dict(color='rgba(128,128,128, 0.3)'), fillcolor='rgba(128,128,128, 0.3)'),
                              row=starting_row, col=1)
            dict_qoi_vs_plot_rows[single_qoi_column]["qoi"] = starting_row
            starting_row += 1

            if "StdDev" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("StdDev", False):
                fig.add_trace(go.Scatter(x=pdTimesteps,
                                         y=[self.result_dict[single_qoi_column][key]["StdDev"] for key in keyIter],
                                         name='std. dev', line_color='darkviolet', mode='lines'),
                              row=starting_row, col=1)
                dict_qoi_vs_plot_rows[single_qoi_column]["StdDev"] = starting_row
                starting_row += 1

            if "Skew" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("Skew", False):
                fig.add_trace(go.Scatter(x=pdTimesteps,
                                         y=[self.result_dict[single_qoi_column][key]["Skew"] for key in keyIter],
                                         name='Skew', mode='lines', ),
                              row=starting_row, col=1)
                dict_qoi_vs_plot_rows[single_qoi_column]["Skew"] = starting_row
                starting_row += 1

            if "Kurt" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("Kurt", False):
                fig.add_trace(go.Scatter(x=pdTimesteps,
                                         y=[self.result_dict[single_qoi_column][key]["Kurt"] for key in keyIter],
                                         name='Kurt', mode='lines', ),
                              row=starting_row, col=1)
                dict_qoi_vs_plot_rows[single_qoi_column]["Kurt"] = starting_row
                starting_row += 1

        for single_qoi_column in self.list_qoi_column:
            if "Sobol_m" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("Sobol_m", False):
                for i in range(len(self.labels)):
                    name = self.labels[i] + "_" + single_qoi_column + "_S_m"
                    fig.add_trace(go.Scatter(
                        x=pdTimesteps,
                        y=[self.result_dict[single_qoi_column][key]["Sobol_m"][i] for key in keyIter],
                        name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                        row=starting_row, col=1)
                dict_qoi_vs_plot_rows[single_qoi_column]["Sobol_m"] = starting_row
                starting_row += 1

        for single_qoi_column in self.list_qoi_column:
            if "Sobol_m2" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("Sobol_m2", False):
                for i in range(len(self.labels)):
                    name = self.labels[i] + "_" + single_qoi_column + "_S_m2"
                    fig.add_trace(go.Scatter(
                        x=pdTimesteps,
                        y=[self.result_dict[single_qoi_column][key]["Sobol_m2"][i] for key in keyIter],
                        name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                        row=starting_row, col=1)
                dict_qoi_vs_plot_rows[single_qoi_column]["Sobol_m2"] = starting_row
                starting_row += 1

        for single_qoi_column in self.list_qoi_column:
            if "Sobol_t" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("Sobol_t", False):
                for i in range(len(self.labels)):
                    name = self.labels[i] + "_" + single_qoi_column + "_S_t"
                    fig.add_trace(go.Scatter(
                        x=pdTimesteps,
                        y=[self.result_dict[single_qoi_column][key]["Sobol_t"][i] for key in keyIter],
                        name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                        row=starting_row, col=1)
                dict_qoi_vs_plot_rows[single_qoi_column]["Sobol_t"] = starting_row
                starting_row += 1

        # fig.update_traces(mode='lines')
        if forcing and self.forcing_data_fetched:
            fig.update_yaxes(title_text="N [mm/h]", showgrid=True, row=1, col=1)
            fig.update_yaxes(autorange="reversed", row=1, col=1)
            fig.update_yaxes(title_text="T [c]", showgrid=True, row=2, col=1)

        for single_qoi_column in self.list_qoi_column:
            fig.update_yaxes(title_text=single_qoi_column, showgrid=True,
                             row=dict_qoi_vs_plot_rows[single_qoi_column]["qoi"], col=1)
            if "StdDev" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("StdDev", False):
                fig.update_yaxes(title_text=f"StdDev {single_qoi_column}", showgrid=True, side='left',
                                 row=dict_qoi_vs_plot_rows[single_qoi_column]["StdDev"], col=1)
            if "Skew" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("Skew", False):
                fig.update_yaxes(title_text=f"Skew {single_qoi_column}", showgrid=True, side='left',
                                 row=dict_qoi_vs_plot_rows[single_qoi_column]["Skew"], col=1)
            if "Kurt" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("Kurt", False):
                fig.update_yaxes(title_text=f"Kurt {single_qoi_column}", showgrid=True, side='left',
                                 row=dict_qoi_vs_plot_rows[single_qoi_column]["Kurt"], col=1)
            if "Sobol_m" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("Sobol_m", False):
                fig.update_yaxes(title_text=f"{single_qoi_column}_m", showgrid=True, range=[0, 1],
                                 row=dict_qoi_vs_plot_rows[single_qoi_column]["Sobol_m"], col=1)
            if "Sobol_m2" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("Sobol_m2", False):
                fig.update_yaxes(title_text=f"{single_qoi_column}_m2", showgrid=True, range=[0, 1],
                                 row=dict_qoi_vs_plot_rows[single_qoi_column]["Sobol_m2"], col=1)
            if "Sobol_t" in self.result_dict[single_qoi_column][keyIter[0]] and dict_what_to_plot.get("Sobol_t", False):
                fig.update_yaxes(title_text=f"{single_qoi_column}_t", showgrid=True, range=[0, 1],
                                 row=dict_qoi_vs_plot_rows[single_qoi_column]["Sobol_t"], col=1)

        width = len(self.list_qoi_column) * 1000
        fig.update_layout(width=width)
        fig.update_layout(title_text=window_title)
        fig.update_layout(xaxis=dict(rangemode='normal', range=[timesteps_min, timesteps_max], type="date"))
        fig.update_xaxes(
            tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
            dtick="M1"                     # Set tick interval to 1 day for denser ticks
        )
        print(f"[HVB STAT INFO] _plotStatisticsDict_plotly function is almost over, just to save the plot!")

        # filename = pathlib.Path(filename)
        plot(fig, filename=filename, auto_open=display)
        return fig

    def prepare_for_plotting(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):
        timestepRange = (min(self.pdTimesteps), max(self.pdTimesteps))

        plot_measured_timeseries = kwargs.get('plot_measured_timeseries', True)
        plot_unaltered_timeseries = kwargs.get('plot_unaltered_timeseries', False)
        plot_forcing_timeseries = kwargs.get('plot_forcing_timeseries', True)
        time_column_name = kwargs.get('time_column_name', self.time_column_name)

        if plot_measured_timeseries and (not self.measured_fetched or self.df_measured is None or self.df_measured.empty):
            self.get_measured_data(
                time_column_name=time_column_name,
                qoi_column_name=self.list_original_model_output_columns)

        if plot_unaltered_timeseries:
            self.get_unaltered_run_data()

        if plot_forcing_timeseries and (not self.forcing_data_fetched or self.forcing_df is None or self.forcing_df.empty):
            precipitation_column_name = kwargs.get('precipitation_df_column_to_draw', "precipitation")
            temperature_column_name = kwargs.get('temperature_df_column_to_draw', "temperature")
            self.get_forcing_data(time_column_name=time_column_name,
                                  precipitation_column_name=precipitation_column_name,
                                  temperature_column_name=temperature_column_name)

    def plotResults_single_qoi(self, single_qoi_column, dict_time_vs_qoi_stat=None, timestep=-1, display=False, fileName="",
                               fileNameIdent="", directory="./", fileNameIdentIsFullName=False, safe=True,
                               dict_what_to_plot=None, **kwargs):
        # TODO - This might be a memory problem, why not just self.result_dict[single_qoi_column]!
        if dict_time_vs_qoi_stat is None:
            dict_time_vs_qoi_stat = self.result_dict[single_qoi_column]

        if fileName == "":
            fileName = single_qoi_column

        single_fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html",
                                                directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)

        fig = self._plotStatisticsDict_plotly_single_qoi(single_qoi_column=single_qoi_column,
                                                         dict_time_vs_qoi_stat=dict_time_vs_qoi_stat,
                                                         unalatered=self.unaltered_computed,
                                                         measured=self.measured_fetched,
                                                         forcing=self.forcing_data_fetched, recalculateTimesteps=False,
                                                         filename=single_fileName, display=display,
                                                         dict_what_to_plot=dict_what_to_plot, **kwargs)
        if display:
            fig.show()
        print(f"[STAT INFO] plotResults for QoI-{single_qoi_column} function is done!")

    def _plotStatisticsDict_plotly_single_qoi(self, single_qoi_column, dict_time_vs_qoi_stat=None, unalatered=False,
                                              measured=False, forcing=False, recalculateTimesteps=False,
                                              window_title='Forward UQ & SA', filename="sim-plotly.html", display=False,
                                              dict_what_to_plot=None, **kwargs):

        pdTimesteps = self.pdTimesteps
        keyIter = list(pdTimesteps)
        timesteps_min = min(pdTimesteps)
        timesteps_max = max(pdTimesteps)

        if single_qoi_column =='Q_cms':
            window_title = window_title + f": QoI - Streamflow[m^3/s]"
        elif single_qoi_column =='AET':
            window_title = window_title + f": QoI - Actual Evapotranspiration"
        else:       
            window_title = window_title + f": QoI - {single_qoi_column}"

        if dict_time_vs_qoi_stat is None:
            dict_time_vs_qoi_stat = self.result_dict[single_qoi_column]

        if dict_what_to_plot is None:
            if self.dict_what_to_plot is not None:
                dict_what_to_plot = self.dict_what_to_plot
            else:
                dict_what_to_plot = {
                    "E_minus_std": False, "E_plus_std": False, "P10": False, "P90": False,
                    "StdDev": False, "Skew": False, "Kurt": False, "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False
                }
                self.dict_what_to_plot = dict_what_to_plot
                
        # self._check_if_Sobol_t_computed(keyIter[0], qoi_column=single_qoi_column)
        # self._check_if_Sobol_m_computed(keyIter[0], qoi_column=single_qoi_column)

        n_rows, starting_row = self._compute_number_of_rows_for_plotting(
            dict_what_to_plot, forcing, list_qoi_column_to_plot=[single_qoi_column,], result_dict=dict_time_vs_qoi_stat)

        fig = make_subplots(rows=n_rows, cols=1,
                            print_grid=True,
                            shared_xaxes=True,
                            vertical_spacing=0.04)

        # HBV - Specific plotting of observed data, i.e., forcing data and measured streamflow
        if forcing and self.forcing_data_fetched:
            reset_index_at_the_end = False
            if self.forcing_df.index.name != self.time_column_name:
                self.forcing_df.set_index(self.time_column_name, inplace=True)
                reset_index_at_the_end = True

            # Precipitation
            column_to_draw = kwargs.get('precipitation_df_column_to_draw', 'precipitation')
            N_max = self.forcing_df[column_to_draw].max()
            fig.add_trace(go.Bar(x=self.forcing_df.index,
                                 y=self.forcing_df[column_to_draw],
                                 name="Precipitation", marker_color='red',
                                 showlegend=False),
                          row=1, col=1)

            # Temperature
            column_to_draw = kwargs.get('temperature_df_column_to_draw', 'temperature')
            fig.add_trace(go.Scatter(x=self.forcing_df.index,
                                     y=self.forcing_df[column_to_draw],
                                     name="Temperature", line_color='blue', mode='lines',        
                                     showlegend=False),
                          row=2, col=1)

            if reset_index_at_the_end:
                self.forcing_df.reset_index(inplace=True)
                self.forcing_df.rename(columns={self.forcing_df.index.name: self.time_column_name}, inplace=True)

            # Streamflow - hard-coded for HBV
            streamflow_df = self._get_measured_streamflow(time_column_name="TimeStamp",
                                                          streamflow_column_name="streamflow")
            column_to_draw = kwargs.get('streamflow_df_column_to_draw', 'streamflow')
            reset_index_at_the_end = False
            if streamflow_df.index.name != self.time_column_name:
                streamflow_df.set_index(self.time_column_name, inplace=True)
                reset_index_at_the_end = True
            fig.add_trace(go.Scatter(x=streamflow_df.index,
                                     y=streamflow_df[column_to_draw],
                                     name="Obs Streamflow", line_color='red', mode='lines'),
                          row=3, col=1)
            if reset_index_at_the_end:
                streamflow_df.reset_index(inplace=True)
                streamflow_df.rename(columns={streamflow_df.index.name: self.time_column_name}, inplace=True)

        
        local_colors = [ '#0072B2', '#E69F00', '#CC79A7', '#009E73', ]

        dict_plot_rows = dict()

        if single_qoi_column in self.list_original_model_output_columns:
            if measured and self.measured_fetched and self.df_measured is not None and not self.df_measured.empty:
                # column_to_draw = self.list_qoi_column_measured[idx]
                # timestamp_column = kwargs.get('measured_df_timestamp_column', 'TimeStamp')
                df_measured_subset = self.df_measured.loc[self.df_measured["qoi"] == single_qoi_column]
                if not df_measured_subset.empty:
                    column_to_draw = "measured"
                    timestamp_column = self.time_column_name
                    fig.add_trace(
                        go.Scatter(x=df_measured_subset[timestamp_column], y=df_measured_subset[column_to_draw],
                                   name=f"{single_qoi_column} (measured)", line_color='red', mode='lines',
                                   line=dict(color='green'), showlegend=True),
                        #line_color='red'
                        row=starting_row, col=1)

        if "E" in dict_time_vs_qoi_stat[keyIter[0]]:
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                    y=[dict_time_vs_qoi_stat[key]["E"] for key in keyIter],
                                    name=f'E[{single_qoi_column}]',
                                    line=dict(color=local_colors[0]), mode='lines'),
                        # line_color='green'
                        row=starting_row, col=1)

        if dict_what_to_plot.get("E_minus_std", False) and "StdDev" in dict_time_vs_qoi_stat[keyIter[0]] and "E" in dict_time_vs_qoi_stat[keyIter[0]]:
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[(dict_time_vs_qoi_stat[key]["E"] \
                                         - dict_time_vs_qoi_stat[key]["StdDev"]) for key in keyIter],
                                     name='mean - std. dev', mode='lines', showlegend=False, line_color='rgba(200, 200, 200, 0.4)',
                                    #  line_color='darkviolet'
                                     ),
                          row=starting_row, col=1)
        if dict_what_to_plot.get("E_plus_std", False) and "StdDev" in dict_time_vs_qoi_stat[keyIter[0]] and "E" in dict_time_vs_qoi_stat[keyIter[0]]:
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[(dict_time_vs_qoi_stat[key]["E"] + \
                                         dict_time_vs_qoi_stat[key]["StdDev"]) for key in keyIter],
                                     name='mean +- std. dev', mode='lines', fill='tonexty', showlegend=True, line_color='rgba(200, 200, 200, 0.4)',
                                    #  line_color='darkviolet'
                                     ),
                          row=starting_row, col=1)
        if dict_what_to_plot.get("E_minus_2std", False) and "StdDev" in dict_time_vs_qoi_stat[keyIter[0]] and "E" in dict_time_vs_qoi_stat[keyIter[0]]:
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                        y=[(dict_time_vs_qoi_stat[key]["E"] \
                                            - 2*dict_time_vs_qoi_stat[key]["StdDev"]) for key in keyIter],
                                            name='mean - 2*std. dev', mode='lines', showlegend=False, line_color='rgba(200, 200, 200, 0.4)',),
                            row=starting_row, col=1)
        if dict_what_to_plot.get("E_plus_2std", False) and "StdDev" in dict_time_vs_qoi_stat[keyIter[0]] and "E" in dict_time_vs_qoi_stat[keyIter[0]]:
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                        y=[(dict_time_vs_qoi_stat[key]["E"] +\
                                            2*dict_time_vs_qoi_stat[key]["StdDev"]) for key in keyIter],
                                            name='mean +- 2*std. dev', mode='lines', fill='tonexty', showlegend=True, line_color='rgba(200, 200, 200, 0.4)',),
                            row=starting_row, col=1)

        # 'rgba(255, 165, 0, 0.3)'
        if "P10" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("P10", False):
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[dict_time_vs_qoi_stat[key]["P10"] for key in keyIter],
                                     name='10th percentile', line_color='rgba(128,128,128, 0.3)', showlegend=False, mode='lines'),
                          row=starting_row, col=1)
        if "P90" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("P90", False):
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[dict_time_vs_qoi_stat[key]["P90"] for key in keyIter],
                                     name='10th percentile-90th percentile', mode='lines', fill='tonexty', showlegend=True, line=dict(color='rgba(128,128,128, 0.3)'), fillcolor='rgba(128,128,128, 0.3)'),
                                    #  line_color='yellow',
                          row=starting_row, col=1)
        dict_plot_rows["qoi"] = starting_row
        starting_row += 1

        # fig.add_trace(go.Scatter(x=pdTimesteps,
        #                          y=[dict_time_vs_qoi_stat[key]["StdDev"] for key in keyIter],
        #                          name='std. dev', line_color='darkviolet', mode='lines'),
        #               row=starting_row, col=1)
        # dict_plot_rows["StdDev"] = starting_row
        # starting_row += 1

        if "StdDev" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("StdDev", False):
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[dict_time_vs_qoi_stat[key]["StdDev"] for key in keyIter],
                                     name='std. dev', line_color='darkviolet', mode='lines'),
                          row=starting_row, col=1)
            dict_plot_rows["StdDev"] = starting_row
            starting_row += 1

        if "Skew" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Skew", False):
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[dict_time_vs_qoi_stat[key]["Skew"] for key in keyIter],
                                     name='Skew', mode='lines', ),
                          row=starting_row, col=1)
            dict_plot_rows["Skew"] = starting_row
            starting_row += 1

        if "Kurt" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Kurt", False):
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[dict_time_vs_qoi_stat[key]["Kurt"] for key in keyIter],
                                     name='Kurt', mode='lines', ),
                          row=starting_row, col=1)
            dict_plot_rows["Kurt"] = starting_row
            starting_row += 1

        if "Sobol_m" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_m", False):
            for i in range(len(self.labels)):
                name = self.labels[i] + "_" + single_qoi_column + "_S_m"
                fig.add_trace(go.Scatter(
                    x=pdTimesteps,
                    y=[dict_time_vs_qoi_stat[key]["Sobol_m"][i] for key in keyIter],
                    name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                    row=starting_row, col=1)
            dict_plot_rows["Sobol_m"] = starting_row
            starting_row += 1

        if "Sobol_m2" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_m2", False):
            for i in range(len(self.labels)):
                name = self.labels[i] + "_" + single_qoi_column + "_S_m"
                fig.add_trace(go.Scatter(
                    x=pdTimesteps,
                    y=[dict_time_vs_qoi_stat[key]["Sobol_m2"][i] for key in keyIter],
                    name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                    row=starting_row, col=1)
            dict_plot_rows["Sobol_m2"] = starting_row
            starting_row += 1

        if "Sobol_t" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_t", False):
            for i in range(len(self.labels)):
                name = self.labels[i] + "_" + single_qoi_column + "_S_t"
                fig.add_trace(go.Scatter(
                    x=pdTimesteps,
                    y=[dict_time_vs_qoi_stat[key]["Sobol_t"][i] for key in keyIter],
                    name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                    row=starting_row, col=1)
            dict_plot_rows["Sobol_t"] = starting_row
            starting_row += 1

        plot_generalized_sobol_indices = False
        for key in dict_time_vs_qoi_stat[keyIter[-1]].keys():
            if key.startswith("generalized_sobol_total_index_"):
                plot_generalized_sobol_indices = True
                break
        if plot_generalized_sobol_indices:  
            for i in range(len(self.labels)):
                name = f"generalized_sobol_total_index_{self.labels[i]}"
                y = []
                for key in keyIter:
                    if name in dict_time_vs_qoi_stat[key]:
                        y.append(dict_time_vs_qoi_stat[key][name])
                if len(y)==1: 
                    y = [y[0],]*len(keyIter)  #[y[0],]*len(pdTimesteps)
                # if self.compute_generalized_sobol_indices_over_time:
                #     y = [dict_time_vs_qoi_stat[key][name] for key in keyIter]
                # else:
                #     y = [dict_time_vs_qoi_stat[keyIter[-1]][name]]*len(keyIter)
                name = self.labels[i] + "_" + single_qoi_column + "generalized_S_t"
                fig.add_trace(go.Scatter(
                    x=pdTimesteps,
                    y=y,
                    name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                    row=starting_row, col=1)
            dict_plot_rows["generalized_sobol_total_index"] = starting_row
            starting_row += 1

        # fig.update_traces(mode='lines')
        if forcing and self.forcing_data_fetched:
            fig.update_yaxes(title_text="Precipitation [mm/day]", showgrid=True, row=1, col=1)
            fig.update_yaxes(autorange="reversed", row=1, col=1)
            fig.update_yaxes(title_text="Temperature [°C]", showgrid=True, row=2, col=1)
            fig.update_yaxes(title_text="Measured Streamflow [m^3/s]", showgrid=True, row=3, col=1)

        fig.update_yaxes(title_text=single_qoi_column, showgrid=True, side='left',
                         row=dict_plot_rows["qoi"], col=1)
        # fig.update_yaxes(title_text=f"Std. Dev. [{single_qoi_column}]", side='left', showgrid=True,
        #                  row=starting_row+1, col=1)
        if "StdDev" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("StdDev", False):
            fig.update_yaxes(title_text=f"StdDev", showgrid=True,
                             row=dict_plot_rows["StdDev"], col=1)
        if "Skew" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Skew", False):
            fig.update_yaxes(title_text=f"Skew", showgrid=True,
                             row=dict_plot_rows["Skew"], col=1)
        if "Kurt" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Kurt", False):
            fig.update_yaxes(title_text=f"Kurt", showgrid=True,
                             row=dict_plot_rows["Kurt"], col=1)
        if "Sobol_m" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_m", False):
            fig.update_yaxes(title_text=f"F. SI", showgrid=True, range=[0, 1],
                             row=dict_plot_rows["Sobol_m"], col=1)
        if "Sobol_m2" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_m2", False):
            fig.update_yaxes(title_text=f"S. SI", showgrid=True, range=[0, 1],
                             row=dict_plot_rows["Sobol_m2"], col=1)
        if "Sobol_t" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_t", False):
            fig.update_yaxes(title_text=f"T. SI", showgrid=True, range=[0, 1],
                             row=dict_plot_rows["Sobol_t"], col=1)
        if plot_generalized_sobol_indices and dict_plot_rows.get("generalized_sobol_total_index", False):
            fig.update_yaxes(title_text=f"Gener. T. SI", showgrid=True, range=[0, 1],
                             row=dict_plot_rows["generalized_sobol_total_index"], col=1)
            
        fig.update_layout(title_text=window_title)
        fig.update_layout(
            xaxis=dict(
                rangemode='normal',
                range=[timesteps_min, timesteps_max],
                type="date"
            ),
            yaxis=dict(
                rangemode='normal',  # Ensures the range is not padded for markers
                autorange=True       # Auto-range is enabled
            )
        )
        fig.update_layout(height=1100, width=1200)
        fig.update_layout(
                # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                showlegend=True,
                # template="plotly_white",
            )
        fig.update_xaxes(
            tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
            dtick="M1"                     # Set tick interval to 1 day for denser ticks
        )
        print(f"[HVB STAT INFO] _plotStatisticsDict_plotly function for Qoi-{single_qoi_column} is almost over, just to save the plot!")

        # filename = pathlib.Path(filename)
        fig.update_layout(
            margin=dict(
                t=10,  # Top margin
                b=10,  # Bottom margin
                l=20,  # Left margin
                r=20   # Right margin
            )
        )
        plot(fig, filename=filename, auto_open=display)
        new_filename = pathlib.Path(filename).with_suffix(".pdf")
        # fig.write_image(str(new_filename), format="pdf", width=1200,)
        return fig

    def plot_forcing_mean_predicted_and_observed_all_qoi(self, directory="./", fileName="simulation_big_plot.html"):
        measured_columns_names_set = set()
        for single_qoi in self.list_qoi_column:
            measured_columns_names_set.add(self.dict_corresponding_original_qoi_column[single_qoi])

        total_number_of_rows = 2 + len(self.list_qoi_column) + len(measured_columns_names_set)
        fig = make_subplots(
            rows=total_number_of_rows, cols=1, shared_xaxes=True
            #     subplot_titles=tuple(self.list_qoi_column)
        )
        n_row = 3

        fig.add_trace(
            go.Bar(
                x=self.forcing_df.index, y=self.forcing_df['precipitation'],
                text=self.forcing_df['precipitation'],
                name="Precipitation"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.forcing_df.index, y=self.forcing_df['temperature'],
                text=self.forcing_df['temperature'],
                name="Temperature", mode='lines'
            ),
            row=2, col=1
        )

        if self.df_statistics is None or self.df_statistics.empty:
            raise Exception(f"You are trying to call a plotting utiltiy function whereas "
                            f"self.df_statistics object is still not computed - make sure to first call"
                            f"self.create_df_from_statistics_data")

        measured_columns_names_set = set()
        for single_qoi in self.list_qoi_column:
            df_statistics_single_qoi = self.df_statistics.loc[
                self.df_statistics['qoi'] == single_qoi]
            corresponding_measured_column = self.dict_corresponding_original_qoi_column[single_qoi]
            if corresponding_measured_column not in measured_columns_names_set:
                measured_columns_names_set.add(corresponding_measured_column)
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
        fig.update_xaxes(
            tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
            dtick="M1"                     # Set tick interval to 1 day for denser ticks
        )
        if not str(directory).endswith("/"):
            directory = str(directory) + "/"
        fileName = directory + fileName
        plot(fig, filename=fileName)
        return fig

    def get_info_for_plotting_forcing_data(self, **kwargs):
        return {
            "n_rows": 2,
            "subplot_titles": ["Temperature [°C]", "Precipitation [mm/day]",]
        }

    def plot_forcing_data(self, df: pd.DataFrame=None, fig=None, add_to_subplot=False, **kwargs):
        if df is None:
            df = self.df_forcing

        if df is None or df.empty:
            print(f"[STAT INFO] DF subset storing forcing data is empty!")
            return

        if fig is None:
            add_to_subplot = False

        if add_to_subplot:
            n_rows=kwargs.get("n_rows", 1)
            n_col=kwargs.get("n_col", 1)
        else:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            n_rows = 1
            n_col = 1

        reset_index_in_the_end = False
        if df.index.name != self.time_column_name:
            df.set_index(self.time_column_name, inplace=True)
            reset_index_in_the_end = True

        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df[self.temperature_column_name],
                text=df[self.temperature_column_name],
                name="Temperature", mode='lines+markers',
                showlegend=False
                # marker_color='blue'
            ), row=n_rows, col=n_col
        )

        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df[self.precipitation_column_name],
                text=df[self.precipitation_column_name],
                name="Precipitation",
                showlegend=False,
                marker_color='red'
                # mode="lines",
                #         line=dict(
                #             color='LightSkyBlue')
            ), row=n_rows+1,col=n_col
        )
        fig.update_yaxes(autorange="reversed", row=2, col=1)
        
        if reset_index_in_the_end:
            df.reset_index(inplace=True)
            df.rename(columns={df.index.name: self.time_column_name}, inplace=True)

        if not add_to_subplot:
            return fig