from collections import defaultdict
import dill
import json
import numpy as np
import pathlib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from distutils.util import strtobool
import time

from common import utility
# import hbvsask_utility as hbv
from hbv_sask import hbvsask_utility as hbv


class HBVSASKModel(object):
    def __init__(self, configurationObject, inputModelDir, workingDir=None, *args, **kwargs):
        if isinstance(configurationObject, dict):
            self.configurationObject = configurationObject
        else:
            with open(configurationObject) as f:
                self.configurationObject = json.load(f)

        if "basis" in kwargs:
            self.basis = kwargs['basis']
        else:
            self.basis = self.configurationObject["model_settings"].get("basis", 'Oldman_Basin')

        self.inputModelDir = pathlib.Path(inputModelDir)
        self.inputModelDir_basis = self.inputModelDir / self.basis

        if workingDir is None:
            workingDir = self.inputModelDir
        self.workingDir = pathlib.Path(workingDir)
        self.workingDir.mkdir(parents=True, exist_ok=True)

        self._setup(**kwargs)

    # TODO Think of moving this _setup into the separate class, e.g., (Larsim)Configurations
    def _setup(self, **kwargs):

        if "run_full_timespan" in kwargs:
            self.run_full_timespan = kwargs['run_full_timespan']
        else:
            self.run_full_timespan = strtobool(self.configurationObject["time_settings"].get("run_full_timespan", 'False'))

        if "writing_results_to_a_file" in kwargs:
            self.writing_results_to_a_file = kwargs['writing_results_to_a_file']
        else:
            self.writing_results_to_a_file = strtobool(self.configurationObject["model_settings"].get(
                "writing_results_to_a_file", True))

        if "plotting" in kwargs:
            self.plotting = kwargs['plotting']
        else:
            self.plotting = strtobool(self.configurationObject["model_settings"].get("plotting", True))

        ######################################################################################################

        # self.initial_condition_file = self.inputModelDir_basis / "initial_condition.inp"
        # initial_condition_file = kwargs.get("initial_condition_file", "state_df.pkl")
        initial_condition_file = kwargs.get("initial_condition_file", "state_const_df.pkl")
        if self.run_full_timespan:  # TODO
            initial_condition_file = kwargs.get("initial_condition_file", "state_const_df.pkl")
        monthly_data_inp = kwargs.get("monthly_data_inp", "monthly_data.inp")
        precipitation_temperature_inp = kwargs.get("precipitation_temperature_inp", "Precipitation_Temperature.inp")
        streamflow_inp = kwargs.get("streamflow_inp", "streamflow.inp")
        factorSpace_txt = kwargs.get("factorSpace_txt", "factorSpace.txt")

        self.initial_condition_file = self.inputModelDir_basis / initial_condition_file
        self.monthly_data_inp = self.inputModelDir_basis / monthly_data_inp
        self.precipitation_temperature_inp = self.inputModelDir_basis / precipitation_temperature_inp
        self.streamflow_inp = self.inputModelDir_basis / streamflow_inp
        self.factorSpace_txt = self.inputModelDir / factorSpace_txt

        self.time_column_name = kwargs.get("time_column_name", "TimeStamp")
        self.streamflow_column_name = kwargs.get("streamflow_column_name", "streamflow")
        self.precipitation_column_name = kwargs.get("precipitation_column_name", "precipitation")
        self.temperature_column_name = kwargs.get("temperature_column_name", "temperature")
        self.long_term_precipitation_column_name = kwargs.get("long_term_precipitation_column_name", "monthly_average_PE")
        self.long_term_temperature_column_name = kwargs.get("long_term_temperature_column_name", "monthly_average_T")

        ######################################################################################################

        # these set of control variables are for UQEF & UQEFPP framework...
        self.uq_method = kwargs.get('uq_method', None)
        self.raise_exception_on_model_break = kwargs.get('raise_exception_on_model_break', False)
        if self.uq_method is not None and self.uq_method == "sc":  # always break when running gPCE simulation
            self.raise_exception_on_model_break = True
        self.disable_statistics = kwargs.get('disable_statistics', False)

        ######################################################################################################
        # TODO Update code with options that self.read_measured_data (self.read_measured_streamflow), self.qoi_column_measured
        # TODO self.qoi, self.qoi_column can be lists as well, self.calculate_GoF

        dict_config_simulation_settings = self.configurationObject["simulation_settings"]

        # self.qoi, self.qoi_column, self.multiple_qoi, self.number_of_qois,
        # self.read_measured_data, self.qoi_column_measured, self.read_measured_streamflow,
        # self.streamflow_column_name, self.calculate_GoF, self.list_calculate_GoF,
        # self.objective_function, self.objective_function_qoi, self.objective_function_names_qoi, \
        # self.list_objective_function_qoi, self.list_qoi_column, self.list_qoi_column_measured,
        # self.list_read_measured_data, self.compute_gradients, self.CD

        if "qoi" in kwargs:
            self.qoi = kwargs['qoi']
        else:
            self.qoi = dict_config_simulation_settings.get("qoi", "Q_cms")

        if "qoi_column" in kwargs:
            self.qoi_column = kwargs['qoi_column']
        else:
            self.qoi_column = dict_config_simulation_settings.get("qoi_column", "Q_cms")

        self.multiple_qoi = False
        self.number_of_qois = 1
        if isinstance(self.qoi, list) or (self.qoi == "GoF" and isinstance(self.qoi_column, list)):
            self.multiple_qoi = True
            self.number_of_qois = len(self.qoi_column)

        if "read_measured_data" in kwargs:
            self.read_measured_data = kwargs['read_measured_data']
        else:
            if self.multiple_qoi:
                self.read_measured_data = []
                try:
                    temp = dict_config_simulation_settings["read_measured_data"]
                except KeyError:
                    temp = ["False"]*self.number_of_qois
                for i in range(self.number_of_qois):
                    self.read_measured_data[i] = strtobool(temp[i])
            else:
                self.read_measured_data = strtobool(dict_config_simulation_settings.get("read_measured_data", False))

        if "qoi_column_measured" in kwargs:
            self.qoi_column_measured = kwargs['qoi_column_measured']
        else:
            if self.multiple_qoi:
                try:
                    self.qoi_column_measured = dict_config_simulation_settings["qoi_column_measured"]
                    for idx, single_qoi_column_measured in enumerate(self.qoi_column_measured):
                        if single_qoi_column_measured == "None":
                            self.qoi_column_measured[idx] = None
                except KeyError:
                    self.qoi_column_measured = [None]*self.number_of_qois
            else:
                self.qoi_column_measured = dict_config_simulation_settings.get("qoi_column_measured", "streamflow")
                if self.qoi_column_measured == "None":
                    self.qoi_column_measured = None

        if self.multiple_qoi:
            for idx, single_read_measured_data in enumerate(self.read_measured_data):
                if single_read_measured_data and self.qoi_column_measured[idx] is None:
                    # raise ValueError
                    self.read_measured_data[idx] = False
        else:
            if self.read_measured_data and self.qoi_column_measured is None:
                # raise ValueError
                self.read_measured_data = False

        if self.multiple_qoi:
            assert len(self.read_measured_data) == len(self.qoi)
            assert len(self.read_measured_data) == len(self.qoi_column_measured)

        # streamflow is of special importance here, since we have saved/measured/ground truth that for it and it is inside input data
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

        self.calculate_GoF = strtobool(dict_config_simulation_settings.get("calculate_GoF", False))
        # self.calculate_GoF has to follow the self.read_measured_data which tells if ground truth data for that qoi is available
        if self.multiple_qoi and self.calculate_GoF:
            self.list_calculate_GoF = []
            for idx, single_read_measured_data in enumerate(self.read_measured_data):
                self.list_calculate_GoF[idx] = single_read_measured_data
        else:
            self.list_calculate_GoF = self.read_measured_data

        self.objective_function = dict_config_simulation_settings.get("objective_function", [])
        self.compute_gradients = strtobool(dict_config_simulation_settings.get("compute_gradients", False))

        self.objective_function_qoi = None
        self.objective_function_names_qoi = None
        if self.qoi == "GoF":
            # take only those Outputs of Interest that have measured data
            if self.multiple_qoi:
                updated_qoi_column = []
                updated_qoi_column_measured = []
                for idx, single_qoi_column in enumerate(self.qoi_column):
                    if self.read_measured_data[idx]:
                        updated_qoi_column.append(single_qoi_column)
                        updated_qoi_column_measured.append(self.qoi_column_measured[idx])
                self.qoi_column = updated_qoi_column
                self.qoi_column_measured = updated_qoi_column_measured
            else:
                if not self.read_measured_data:
                    raise ValueError
            self.objective_function_qoi = dict_config_simulation_settings.get("objective_function_qoi", "all")
            self.objective_function_qoi = utility.gof_list_to_function_names(
                self.objective_function_qoi)
            if isinstance(self.objective_function_qoi, list):
                self.objective_function_names_qoi = [
                    single_gof.__name__ if callable(single_gof) else single_gof \
                    for single_gof in self.objective_function_qoi]
                self.list_objective_function_qoi = self.objective_function_qoi
            else:
                self.list_objective_function_qoi = [self.objective_function_qoi,]
                if callable(self.objective_function_qoi):
                    self.objective_function_names_qoi = self.objective_function_qoi.__name__
                else:
                    self.objective_function_names_qoi = self.objective_function_qoi

        # Create a list version of some configuration parameters which might be needed when computing GoF
        if self.qoi == "GoF" or self.calculate_GoF:
            if not isinstance(self.qoi_column, list):
                self.list_qoi_column = [self.qoi_column, ]
            else:
                self.list_qoi_column = self.qoi_column
            if not isinstance(self.qoi_column_measured, list):
                self.list_qoi_column_measured = [self.qoi_column_measured, ]
            else:
                self.list_qoi_column_measured = self.qoi_column_measured
            if not isinstance(self.read_measured_data, list):
                self.list_read_measured_data = [self.read_measured_data, ]
            else:
                self.list_read_measured_data = self.read_measured_data

        if self.compute_gradients:
            gradients_method = dict_config_simulation_settings.get("gradients_method", "Forward Difference")
            if gradients_method == "Central Difference":
                self.CD = 1  # flag for using Central Differences (with 2 * num_evaluations)
            elif gradients_method == "Forward Difference":
                self.CD = 0  # flag for using Forward Differences (with num_evaluations)
            else:
                raise Exception(f"NUMERICAL GRADIENT EVALUATION ERROR: "
                                f"Only \"Central Difference\" and \"Forward Difference\" supported")
            self.eps_val_global = dict_config_simulation_settings.get("eps_gradients", 1e-4)

            self.compute_active_subspaces = strtobool(
                dict_config_simulation_settings.get("compute_active_subspaces", "False"))
            self.save_gradient_related_runs = strtobool(
                dict_config_simulation_settings.get("save_gradient_related_runs", "False"))

        ######################################################################################################

        self._timespan_setup(**kwargs)
        self._input_data_setup(time_column_name=self.time_column_name,
                               precipitation_column_name=self.precipitation_column_name,
                               temperature_column_name=self.temperature_column_name,
                               long_term_precipitation_column_name=self.long_term_precipitation_column_name,
                               long_term_temperature_column_name=self.long_term_temperature_column_name,
                               read_measured_streamflow=self.read_measured_streamflow,
                               streamflow_column_name=self.streamflow_column_name)

        if self.plotting:
            figure = self._plot_input_data(time_column_name=self.time_column_name,
                                           precipitation_column_name=self.precipitation_column_name,
                                           temperature_column_name=self.temperature_column_name,
                                           read_measured_streamflow=self.read_measured_streamflow,
                                           streamflow_column_name=self.streamflow_column_name)

    def _timespan_setup(self, **kwargs):
        if self.run_full_timespan:
            self.start_date, self.end_date = hbv._get_full_time_span(self.basis)
        else:
            try:
                self.start_date = pd.Timestamp(
                    year=self.configurationObject["time_settings"]["start_year"],
                    month=self.configurationObject["time_settings"]["start_month"],
                    day=self.configurationObject["time_settings"]["start_day"],
                    hour=self.configurationObject["time_settings"].get("start_hour", 0)
                )
                self.end_date = pd.Timestamp(
                    year=self.configurationObject["time_settings"]["end_year"],
                    month=self.configurationObject["time_settings"]["end_month"],
                    day=self.configurationObject["time_settings"]["end_day"],
                    hour=self.configurationObject["time_settings"].get("end_hour", 0)
                )
            except KeyError:
                self.start_date, self.end_date = hbv._get_full_time_span(self.basis)

        if "spin_up_length" in kwargs:
            self.spin_up_length = kwargs["spin_up_length"]
        else:
            try:
                self.spin_up_length = self.configurationObject["time_settings"]["spin_up_length"]
            except KeyError:
                self.spin_up_length = 0  # 365*3

        # note: one has to omit simulation_length both from kwargs and configurationObject
        # if you want that run_full_timespan has an effect
        if "simulation_length" in kwargs:
            self.simulation_length = kwargs["simulation_length"]
        else:
            try:
                self.simulation_length = self.configurationObject["time_settings"]["simulation_length"]
            except KeyError:
                self.simulation_length = (self.end_date - self.start_date).days - self.spin_up_length
                if self.simulation_length <= 0:
                    self.simulation_length = 365

        self.start_date_predictions = pd.to_datetime(self.start_date) + pd.DateOffset(days=self.spin_up_length)
        self.end_date = pd.to_datetime(self.start_date_predictions) + pd.DateOffset(days=self.simulation_length)
        self.full_data_range = pd.date_range(start=self.start_date, end=self.end_date, freq="1D")
        self.simulation_range = pd.date_range(start=self.start_date_predictions, end=self.end_date, freq="1D")

        self.start_date = pd.Timestamp(self.start_date)
        self.end_date = pd.Timestamp(self.end_date)
        self.start_date_predictions = pd.Timestamp(self.start_date_predictions)

        # print(f"start_date-{self.start_date}; spin_up_length-{self.spin_up_length};
        # start_date_predictions-{self.start_date_predictions}")
        # print(
        #     f"start_date_predictions-{self.start_date_predictions}; simulation_length-{self.simulation_length}; end_date-{self.end_date}")
        # print(len(self.simulation_range), (self.end_date - self.start_date_predictions).days)
        # assert len(self.time_series_measured_data_df[self.start_date:self.end_date]) == len(self.full_data_range)

    def _input_data_setup(self, time_column_name="TimeStamp", precipitation_column_name="precipitation",
                          temperature_column_name="temperature",
                          long_term_precipitation_column_name="monthly_average_PE",
                          long_term_temperature_column_name="monthly_average_T", read_measured_streamflow=True,
                          streamflow_column_name="streamflow"):
        # Reading the input data

        self.precipitation_temperature_df = hbv.read_precipitation_temperature(
            self.precipitation_temperature_inp, time_column_name=time_column_name,
            precipitation_column_name=precipitation_column_name, temperature_column_name=temperature_column_name
        )

        if read_measured_streamflow:
            self.streamflow_df = hbv.read_streamflow(
                self.streamflow_inp, time_column_name=time_column_name, streamflow_column_name=streamflow_column_name
            )
            self.time_series_measured_data_df = pd.merge(
                self.streamflow_df, self.precipitation_temperature_df,  left_index=True, right_index=True
            )
        else:
            self.time_series_measured_data_df = self.precipitation_temperature_df

        self.precipitation_temperature_monthly_df = hbv.read_long_term_data(
            self.monthly_data_inp, time_column_name=time_column_name,
            precipitation_column_name=long_term_precipitation_column_name,
            temperature_column_name=long_term_temperature_column_name
        )
        self.param_setup_dict = hbv.read_param_setup_dict(self.factorSpace_txt)

        # Parse input based on some timeframe
        if time_column_name in self.time_series_measured_data_df.columns:
            self.time_series_measured_data_df = self.time_series_measured_data_df.loc[
                (self.time_series_measured_data_df[time_column_name] >= self.start_date) & (self.time_series_measured_data_df[time_column_name] <= self.end_date)]
        else:
            self.time_series_measured_data_df = self.time_series_measured_data_df[self.start_date:self.end_date]

        # self.initial_condition_df = read_initial_conditions(self.initial_condition_file, return_dict_or_df="df")
        self.initial_condition_df = hbv.read_initial_conditions(
            self.initial_condition_file, timestamp=self.start_date, time_column_name=time_column_name)
        # print(self.initial_condition_df)

        # self.default_par_values_dict = {'TT': 0.0, 'C0': 5.0, 'ETF': 0.5, 'LP': 0.5, 'FC': 100,
        #                                 'beta': 2.0, 'FRAC': 0.5, 'K1': 0.5, 'alpha': 2.0, 'K2': 0.025,
        #                                 'UBAS': 1, 'PM': 1}

    def _plot_input_data(self, time_column_name="TimeStamp", precipitation_column_name="precipitation",
                         temperature_column_name="temperature", read_measured_streamflow=True,
                         streamflow_column_name="streamflow"):
        if self.time_series_measured_data_df is None:
            return

        if read_measured_streamflow:
            n_rows = 3
        else:
            n_rows = 2
        fig = make_subplots(rows=n_rows, cols=1)

        if time_column_name in self.time_series_measured_data_df.columns:
            fig.add_trace(
                go.Scatter(x=self.time_series_measured_data_df[time_column_name], y=self.time_series_measured_data_df[precipitation_column_name],
                           name="P"), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=self.time_series_measured_data_df[time_column_name], y=self.time_series_measured_data_df[temperature_column_name],
                           name="T"), row=2, col=1)
            if read_measured_streamflow:
                fig.add_trace(
                    go.Scatter(x=self.time_series_measured_data_df[time_column_name], y=self.time_series_measured_data_df[streamflow_column_name], name="Q_cms"), row=3, col=1)
        else:
            fig.add_trace(
                go.Scatter(x=self.time_series_measured_data_df.index, y=self.time_series_measured_data_df[precipitation_column_name],
                           name="P"), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=self.time_series_measured_data_df.index, y=self.time_series_measured_data_df[temperature_column_name],
                           name="T"), row=2, col=1)
            if read_measured_streamflow:
                fig.add_trace(
                    go.Scatter(x=self.time_series_measured_data_df.index, y=self.time_series_measured_data_df[streamflow_column_name], name="Q_cms"), row=3, col=1)

        plot_filename = self.workingDir / f"forcing_data.html"
        plot(fig, filename=str(plot_filename), auto_open=False)
        return fig

    def prepare(self, *args, **kwargs):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def timesteps(self):
        return list(self.full_data_range)

    def run(self, i_s=[0, ], parameters=None, raise_exception_on_model_break=None, *args, **kwargs):
        print(f"[HVBSASK INFO] {i_s} parameter: {parameters}")

        if raise_exception_on_model_break is None:
            raise_exception_on_model_break = self.raise_exception_on_model_break
        take_direct_value = kwargs.get("take_direct_value", False)
        createNewFolder = kwargs.get("createNewFolder", False)
        deleteFolderAfterwards = kwargs.get("deleteFolderAfterwards", True)
        writing_results_to_a_file = kwargs.get("writing_results_to_a_file", self.writing_results_to_a_file)
        plotting = kwargs.get("plotting", self.plotting)

        # TODO Change this now when multiple QoI are supported
        merge_output_with_measured_data = kwargs.get("merge_output_with_measured_data", False)
        if not self.read_measured_data:
            merge_output_with_measured_data = False
        if self.calculate_GoF:
            merge_output_with_measured_data = True

        results_array = []
        for ip in range(0, len(i_s)):  # for each peace of work
            i = i_s[ip]  # i is unique index run

            if parameters is not None:
                parameter = parameters[ip]
            else:
                parameter = None  # an unaltered run will be executed

            id_dict = {"index_run": i}

            parameters_dict = hbv.parameters_configuration(
                parameters=parameter,
                configurationObject=self.configurationObject,
                take_direct_value=take_direct_value
            )

            start = time.time()

            # create local directory for this particular run
            if createNewFolder:
                curr_working_dir = self.workingDir / f"run_{i}"
                curr_working_dir.mkdir(parents=True, exist_ok=True)
            else:
                curr_working_dir = self.workingDir

            # Running the model
            flux, state = hbv.HBV_SASK(
                forcing=self.time_series_measured_data_df,
                long_term=self.precipitation_temperature_monthly_df,
                par_values_dict=parameters_dict,
                initial_condition_df=self.initial_condition_df,
                printing=False,
                time_column_name=self.time_column_name,
                precipitation_column_name=self.precipitation_column_name,
                temperature_column_name=self.temperature_column_name,
                long_term_precipitation_column_name=self.long_term_precipitation_column_name,
                long_term_temperature_column_name=self.long_term_temperature_column_name
            )

            ######################################################################################################
            # Processing model output
            ######################################################################################################

            time_series_list = list(self.full_data_range)  # list(self.simulation_range)
            last_date = time_series_list[-1]
            time_series_list_plus_one_day = time_series_list.copy()
            time_series_list_plus_one_day.append(pd.to_datetime(last_date) + pd.DateOffset(days=1))

            assert len(list(self.full_data_range)) == len(flux["Q_cms"])

            # Create a final df - flux
            flux_df = pd.DataFrame(
                list(zip(time_series_list, flux["Q_cms"], flux["Q_mm"], flux["AET"], flux["PET"], flux["Q1"],
                         flux["Q1_routed"],
                         flux["Q2"], flux["ponding"])),
                columns=[self.time_column_name, 'Q_cms', 'Q_mm', 'AET', 'PET', 'Q1', 'Q1_routed', 'Q2', "ponding"]
            )
            flux_df['Index_run'] = i

            # Create a final df - state
            state_df = pd.DataFrame(
                list(zip(time_series_list_plus_one_day, state["SWE"], state["SMS"], state["S1"], state["S2"])),
                columns=[self.time_column_name, 'initial_SWE', 'initial_SMS', 'S1', 'S2', ]
            )
            state_df['WatershedArea_km2'] = self.initial_condition_df["WatershedArea_km2"].values[0]
            state_df['Index_run'] = i

            # Parse flux_df between start_date_predictions, end_date
            flux_df.set_index(self.time_column_name, inplace=True)
            flux_df = flux_df.loc[self.simulation_range]  # flux_df[self.start_date_predictions:self.end_date]

            # Append measured data to flux_df, i.e., merge flux_df and self.time_series_measured_data_df[self.qoi_column_measured]
            # TODO Change this now when multiple QoI are supported
            if merge_output_with_measured_data:
                flux_df = flux_df.merge(
                    self.time_series_measured_data_df[[self.qoi_column_measured, ]], left_index=True, right_index=True)

            # Parse state_df between start_date_predictions, end_date + 1
            state_df.set_index(self.time_column_name, inplace=True)
            state_df = state_df[self.start_date_predictions:]

            # reset the index
            flux_df.reset_index(inplace=True)
            flux_df.rename(columns={"index": self.time_column_name}, inplace=True)
            state_df.reset_index(inplace=True)
            state_df.rename(columns={"index": self.time_column_name}, inplace=True)

            ######################################################################################################

            index_run_and_parameters_dict = {**id_dict, **parameters_dict}

            index_parameter_gof_DF = None
            index_parameter_gof_dictof_DFs = None
            if self.calculate_GoF:
                if self.multiple_qoi:
                    index_parameter_gof_list_of_dicts = []
                    for idx, single_qoi_column in enumerate(self.list_qoi_column):
                        if self.list_calculate_GoF[idx] and self.list_read_measured_data[idx]:
                            # TODO instead of this create one big DF with column qoi
                            index_parameter_gof_dict_single_qoi = self._calculate_GoF(
                                measuredDF=self.time_series_measured_data_df, predictedDF=flux_df,
                                gof_list=self.objective_function,
                                measuredDF_time_column_name=self.time_column_name,
                                simulatedDF_time_column_name=self.time_column_name,
                                measuredDF_column_name=self.list_qoi_column_measured[idx],
                                simulatedDF_column_name=single_qoi_column,
                                parameters_dict=index_run_and_parameters_dict,
                                return_dict=True
                            )
                            index_parameter_gof_list_of_dicts.append(index_parameter_gof_dict_single_qoi)
                    index_parameter_gof_DF = pd.DataFrame(index_parameter_gof_list_of_dicts)
                else:
                    index_parameter_gof_DF = self._calculate_GoF(
                        measuredDF=self.time_series_measured_data_df, predictedDF=flux_df,
                        parameters_dict=index_run_and_parameters_dict
                    )

            # TODO - Different QoI (e.g., some likelihood); Different purpose - ActiveSubspaces
            # self.qoi = "GoF" | "Q" | ["Q_cms","AET"]
            # self.mode = "continuous" | "sliding_window" | "resampling"

            ######################################################################################################
            # Computing gradients
            ######################################################################################################

            # It makes sence to compute gradients only if once of the following flags are set to True
            if not self.save_gradient_related_runs and not self.compute_active_subspaces:
                self.compute_gradients = False

            if self.compute_gradients:
                h_vector = []
                gradient_vectors_dict = defaultdict(list)
                dict_of_grad_estimation_vector = defaultdict(list)
                gradient_vectors_param_dict = defaultdict(list)

                list_of_columns_to_filter = [self.time_column_name, ] + self.list_qoi_column

                dict_param_info_from_configurationObject = utility.get_param_info_dict_from_configurationObject(
                    self.configurationObject)

                # CD = 1 central differences; CD = 0 forward differences
                # Assumption: parameters_dict is a dictionary of parameters of interest already computed above
                parameter_index_to_perturb = 0  # this 0 id will mark the run with unchanged parameters vector
                for single_param_name, single_param_value in parameters_dict.items():
                    parameter_index_to_perturb += 1

                    updated_parameter_dict = parameters_dict.copy()

                    # 2.1 Update parameter value
                    single_dict_param_info = dict_param_info_from_configurationObject[single_param_name]
                    parameter_lower_limit = single_dict_param_info["lower_limit"]
                    parameter_upper_limit = single_dict_param_info["upper_limit"]
                    if parameter_lower_limit is None or parameter_upper_limit is None:
                        raise Exception(
                            'ERROR in computing a gradient of QoI wrt parameter: '
                            'parameter_lower_limit and/or parameter_upper_limit are not specified in configurationObject!')
                    else:
                        # TODO don't forget -param_h when self.CD
                        param_h = self.eps_val_global * (parameter_upper_limit - parameter_lower_limit)
                        # TODO is this necessary?
                        parameter_lower_limit += param_h
                        parameter_upper_limit -= param_h

                    updated_parameter_dict[single_param_name] = single_param_value + param_h

                    # 2.2 Run the model & 2.3. Do some postprocessing
                    flux_plus_h, _ = hbv.HBV_SASK(
                        forcing=self.time_series_measured_data_df,
                        long_term=self.precipitation_temperature_monthly_df,
                        par_values_dict=updated_parameter_dict,
                        initial_condition_df=self.initial_condition_df,
                        printing=False,
                        time_column_name=self.time_column_name,
                        precipitation_column_name=self.precipitation_column_name,
                        temperature_column_name=self.temperature_column_name,
                        long_term_precipitation_column_name=self.long_term_precipitation_column_name,
                        long_term_temperature_column_name=self.long_term_temperature_column_name
                    )
                    h = param_h

                    # Create a final df - flux with +dh parametr value
                    flux_plus_h_df = pd.DataFrame(
                        list(zip(time_series_list,
                                 flux_plus_h["Q_cms"], flux_plus_h["Q_mm"], flux_plus_h["AET"], flux_plus_h["PET"],
                                 flux_plus_h["Q1"], flux_plus_h["Q1_routed"], flux_plus_h["Q2"],
                                 flux_plus_h["ponding"])),
                        columns=[self.time_column_name, 'Q_cms', 'Q_mm', 'AET', 'PET', 'Q1', 'Q1_routed', 'Q2',
                                 "ponding"]
                    )
                    # Preparations before computing GoF
                    flux_plus_h_df = flux_plus_h_df[list_of_columns_to_filter]
                    flux_plus_h_df['Index_run'] = i
                    flux_plus_h_df["Parameter_index_to_perturb"] = parameter_index_to_perturb
                    flux_plus_h_df['Sub_index_run'] = 0
                    flux_plus_h_df = flux_plus_h_df[
                        flux_plus_h_df[self.time_column_name].isin(self.simulation_range)]

                    if self.CD:
                        updated_parameter_dict[single_param_name] = single_param_value - param_h

                        # Run the model for -dh
                        flux_minus_h, _ = hbv.HBV_SASK(
                            forcing=self.time_series_measured_data_df,
                            long_term=self.precipitation_temperature_monthly_df,
                            par_values_dict=updated_parameter_dict,
                            initial_condition_df=self.initial_condition_df,
                            printing=False,
                            time_column_name=self.time_column_name,
                            precipitation_column_name=self.precipitation_column_name,
                            temperature_column_name=self.temperature_column_name,
                            long_term_precipitation_column_name=self.long_term_precipitation_column_name,
                            long_term_temperature_column_name=self.long_term_temperature_column_name
                        )
                        h = 2*param_h

                        # Create a final df - flux with -dh parametr value
                        flux_minus_h_df = pd.DataFrame(
                            list(zip(time_series_list,
                                     flux_minus_h["Q_cms"], flux_minus_h["Q_mm"], flux_minus_h["AET"], flux_minus_h["PET"],
                                     flux_minus_h["Q1"], flux_minus_h["Q1_routed"], flux_minus_h["Q2"],
                                     flux_minus_h["ponding"])),
                            columns=[self.time_column_name, 'Q_cms', 'Q_mm', 'AET', 'PET', 'Q1', 'Q1_routed', 'Q2',
                                     "ponding"]
                        )
                        # Preparations before computing GoF
                        flux_minus_h_df = flux_minus_h_df[list_of_columns_to_filter]
                        flux_minus_h_df['Index_run'] = i
                        flux_minus_h_df["Parameter_index_to_perturb"] = parameter_index_to_perturb
                        flux_minus_h_df['Sub_index_run'] = 1
                        flux_minus_h_df = flux_minus_h_df[
                            flux_minus_h_df[self.time_column_name].isin(self.simulation_range)]

                    h_vector.append(h)

                    # Compute goodness of fit (GoF) when self._calculate_GoF is True, Think if this is necessary
                    # 2.4. Compute goodness of fit (GoF) when qoi=GoF&
                    if self.qoi == "GoF":
                        for idx, single_qoi_column in enumerate(self.list_qoi_column):
                            if self.list_read_measured_data[idx]:
                                if self.CD:
                                    dict_single_qoi_minus_h = self._calculate_GoF(
                                        measuredDF=self.time_series_measured_data_df, predictedDF=flux_minus_h_df,
                                        gof_list=self.list_objective_function_qoi,
                                        measuredDF_time_column_name=self.time_column_name,
                                        simulatedDF_time_column_name=self.time_column_name,
                                        measuredDF_column_name=self.list_qoi_column_measured[idx],
                                        simulatedDF_column_name=single_qoi_column,
                                        parameters_dict=None,
                                        return_dict=True
                                    )

                                dict_single_qoi_plus_h = self._calculate_GoF(
                                    measuredDF=self.time_series_measured_data_df, predictedDF=flux_plus_h_df,
                                    gof_list=self.list_objective_function_qoi,
                                    measuredDF_time_column_name=self.time_column_name,
                                    simulatedDF_time_column_name=self.time_column_name,
                                    measuredDF_column_name=self.list_qoi_column_measured[idx],
                                    simulatedDF_column_name=single_qoi_column,
                                    parameters_dict=None,
                                    return_dict=True
                                )

                                for single_objective_function_qoi in self.list_objective_function_qoi:
                                    if self.CD:
                                        f_x_ij_m_h = dict_single_qoi_minus_h[single_objective_function_qoi]
                                        f_x_ij_p_h = dict_single_qoi_plus_h[single_objective_function_qoi]
                                        grad = (f_x_ij_p_h - f_x_ij_m_h)/h
                                    else:
                                        f_x_ij_p_h = dict_single_qoi_plus_h[single_objective_function_qoi]
                                        f_x_ij = \
                                            index_parameter_gof_DF.loc[(index_parameter_gof_DF["qoi"] == single_qoi_column)][
                                                single_objective_function_qoi].values[0]
                                        grad = (f_x_ij_p_h - f_x_ij) / h

                                dict_of_grad_estimation_vector[
                                    (single_qoi_column,single_objective_function_qoi)].append(grad)
                    else:
                        for idx, single_qoi_column in enumerate(self.list_qoi_column):
                            if self.list_read_measured_data[idx]:
                                if self.CD:
                                    flux_plus_h_df, flux_minus_h_df = utility.filter_two_DF_on_common_timesteps(
                                        flux_plus_h_df, flux_minus_h_df, column_name=self.time_column_name)
                                    grad = (flux_plus_h_df[single_qoi_column] - flux_minus_h_df[single_qoi_column]) / h
                                else:
                                    flux_df, flux_plus_h_df = utility.filter_two_DF_on_common_timesteps(
                                        flux_df, flux_plus_h_df, column_name=self.time_column_name)
                                    grad = (flux_plus_h_df[single_qoi_column] - flux_df[single_qoi_column]) / h
                                dict_of_grad_estimation_vector[single_qoi_column].append(grad) # TODO .values?
                            if self.save_gradient_related_runs:
                                new_column_name = "d_" + single_qoi_column + "_d_" + single_param_name
                                flux_df[new_column_name] = grad
                                # TODO Thinks about saving the output of 'gradient' runs
                                # flux_df.set_index(self.time_column_name, inplace=True)
                                # flux_plus_h_df.set_index(self.time_column_name, inplace=True)
                                # flux_df = flux_df.merge(flux_plus_h_df[[single_qoi_column, ]], left_index=True, right_index=True)
                                # flux_df.reset_index(inplace=True)
                                # flux_df.rename(columns={"index": self.time_column_name}, inplace=True)

                # 3. Process data for generating gradient matrices
                gradient_matrix_dict = dict()
                if self.compute_active_subspaces:
                    for idx, single_qoi_column in enumerate(self.list_qoi_column):
                        if self.qoi == "GoF":
                            for single_objective_function_qoi in self.list_objective_function_qoi:
                                if self.list_read_measured_data[idx]:
                                    grad_estimation_vector = dict_of_grad_estimation_vector[(single_qoi_column, single_objective_function_qoi)]
                                    gradient_matrix_dict[(single_qoi_column, single_objective_function_qoi)] = np.outer(grad_estimation_vector,grad_estimation_vector)
                        else:
                            # TODO Transform the long vector into the matrix such that time is a 2nd/3rd dimension
                            data = np.array(dict_of_grad_estimation_vector[single_qoi_column])
                            grad_estimation_matrix = data.reshape((parameter_index_to_perturb,len(self.simulation_range))).transpose()
                            # TODO I am not sure about this!
                            gradient_matrix_dict[single_qoi_column] = np.outer(grad_estimation_matrix,grad_estimation_matrix)
            ######################################################################################################
            # Final savings and plots
            ######################################################################################################
            end = time.time()
            runtime = end - start

            result_dict = {"run_time": runtime,
                           "result_time_series": flux_df,
                           "parameters_dict": index_run_and_parameters_dict,
                           "state_df": state_df, }

            if self.calculate_GoF:
                result_dict["gof_df"] = index_parameter_gof_DF

            if self.compute_gradients and self.compute_active_subspaces and not len(gradient_matrix_dict) == 0:
                result_dict["grad_matrix"] = gradient_matrix_dict

            results_array.append((result_dict, runtime))
            print(f"[HVBSASK INFO] Process {i} returned / appended it's results")

            # TODO Change this now when multiple QoI are supported
            if writing_results_to_a_file and curr_working_dir is not None:
                file_path = curr_working_dir / f"flux_df_{i}.pkl"
                flux_df.to_pickle(file_path, compression="gzip")
                file_path = curr_working_dir / f"state_df_{i}.pkl"
                state_df.to_pickle(file_path, compression="gzip")
                if index_run_and_parameters_dict is not None:  # TODO seems as parameters_dict is never None!
                    file_path = curr_working_dir / f"parameters_HBVSASK_run_{i}.pkl"
                    with open(file_path, 'wb') as f:
                        dill.dump(index_run_and_parameters_dict, f)

                if self.calculate_GoF:
                    if index_parameter_gof_DF is not None:
                        file_path = curr_working_dir / f"gof_{i}.pkl"
                        index_parameter_gof_DF.to_pickle(file_path, compression="gzip")

                if self.compute_gradients and self.compute_active_subspaces and not len(gradient_matrix_dict) == 0:
                    file_path = curr_working_dir / f"gradient_matrix_dict_run_{i}.pkl"
                    with open(file_path, 'wb') as f:
                        dill.dump(gradient_matrix_dict, f)

            if plotting:
                if self.multiple_qoi:
                    for idx, single_qoi_column in enumerate(self.list_qoi_column):
                        fig = hbv._plot_output_data_and_precipitation(input_data_df=self.time_series_measured_data_df,
                                                                      simulated_data_df=flux_df,
                                                                      input_data_time_column=self.time_column_name,
                                                                      simulated_time_column=self.time_column_name,
                                                                      measured_data_column=self.list_qoi_column_measured[idx],
                                                                      simulated_column=single_qoi_column,
                                                                      precipitation_columns=self.precipitation_column_name,
                                                                      additional_columns=None,
                                                                      plot_measured_data=self.list_read_measured_data[idx])
                        # fig.add_trace(go.Scatter(x=flux_df.index, y=flux_df["Q_cms"], name="Q_cms"))
                        plot_filename = curr_working_dir / f"hbv_sask_{self.basis}_{i}_{single_qoi_column}.html"
                        plot(fig, filename=str(plot_filename), auto_open=False)
                else:
                    fig = hbv._plot_output_data_and_precipitation(input_data_df=self.time_series_measured_data_df,
                                                                  simulated_data_df=flux_df,
                                                                  input_data_time_column=self.time_column_name,
                                                                  simulated_time_column=self.time_column_name,
                                                                  measured_data_column=self.qoi_column_measured,
                                                                  simulated_column=self.qoi_column,
                                                                  precipitation_columns=self.precipitation_column_name,
                                                                  additional_columns=None,
                                                                  plot_measured_data=self.read_measured_data)
                    # fig.add_trace(go.Scatter(x=flux_df.index, y=flux_df["Q_cms"], name="Q_cms"))
                    plot_filename = curr_working_dir / f"hbv_sask_{self.basis}_{i}.html"
                    plot(fig, filename=str(plot_filename), auto_open=False)
                # fig.show()

        return results_array

    # TODO Change this now when multiple QoI are supported
    def _calculate_GoF(self, measuredDF, predictedDF,
                       gof_list=None, measuredDF_time_column_name=None, measuredDF_column_name=None,
                       simulatedDF_time_column_name=None, simulatedDF_column_name=None,
                       parameters_dict=None, return_dict=False):
        """
        Assumption - that predictedDF stores as well measured data
        """
        if gof_list is None:
            gof_list = self.objective_function
        if measuredDF_time_column_name is None:
            measuredDF_time_column_name = self.time_column_name
        if measuredDF_column_name is None:
            measuredDF_column_name = self.qoi_column_measured
        if simulatedDF_time_column_name is None:
            simulatedDF_time_column_name = self.time_column_name
        if simulatedDF_column_name is None:
            simulatedDF_column_name = self.qoi_column

        gof_dict = utility.calculateGoodnessofFit_simple(
            measuredDF=measuredDF,
            predictedDF=predictedDF,
            gof_list=gof_list,
            measuredDF_time_column_name=measuredDF_time_column_name,
            measuredDF_column_name=measuredDF_column_name,
            simulatedDF_time_column_name=simulatedDF_time_column_name,
            simulatedDF_column_name=simulatedDF_column_name,
            return_dict=True,
        )
        gof_dict["qoi"] = self.qoi_column

        if return_dict:
            if parameters_dict is None:
                return gof_dict
            else:
                return {**parameters_dict, **gof_dict}
        else:
            if parameters_dict is None:
                # return pd.DataFrame([{**gof_dict},])
                return pd.DataFrame({**gof_dict})
            else:
                # return pd.DataFrame([{**parameters_dict, **gof_dict}, ])
                return pd.DataFrame({**parameters_dict, **gof_dict})