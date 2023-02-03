import json
import pathlib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from distutils.util import strtobool
import time
import dill

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

        self.uq_method = kwargs.get('uq_method', None)
        self.raise_exception_on_model_break = kwargs.get('raise_exception_on_model_break', False)
        if self.uq_method is not None and self.uq_method == "sc":  # always break when running gPCE simulation
            self.raise_exception_on_model_break = True
        self.disable_statistics = kwargs.get('disable_statistics', False)

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

        try:
            self.calculate_GoF = strtobool(self.configurationObject["output_settings"].get("calculate_GoF", True))
            self.objective_function = self.configurationObject["output_settings"].get("objective_function", "all")
        except KeyError:
            self.calculate_GoF = False
            self.objective_function = []

        self._timespan_setup(**kwargs)
        self._input_data_setup(
            time_column_name=self.time_column_name, streamflow_column_name=self.streamflow_column_name,
            precipitation_column_name=self.precipitation_column_name, temperature_column_name=self.temperature_column_name,
            long_term_precipitation_column_name=self.long_term_precipitation_column_name,
            long_term_temperature_column_name=self.long_term_temperature_column_name

        )

        if self.plotting:
            self._plot_input_data(
                time_column_name=self.time_column_name, streamflow_column_name=self.streamflow_column_name,
                precipitation_column_name=self.precipitation_column_name, temperature_column_name=self.temperature_column_name
            )

    def _timespan_setup(self, **kwargs):
        if self.run_full_timespan:
            self.start_date, self.end_date = hbv._get_full_time_span(self.basis)
        else:
            try:
                self.start_date = pd.Timestamp(
                    year=self.configurationObject["time_settings"]["start_year"],
                    month=self.configurationObject["time_settings"]["start_month"],
                    day=self.configurationObject["time_settings"]["start_day"],
                    hour=self.configurationObject["time_settings"]["start_hour"]
                )
                self.end_date = pd.Timestamp(
                    year=self.configurationObject["time_settings"]["end_year"],
                    month=self.configurationObject["time_settings"]["end_month"],
                    day=self.configurationObject["time_settings"]["end_day"],
                    hour=self.configurationObject["time_settings"]["end_hour"]
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
        # assert len(self.time_series_data_df[self.start_date:self.end_date]) == len(self.full_data_range)

    def _input_data_setup(self, time_column_name="TimeStamp", streamflow_column_name="streamflow",
                          precipitation_column_name="precipitation", temperature_column_name="temperature",
                          long_term_precipitation_column_name="monthly_average_PE",
                          long_term_temperature_column_name="monthly_average_T"
                          ):
        # Reading the input data
        self.streamflow_df = hbv.read_streamflow(
            self.streamflow_inp, time_column_name=time_column_name, streamflow_column_name=streamflow_column_name
        )
        self.precipitation_temperature_df = hbv.read_precipitation_temperature(
            self.precipitation_temperature_inp, time_column_name=time_column_name,
            precipitation_column_name=precipitation_column_name, temperature_column_name=temperature_column_name
        )
        self.time_series_data_df = pd.merge(
            self.streamflow_df, self.precipitation_temperature_df,  left_index=True, right_index=True
        )
        self.precipitation_temperature_monthly_df = hbv.read_long_term_data(
            self.monthly_data_inp, time_column_name=time_column_name,
            precipitation_column_name=long_term_precipitation_column_name,
            temperature_column_name=long_term_temperature_column_name
        )
        self.param_setup_dict = hbv.read_param_setup_dict(self.factorSpace_txt)

        # Parse input based on some timeframe
        if time_column_name in self.time_series_data_df.columns:
            self.time_series_data_df = self.time_series_data_df.loc[
                (self.time_series_data_df[time_column_name] >= self.start_date) & (self.time_series_data_df[time_column_name] <= self.end_date)]
        else:
            self.time_series_data_df = self.time_series_data_df[self.start_date:self.end_date]

        # self.initial_condition_df = read_initial_conditions(self.initial_condition_file, return_dict_or_df="df")
        self.initial_condition_df = hbv.read_initial_conditions(
            self.initial_condition_file, timestamp=self.start_date, time_column_name=time_column_name)
        # print(self.initial_condition_df)

        # self.default_par_values_dict = {'TT': 0.0, 'C0': 5.0, 'ETF': 0.5, 'LP': 0.5, 'FC': 100,
        #                                 'beta': 2.0, 'FRAC': 0.5, 'K1': 0.5, 'alpha': 2.0, 'K2': 0.025,
        #                                 'UBAS': 1, 'PM': 1}

    def _plot_input_data(self, time_column_name="TimeStamp", streamflow_column_name="streamflow",
                         precipitation_column_name="precipitation", temperature_column_name="temperature"):
        if self.time_series_data_df is None:
            return

        fig = make_subplots(rows=3, cols=1)
        if time_column_name in self.time_series_data_df.columns:
            fig.add_trace(
                go.Scatter(x=self.time_series_data_df[time_column_name], y=self.time_series_data_df[precipitation_column_name],
                           name="P"), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=self.time_series_data_df[time_column_name], y=self.time_series_data_df[temperature_column_name],
                           name="T"), row=2, col=1)
            fig.add_trace(
                go.Scatter(x=self.time_series_data_df[time_column_name], y=self.time_series_data_df[streamflow_column_name], name="Q_cms"), row=3, col=1)
        else:
            fig.add_trace(
                go.Scatter(x=self.time_series_data_df.index, y=self.time_series_data_df[precipitation_column_name],
                           name="P"), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=self.time_series_data_df.index, y=self.time_series_data_df[temperature_column_name],
                           name="T"), row=2, col=1)
            fig.add_trace(
                go.Scatter(x=self.time_series_data_df.index, y=self.time_series_data_df[streamflow_column_name], name="Q_cms"), row=3, col=1)

        plot_filename = self.workingDir / f"forcing_data.html"
        plot(fig, filename=str(plot_filename), auto_open=False)

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

        merge_output_with_measured_data = kwargs.get("merge_output_with_measured_data", False)
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
                forcing=self.time_series_data_df,
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

            parameters_dict = {**id_dict, **parameters_dict}

            # Create a final df - flux
            time_series_list = list(self.full_data_range)  # list(self.simulation_range)
            assert len(list(self.full_data_range)) == len(flux["Q_cms"])

            flux_df = pd.DataFrame(
                list(zip(time_series_list, flux["Q_cms"], flux["Q_mm"], flux["AET"], flux["PET"], flux["Q1"],
                         flux["Q1_routed"],
                         flux["Q2"], flux["ponding"])),
                columns=[self.time_column_name, 'Q_cms', 'Q_mm', 'AET', 'PET', 'Q1', 'Q1_routed', 'Q2', "ponding"]
            )
            flux_df['Index_run'] = i

            # Create a final df - state
            last_date = time_series_list[-1]
            time_series_list.append(pd.to_datetime(last_date) + pd.DateOffset(days=1))
            state_df = pd.DataFrame(
                list(zip(time_series_list, state["SWE"], state["SMS"], state["S1"], state["S2"])),
                columns=[self.time_column_name, 'initial_SWE', 'initial_SMS', 'S1', 'S2', ]
            )
            state_df['WatershedArea_km2'] = self.initial_condition_df["WatershedArea_km2"].values[0]
            state_df['Index_run'] = i

            # Parse flux_df between start_date_predictions, end_date
            flux_df.set_index(self.time_column_name, inplace=True)
            flux_df = flux_df.loc[self.simulation_range]  # flux_df[self.start_date_predictions:self.end_date]

            # Append measured streamflow to flux_df, i.e., merge flux_df and self.time_series_data_df[self.streamflow_column_name]
            if merge_output_with_measured_data:
                flux_df = flux_df.merge(
                    self.time_series_data_df[[self.streamflow_column_name, ]], left_index=True, right_index=True)

            # Parse state_df between start_date_predictions, end_date + 1
            state_df.set_index(self.time_column_name, inplace=True)
            state_df = state_df[self.start_date_predictions:]

            # reset the index
            flux_df.reset_index(inplace=True)
            flux_df.rename(columns={"index": self.time_column_name}, inplace=True)
            state_df.reset_index(inplace=True)
            state_df.rename(columns={"index": self.time_column_name}, inplace=True)

            # TODO - Different QoI (e.g., some likelihood); Different purpose - ActiveSubspaces
            # self.qoi = "GoF" | "Q" | ["Q_cms","AET"]
            # self.mode = "continuous" | "sliding_window" | "resampling"
            # self.compute_gradients = True | False

            end = time.time()
            runtime = end - start

            result_dict = {"run_time": runtime,
                           "result_time_series": flux_df,
                           "parameters_dict": parameters_dict,
                           "state_df": state_df, }

            # Compute Metrics
            index_parameter_gof_DF = None
            if self.calculate_GoF:
                # TODO measuredDF should be self.time_series_data_df?
                index_parameter_gof_DF = self._calculate_GoF(measuredDF=self.time_series_data_df, predictedDF=flux_df, parameters_dict=parameters_dict)
                # index_parameter_gof_DF = self._calculate_GoF(measuredDF=flux_df, predictedDF=flux_df, parameters_dict=parameters_dict)
            result_dict["gof_df"] = index_parameter_gof_DF

            results_array.append((result_dict, runtime))
            print(f"[HVBSASK INFO] Process {i} returned / appended it's results")

            # Write to a file
            if writing_results_to_a_file and curr_working_dir is not None:
                file_path = curr_working_dir / f"flux_df_{i}.pkl"
                flux_df.to_pickle(file_path, compression="gzip")
                file_path = curr_working_dir / f"state_df_{i}.pkl"
                state_df.to_pickle(file_path, compression="gzip")
                if parameters_dict is not None:  # TODO seems as parameters_dict is never None!
                    file_path = curr_working_dir / f"parameters_HBVSASK_run_{i}.pkl"
                    with open(file_path, 'wb') as f:
                        dill.dump(parameters_dict, f)
                if index_parameter_gof_DF is not None and self.calculate_GoF:
                    file_path = curr_working_dir / f"gof_{i}.pkl"
                    index_parameter_gof_DF.to_pickle(file_path, compression="gzip")

            if plotting:
                fig = hbv._plot_streamflow_and_precipitation(
                    input_data_df=self.time_series_data_df,
                    simulated_data_df=flux_df,
                    input_data_time_column=self.time_column_name,
                    simulated_time_column=self.time_column_name,
                    observed_streamflow_column=self.streamflow_column_name,
                    simulated_streamflow_column="Q_cms",
                    precipitation_columns=self.precipitation_column_name,
                    additional_columns=None
                )
                # fig.add_trace(go.Scatter(x=flux_df.index, y=flux_df["Q_cms"], name="Q_cms"))
                plot_filename = curr_working_dir / f"hbv_sask_{self.basis}_{i}.html"
                plot(fig, filename=str(plot_filename), auto_open=False)
                # fig.show()

        return results_array

    def _calculate_GoF(self, measuredDF, predictedDF, parameters_dict):
        """
        Assumption - that predictedDF stores as well measured data
        """
        gof_dict = utility.calculateGoodnessofFit_simple(
            measuredDF=measuredDF,
            predictedDF=predictedDF,
            gof_list=self.objective_function,
            measuredDF_time_column_name=self.time_column_name,
            measuredDF_column_name=self.streamflow_column_name,
            simulatedDF_time_column_name=self.time_column_name,
            simulatedDF_column_name='Q_cms',
            return_dict=True,
        )
        index_parameter_gof_dict = {**parameters_dict, **gof_dict}
        return pd.DataFrame([index_parameter_gof_dict,])
