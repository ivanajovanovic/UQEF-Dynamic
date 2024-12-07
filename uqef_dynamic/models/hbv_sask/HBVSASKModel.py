# Standard library imports
from collections import defaultdict
from distutils.util import strtobool
import json
from pathlib import Path
import time
from typing import List, Optional, Dict, Any, Union

# Third party imports
import dill
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

# Local application imports
from uqef_dynamic.utils import utility
from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv

class HBVSASKModelConfigurations:
    """
        This is the class for the HBVSASK model configurations.

    """
    RUN_FULL_TIMESPAN = "run_full_timespan"
    WRITING_RESULTS_TO_A_FILE = "writing_results_to_a_file"
    PLOTTING = "plotting"
    CORRUPT_FORCING_DATA = "corrupt_forcing_data"
    TIME_SETTINGS = "time_settings"
    MODEL_SETTINGS = "model_settings"
    UQ_METHOD = "uq_method"
    RAISE_EXCEPTION_ON_MODEL_BREAK = "raise_exception_on_model_break"
    DISABLE_STATISTICS = "disable_statistics"
    INITIAL_CONDITION_FILE = "initial_condition_file"
    MONTHLY_DATA_INP = "monthly_data_inp"
    PRECIPITATION_TEMPERATURE_INP = "precipitation_temperature_inp"
    STREAMFLOW_INP = "streamflow_inp"
    FACTORSPACE_TXT = "factorSpace_txt"
    TIME_COLUMN_NAME = "time_column_name"
    STREAMFLOW_COLUMN_NAME = "streamflow_column_name"
    PRECIPITATION_COLUMN_NAME = "precipitation_column_name"
    TEMPERATURE_COLUMN_NAME = "temperature_column_name"
    LONG_TERM_PRECIPITATION_COLUMN_NAME = "long_term_precipitation_column_name"
    LONG_TERM_TEMPERATURE_COLUMN_NAME = "long_term_temperature_column_name"

    def __init__(self, configurationObject: dict, *args: Any, **kwargs: Any):
        self.configurationObject = configurationObject
        
        self.run_full_timespan = self.get_value_from_kwargs_or_config_dict_bool(self.RUN_FULL_TIMESPAN, self.TIME_SETTINGS, kwargs, default='False')
        self.writing_results_to_a_file = self.get_value_from_kwargs_or_config_dict_bool(self.WRITING_RESULTS_TO_A_FILE, self.MODEL_SETTINGS, kwargs, default="True")
        self.plotting = self.get_value_from_kwargs_or_config_dict_bool(self.PLOTTING, self.MODEL_SETTINGS, kwargs, default="True")
        self.corrupt_forcing_data = self.get_value_from_kwargs_or_config_dict_bool(self.CORRUPT_FORCING_DATA, self.MODEL_SETTINGS, kwargs,default="False")
        
        self.uq_method = kwargs.get(self.UQ_METHOD, None)
        self.raise_exception_on_model_break = kwargs.get(self.RAISE_EXCEPTION_ON_MODEL_BREAK, False)
        if self.uq_method is not None and self.uq_method == "sc":  # always break when running gPCE simulation
            self.raise_exception_on_model_break = True
        self.disable_statistics = kwargs.get(self.DISABLE_STATISTICS, False)
        # if not self.disable_statistics:
        #     self.writing_results_to_a_file = False
        #####################################
        # self.initial_condition_file = self.inputModelDir_basis / "initial_condition.inp"
        initial_condition_file = kwargs.get("initial_condition_file", "state_df.pkl")
        # initial_condition_file = kwargs.get("initial_condition_file", "state_const_df.pkl")
        # if self.run_full_timespan:
        #     initial_condition_file = kwargs.get("initial_condition_file", "state_const_df.pkl")
        monthly_data_inp = kwargs.get("monthly_data_inp", "monthly_data.inp")
        precipitation_temperature_inp = kwargs.get("precipitation_temperature_inp", "Precipitation_Temperature.inp")
        streamflow_inp = kwargs.get("streamflow_inp", "streamflow.inp")
        factorSpace_txt = kwargs.get("factorSpace_txt", "factorSpace.txt")

        self.time_column_name = kwargs.get('time_column_name', utility.TIME_COLUMN_NAME)
        self.index_column_name = kwargs.get('index_column_name', utility.INDEX_COLUMN_NAME)

        self.simulation_config = utility.read_simulation_settings_from_configuration_object(self.configurationObject, **kwargs)
        self.assign_values(self.simulation_config)

    def assign_values(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def get_value(self, key: str, default: Optional[Any] = None):
        return self.configurationObject.get(key, default)
    
    def get_value_from_kwargs_or_config_dict_bool(self, key: str, section: str, kwargs: dict, default: Optional[str] = "False"):
        if key in kwargs:
            return kwargs[key]
        else:
            try:
                return strtobool(self.configurationObject[section].get(key, default))
            except KeyError:
                return default
    
    def get_value_from_kwargs_or_config_dict(self, key: str, section: str, kwargs: dict, default: Optional[Any] = None):
        if key in kwargs:
            return kwargs[key]
        else:
            try:
                return self.configurationObject[section].get(key, default)
            except KeyError:
                return default
        
        
class HBVSASKModel(object):
    def __init__(self, configurationObject: Union[dict, str, Path], inputModelDir: Union[str, Path], workingDir: Optional[Union[str, Path]] = None, *args, **kwargs):
        """
        This is the main class for the HBVSASK model.

        The HBVSASKModel class encapsulates the functionality of the HBVSASK hydrological model.
        It provides methods for configuring the model, running simulations, and retrieving the results.

        Attributes:
            configurationObject (dict): A dictionary containing the configuration parameters for the model.
            inputModelDir (Union[str, Path]): The directory of the input model. Can be either a string or a Path object.
            workingDir (Optional[str]): The working directory. If not provided, defaults to None.
        """
        self.configurationObject = None
        if isinstance(configurationObject, dict):
            self.configurationObject = configurationObject
        else:
            with open(configurationObject) as f:
                self.configurationObject = json.load(f)
        
        self.configurations = HBVSASKModelConfigurations(self.configurationObject, **kwargs)

        self.inputModelDir = Path(inputModelDir)

        if "basis" in kwargs:
            self.basis = kwargs['basis']
        else:
            self.basis = self.configurationObject["model_settings"].get("basis", 'Oldman_Basin')
        self.inputModelDir_basis = self.inputModelDir / self.basis

        if workingDir is None:
            workingDir = self.inputModelDir
        self.workingDir = Path(workingDir)
        self.workingDir.mkdir(parents=True, exist_ok=True)

        self._setup(**kwargs)

    def assign_values(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        
    # TODO Finish moving this _setup into the separate class, e.g., HBVSASKModelConfigurations
    def _setup(self, **kwargs):

        if "run_full_timespan" in kwargs:
            self.run_full_timespan = kwargs['run_full_timespan']
        else:
            if "time_settings" in self.configurationObject:
                self.run_full_timespan = strtobool(self.configurationObject["time_settings"].get(
                    "run_full_timespan", 'False'))
            else:
                self.run_full_timespan = False

        if "writing_results_to_a_file" in kwargs:
            self.writing_results_to_a_file = kwargs['writing_results_to_a_file']
        else:
            if "model_settings" in self.configurationObject:
                self.writing_results_to_a_file = strtobool(self.configurationObject["model_settings"].get(
                    "writing_results_to_a_file", "True"))
            else:
                self.writing_results_to_a_file = True

        if "plotting" in kwargs:
            self.plotting = kwargs['plotting']
        else:
            if "model_settings" in self.configurationObject:
                self.plotting = strtobool(self.configurationObject["model_settings"].get(
                    "plotting", "True"))
            else:
                self.plotting = True    

        if "printing" in kwargs:
            self.printing = kwargs['printing']
        else:
            if "model_settings" in self.configurationObject:
                self.printing = strtobool(self.configurationObject["model_settings"].get(
                    "printing", "False"))
            else:
                self.printing = False 

        if "corrupt_forcing_data" in kwargs:
            self.corrupt_forcing_data = kwargs['corrupt_forcing_data']
        else:
            if "model_settings" in self.configurationObject:
                self.corrupt_forcing_data = strtobool(self.configurationObject["model_settings"].get(
                    "corrupt_forcing_data", "False"))
            else:
                self.corrupt_forcing_data = False

        #####################################
        # these set of control variables are for UQEF & UQEF-Dynamic framework...
        #####################################

        self.uq_method = kwargs.get('uq_method', None)
        self.raise_exception_on_model_break = kwargs.get('raise_exception_on_model_break', False)
        if self.uq_method is not None and self.uq_method == "sc":  # always break when running gPCE simulation
            self.raise_exception_on_model_break = True
        self.disable_statistics = kwargs.get('disable_statistics', False)
        # if not self.disable_statistics:
        #     self.writing_results_to_a_file = False
        
        #####################################
        # self.initial_condition_file = self.inputModelDir_basis / "initial_condition.inp"
        initial_condition_file = kwargs.get("initial_condition_file", "state_df.pkl")
        # initial_condition_file = kwargs.get("initial_condition_file", "state_const_df.pkl")
        # if self.run_full_timespan:
        #     initial_condition_file = kwargs.get("initial_condition_file", "state_const_df.pkl")
        monthly_data_inp = kwargs.get("monthly_data_inp", "monthly_data.inp")
        precipitation_temperature_inp = kwargs.get("precipitation_temperature_inp", "Precipitation_Temperature.inp")
        streamflow_inp = kwargs.get("streamflow_inp", "streamflow.inp")
        factorSpace_txt = kwargs.get("factorSpace_txt", "factorSpace.txt")

        self.initial_condition_file = self.inputModelDir_basis / initial_condition_file
        self.monthly_data_inp = self.inputModelDir_basis / monthly_data_inp
        self.precipitation_temperature_inp = self.inputModelDir_basis / precipitation_temperature_inp
        self.streamflow_inp = self.inputModelDir_basis / streamflow_inp
        self.factorSpace_txt = self.inputModelDir / factorSpace_txt

        self.time_column_name = kwargs.get('time_column_name', utility.TIME_COLUMN_NAME)
        self.index_column_name = kwargs.get('index_column_name', utility.INDEX_COLUMN_NAME)

        self.streamflow_column_name = kwargs.get("streamflow_column_name", "streamflow")
        self.precipitation_column_name = kwargs.get("precipitation_column_name", "precipitation")
        self.temperature_column_name = kwargs.get("temperature_column_name", "temperature")
        self.long_term_precipitation_column_name = kwargs.get("long_term_precipitation_column_name", "monthly_average_PE")
        self.long_term_temperature_column_name = kwargs.get("long_term_temperature_column_name", "monthly_average_T")

        #####################################
        config_dict = utility.read_simulation_settings_from_configuration_object(self.configurationObject, **kwargs)
        self.assign_values(config_dict)

        if self.autoregressive_model_first_order and (self.qoi == "GoF" or self.mode == "sliding_window"):
            print(f"Possible error in the configuration file - autoregressive_model_first_order is set to True, \
            but qoi is GoF or mode is sliding_window. Setting autoregressive_model_first_order to False!")
            self.autoregressive_model_first_order = False

        #####################################
        # streamflow is of special importance here (i.e., for HBVSASK), 
        # since we have saved/measured/ground truth that for it and it is inside input data
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

        #####################################

        self._timespan_setup(**kwargs)
        self._input_and_measured_data_setup(time_column_name=self.time_column_name,
                                            precipitation_column_name=self.precipitation_column_name,
                                            temperature_column_name=self.temperature_column_name,
                                            long_term_precipitation_column_name=self.long_term_precipitation_column_name,
                                            long_term_temperature_column_name=self.long_term_temperature_column_name,
                                            read_measured_streamflow=self.read_measured_streamflow,
                                            streamflow_column_name=self.streamflow_column_name)

        if self.plotting:
            figure = self.plot_input_data(time_column_name=self.time_column_name,
                                           precipitation_column_name=self.precipitation_column_name,
                                           temperature_column_name=self.temperature_column_name,
                                           read_measured_streamflow=self.read_measured_streamflow,
                                           streamflow_column_name=self.streamflow_column_name)

    #####################################
    # time setups
    ##################################### 
    def _timespan_setup(self, **kwargs):
        self.get_start_end_dates()
        self.get_resolution(**kwargs)
        self.get_spin_up_length(**kwargs)
        self.set_simulation_length(simulation_length=None, **kwargs)
        self.set_date_ranges()
        
    def get_start_end_dates(self):
        # TODO think about refactoring this, to get start_date and end_date as input arguments as well
        # note: run_full_timespan has a power over start and end dates
        if self.run_full_timespan:
            self.start_date, self.end_date = hbv._get_full_time_span(self.basis)
        else:
            try:
                self.start_date = self.get_timestamp_from_configuration_dict("start")
                self.end_date = self.get_timestamp_from_configuration_dict("end")
            except KeyError:
                self.start_date, self.end_date = hbv._get_full_time_span(self.basis)

    def get_timestamp_from_configuration_dict(self, time_type):
        return pd.Timestamp(
            year=self.configurationObject["time_settings"][f"{time_type}_year"],
            month=self.configurationObject["time_settings"][f"{time_type}_month"],
            day=self.configurationObject["time_settings"][f"{time_type}_day"],
            hour=self.configurationObject["time_settings"].get(f"{time_type}_hour", 0)
        )

    def get_resolution(self, **kwargs):
        if "resolution" in kwargs:
            self.resolution = kwargs["resolution"]
        else:
            if "time_settings" in self.configurationObject:
                self.resolution = self.configurationObject["time_settings"].get("resolution", "daily")
            else:
                self.resolution = "daily"
        if self.resolution not in ["daily", "hourly", "minute"]:
            raise Exception(f"Error in Statistics class - resolution is not daily, hourly or minute")

    def get_spin_up_length(self, **kwargs):
        if "spin_up_length" in kwargs:
            self.spin_up_length = kwargs["spin_up_length"]
        elif "warm_up_length" in kwargs:
            self.spin_up_length = kwargs["warm_up_length"]
        else:
            try:
                if "spin_up_length" in self.configurationObject["time_settings"]:
                    self.spin_up_length = self.configurationObject["time_settings"]["spin_up_length"]
                elif "warm_up_length" in self.configurationObject["time_settings"]:
                    self.spin_up_length = self.configurationObject["time_settings"]["warm_up_length"]
                else:
                    raise KeyError
            except KeyError:
                self.spin_up_length = 0  # 365*3
    
    def set_simulation_length(self, simulation_length: int=None, **kwargs):
        # note: one has to omit simulation_length both from kwargs and configurationObject
        # if you want that run_full_timespan has an effect
        if simulation_length is not None:
            self.simulation_length = simulation_length
        elif "simulation_length" in kwargs:
            self.simulation_length = kwargs["simulation_length"]
        else:
            try:
                self.simulation_length = self.configurationObject["time_settings"]["simulation_length"]
            except KeyError:
                self.simulation_length = (self.end_date - self.start_date).days - self.spin_up_length
        if self.simulation_length <= 0:
            self.simulation_length = 365
    
    def set_date_ranges(self):
        # note: self.simulation_length actually has power over previously set end_date
        # note: for now, the resolutin is set to daily
        self.start_date_predictions = pd.to_datetime(self.start_date) + pd.DateOffset(days=self.spin_up_length)
        self.end_date = pd.to_datetime(self.start_date_predictions) + pd.DateOffset(days=self.simulation_length)
        self.full_data_range = pd.date_range(start=self.start_date, end=self.end_date, freq="1D")
        self.simulation_range = pd.date_range(start=self.start_date_predictions, end=self.end_date, freq="1D")

        self.start_date = pd.Timestamp(self.start_date)
        self.end_date = pd.Timestamp(self.end_date)
        self.start_date_predictions = pd.Timestamp(self.start_date_predictions)
    
    def set_run_full_timespan(self, run_full_timespan: bool):
        self.run_full_timespan = run_full_timespan
         
    def set_start_date(self, start_date: pd.Timestamp):
        self.start_date = start_date

    def set_end_date(self, end_date: pd.Timestamp):
        self.end_date = end_date

    def set_resolution(self, resolution: str):
        self.resolution = resolution
        if self.resolution not in ["daily", "hourly", "minute"]:
            raise Exception(f"Error in Statistics class - resolution is not daily, hourly or minute")

    def set_spin_up_length(self, spin_up_length: int):
        self.spin_up_length = spin_up_length
        if self.spin_up_length < 0 or self.spin_up_length > 365*3 or self.spin_up_length is None:
            self.spin_up_length = 0

    def set_start_date_predictions(self, start_date_predictions: pd.Timestamp):
        self.start_date_predictions = start_date_predictions
      
    def set_timespan_setup_from_start_end_and_spin_up_length(self, start_date, end_date, spin_up_length):
        run_full_timespan = False
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        # dict_with_dates_setup = {"start_date": start_date, "end_date": end_date, "spin_up_length":spin_up_length}
        self.set_run_full_timespan(run_full_timespan)
        self.set_start_date(start_date)
        self.set_end_date(end_date)
        self.set_spin_up_length(spin_up_length)
        simulation_length = (self.end_date - self.start_date).days - self.spin_up_length
        if simulation_length <= 0:
            simulation_length = 365
        self.set_simulation_length(simulation_length)
        self.set_date_ranges()
    
    #####################################
    def _input_and_measured_data_setup(self, time_column_name="TimeStamp", precipitation_column_name="precipitation",
                                       temperature_column_name="temperature",
                                       long_term_precipitation_column_name="monthly_average_PE",
                                       long_term_temperature_column_name="monthly_average_T", read_measured_streamflow=None,
                                       streamflow_column_name="streamflow"):
        # Reading the input data

        # % ********  Forcing (Precipitation and Temperature)  *********
        self.precipitation_temperature_df = hbv.read_precipitation_temperature(
            self.precipitation_temperature_inp, time_column_name=time_column_name,
            precipitation_column_name=precipitation_column_name, temperature_column_name=temperature_column_name
        )

        # % ********  Evapotranspiration  *********
        self.precipitation_temperature_monthly_df = hbv.read_long_term_data(
            self.monthly_data_inp, time_column_name=time_column_name,
            precipitation_column_name=long_term_precipitation_column_name,
            temperature_column_name=long_term_temperature_column_name
        )

        # % ********  Initial Condition  *********
        # self.initial_condition_df = read_initial_conditions(self.initial_condition_file, return_dict_or_df="df")
        self.initial_condition_df = hbv.read_initial_conditions(
            self.initial_condition_file, timestamp=self.start_date, time_column_name=time_column_name)
        # print(self.initial_condition_df)

        # self.default_par_values_dict = {'TT': 0.0, 'C0': 5.0, 'ETF': 0.5, 'LP': 0.5, 'FC': 100,
        #                                 'beta': 2.0, 'FRAC': 0.5, 'K1': 0.5, 'alpha': 2.0, 'K2': 0.025,
        #                                 'UBAS': 1, 'PM': 1}

        # % ********  Parameters  *********
        self.param_setup_dict = hbv.read_param_setup_dict(self.factorSpace_txt)

        # % ********  Observed Streamflow  *********
        if read_measured_streamflow is None:
            read_measured_streamflow = self.read_measured_streamflow
        if read_measured_streamflow:
            self.streamflow_df = hbv.read_streamflow(
                self.streamflow_inp, time_column_name=time_column_name, streamflow_column_name=streamflow_column_name
            )
            self.time_series_measured_data_df = pd.merge(
                self.streamflow_df, self.precipitation_temperature_df,  left_index=True, right_index=True
            )
        else:
            self.time_series_measured_data_df = self.precipitation_temperature_df

        # % ********  Parse input (and observed streamflow) - everything stored in self.time_series_measured_data_df  *********
        if time_column_name in self.time_series_measured_data_df.columns:
            self.time_series_measured_data_df = self.time_series_measured_data_df.loc[
                (self.time_series_measured_data_df[time_column_name] >= self.start_date) & (self.time_series_measured_data_df[time_column_name] <= self.end_date)]
        else:
            self.time_series_measured_data_df = self.time_series_measured_data_df[self.start_date:self.end_date]
            # print(f"DEBUGGING - self.end_date {self.end_date}")
            # print(f"DEBUGGING - self.time_series_measured_data_df.index-{self.time_series_measured_data_df.index}\n")
            # self.time_series_measured_data_df = self.time_series_measured_data_df.loc[self.simulation_range]

    def plot_input_data(self, time_column_name="TimeStamp", precipitation_column_name="precipitation",
                         temperature_column_name="temperature", read_measured_streamflow=None,
                         streamflow_column_name="streamflow", **kwargs):
        
        if "plot_whole_simulation_period" in kwargs:
            plot_whole_simulation_period = kwargs["plot_whole_simulation_period"]
        else:
            plot_whole_simulation_period = True
    
        if self.time_series_measured_data_df is None:
            return

        if read_measured_streamflow is None:
            read_measured_streamflow = self.read_measured_streamflow

        if read_measured_streamflow:
            n_rows = 3
        else:
            n_rows = 2
        fig = make_subplots(rows=n_rows, cols=1)

        time_series_measured_data_df = self.time_series_measured_data_df

        condition_for_plotting_streamflow = read_measured_streamflow and streamflow_column_name in time_series_measured_data_df.columns

        if time_column_name is None:
            time_column_name = self.time_column_name

        if time_column_name in time_series_measured_data_df.columns:
            if not plot_whole_simulation_period:
                time_series_measured_data_df = self.time_series_measured_data_df.loc[\
                    self.time_series_measured_data_df[time_column_name].isin(\
                    pd.date_range(start=self.start_date_predictions,end=self.end_date))]
            fig.add_trace(
                go.Scatter(x=time_series_measured_data_df[time_column_name], y=time_series_measured_data_df[precipitation_column_name],
                           name="Precipitation"), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=time_series_measured_data_df[time_column_name], y=time_series_measured_data_df[temperature_column_name],
                           name="Temperature"), row=2, col=1)
            if condition_for_plotting_streamflow:
                fig.add_trace(
                    go.Scatter(x=time_series_measured_data_df[time_column_name], y=time_series_measured_data_df[streamflow_column_name], name="Q_cms (Measured)"), row=3, col=1)
        else:
            if not plot_whole_simulation_period:
                time_series_measured_data_df = self.time_series_measured_data_df.loc[self.start_date_predictions:self.end_date,:]
            fig.add_trace(
                go.Scatter(x=time_series_measured_data_df.index, y=time_series_measured_data_df[precipitation_column_name],
                           name="Precipitation"), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=time_series_measured_data_df.index, y=time_series_measured_data_df[temperature_column_name],
                           name="Temperature"), row=2, col=1)
            if condition_for_plotting_streamflow:
                fig.add_trace(
                    go.Scatter(x=time_series_measured_data_df.index, y=time_series_measured_data_df[streamflow_column_name], name="Q_cms (Measured)"), row=3, col=1)

        fig.update_layout(
            # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            showlegend=True,
            # template="plotly_white",
        )
        if "title" in kwargs:
            fig.update_layout(title=kwargs["title"])
        fileName = kwargs.get('fileName')
        if fileName is None or not fileName.endswith(".html"):
            fileName = f"forcing_data.html"
        plot_filename = self.workingDir / fileName
        plot(fig, filename=str(plot_filename), auto_open=False)
        return fig

    def redo_input_and_measured_data_setup(self):
        """
        Redo input and measured data setup, e.g., after the time settings have been changed.
        """
        self._input_and_measured_data_setup(
            time_column_name=self.time_column_name,
            precipitation_column_name=self.precipitation_column_name,
            temperature_column_name=self.temperature_column_name,
            long_term_precipitation_column_name=self.long_term_precipitation_column_name,
            long_term_temperature_column_name=self.long_term_temperature_column_name,
            read_measured_streamflow=self.read_measured_streamflow,
            streamflow_column_name=self.streamflow_column_name
            )

        if self.plotting:
            figure = self.plot_input_data(time_column_name=self.time_column_name,
                                           precipitation_column_name=self.precipitation_column_name,
                                           temperature_column_name=self.temperature_column_name,
                                           read_measured_streamflow=self.read_measured_streamflow,
                                           streamflow_column_name=self.streamflow_column_name)

    #####################################
    def prepare(self, *args, **kwargs):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def get_full_data_range(self):
        return self.full_data_range
    
    def get_simulation_range(self):
        return self.simulation_range
    
    def get_start_date(self):
        return self.start_date
    
    def get_end_date(self):
        return self.end_date
    
    def get_start_date_predictions(self):
        return self.start_date_predictions
    
    def timesteps(self):
        return list(self.full_data_range)

    def run(
            self, i_s: Optional[List[int]] = [0, ], 
            parameters: Optional[Union[Dict[str, Any], List[Any]]] = None,
            raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs
            ):
        """
            This is the main function to run the HBVSASK model.

            Parameters:
            :i_s (Optional[List[int]]): A list of unique model run ids. Default is [0].
            :parameters (Optional[Union[Dict[str, Any], List[Any]]]): A dictionary of parameters to be used in the model;
            or a list of values for the parameters specified in the configuration file. Default is None.
            :raise_exception_on_model_break (Optional[Union[bool, Any]]): If True, the function will raise an exception when the model breaks.
            Important when uq_method is gPCE. Default is None.
            :*args: Variable length argument list.
            :**kwargs (dict, optional): Additional keyword arguments.
            Can include:
                - take_direct_value (bool): 
                    If True, the function will take the direct value of the parameter 
                    (e.g., it is expeced that parameters[unique_run_index] is a dict with keys being paramter name and values being parameter values). 
                    If take_direct_value is False then parameters[unique_run_index] should be a list of parameter values corresponding to the order of the parameters in the configuration file
                    Default is False.
                - createNewFolder (bool): If True, the function will create a new folder for the results. Default is False.
                - deleteFolderAfterwards (bool): If True, the function will delete the folder after the results are retrieved. Default is True.
                - writing_results_to_a_file (bool): If True, the function will write the results to a file. Default is self.writing_results_to_a_file.
                - plotting (bool): If True, the function will plot the results. Default is self.plotting.
                - printing (bool): If True, the function will print the results. Default is self.printing.
                - corrupt_forcing_data (bool): If True, the function will corrupt the forcing data. Default is self.corrupt_forcing_data.
                - merge_output_with_measured_data (bool): If True, the function will merge the output with the measured data. Default is False.
                - start_date (pd.Timestamp): The start date of the model run. Default is self.start_date.
                - start_date_predictions (pd.Timestamp): The start date of the model run predictions. Default is self.start_date_predictions.
                - end_date (pd.Timestamp): The end date of the model run. Default is self.end_date.
                - forcing (pd.DataFrame): The forcing dataframe of the model run. Default is self.forcing.
                - long_term (pd.DataFrame): The long-term dataframe of the model run. Default is self.long_term.
                - initial_condition_df (pd.DataFrame): The initial condition dataframe of the model run. Default is self.initial_condition_df.

            Returns: 
            List[Tuple[Dict[str, Any], float]]: A list of tuples for each model run; each tuple is in the form (result_dict, runtime); 
            - result_dict is a dictionary that might contain the following key-value entries (depending on the configuration file):
            - ("result_time_series", flux_df): a dataframe containing the model output for the time period specified in the configuration file.
            - ("state_df", state_df): a dataframe containing the model state for the time period specified in the configuration file.
            - ("gof_df", index_parameter_gof_DF): a dataframe containing the goodness-of-fit values for the time period specified in the configuration file.
            - ("parameters_dict", index_run_and_parameters_dict): a dictionary containing the parameter values for the time period specified in the configuration file.
            - ("run_time", runtime): the runtime of a single model run; should have the same value as runtime variable
            - ("grad_matrix", gradient_matrix_dict): a dictionary containing the gradient vectors for the time period specified in the configuration file.
        """
        # print(f"[HVBSASK INFO] {i_s} parameters: {parameters}")

        if raise_exception_on_model_break is None:
            raise_exception_on_model_break = self.raise_exception_on_model_break
        take_direct_value = kwargs.get("take_direct_value", False)
        if self.uq_method == "ensemble":
            take_direct_value = True
        createNewFolder = kwargs.get("createNewFolder", False)
        deleteFolderAfterwards = kwargs.get("deleteFolderAfterwards", True)
        writing_results_to_a_file = kwargs.get("writing_results_to_a_file", self.writing_results_to_a_file)
        plotting = kwargs.get("plotting", self.plotting)
        printing = kwargs.get("printing", self.printing)
        corrupt_forcing_data = kwargs.get("corrupt_forcing_data", self.corrupt_forcing_data)

        # TODO Think if merge_output_with_measured_data should always be True
        merge_output_with_measured_data = kwargs.get("merge_output_with_measured_data", False)
        # TODO - add an extra opion when autoregressive_model_first_order is set to True to choose between measred data and previous model output
        if any(self.list_calculate_GoF) or self.autoregressive_model_first_order or self.qoi == "GoF":
            merge_output_with_measured_data = True
        # if any(self.list_calculate_GoF):
        #     merge_output_with_measured_data = True
        # if not any(self.list_read_measured_data):
        #     merge_output_with_measured_data = False

        # HBV Specific input data
        forcing = kwargs.get("forcing", self.time_series_measured_data_df)
        long_term = kwargs.get("long_term", self.precipitation_temperature_monthly_df)
        initial_condition_df = kwargs.get("initial_condition_df", self.initial_condition_df)

        ######################################################################################################
        # Each model run can be configured with different start_date, start_date_predictions, end_date
        # however, these dates have to be within the self.full_data_range
        start_date = pd.Timestamp(kwargs.get("start_date", self.start_date))
        start_date_predictions = pd.Timestamp(kwargs.get("start_date_predictions", self.start_date_predictions))
        end_date = pd.Timestamp(kwargs.get("end_date", self.end_date))

        # Compute the dates in the output of the model; get the time-span from forcing data
        if self.time_column_name in forcing.columns:
            time_series_list = list(forcing[self.time_column_name])
        else:
            time_series_list = list(forcing.index)

        if min(time_series_list) > start_date:
            start_date = pd.Timestamp(min(time_series_list))
        if max(time_series_list) < end_date:
            end_date = pd.Timestamp(max(time_series_list))
        if min(time_series_list) > start_date_predictions:
            start_date_predictions = pd.Timestamp(min(time_series_list))

        # Make sure that local time-settings for tha particular run are within the 'global' time set-up (i.e., self.full_data_range)
        assert start_date >= self.start_date, "start_date has to be within the self.full_data_range"
        assert start_date_predictions >= self.start_date_predictions, f"start_date_predictions has to be within the self.full_data_range, greater than {self.start_date_predictions}"
        assert end_date <= self.end_date, f"end_date has to be within the self.full_data_range, less than {self.end_date}"

        # self.resolution
        if self.resolution == "daily":
            freq = "1D"
        elif self.resolution == "hourly":
            freq = "1H"
        else:
            freq = None
        full_data_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        simulation_range = pd.date_range(start=start_date_predictions, end=end_date, freq=freq)

        # print(f"DEBUGGING - start_date {start_date}")
        # print(f"DEBUGGING - start_date_predictions {start_date_predictions}")
        # print(f"DEBUGGING - end_date {end_date}")
        # print(f"DEBUGGING - time_series_list {time_series_list}")
        # print(f"DEBUGGING - full_data_range {full_data_range}")
        # print(f"DEBUGGING - simulation_range {simulation_range}")

        ######################################################################################################
        results_array = []
        for ip in range(0, len(i_s)):  # for each piece of work
            unique_run_index = i_s[ip]  # i is unique index run

            if parameters is not None:
                parameter = parameters[ip]
            else:
                parameter = None  # an unaltered run will be executed

            id_dict = {self.index_column_name: unique_run_index}

            # this indeed represents the number of parameters considered to be uncertain, later on parameters_dict might
            # be extanded with fixed parameters that occure in configurationObject
            if parameter is None:
                number_of_uncertain_params = 0
            elif isinstance(parameter, dict):
                number_of_uncertain_params = len(list(parameter.keys()))
            elif isinstance(parameter, list) or isinstance(parameter, np.ndarray) or isinstance(parameter, pd.DataFrame) or isinstance(parameter, pd.Series):
                number_of_uncertain_params = len(parameter)
            elif isinstance(parameter, (int, float, complex)):
                number_of_uncertain_params = 1
            else:
                number_of_uncertain_params = 0

            parameters_dict = utility.configuring_parameter_values(
                parameters=parameter,
                configurationObject=self.configurationObject.get("parameters", None),
                default_par_info_dict=hbv.DEFAULT_PAR_VALUES_DICT,
                take_direct_value=take_direct_value
            )

            if printing:
                print(f"[HVBSASK INFO] {i_s} parameters_dict - {parameters_dict} \n")
            else:
                print(f"model execution run id - {unique_run_index}...")

            start = time.time()

            # create local directory for this particular run
            if createNewFolder:
                curr_working_dir = self.workingDir / f"run_{unique_run_index}"
                curr_working_dir.mkdir(parents=True, exist_ok=True)
            else:
                curr_working_dir = self.workingDir

            # Running the model
            try:
                flux, state = hbv.HBV_SASK(
                    forcing=forcing,
                    long_term=long_term,
                    par_values_dict=parameters_dict,
                    initial_condition_df=initial_condition_df,
                    printing=False,
                    time_column_name=self.time_column_name,
                    precipitation_column_name=self.precipitation_column_name,
                    temperature_column_name=self.temperature_column_name,
                    long_term_precipitation_column_name=self.long_term_precipitation_column_name,
                    long_term_temperature_column_name=self.long_term_temperature_column_name,
                    corrupt_forcing_data=corrupt_forcing_data
                )
            except:
                if raise_exception_on_model_break:
                    raise Exception(f"Model broke for unique_run_index {unique_run_index}")
                else:
                    index_run_and_parameters_dict = {**id_dict, **parameters_dict, "successful_run": False}
                    end = time.time()
                    runtime = end - start
                    result_dict = {"run_time": runtime,
                                "result_time_series": None,
                                "parameters_dict": index_run_and_parameters_dict,
                                "state_df": None, }
                    results_array.append((result_dict, runtime))
                    continue

            assert len(list(time_series_list)) == len(flux["Q_cms"]), "Error in HBVSASKModel - flux and time_series_list have different lengths"
            ######################################################################################################
            # Processing model output
            ######################################################################################################

            # Create a final df - flux
            flux_df = self._create_flux_df(flux, time_series_list)

            if corrupt_forcing_data and "precipitation" in flux:
                flux_df['precipitation'] = flux["precipitation"]
            flux_df['Index_run'] = unique_run_index
            # Parse flux_df between start_date_predictions, end_date
            flux_df.set_index(self.time_column_name, inplace=True)
            # flux_df = flux_df.loc[simulation_range]  # flux_df[start_date_predictions:end_date]
            flux_df = flux_df[flux_df.index.isin(simulation_range)]  # Sample only the simulation_range

            # Create a final df - state
            last_date = time_series_list[-1]
            time_series_list_plus_one_day = time_series_list.copy()
            # time_series_list_plus_one_day.append(pd.to_datetime(last_date) + pd.DateOffset(days=1))
            time_series_list_plus_one_day.append(pd.to_datetime(last_date) + pd.Timedelta(days=1))
            state_df = pd.DataFrame(
                list(zip(time_series_list_plus_one_day, state["SWE"], state["SMS"], state["S1"], state["S2"])),
                columns=[self.time_column_name, 'SWE', 'SMS', 'S1', 'S2', ]
            )
            state_df['WatershedArea_km2'] = self.initial_condition_df["WatershedArea_km2"].values[0]
            state_df['Index_run'] = unique_run_index
            # Parse state_df between start_date_predictions, end_date + 1
            state_df.set_index(self.time_column_name, inplace=True)
            state_df = state_df[start_date_predictions:]  #  state_df = state_df[simulation_range]
    
            # Append measured data to flux_df, i.e., merge flux_df and self.time_series_measured_data_df[self.qoi_column_measured]
            if merge_output_with_measured_data:
                list_qoi_column_measured_to_filter = [
                    single_qoi_column_measured for single_qoi_column_measured in self.list_qoi_column_measured \
                    if single_qoi_column_measured is not None and single_qoi_column_measured != "None"]
                flux_df = flux_df.merge(
                    self.time_series_measured_data_df[list_qoi_column_measured_to_filter], left_index=True, right_index=True)

            ######################################################################################################
            # Some basic transformation of model output
            ######################################################################################################
            for idx, single_qoi_column in enumerate(self.list_qoi_column):
                single_transformation = self.list_transform_model_output[idx]
                if single_transformation is not None and single_transformation != "None":
                    # new_column_name = single_transformation + "_" + single_qoi_column
                    new_column_name = single_qoi_column
                    utility.transform_column_in_df(flux_df, transformation_function_str=single_transformation,
                                                   column_name=single_qoi_column, new_column_name=new_column_name)
                    # flux_df.drop(labels=single_qoi_column, inplace=False)
                    # flux_df.rename(columns={new_column_name: single_qoi_column}, inplace=False)
                    if self.list_read_measured_data[idx]:
                        # new_column_name = single_transformation + "_" + self.list_qoi_column_measured[idx]
                        new_column_name = self.list_qoi_column_measured[idx]
                        utility.transform_column_in_df(
                            self.time_series_measured_data_df, transformation_function_str=single_transformation,
                            column_name=self.list_qoi_column_measured[idx], new_column_name=new_column_name)
                        # self.time_series_measured_data_df.drop(labels=self.list_qoi_column_measured[idx], inplace=False)
                        # self.time_series_measured_data_df.rename(columns={
                        #     new_column_name: self.list_qoi_column_measured[idx]}, inplace=False)
                        if merge_output_with_measured_data:
                            # new_column_name = single_transformation + "_" + self.list_qoi_column_measured[idx]
                            new_column_name = self.list_qoi_column_measured[idx]
                            utility.transform_column_in_df(flux_df, transformation_function_str=single_transformation,
                                                           column_name=self.list_qoi_column_measured[idx],
                                                           new_column_name=new_column_name)
                            # flux_df.drop(labels=self.list_qoi_column_measured[idx], inplace=False)
                            # flux_df.rename(columns={new_column_name: self.list_qoi_column_measured[idx]},
                            #                inplace=False)

            ######################################################################################################
            # Compute GoFs for the whole time-span in certain set-ups
            ######################################################################################################

            index_run_and_parameters_dict = {**id_dict, **parameters_dict, "successful_run": True}

            ######################################################################################################

            index_parameter_gof_DF = None
            # Note - it does not make sense to have both qoi=GoF and calculate_GoF=True at the same time
            # condition_for_computing_index_parameter_gof_DF = \
            #     (self.calculate_GoF and not self.qoi == "GoF") or \
            #     (self.calculate_GoF and self.qoi == "GoF" and self.mode == "sliding_window")
            condition_for_computing_index_parameter_gof_DF = \
                (any(self.list_calculate_GoF) and not self.qoi == "GoF") or \
                (any(self.list_calculate_GoF) and self.qoi == "GoF" and self.mode == "sliding_window")
            if condition_for_computing_index_parameter_gof_DF:
                index_parameter_gof_list_of_dicts = []
                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    if self.list_calculate_GoF[idx] and self.list_read_measured_data[idx]:
                        index_parameter_gof_dict_single_qoi = self._calculate_GoF(
                            measuredDF=flux_df, #self.time_series_measured_data_df,
                            simulatedDF=flux_df,
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

            ######################################################################################################
            # process result to compute the final QoI - this part is if QoI should be something
            # different from the model output itself
            # self.qoi = "GoF" | "Q" | ["Q_cms","AET"]
            # self.mode = "continuous" | "sliding_window" | "resampling"
            ######################################################################################################

            processed_time_series_results = None
            if self.mode == "continuous":
                if self.qoi == "GoF":
                    index_parameter_gof_list_of_dicts = []
                    for idx, single_qoi_column in enumerate(self.list_qoi_column):
                        if self.list_read_measured_data[idx]:
                            index_parameter_gof_dict_single_qoi = self._calculate_GoF(
                                measuredDF=self.time_series_measured_data_df,  # flux_df
                                simulatedDF=flux_df,
                                gof_list=self.objective_function_qoi,
                                measuredDF_time_column_name=self.time_column_name,
                                simulatedDF_time_column_name=self.time_column_name,
                                measuredDF_column_name=self.list_qoi_column_measured[idx],
                                simulatedDF_column_name=single_qoi_column,
                                parameters_dict=index_run_and_parameters_dict,
                                return_dict=True
                            )
                            index_parameter_gof_list_of_dicts.append(index_parameter_gof_dict_single_qoi)
                            for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
                                new_column_name = single_objective_function_name_qoi + "_" + single_qoi_column
                                flux_df[new_column_name] = index_parameter_gof_dict_single_qoi[
                                    single_objective_function_name_qoi]
                    index_parameter_gof_DF = pd.DataFrame(index_parameter_gof_list_of_dicts)
                elif self.autoregressive_model_first_order:
                    self._compute_autoregressive_model_first_order(flux_df=flux_df)
                    simulation_range = self._dropna_from_df_and_update_simulation_range(flux_df, update_simulation_range=True)

            elif self.mode == "sliding_window":
                if self.center == "center":
                    center = True
                else:
                    center = False
                if self.qoi == "GoF":
                    for idx, single_qoi_column in enumerate(self.list_qoi_column):
                        if self.list_read_measured_data[idx]:
                            for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
                                new_column_name = single_objective_function_name_qoi + "_" + single_qoi_column + \
                                                  "_sliding_window"
                                rol = flux_df[single_qoi_column].rolling(
                                    window=self.interval, min_periods=self.min_periods,
                                    center=center, win_type=None
                                )
                                flux_df[new_column_name] = rol.apply(
                                    self._calculate_GoF_on_data_subset, raw=False,
                                    args=(flux_df, single_qoi_column, idx, single_objective_function_name_qoi)
                                )
                                # print(f"DEBUGGING - new_column_name {new_column_name}")
                                # print(f"DEBUGGING - flux_df[new_column_name] {flux_df[new_column_name]}")
                else:
                    for idx, single_qoi_column in enumerate(self.list_qoi_column):
                        ser, new_column_name = self._compute_rolling_function_over_qoi(flux_df, single_qoi_column,
                                                                                       center=center, win_type=None)
                        flux_df[new_column_name] = ser
                simulation_range = self._dropna_from_df_and_update_simulation_range(flux_df, update_simulation_range=True)

            elif self.mode == "resampling":
                pass
            else:
                raise Exception(f"[ERROR] mode should have one of the following values:"
                                f" \"continuous\" or \"sliding_window\" or \"resampling\"")

            ######################################################################################################
            # Computing gradients
            ######################################################################################################

            if self.compute_gradients or self.compute_active_subspaces:
                if self.center == "center":
                    center = True
                else:
                    center = False
                flux_df, gradient_matrix_dict = self._compute_gradient_matrix(
                    unique_run_index=unique_run_index, 
                    flux_df=flux_df, 
                    parameters_dict=parameters_dict, 
                    index_parameter_gof_DF=index_parameter_gof_DF, 
                    time_series_list=time_series_list, 
                    center=center,
                    simulation_range=simulation_range,
                    **kwargs
                )

            ######################################################################################################
            # Final savings and plots
            ######################################################################################################
            end = time.time()
            runtime = end - start

            simulation_range = self._dropna_from_df_and_update_simulation_range(flux_df, update_simulation_range=True)
            # flux_df = flux_df.loc[simulation_range]
            flux_df = flux_df[flux_df.index.isin(simulation_range)]
            #state_df = state_df[state_df.index.isin(simulation_range)]

            result_dict = {"run_time": runtime,
                           "result_time_series": flux_df,
                           "parameters_dict": index_run_and_parameters_dict,
                           "state_df": state_df, }

            # if self.calculate_GoF or self.qoi == "GoF":
            # if condition_for_computing_index_parameter_gof_DF:
            if index_parameter_gof_DF is not None:
                result_dict["gof_df"] = index_parameter_gof_DF

            if self.compute_gradients and self.compute_active_subspaces and not len(gradient_matrix_dict) == 0:
                result_dict["grad_matrix"] = gradient_matrix_dict

            results_array.append((result_dict, runtime))

            if writing_results_to_a_file and curr_working_dir is not None:
                file_path = curr_working_dir / f"flux_df_{unique_run_index}.pkl"
                flux_df.to_pickle(file_path, compression="gzip")
                file_path = curr_working_dir / f"state_df_{unique_run_index}.pkl"
                state_df.to_pickle(file_path, compression="gzip")
                if index_run_and_parameters_dict is not None:  # TODO seems as parameters_dict is never None!
                    file_path = curr_working_dir / f"parameters_HBVSASK_run_{unique_run_index}.pkl"
                    with open(file_path, 'wb') as f:
                        dill.dump(index_run_and_parameters_dict, f)

                # if self.calculate_GoF or self.qoi == "GoF":
                # if condition_for_computing_index_parameter_gof_DF:
                if index_parameter_gof_DF is not None:
                    file_path = curr_working_dir / f"gof_{unique_run_index}.pkl"
                    index_parameter_gof_DF.to_pickle(file_path, compression="gzip")

                if self.compute_gradients and self.compute_active_subspaces and not len(gradient_matrix_dict) == 0:
                    file_path = curr_working_dir / f"gradient_matrix_dict_run_{unique_run_index}.pkl"
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
                        plot_filename = curr_working_dir / f"hbv_sask_{self.basis}_{unique_run_index}_{single_qoi_column}.html"
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
                    plot_filename = curr_working_dir / f"hbv_sask_{self.basis}_{unique_run_index}.html"
                    plot(fig, filename=str(plot_filename), auto_open=False)
                # fig.show()

        return results_array

    def run_model_single_time_stamp(self, date_of_interest, i_s: Optional[List[int]] = [0, ],
                                    parameters: Optional[Union[Dict[str, Any], List[Any]]] = None,
                                    raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs):
        """
        This function runs the HBVSASK model for a single time stamp. It is a wrapper around the self.run() function.

        Parameters:
            Look up at the definition of the run funtion for more details.

        Returns:
            - The self.run() function is called and returned

        """
        assert date_of_interest >= self.start_date_predictions, \
            "Sorry, the date of interest is before start_date_predictions"
        assert date_of_interest <= self.end_date, "Sorry, the date of interest is after end_date"

        # HBV Specific input data - modify the forcing data and initial condition data based on the date of interest
        forcing = kwargs.get("forcing", self.time_series_measured_data_df)
        forcing = forcing.loc[[date_of_interest, ], :]  # .copy()
        initial_condition_df = kwargs.get("initial_condition_df", None)
        if initial_condition_df is None:
            date_of_interest_minus_one = pd.to_datetime(date_of_interest) - pd.DateOffset(days=1)
            initial_condition_df = hbv.read_initial_conditions(
                self.initial_condition_file, timestamp=date_of_interest_minus_one,
                time_column_name=self.time_column_name)
        kwargs["forcing"] = forcing
        kwargs["initial_condition_df"] = initial_condition_df

        return self.run(
            i_s=i_s, parameters=parameters, raise_exception_on_model_break=raise_exception_on_model_break,
            *args, **kwargs
        )

    def _create_flux_df(self, model_output_flux_dict, time_series_list):
        results = pd.DataFrame(
            list(zip(time_series_list,
                     model_output_flux_dict["Q_cms"], model_output_flux_dict["Q_mm"], model_output_flux_dict["AET"],
                     model_output_flux_dict["PET"], model_output_flux_dict["Q1"], model_output_flux_dict["Q1_routed"],
                     model_output_flux_dict["Q2"], model_output_flux_dict["ponding"])),
            columns=[self.time_column_name, 'Q_cms', 'Q_mm', 'AET', 'PET', 'Q1', 'Q1_routed', 'Q2',
                     "ponding"]
        )
        # results.set_index(self.time_column_name, inplace=True)
        return results

    def _dropna_from_df_and_update_simulation_range(self, df, update_simulation_range=False):
        df.dropna(inplace=True)
        # update simulation_range after dropping some rows...
        # TODO Why not just simulation_range = df.time_column_name.values or something like that?
        if update_simulation_range:
            if not df.index.name == self.time_column_name:
                df.set_index(self.time_column_name, inplace=True)
                simulation_range = df.index
                df.reset_index(inplace=True)
                df.rename(columns={"index": self.time_column_name}, inplace=True)
            else:
                simulation_range = df.index
            return simulation_range
        else:
            return None

    def _compute_autoregressive_model_first_order(self, flux_df):
        reset_index = False
        # Make sure that flux_df has time_column_name as index and that it is sorted
        if not flux_df.index.name == self.time_column_name:
            flux_df.set_index(self.time_column_name, inplace=True)
            reset_index = True
        flux_df.sort_index(inplace=True)
        if self.scale_factor_autoregressive_model_first_order is None:
            self.scale_factor_autoregressive_model_first_order = 1.0
        for idx, single_qoi_column in enumerate(self.list_qoi_column):
            new_column_name = "delta_" + single_qoi_column
            if self.list_read_measured_data[idx] and self.list_qoi_column_measured[idx] is not None:
                single_qoi_column_measured = self.list_qoi_column_measured[idx]
            else:
                single_qoi_column_measured = None
            if single_qoi_column_measured is not None and single_qoi_column_measured in flux_df.columns:
                for timestamp in flux_df.index[1:]:
                    previous_timestamp = utility.compute_previous_timestamp(timestamp, self.resolution)
                    flux_df.at[timestamp, new_column_name] = flux_df.at[timestamp, single_qoi_column] - self.scale_factor_autoregressive_model_first_order*flux_df.at[previous_timestamp, single_qoi_column_measured]
            else:
                flux_df[new_column_name] = flux_df[single_qoi_column].diff()
        if reset_index:
            flux_df.reset_index(inplace=True)
            flux_df.rename(columns={"index": self.time_column_name}, inplace=True)

    def _calculate_GoF_on_data_subset(self, ser, df, qoi_column, qoi_column_idx, objective_function_name_qoi):
        """
        This function is used to calculate the GoF on a subset of data, i.e., on a sliding window.

        Parameters:
        :ser (pd.Series): A pandas series representing a subset of data.
        :df (pd.DataFrame): A pandas dataframe representing the whole data.
        :qoi_column (str): The name of the column in the dataframe df that is used for the GoF calculation.
        :qoi_column_idx (int): The index of the column in the list of columns that is used for the GoF calculation.
        :objective_function_name_qoi (str): The name of the objective function that is used for the GoF calculation.
        
        Returns:
        float: The value of the GoF for the subset of data.
        Important assumtion is that qoi_column represents a single column in the dataframe df.
        """
        # df_subset = df.iloc[ser.index]
        df_subset = df.loc[ser.index].copy()
        gof_dict = self._calculate_GoF(
            measuredDF=self.time_series_measured_data_df.copy(), # df_subset,  # self.time_series_measured_data_df.copy(),  
            simulatedDF=df_subset,
            gof_list=objective_function_name_qoi,
            measuredDF_time_column_name=self.time_column_name,
            measuredDF_column_name=self.list_qoi_column_measured[qoi_column_idx],
            simulatedDF_time_column_name=self.time_column_name,
            simulatedDF_column_name=qoi_column,
            parameters_dict=None,
            return_dict=True
        )
        return gof_dict[objective_function_name_qoi]

    def _compute_rolling_function_over_qoi(self, df, single_qoi_column, center=False, win_type=None):
        if self.method == "avrg":
            # new_column_name = single_qoi_column + "_avrg_sliding_window"
            if self.center == "left":
                result = df[single_qoi_column].loc[::-1].rolling(
                    window=self.interval, min_periods=self.min_periods,
                    center=False, win_type=win_type).mean()
            else:
                result = df.rolling(window=self.interval, min_periods=self.min_periods,
                                    center=center, win_type=win_type)[single_qoi_column].mean()
        elif self.method == "min":
            # new_column_name = single_qoi_column + "_min_sliding_window"
            if self.center == "left":
                result = df[single_qoi_column].loc[::-1].rolling(
                    window=self.interval, min_periods=self.min_periods,
                    center=False, win_type=win_type).min()
            else:
                result = df.rolling(window=self.interval, min_periods=self.min_periods,
                                    center=center, win_type=win_type)[single_qoi_column].min()
        elif self.method == "max":
            # new_column_name = single_qoi_column + "_max_sliding_window"
            if self.center == "left":
                result = df[single_qoi_column].loc[::-1].rolling(
                    window=self.interval, min_periods=self.min_periods,
                    center=False, win_type=win_type).max()
            else:
                result = df.rolling(window=self.interval, min_periods=self.min_periods,
                                    center=center, win_type=win_type)[single_qoi_column].max()
        else:
            raise Exception(f"[ERROR:] method should be either \"avrg\" or \"max\" or \"min\"")

        new_column_name = single_qoi_column + "_" + self.method + "_sliding_window"

        return result, new_column_name

    def _calculate_GoF(self, measuredDF, simulatedDF,
                       gof_list=None, measuredDF_time_column_name=None, measuredDF_column_name=None,
                       simulatedDF_time_column_name=None, simulatedDF_column_name=None,
                       parameters_dict=None, return_dict=False):
        """
        Important sssumption - that simulatedDF stores as well measured data; 
        also qoi_column represents a SINGLE column in the dataframe df.
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
            simulatedDF=simulatedDF,
            gof_list=gof_list,
            measuredDF_time_column_name=measuredDF_time_column_name,
            measuredDF_column_name=measuredDF_column_name,
            simulatedDF_time_column_name=simulatedDF_time_column_name,
            simulatedDF_column_name=simulatedDF_column_name,
            return_dict=True,
        )

        gof_dict["qoi"] = simulatedDF_column_name

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
            
    def _compute_gradient_matrix(
            self, unique_run_index, flux_df, parameters_dict, index_parameter_gof_DF,
            time_series_list, simulation_range, center=False, **kwargs):

        h_vector = []
        dict_of_grad_estimation_vector = defaultdict(list)
        # gradient_vectors_dict = defaultdict(list)
        # gradient_vectors_param_dict = defaultdict(list)

        list_of_columns_to_filter = [self.time_column_name, ] + self.list_qoi_column

        # flux_df.set_index(self.time_column_name, inplace=True)

        forcing = kwargs.get("forcing", self.time_series_measured_data_df)
        long_term = kwargs.get("long_term", self.precipitation_temperature_monthly_df)
        initial_condition_df = kwargs.get("initial_condition_df", self.initial_condition_df)

        # dict_param_info_from_configurationObject = utility.get_param_info_dict_from_configurationObject(
        #     self.configurationObject)
        dict_param_info = utility.get_param_info_dict(
            default_par_info_dict=hbv.DEFAULT_PAR_INFO_DICT,
            configurationObject=self.configurationObject['parameters'], 
            )

        # CD = 1 central differences; CD = 0 forward differences
        # Assumption: parameters_dict is a dictionary of parameters of interest already computed above
        parameter_index_to_perturb = 0  # this 0 id will mark the run with unchanged parameters vector
        for single_param_name, single_param_value in parameters_dict.items():
            parameter_index_to_perturb += 1

            updated_parameter_dict = parameters_dict.copy()

            # 2.1 Update parameter value
            single_dict_param_info = dict_param_info[single_param_name]
            parameter_lower_limit = single_dict_param_info["lower_limit"]
            parameter_upper_limit = single_dict_param_info["upper_limit"]
            if parameter_lower_limit is None or parameter_upper_limit is None:
                raise Exception(
                    'ERROR in computing a gradient of QoI wrt parameter: '
                    'parameter_lower_limit and/or parameter_upper_limit are not specified in configurationObject!')
            else:
                param_h = self.eps_val_global * (parameter_upper_limit - parameter_lower_limit)
                parameter_lower_limit += param_h
                parameter_upper_limit -= param_h

            updated_parameter_dict[single_param_name] = single_param_value + param_h

            # 2.2 Run the model; forward run always; backward run only when CD=1 & 2.3. Do some postprocessing
            flux_plus_h, _ = hbv.HBV_SASK(
                forcing=forcing,
                long_term=long_term,
                par_values_dict=updated_parameter_dict,
                initial_condition_df=initial_condition_df,
                printing=False,
                time_column_name=self.time_column_name,
                precipitation_column_name=self.precipitation_column_name,
                temperature_column_name=self.temperature_column_name,
                long_term_precipitation_column_name=self.long_term_precipitation_column_name,
                long_term_temperature_column_name=self.long_term_temperature_column_name
            )
            h = param_h

            # Create a final df - flux with +dh parameter value
            # flux_plus_h_df = pd.DataFrame(
            #     list(zip(time_series_list,
            #              flux_plus_h["Q_cms"], flux_plus_h["Q_mm"], flux_plus_h["AET"], flux_plus_h["PET"],
            #              flux_plus_h["Q1"], flux_plus_h["Q1_routed"], flux_plus_h["Q2"],
            #              flux_plus_h["ponding"])),
            #     columns=[self.time_column_name, 'Q_cms', 'Q_mm', 'AET', 'PET', 'Q1', 'Q1_routed', 'Q2',
            #              "ponding"]
            # )
            flux_plus_h_df = self._create_flux_df(flux_plus_h, time_series_list)

            # Preparations before computing GoF
            flux_plus_h_df = flux_plus_h_df[list_of_columns_to_filter]
            flux_plus_h_df['Index_run'] = unique_run_index
            flux_plus_h_df["Parameter_index_to_perturb"] = parameter_index_to_perturb
            flux_plus_h_df['Sub_index_run'] = 0
            flux_plus_h_df = flux_plus_h_df[
                flux_plus_h_df[self.time_column_name].isin(simulation_range)]
            flux_plus_h_df.set_index(self.time_column_name, inplace=True)

            for idx, single_qoi_column in enumerate(self.list_qoi_column):
                single_transformation = self.list_transform_model_output[idx]
                if single_transformation is not None:
                    # new_column_name = single_transformation + "_" + single_qoi_column
                    new_column_name = single_qoi_column
                    utility.transform_column_in_df(
                        flux_plus_h_df, transformation_function_str=single_transformation,
                        column_name=single_qoi_column, new_column_name=new_column_name)

            if self.CD:
                updated_parameter_dict[single_param_name] = single_param_value - param_h

                # Run the model for -dh
                flux_minus_h, _ = hbv.HBV_SASK(
                    forcing=forcing,
                    long_term=long_term,
                    par_values_dict=updated_parameter_dict,
                    initial_condition_df=initial_condition_df,
                    printing=False,
                    time_column_name=self.time_column_name,
                    precipitation_column_name=self.precipitation_column_name,
                    temperature_column_name=self.temperature_column_name,
                    long_term_precipitation_column_name=self.long_term_precipitation_column_name,
                    long_term_temperature_column_name=self.long_term_temperature_column_name
                )
                h = 2*param_h

                # Create a final df - flux with -dh parameter value
                # flux_minus_h_df = pd.DataFrame(
                #     list(zip(time_series_list,
                #              flux_minus_h["Q_cms"], flux_minus_h["Q_mm"], flux_minus_h["AET"], flux_minus_h["PET"],
                #              flux_minus_h["Q1"], flux_minus_h["Q1_routed"], flux_minus_h["Q2"],
                #              flux_minus_h["ponding"])),
                #     columns=[self.time_column_name, 'Q_cms', 'Q_mm', 'AET', 'PET', 'Q1', 'Q1_routed', 'Q2',
                #              "ponding"]
                # )
                flux_minus_h_df = self._create_flux_df(flux_minus_h, time_series_list)

                # Preparations before computing GoF
                flux_minus_h_df = flux_minus_h_df[list_of_columns_to_filter]
                flux_minus_h_df['Index_run'] = unique_run_index
                flux_minus_h_df["Parameter_index_to_perturb"] = parameter_index_to_perturb
                flux_minus_h_df['Sub_index_run'] = 1
                flux_minus_h_df = flux_minus_h_df[
                    flux_minus_h_df[self.time_column_name].isin(simulation_range)]
                flux_minus_h_df.set_index(self.time_column_name, inplace=True)

                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    single_transformation = self.list_transform_model_output[idx]
                    if single_transformation is not None:
                        # new_column_name = single_transformation + "_" + single_qoi_column
                        new_column_name = single_qoi_column
                        utility.transform_column_in_df(flux_minus_h_df,
                                                        transformation_function_str=single_transformation,
                                                        column_name=single_qoi_column,
                                                        new_column_name=new_column_name)

            h_vector.append(h)

            # 2.4. Compute gradients
            if self.qoi == "GoF":
                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    if self.list_read_measured_data[idx]:
                        if self.mode == "continuous":
                            if self.CD:
                                dict_single_qoi_minus_h = self._calculate_GoF(
                                    measuredDF=self.time_series_measured_data_df,
                                    simulatedDF=flux_minus_h_df,
                                    gof_list=self.objective_function_qoi,
                                    measuredDF_time_column_name=self.time_column_name,
                                    simulatedDF_time_column_name=self.time_column_name,
                                    measuredDF_column_name=self.list_qoi_column_measured[idx],
                                    simulatedDF_column_name=single_qoi_column,
                                    parameters_dict=None,
                                    return_dict=True
                                )

                            dict_single_qoi_plus_h = self._calculate_GoF(
                                measuredDF=self.time_series_measured_data_df,
                                simulatedDF=flux_plus_h_df,
                                gof_list=self.objective_function_qoi,
                                measuredDF_time_column_name=self.time_column_name,
                                simulatedDF_time_column_name=self.time_column_name,
                                measuredDF_column_name=self.list_qoi_column_measured[idx],
                                simulatedDF_column_name=single_qoi_column,
                                parameters_dict=None,
                                return_dict=True
                            )

                            for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
                                if self.CD:
                                    f_x_ij_m_h = dict_single_qoi_minus_h[single_objective_function_name_qoi]
                                    f_x_ij_p_h = dict_single_qoi_plus_h[single_objective_function_name_qoi]
                                    grad = (f_x_ij_p_h - f_x_ij_m_h)/h
                                else:
                                    f_x_ij_p_h = dict_single_qoi_plus_h[single_objective_function_name_qoi]
                                    f_x_ij = \
                                        index_parameter_gof_DF.loc[(index_parameter_gof_DF["qoi"] == single_qoi_column)][
                                            single_objective_function_name_qoi].values[0]
                                    grad = (f_x_ij_p_h - f_x_ij) / h

                                new_column_name = "d_" + single_objective_function_name_qoi + "_" + \
                                                    single_qoi_column + "_d_" + single_param_name
                                flux_df[new_column_name] = grad

                                # in this case grad should be a single float number
                                dict_of_grad_estimation_vector[
                                    (single_qoi_column,single_objective_function_name_qoi)].append(grad)

                        elif self.mode == "sliding_window":
                            for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
                                # def _calculate_GoF_on_data_subset(ser, df):
                                #     df_subset = df.iloc[ser.index]
                                #     gof_dict = self._calculate_GoF(
                                #         measuredDF=self.time_series_measured_data_df,
                                #         simulatedDF=df_subset,
                                #         gof_list=single_objective_function_name_qoi,
                                #         measuredDF_time_column_name=self.time_column_name,
                                #         simulatedDF_time_column_name=self.time_column_name,
                                #         measuredDF_column_name=self.list_qoi_column_measured[idx],
                                #         simulatedDF_column_name=single_qoi_column,
                                #         parameters_dict=None,
                                #         return_dict=True
                                #     )
                                #     return gof_dict[single_objective_function_name_qoi]

                                new_column_name = single_objective_function_name_qoi + "_" + single_qoi_column + \
                                                    "_sliding_window"

                                if self.CD:
                                    rol = flux_minus_h_df[single_qoi_column].rolling(
                                        window=self.interval,
                                        min_periods=self.min_periods,
                                        center=center, win_type=None
                                    )
                                    flux_minus_h_df[new_column_name] = rol.apply(
                                        self._calculate_GoF_on_data_subset, raw=False,
                                        args=(flux_minus_h_df, single_qoi_column, idx,
                                                single_objective_function_name_qoi)
                                    )

                                rol = flux_plus_h_df[single_qoi_column].rolling(
                                    window=self.interval,
                                    min_periods=self.min_periods,
                                    center=center, win_type=None
                                )
                                flux_plus_h_df[new_column_name] = rol.apply(
                                    self._calculate_GoF_on_data_subset, raw=False,
                                    args=(flux_plus_h_df, single_qoi_column, idx,
                                            single_objective_function_name_qoi)
                                )

                                # flux_df = flux_df.loc[simulation_range]
                                if self.CD:
                                    grad = (flux_plus_h_df[new_column_name] - flux_minus_h_df[new_column_name]) / h
                                else:
                                    grad = (flux_plus_h_df[new_column_name] - flux_df[new_column_name]) / h

                                new_column_name = "d_" + single_objective_function_name_qoi + "_" + \
                                                    single_qoi_column + "_sliding_window" \
                                                    + "_d_" + single_param_name
                                flux_df[new_column_name] = grad

                                grad = grad.dropna()
                                dict_of_grad_estimation_vector[
                                    (single_qoi_column, single_objective_function_name_qoi)].append(
                                    grad.values.tolist())

                        elif self.mode == "resampling":
                            raise NotImplementedError
            else:
                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    if self.mode == "continuous":
                        if self.CD:
                            flux_plus_h_df, flux_minus_h_df = utility.filter_two_DF_on_common_timesteps(
                                flux_plus_h_df, flux_minus_h_df, column_name_df1=self.time_column_name,
                                column_name_df2=self.time_column_name)
                            grad = (flux_plus_h_df[single_qoi_column] - flux_minus_h_df[single_qoi_column]) / h
                        else:
                            # flux_df, flux_plus_h_df = utility.filter_two_DF_on_common_timesteps(
                            #     flux_df, flux_plus_h_df, column_name_df1=self.time_column_name)
                            # grad = (flux_plus_h_df[single_qoi_column].tolist() - flux_df[single_qoi_column].tolist()) / h
                            grad = (flux_plus_h_df[single_qoi_column] - flux_df[single_qoi_column]) / h

                        new_column_name = "d_" + single_qoi_column + "_d_" + single_param_name
                        flux_df[new_column_name] = grad

                        grad = grad.dropna()
                        dict_of_grad_estimation_vector[single_qoi_column].append(grad.values.tolist())

                    elif self.mode == "sliding_window":
                        if self.CD:
                            ser, new_column_name = self._compute_rolling_function_over_qoi(flux_minus_h_df,
                                                                                            single_qoi_column,
                                                                                            center=center,
                                                                                            win_type=None)
                            flux_minus_h_df[new_column_name] = ser

                        ser, new_column_name = self._compute_rolling_function_over_qoi(flux_plus_h_df,
                                                                                        single_qoi_column,
                                                                                        center=center,
                                                                                        win_type=None)
                        flux_plus_h_df[new_column_name] = ser

                        if self.CD:
                            grad = (flux_plus_h_df[new_column_name] - flux_minus_h_df[new_column_name]) / h
                        else:
                            grad = (flux_plus_h_df[new_column_name] - flux_df[new_column_name]) / h

                        new_column_name = "d_" + single_qoi_column + "_" + self.method + "_sliding_window" + \
                                            "_d_" + single_param_name
                        flux_df[new_column_name] = grad

                        grad = grad.dropna()
                        dict_of_grad_estimation_vector[single_qoi_column].append(grad.values.tolist())

                    elif self.mode == "resampling":
                        raise NotImplementedError()

            # 2.4.1. Save pure runs
            if self.save_gradient_related_runs:
                # for idx, single_qoi_column in enumerate(self.list_qoi_column):
                suffixes_name = '_' + single_param_name
                flux_df = flux_df.merge(
                    flux_plus_h_df[self.list_qoi_column], left_index=True, right_index=True,
                    suffixes=(None, suffixes_name)
                )
                if self.CD:
                    suffixes_name = '_' + single_param_name + "_m"
                    flux_df = flux_df.merge(
                        flux_minus_h_df[self.list_qoi_column], left_index=True, right_index=True,
                        suffixes=(None, suffixes_name)
                    )

        # 3. Process data for generating gradient matrices when computing active subspaces; the computation of active subspaces
        # is done in the Statistics class where all the model runs are processed together
        gradient_matrix_dict = dict()
        if self.compute_active_subspaces:
            for idx, single_qoi_column in enumerate(self.list_qoi_column):
                if self.qoi == "GoF":
                    if self.mode == "continuous":
                        for single_objective_function_qoi in self.list_objective_function_names_qoi:
                            if self.list_read_measured_data[idx]:
                                grad_estimation_vector = dict_of_grad_estimation_vector[
                                    (single_qoi_column, single_objective_function_qoi)]
                                # TODO Check this!
                                gradient_matrix_dict[(single_qoi_column, single_objective_function_qoi)] = \
                                    np.outer(grad_estimation_vector,grad_estimation_vector)
                    elif self.mode == "sliding_window":
                        # TODO implement this in a similar way as below when qoi is some model output
                        raise NotImplementedError()
                    elif self.mode == "resampling":
                        raise NotImplementedError()
                else:
                    gradient_matrix_dict[single_qoi_column] = []
                    # TODO Transform the long vector into the matrix such that time is a 2nd/3rd dimension
                    print(f"[DEBUGGING] type(dict_of_grad_estimation_vector[single_qoi_column]) - {type(dict_of_grad_estimation_vector[single_qoi_column])}")
                    data = np.array(dict_of_grad_estimation_vector[single_qoi_column])
                    print(f"[DEBUGGING] dict_of_grad_estimation_vector[single_qoi_column].shape - {data.shape}")
                    gradient_data_in_matrix_form = data.reshape(
                        (parameter_index_to_perturb, len(simulation_range))).transpose()
                    print(f"[DEBUGGING] gradient_data_in_matrix_form.shape - {gradient_data_in_matrix_form.shape}")
                    # TODO I am not sure about this!
                    for single_time_step in range(len(simulation_range)):
                        gradient_matrix_for_single_time_step = np.outer(
                            gradient_data_in_matrix_form[single_time_step], gradient_data_in_matrix_form[single_time_step])
                        gradient_matrix_dict[single_qoi_column].append(gradient_matrix_for_single_time_step)

        # flux_df.reset_index(inplace=True)
        # flux_df.rename(columns={"index": self.time_column_name}, inplace=True)

        return flux_df, gradient_matrix_dict