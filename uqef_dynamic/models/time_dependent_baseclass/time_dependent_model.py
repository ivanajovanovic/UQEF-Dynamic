"""
This module is still work-in-progress

It should eventually contain the general code extracted from HBVSASKModel
"""
from abc import ABC, abstractmethod
import copy
import datetime
import dill
from distutils.util import strtobool
import json
import pandas as pd
from pathlib import Path
import time
from typing import List, Optional, Dict, Any, Union

from uqef.model import Model
from uqef_dynamic.utils import utility


class TimeDependentModelConfig(object):
    def __init__(self, configurationObject, deep_copy=False, *args, **kwargs):
        if configurationObject is None:
            self.configurationObject = dict()
        elif not isinstance(configurationObject, dict):
            self.configurationObject = utility.return_configuration_object(configurationObject)
        elif deep_copy:
            self.configurationObject = copy.deepcopy(configurationObject)
        else:
            self.configurationObject = configurationObject

        self.dict_config_time_settings = self.configurationObject.get("time_settings", dict())
        self.timeframe = None
        self.warm_up_duration = None
        self.cut_runs = False
        self.timestep = 5
        self.timestep_in_hours = False

        self.dict_config_model_settings = self.configurationObject.get("model_settings", dict())
        self.boolean_writing_results_to_a_file = False
        self.boolean_make_local_copy_of_master_dir = False
        self.boolean_run_unaltered_sim = False
        self.raise_exception_on_model_break = True
        self.max_retries = None

        self.dict_config_model_paths = self.configurationObject.get("model_paths", dict())

        dict_config_output_settings = self.configurationObject.get("output_settings", dict())
        self.calculate_GoF = True

        dict_config_simulation_settings = self.configurationObject.get("simulation_settings", dict())

        dict_config_parameters_settings = self.configurationObject.get("parameters_settings", dict())
        dict_config_parameters = self.configurationObject.get("parameters", dict())

        #####################################
        # TODO add that first parameter values are read from kwargs
        self.timeframe = utility.parse_datetime_configuration(self.dict_config_time_settings)
        self.resolution = self.dict_config_time_settings.get("resolution", "integer")
        self.warm_up_duration = self.dict_config_time_settings.get("warm_up_duration", None)
        self.cut_runs = strtobool(self.dict_config_time_settings.get("cut_runs", "False"))
        self.timestep = self.dict_config_time_settings.get("timestep", 5)
        self.timestep_in_hours = strtobool(self.dict_config_time_settings.get("timestep_in_hours", "False"))

        if "run_full_timespan" in kwargs:
            self.run_full_timespan = kwargs['run_full_timespan']
        else:
            self.run_full_timespan = strtobool(self.dict_config_time_settings.get(
                "run_full_timespan", 'False'))

        #####################################
        if "writing_results_to_a_file" in kwargs:
            self.boolean_writing_results_to_a_file = kwargs['writing_results_to_a_file']
        else:
            self.boolean_writing_results_to_a_file = strtobool(self.dict_config_model_settings.get("writing_results_to_a_file", "False"))
        self.writing_results_to_a_file = self.boolean_writing_results_to_a_file  # legacy

        if "make_local_copy_of_master_dir" in kwargs:
            self.boolean_make_local_copy_of_master_dir = kwargs['make_local_copy_of_master_dir']
        else:
            self.boolean_make_local_copy_of_master_dir = strtobool(
                self.dict_config_model_settings.get("make_local_copy_of_master_dir", "False"))

        if "run_unaltered_sim" in kwargs:
            self.boolean_run_unaltered_sim = kwargs['run_unaltered_sim']
        else:
            self.boolean_run_unaltered_sim = strtobool(self.dict_config_model_settings.get("run_unaltered_sim", "False"))

        if "raise_exception_on_model_break" in kwargs:
            self.raise_exception_on_model_break = kwargs['raise_exception_on_model_break']
        else:
            self.raise_exception_on_model_break = strtobool(
                self.dict_config_model_settings.get("raise_exception_on_model_break", "True"))

        if "max_retries" in kwargs:
            self.max_retries = kwargs['max_retries']
        else:
            self.max_retries = self.dict_config_model_settings.get("max_retries", None)

        if "plotting" in kwargs:
            self.plotting = kwargs['plotting']
        else:
            if "model_settings" in self.configurationObject:
                self.plotting = strtobool(self.configurationObject["model_settings"].get(
                    "plotting", "True"))
            else:
                self.plotting = False 

        if "printing" in kwargs:
            self.printing = kwargs['printing']
        else:
            if "model_settings" in self.configurationObject:
                self.printing = strtobool(self.configurationObject["model_settings"].get(
                    "printing", "True"))
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
        # TODO maybe this is not necessary here...
        self.calculate_GoF = strtobool(dict_config_output_settings.get("calculate_GoF", "True"))
        if self.calculate_GoF:
            self.objective_function = dict_config_output_settings.get("objective_function", "all")
            self.objective_function = utility.gof_list_to_function_names(self.objective_function)

        #####################################

    def toJSON(self):
        def json_default(value):
            if isinstance(value, datetime.datetime):
                return dict(year=value.year, month=value.month, day=value.day, hour=value.hour, minute=value.minute)
            else:
                return value.__dict__
        return json.dumps(self, default=json_default, sort_keys=True, indent=4) #cls=CJsonEncoder, ensure_ascii=False .encode('utf-8')

    @staticmethod
    def json_extract(obj, key):
        """Recursively fetch values from nested JSON."""
        arr = []

        def extract(obj, arr, key):
            """Recursively search for values of key in JSON tree."""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict):
                        extract(v, arr, key)
                    elif k == key:
                        arr.append(v)
            return arr

        values = extract(obj, arr, key)
        return values


class TimeDependentModel(ABC, Model):
    def __init__(self, configurationObject, inputModelDir, workingDir=None, *args, **kwargs):
        if isinstance(configurationObject, TimeDependentModelConfig):
            self.timeDependentModelConfig = configurationObject
        else:
            self.timeDependentModelConfig = TimeDependentModelConfig(configurationObject, deep_copy=False, *args, **kwargs)
        for attr, value in self.timeDependentModelConfig.__dict__.items():
            setattr(self, attr, value)
        # self.configurationObject = self.timeDependentModelConfig.configurationObject  # TODO - remove this eventually
        
        if inputModelDir is not None:
            self.inputModelDir = Path(inputModelDir)
        else:
            self.inputModelDir = None

        if workingDir is None:
            if self.inputModelDir is not None:
                self.workingDir = self.inputModelDir
            else:
                self.workingDir = None #Path.cwd()  
        else:
            self.workingDir = Path(workingDir)
        if self.workingDir is not None:
            self.workingDir.mkdir(parents=True, exist_ok=True)

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

        self.time_column_name = kwargs.get('time_column_name', utility.TIME_COLUMN_NAME)
        self.index_column_name = kwargs.get('index_column_name', utility.INDEX_COLUMN_NAME)

        #####################################
        self.dict_processed_simulation_settings_from_config_file = None

        self._setup(**kwargs)

    def _setup(self, **kwargs):
        self.set_attributes_based_on_dict_processed_simulation_settings_from_config_file(**kwargs)
        self._setup_model_related(**kwargs)
        self._timespan_setup(**kwargs)

    def _setup_model_related(self, **kwargs):
        pass

    def set_attributes_based_on_dict_processed_simulation_settings_from_config_file(self, **kwargs):
        if self.dict_processed_simulation_settings_from_config_file is None:
            self.set_dict_processed_simulation_settings_from_config_file(**kwargs)

        self.assign_values(self.dict_processed_simulation_settings_from_config_file)

    def set_dict_processed_simulation_settings_from_config_file(
            self, dict_processed_simulation_settings_from_config_file=None, **kwargs):
        if dict_processed_simulation_settings_from_config_file is None:
            self.dict_processed_simulation_settings_from_config_file = \
                utility.read_simulation_settings_from_configuration_object(
            self.configurationObject, **kwargs)
        else:
            self.dict_processed_simulation_settings_from_config_file = dict_processed_simulation_settings_from_config_file

    def assign_values(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def _timespan_setup(self, **kwargs):
        """
        TODO Rewrite this; this is too specific for HBV model
        TODO make sure it works both for hourly and daily resolution!
        :param kwargs:
        :return:
        """
        # self.timesteps = None
        if self.run_full_timespan:
            self.start_date, self.end_date = self._get_full_time_span(self.basis)
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
                self.start_date, self.end_date = self._get_full_time_span(self.basis)

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

    def _get_full_time_span(self):
        raise NotImplementedError

    def prepare(self, *args, **kwargs):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def timesteps(self):
        # TODO Rewrite this
        if hasattr(self, "full_data_range"):
            return list(self.full_data_range)
        elif hasattr(self, "t_sol"):
            return self.t_sol
        elif hasattr(self, "t"):
            return self.t
        elif hasattr(self, "list_of_dates_of_interest"):
            return self.list_of_dates_of_interest
        else:
            return None

    def __call__(self, i_s: Optional[List[int]] = [0, ], parameters: Optional[Union[Dict[str, Any], List[Any]]] = None, 
    raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs):
        return self.run(i_s=i_s, parameters=parameters, raise_exception_on_model_break=raise_exception_on_model_break, *args, **kwargs)

    def run(
            self, i_s: Optional[List[int]] = [0, ], 
            parameters: Optional[Union[Dict[str, Any], List[Any]]] = None,
            raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs
            ):
        if raise_exception_on_model_break is None:
            raise_exception_on_model_break = self.raise_exception_on_model_break
        take_direct_value = kwargs.get("take_direct_value", False)
        if self.uq_method == "ensemble":
            take_direct_value = True
        createNewFolder = kwargs.get("createNewFolder", False)
        deleteFolderAfterwards = kwargs.get("deleteFolderAfterwards", True)
        writing_results_to_a_file = kwargs.get("writing_results_to_a_file", self.writing_results_to_a_file)
        printing = kwargs.get("printing", self.printing)
        plotting = kwargs.get("plotting", self.plotting)

        results_array = []
        # parameter = None  # Initialize parameter here
        for ip in range(0, len(i_s)):  # for each piece of work
            start = time.time()

            unique_run_index = i_s[ip]  # Note: i is used in some legacy code as a unique index run

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
            else:
                number_of_uncertain_params = len(parameter)

            parameters_dict = self._parameters_configuration(parameters=parameter, take_direct_value=take_direct_value)
            
            if printing:
                print(f"{unique_run_index} parameters_dict - {parameters_dict} \n")

            # create local directory for this particular run
            if self.workingDir is not None:
                if createNewFolder:
                    curr_working_dir = self.workingDir / f"run_{unique_run_index}"
                    curr_working_dir.mkdir(parents=True, exist_ok=True)
                else:
                    curr_working_dir = self.workingDir
            else:
                curr_working_dir = None

            try:
                model_output = self._model_run(parameters_dict=parameters_dict)
            except:
                result_df = None
                index_run_and_parameters_dict = {**id_dict, **parameters_dict, "successful_run": False}
                if raise_exception_on_model_break:
                    raise Exception(f"[ERROR] Model broke for run {unique_run_index}")
            else:
                index_run_and_parameters_dict = {**id_dict, **parameters_dict, "successful_run": True}
                result_df = self._process_model_output(model_output, unique_run_index)
                self._transform_model_output(result_df)

                if self.index_column_name not in result_df.columns:
                    result_df[self.index_column_name] = unique_run_index
                if self.time_column_name not in result_df.columns:
                    result_df[self.time_column_name] = self.timesteps()
                
                if plotting and curr_working_dir is not None:
                    self._plotting(result_df, unique_run_index, curr_working_dir)
            
            end = time.time()
            runtime = end - start

            if writing_results_to_a_file and curr_working_dir is not None:
                # TODO finish this
                pass

            result_dict = {
                "run_time": runtime,
                "result_time_series":result_df,
                "parameters_dict": index_run_and_parameters_dict
            }

            results_array.append((result_dict, runtime))

        return results_array

    @abstractmethod
    def _parameters_configuration(self, parameters, take_direct_value, *args, **kwargs):
        """
        This function should return a dictionary of parameters to be used in the model.
        This is the first argument of the model_run function.

        Note: it should contain only uncertain parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def _model_run(self, parameters_dict, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _process_model_output(self, model_output, unique_run_index, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _transform_model_output(self, model_output_df, *args, **kwargs):
        raise NotImplementedError
    
    def _plotting(self, result_df, unique_run_index, curr_working_dir):
        pass

    # ====================================================================
    # Code below still requires refactoring
    # ====================================================================

    def run_detailed(
            self, i_s: Optional[List[int]] = [0, ], 
            parameters: Optional[Union[Dict[str, Any], List[Any]]] = None,
            raise_exception_on_model_break: Optional[Union[bool, Any]] = None, *args, **kwargs
            ):
        """
            This is the main function to run the model.

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
        if raise_exception_on_model_break is None:
            raise_exception_on_model_break = self.raise_exception_on_model_break
        take_direct_value = kwargs.get("take_direct_value", False)
        if self.uq_method == "ensemble":
            take_direct_value = True
        createNewFolder = kwargs.get("createNewFolder", False)
        deleteFolderAfterwards = kwargs.get("deleteFolderAfterwards", True)
        writing_results_to_a_file = kwargs.get("writing_results_to_a_file", self.writing_results_to_a_file)
        plotting = kwargs.get("plotting", self.plotting)
        corrupt_forcing_data = kwargs.get("corrupt_forcing_data", self.corrupt_forcing_data)
        merge_output_with_measured_data = kwargs.get("merge_output_with_measured_data", False)
        if any(self.list_calculate_GoF):
            merge_output_with_measured_data = True
        # if not any(self.list_read_measured_data):
        #     merge_output_with_measured_data = False

        results_array = []
        for ip in range(0, len(i_s)):  # for each peace of work
            i = i_s[ip]  # i is unique index run

            if parameters is not None:
                parameter = parameters[ip]
            else:
                parameter = None  # an unaltered run will be executed

            id_dict = {"index_run": i}

            # this indeed represents the number of parameters considered to be uncertain, later on parameters_dict might
            # be extanded with fixed parameters that occure in configurationObject
            if parameter is None:
                number_of_uncertain_params = 0
            elif isinstance(parameter, dict):
                number_of_uncertain_params = len(list(parameter.keys()))
            else:
                number_of_uncertain_params = len(parameter)

            parameters_dict = self._parameters_configuration(parameters=parameter, take_direct_value=take_direct_value)
            # print(f"parameters_dict - {parameters_dict} \n")

            start = time.time()

            # create local directory for this particular run
            if self.workingDir is not None:
                if createNewFolder:
                    curr_working_dir = self.workingDir / f"run_{i}"
                    curr_working_dir.mkdir(parents=True, exist_ok=True)
                else:
                    curr_working_dir = self.workingDir
            else:
                curr_working_dir = None

            # Running the model
            model_output, state = self._model_run(
                par_values_dict=parameters_dict, printing=False, corrupt_forcing_data=corrupt_forcing_data)

            ######################################################################################################
            # Processing model output
            ######################################################################################################
            model_output_df, state_df = self._process_model_output_and_states(model_output, state)

            ######################################################################################################
            # Some basic transformation of model output
            ######################################################################################################
            self._transform_model_output_and_states(model_output_df, merge_output_with_measured_data)

            ######################################################################################################
            # Compute GoFs for the whole time-span in certain set-ups
            ######################################################################################################

            index_run_and_parameters_dict = {**id_dict, **parameters_dict}
            # Note - it does not make sense to have both qoi=GoF and calculate_GoF=True at the same time;
            # logic is inside the function; in other words, this function does not always have a side effect
            index_parameter_gof_DF = None
            condition_for_computing_index_parameter_gof_DF = \
                (self.calculate_GoF and not self.qoi == "GoF") or \
                (self.calculate_GoF and self.qoi == "GoF" and self.mode == "sliding_window")
            if condition_for_computing_index_parameter_gof_DF:
                index_parameter_gof_DF = self._compute_index_parameter_gof_DF(model_output_df, index_run_and_parameters_dict)

            ######################################################################################################
            # process result to compute the final QoI - this part is if QoI should be something
            # different than the model output itself
            # self.qoi = "GoF" | "Q" | ["Q_cms","AET"]
            # self.mode = "continuous" | "sliding_window" | "resampling"
            ######################################################################################################

            processed_time_series_results = None
            # TODO finish this, not nice
            side_effect_return = self._compute_index_parameter_gof_DF(model_output_df, index_run_and_parameters_dict)
            if side_effect_return is not None:
                index_parameter_gof_DF = side_effect_return
            
            ######################################################################################################
            # Computing gradients
            ######################################################################################################
            # TODO mitigate this part
            gradient_matrix_dict = None
            
            ######################################################################################################
            # Final savings and plots
            ######################################################################################################
            end = time.time()
            runtime = end - start

            self._dropna_from_df_and_update_simulation_range(model_output_df, update_simulation_range=True)
            model_output_df = model_output_df.loc[self.simulation_range]

            result_dict = {"run_time": runtime,
                           "result_time_series": model_output_df,
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
                file_path = curr_working_dir / f"model_output_df_{i}.pkl"
                model_output_df.to_pickle(file_path, compression="gzip")
                file_path = curr_working_dir / f"state_df_{i}.pkl"
                state_df.to_pickle(file_path, compression="gzip")
                if index_run_and_parameters_dict is not None:  # TODO seems as parameters_dict is never None!
                    file_path = curr_working_dir / f"parameters_run_{i}.pkl"
                    with open(file_path, 'wb') as f:
                        dill.dump(index_run_and_parameters_dict, f)

                # if self.calculate_GoF or self.qoi == "GoF":
                # if condition_for_computing_index_parameter_gof_DF:
                if index_parameter_gof_DF is not None:
                    file_path = curr_working_dir / f"gof_{i}.pkl"
                    index_parameter_gof_DF.to_pickle(file_path, compression="gzip")

                if self.compute_gradients and self.compute_active_subspaces and not len(gradient_matrix_dict) == 0:
                    file_path = curr_working_dir / f"gradient_matrix_dict_run_{i}.pkl"
                    with open(file_path, 'wb') as f:
                        dill.dump(gradient_matrix_dict, f)

            if plotting:
                # TODO add option for plotting
                pass

        return results_array
    
    def _model_run_depr(self, par_values_dict, printing=False, corrupt_forcing_data=False, **kwargs):
        raise NotImplementedError

    def _process_model_output_and_states(self, model_output, state):
        raise NotImplementedError

    def _transform_model_output_and_states(self, model_output_df, merge_output_with_measured_data):
        for idx, single_qoi_column in enumerate(self.list_qoi_column):
            single_transformation = self.list_transform_model_output[idx]
            if single_transformation is not None and single_transformation != "None":
                # new_column_name = single_transformation + "_" + single_qoi_column
                new_column_name = single_qoi_column
                utility.transform_column_in_df(model_output_df, transformation_function_str=single_transformation,
                                               column_name=single_qoi_column, new_column_name=new_column_name)
                # model_output_df.drop(labels=single_qoi_column, inplace=False)
                # model_output_df.rename(columns={new_column_name: single_qoi_column}, inplace=False)
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
                        utility.transform_column_in_df(model_output_df,
                                                       transformation_function_str=single_transformation,
                                                       column_name=self.list_qoi_column_measured[idx],
                                                       new_column_name=new_column_name)
                        # model_output_df.drop(labels=self.list_qoi_column_measured[idx], inplace=False)
                        # model_output_df.rename(columns={new_column_name: self.list_qoi_column_measured[idx]},
                        #                inplace=False)

    def _compute_index_parameter_gof_DF(self, model_output_df, index_run_and_parameters_dict):
        index_parameter_gof_list_of_dicts = []
        for idx, single_qoi_column in enumerate(self.list_qoi_column):
            if self.list_calculate_GoF[idx] and self.list_read_measured_data[idx]:
                index_parameter_gof_dict_single_qoi = self._calculate_GoF(
                    measuredDF=self.time_series_measured_data_df,
                    predictedDF=model_output_df,
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
        return index_parameter_gof_DF

    def _compute_qoi_based_on_model_output(self, model_output_df, index_run_and_parameters_dict):
        if self.mode == "continuous":
            if self.qoi == "GoF":
                index_parameter_gof_list_of_dicts = []
                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    if self.list_read_measured_data[idx]:
                        index_parameter_gof_dict_single_qoi = self._calculate_GoF(
                            measuredDF=self.time_series_measured_data_df,
                            predictedDF=model_output_df,
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
                            model_output_df[new_column_name] = index_parameter_gof_dict_single_qoi[
                                single_objective_function_name_qoi]
                index_parameter_gof_DF = pd.DataFrame(index_parameter_gof_list_of_dicts)
                return index_parameter_gof_DF
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
                            rol = model_output_df[single_qoi_column].rolling(
                                window=self.interval, min_periods=self.min_periods,
                                center=center, win_type=None
                            )
                            model_output_df[new_column_name] = rol.apply(
                                self._calculate_GoF_on_data_subset, raw=False,
                                args=(model_output_df, single_qoi_column, idx, single_objective_function_name_qoi)
                            )
            else:
                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    ser, new_column_name = self._compute_rolling_function_over_qoi(model_output_df, single_qoi_column,
                                                                                   center=center, win_type=None)
                    model_output_df[new_column_name] = ser
            self._dropna_from_df_and_update_simulation_range(model_output_df, update_simulation_range=True)
        elif self.mode == "resampling":
            pass
        else:
            raise Exception(f"[ERROR] mode should have one of the following values:"
                            f" \"continuous\" or \"sliding_window\" or \"resampling\"")
        return None

    def _dropna_from_df_and_update_simulation_range(self, df, update_simulation_range=False):
        df.dropna(inplace=True)
        # update simulation_range after dropping some rows...
        # TODO Why not just self.simulation_range = df.time_column_name.values or something like that?
        if update_simulation_range:
            if not df.index.name == self.time_column_name:
                df.set_index(self.time_column_name, inplace=True)
                self.simulation_range = df.index
                df.reset_index(inplace=True)
                df.rename(columns={"index": self.time_column_name}, inplace=True)
            else:
                self.simulation_range = df.index

    def _calculate_GoF_on_data_subset(self, ser, df, qoi_column, qoi_column_idx, objective_function_name_qoi):
        # df_subset = df.iloc[ser.index]
        df_subset = df.loc[ser.index]
        gof_dict = self._calculate_GoF(
            measuredDF=self.time_series_measured_data_df,
            predictedDF=df_subset,
            gof_list=objective_function_name_qoi,
            measuredDF_time_column_name=self.time_column_name,
            simulatedDF_time_column_name=self.time_column_name,
            measuredDF_column_name=self.list_qoi_column_measured[qoi_column_idx],
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




