import copy
import datetime
import dill
from distutils.util import strtobool
import json
import os.path as osp
import pathlib
import pandas as pd
import time

from common import utility


class HydroModelConfig(object):
    def __init__(self, configurationObject, deep_copy=False, *args, **kwargs):
        if configurationObject is None:
            self.configurationObject = dict()
        elif not isinstance(configurationObject, dict):
            self.configurationObject = utility.return_configuration_object(configurationObject)
        elif deep_copy:
            self.configurationObject = copy.deepcopy(configurationObject)
        else:
            self.configurationObject = configurationObject

        dict_config_time_settings = self.configurationObject.get("time_settings", dict())
        self.timeframe = None
        self.warm_up_duration = None
        self.cut_runs = False
        self.timestep = 5
        self.timestep_in_hours = False

        dict_config_model_settings = self.configurationObject.get("model_settings", dict())
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
        # TODO add that first parameter values is read from kwargs
        self.timeframe = utility.parse_datetime_configuration(dict_config_time_settings)
        self.warm_up_duration = dict_config_time_settings.get("warm_up_duration", None)
        self.cut_runs = strtobool(dict_config_time_settings.get("cut_runs", "False"))
        self.timestep = dict_config_time_settings.get("timestep", 5)
        self.timestep_in_hours = strtobool(dict_config_time_settings.get("timestep_in_hours", "False"))

        #####################################
        if "writing_results_to_a_file" in kwargs:
            self.boolean_writing_results_to_a_file = kwargs['writing_results_to_a_file']
        else:
            self.boolean_writing_results_to_a_file = strtobool(dict_config_model_settings.get("writing_results_to_a_file", "False"))

        if "make_local_copy_of_master_dir" in kwargs:
            self.boolean_make_local_copy_of_master_dir = kwargs['make_local_copy_of_master_dir']
        else:
            self.boolean_make_local_copy_of_master_dir = strtobool(
                dict_config_model_settings.get("make_local_copy_of_master_dir", "False"))

        if "run_unaltered_sim" in kwargs:
            self.boolean_run_unaltered_sim = kwargs['run_unaltered_sim']
        else:
            self.boolean_run_unaltered_sim = strtobool(dict_config_model_settings.get("run_unaltered_sim", "False"))

        if "raise_exception_on_model_break" in kwargs:
            self.raise_exception_on_model_break = kwargs['raise_exception_on_model_break']
        else:
            self.raise_exception_on_model_break = strtobool(
                dict_config_model_settings.get("raise_exception_on_model_break", "True"))

        if "max_retries" in kwargs:
            self.max_retries = kwargs['max_retries']
        else:
            self.max_retries = dict_config_model_settings.get("max_retries", None)

        #####################################
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




