import copy
from collections import defaultdict
import datetime
import dill
from distutils.util import strtobool
from functools import reduce
import math
import numpy as np
import os
import os.path as osp
import pandas as pd
import pathlib
import subprocess
import time

from uqef.model import Model

import LarsimUtilityFunctions.larsimPaths as paths

from LarsimUtilityFunctions import larsimConfigurationSettings
from LarsimUtilityFunctions import larsimDataPostProcessing
from LarsimUtilityFunctions import larsimDataPreparation
from LarsimUtilityFunctions import larsimInputOutputUtilities
from LarsimUtilityFunctions import larsimTimeUtility
from LarsimUtilityFunctions import Utils as utils


class LarsimConfigurations:
    def __init__(self, configurationObject, deep_copy=False,
                 log_level: int = utils.log_levels.INFO,
                 print_level: int = utils.print_levels.INFO,
                 **kwargs):

        self.log_util = utils.LogUtility(log_level=log_level, print_level=print_level)
        self.log_util.set_print_prefix('LarsimConfigurations')
        self.log_util.set_log_prefix('LarsimConfigurations')

        #####################################
        # Read some configurations from larsimConfigurationSettings file/object
        #####################################

        if not isinstance(configurationObject, dict):
            self.configurationObject = larsimConfigurationSettings.return_configuration_object(configurationObject)
        elif deep_copy:
            self.configurationObject = copy.deepcopy(configurationObject)
        else:
            self.configurationObject = configurationObject

        if "configure_tape10" in kwargs:
            self.boolean_configure_tape10 = kwargs['configure_tape10']
        else:
            try:
                self.boolean_configure_tape10 = strtobool(self.configurationObject["model_settings"]["configure_tape10"])
            except KeyError:
                self.boolean_configure_tape10 = False

        if "copy_and_configure_whms" in kwargs:
            self.boolean_copy_and_configure_whms = kwargs['copy_and_configure_whms']
        else:
            try:
                self.boolean_copy_and_configure_whms = strtobool(
                    self.configurationObject["model_settings"]["copy_and_configure_whms"])
            except KeyError:
                self.boolean_copy_and_configure_whms = False

        if "copy_and_configure_lila_input_files" in kwargs:
            self.boolean_copy_and_configure_lila_input_files = kwargs['copy_and_configure_lila_input_files']
        else:
            try:
                self.boolean_copy_and_configure_lila_input_files = strtobool(
                    self.configurationObject["model_settings"]["copy_and_configure_lila_input_files"])
            except KeyError:
                self.boolean_copy_and_configure_lila_input_files = False

        if "copy_and_configure_kala_input_files" in kwargs:
            self.boolean_copy_and_configure_kala_input_files = kwargs['copy_and_configure_kala_input_files']
        else:
            try:
                self.boolean_copy_and_configure_kala_input_files = strtobool(
                    self.configurationObject["model_settings"]["copy_and_configure_kala_input_files"])
            except KeyError:
                self.boolean_copy_and_configure_kala_input_files = False

        if "make_local_copy_of_master_dir" in kwargs:
            self.boolean_make_local_copy_of_master_dir = kwargs['make_local_copy_of_master_dir']
        else:
            try:
                self.boolean_make_local_copy_of_master_dir = strtobool(
                    self.configurationObject["model_settings"]["make_local_copy_of_master_dir"])
            except KeyError:
                self.boolean_make_local_copy_of_master_dir = False

        if "get_measured_discharge" in kwargs:
            self.boolean_get_measured_discharge = kwargs['get_measured_discharge']
        else:
            try:
                self.boolean_get_measured_discharge = strtobool(
                    self.configurationObject["model_settings"]["get_measured_discharge"])
            except KeyError:
                self.boolean_get_measured_discharge = False

        if "read_date_from_saved_data_dir" in kwargs:
            self.boolean_read_date_from_saved_data_dir = kwargs['read_date_from_saved_data_dir']
        else:
            try:
                self.boolean_read_date_from_saved_data_dir = strtobool(
                    self.configurationObject["model_settings"]["read_date_from_saved_data_dir"])
            except KeyError:
                self.boolean_read_date_from_saved_data_dir = False

        if "get_saved_simulations" in kwargs:
            self.boolean_get_saved_simulations = kwargs['get_saved_simulations']
        else:
            try:
                self.boolean_get_saved_simulations = strtobool(
                    self.configurationObject["model_settings"]["get_saved_simulations"])
            except KeyError:
                self.boolean_get_saved_simulations = False
        self.boolean_get_saved_simulations = self.boolean_get_saved_simulations \
                                                    and self.boolean_read_date_from_saved_data_dir

        if "run_unaltered_sim" in kwargs:
            self.boolean_run_unaltered_sim = kwargs['run_unaltered_sim']
        else:
            try:
                self.boolean_run_unaltered_sim = strtobool(
                    self.configurationObject["model_settings"]["run_unaltered_sim"])
            except KeyError:
                self.boolean_run_unaltered_sim = False

        if "raise_exception_on_model_break" in kwargs:
            self.raise_exception_on_model_break = kwargs['raise_exception_on_model_break']
        else:
            try:
                self.raise_exception_on_model_break = strtobool(
                    self.configurationObject["model_settings"]["raise_exception_on_model_break"])
            except KeyError:
                self.raise_exception_on_model_break = True

        if "max_retries" in kwargs:
            self.max_retries = kwargs['max_retries']
        else:
            try:
                self.max_retries = self.configurationObject["model_settings"]["max_retries"]
            except KeyError:
                self.max_retries = None
        #####################################
        # Set of configuration variables propagated via configuration file related to the output of the model
        # i.e., time settings, define the purpose of the model run, model ouput setting...
        #####################################

        # getting the time span for running the model from the json configuration file
        self.timeframe = None
        self.t = None
        if "time_settings" in self.configurationObject:
            self.timeframe = larsimTimeUtility.parse_datetime_configuration(self.configurationObject)
            self.t = larsimTimeUtility.get_tape10_timesteps(self.timeframe)
        else:
            self.log_util.log_error("time_settings specification is missing from the configuration object!")
            raise Exception(f"[LarsimConfigurations ERROR:] time_settings specification is missing from the configuration object!")

        try:
            self.cut_runs = strtobool(self.configurationObject["time_settings"]["cut_runs"])
        except KeyError:
            self.cut_runs = False

        try:
            self.warm_up_duration = self.configurationObject["time_settings"]["warm_up_duration"]
        except KeyError:
            self.warm_up_duration = None

        # how long one consecutive run should take - used later on in each Larsim run, makes sense only if cut_runs=True
        try:
            self.timestep = self.configurationObject["time_settings"]["timestep"]
        except KeyError:
            self.timestep = 5
        #####################################
        try:
            self.station_of_Interest = self.configurationObject["output_settings"]["station_calibration_postproc"]
        except KeyError:
            self.station_of_Interest = "MARI"

        try:
            self.station_for_model_runs = self.configurationObject["output_settings"]["station_model_runs"]
        except KeyError:
            self.station_for_model_runs = "all"

        try:
            self.type_of_output_of_Interest = self.configurationObject["output_settings"]["type_of_output"]
        except KeyError:
            self.type_of_output_of_Interest = "Abfluss Messung + Vorhersage"

        try:
            self.type_of_output_of_Interest_measured = self.configurationObject["output_settings"]["type_of_output_measured"]
        except KeyError:
            self.type_of_output_of_Interest_measured = "Ground Truth"

        try:
            self.calculate_GoF = strtobool(self.configurationObject["output_settings"]["calculate_GoF"])
        except KeyError:
            self.calculate_GoF = True

        if self.calculate_GoF:
            try:
                self.objective_function = self.configurationObject["output_settings"]["objective_function"]
            except KeyError:
                self.objective_function = "all"
            self.objective_function = larsimDataPostProcessing.gof_list_to_function_names(self.objective_function)

        try:
            self.run_and_save_simulations = strtobool(
                self.configurationObject["output_settings"]["run_and_save_simulations"])
        except KeyError:
            self.run_and_save_simulations = True

        try:
            self.always_save_original_model_runs = strtobool(
                self.configurationObject["output_settings"]["always_save_original_model_runs"])
        except KeyError:
            self.always_save_original_model_runs = False

        # this variable controls if post-processing of the result time series should be done for
        # only self.station_of_Interest (False) or all self.station_for_model_runs (True)
        try:
            self.get_all_possible_stations = strtobool(
                self.configurationObject["output_settings"]["post_processing_for_all_stations"])
        except KeyError:
            self.get_all_possible_stations = True
        #####################################
        # this variable stands for the purpose of LarsimModel run
        # distinguish between different modes / purposes of LarsimModel runs:
        #               calibration, run_and_save_simulations, gradient_computation, UQ_analysis
        # These modes do not have to be mutually exclusive!
        #####################################
        try:
            self.qoi = self.configurationObject["model_run_settings"]["QOI"]
        except KeyError:
            self.qoi = "Q"
        if self.qoi != "Q" and self.qoi != "GoF":
            raise Exception(f"[LarsimConfigurations ERROR] self.qoi should either be \"Q\" or \"GoF\" ")

        try:
            self.mode = self.configurationObject["model_run_settings"]["mode"]
        except KeyError:
            self.mode = "continuous"
        if self.mode != "continuous" and self.mode != "sliding_window" and self.mode != "resampling":
            raise Exception(f"[LarsimConfigurations ERROR] self.mode should have one of the following values:"
                            f" \"continuous\" or \"sliding_window\" or \"resampling\"")

        if self.mode == "sliding_window" or self.mode == "resampling":
            try:
                self.interval = self.configurationObject["model_run_settings"]["interval"]
            except KeyError:
                self.interval = 24
            # if self.interval == "whole":
            #     self.configurationObject["output_settings"]["dailyOutput"] = "True"
            try:
                self.min_periods = self.configurationObject["model_run_settings"]["min_periods"]
            except KeyError:
                self.min_periods = 1
            if self.qoi == "Q":
                try:
                    self.method = self.configurationObject["model_run_settings"]["method"]
                except KeyError:
                    self.method = "avrg"
                if self.method != "avrg" and self.method != "max" and self.method != "min":
                    raise Exception(f"[LarsimConfigurations ERROR:] self.method should be "
                                    f"either \"avrg\" or \"max\" or \"min\"")

        if self.mode == "resampling":
            raise Exception(f"[LarsimConfigurations ERROR] resampling mode is still not implemented")

        # if we want to compute the gradient (of some likelihood fun or output itself) w.r.t parameters
        try:
            self.compute_gradients = strtobool(self.configurationObject["model_run_settings"]["compute_gradients"])
        except KeyError:
            self.compute_gradients = False
        if self.compute_gradients:
            try:
                gradients_method = self.configurationObject["model_run_settings"]["gradients_method"]
            except KeyError:
                gradients_method = "Forward Difference"

            if gradients_method == "Central Difference":
                self.CD = 1  # flag for using Central Differences (with 2 * num_evaluations)
            elif gradients_method == "Forward Difference":
                self.CD = 0  # flag for using Forward Differences (with num_evaluations)
            else:
                raise Exception(f"[LarsimConfigurations ERROR] NUMERICAL GRADIENT EVALUATION ERROR: "
                                f"Only \"Central Difference\" and \"Forward Difference\" supported")
            try:
                # difference for gradient computation
                self.eps_val_global = self.configurationObject["model_run_settings"]["eps_gradients"]
            except KeyError:
                self.eps_val_global = 1e-4

        # if calibration is True some likelihood / objective functions / GoF functio should be calculated
        # from model run and propageted further
        if self.qoi == "GoF" or self.compute_gradients:
            try:
                self.objective_function_qoi = self.configurationObject["model_run_settings"]["objective_function_qoi"]
            except KeyError:
                self.objective_function_qoi = "all"
            self.objective_function_qoi = larsimDataPostProcessing.gof_list_to_function_names(
                self.objective_function_qoi)

        #####################################
        if "tuples_parameters_info" not in self.configurationObject:
            larsimConfigurationSettings.update_configurationObject_with_parameters_info(self.configurationObject)

        self.variable_names = []
        if "tuples_parameters_info" in self.configurationObject:
            for i in self.configurationObject["tuples_parameters_info"]:
                self.variable_names.append(i["name"])
        else:
            try:
                for i in self.configurationObject["parameters"]:
                    self.variable_names.append(i["name"])
                # larsimConfigurationSettings.update_configurationObject_with_parameters_info(self.configurationObject)
            except KeyError:
                self.log_util.log_info("There is no parameters entry in the configurationObject!")
                print(f"[LarsimModel Info] This Larsim Model object has empty variable_names list")


# TODO refactor to do most of the stuff in some function and not in the constructor
class LarsimModelSetUp:
    def __init__(self, configurationObject, inputModelDir, workingDir=None, log_level: int = utils.log_levels.INFO,
                 print_level: int = utils.print_levels.INFO, debug: bool = False, *args, **kwargs):

        #####################################
        # set-up utility variables
        #####################################
        self.log_util = utils.LogUtility(log_level=log_level, print_level=print_level)
        self.debug = debug
        if self.debug:
            self.log_util.log_debug(f'LarsimModelSetUp debug: {self.debug}')
        self.log_util.set_print_prefix('LarsimModelSetUp')
        self.log_util.set_log_prefix('LarsimModelSetUp')

        if isinstance(configurationObject, LarsimConfigurations):
            self.larsimConfObject = configurationObject
        else:
            self.larsimConfObject = LarsimConfigurations(configurationObject, False,
                                                         utils.log_levels.INFO, utils.print_levels.INFO,
                                                         **kwargs)
        #####################################
        self.df_past_sim = None
        self.df_measured = None
        self.df_unaltered_ergebnis = None
        self.df_gof_unaltered_meas = None
        self.df_gof_sim_meas = None

        #####################################
        # Specification of different directories - some are machine / location dependent,
        # adjust path in larsimPaths module and configuration file/object accordingly
        #####################################
        self.inputModelDir = pathlib.Path(inputModelDir)

        if self.larsimConfObject.dict_params_for_defining_paths is not None:
            self.larsimPathsObject = paths.LarsimPathsObject(root_data_dir=inputModelDir,
                                                             **self.larsimConfObject.dict_params_for_defining_paths)
        else:
            self.larsimPathsObject = paths.LarsimPathsObject(root_data_dir=inputModelDir)

        self.sourceDir = pathlib.Path(kwargs.get('sourceDir')) \
            if 'sourceDir' in kwargs and pathlib.Path(kwargs.get('sourceDir')).is_absolute() \
            else pathlib.Path(__file__).resolve().parents[1]

        self.workingDir = pathlib.Path(workingDir)
        if self.workingDir is None:
            self.workingDir = pathlib.Path(paths.workingDir)

        if self.larsimConfObject.boolean_make_local_copy_of_master_dir:
            self.master_dir = self.workingDir / 'master_configuration'
        else:
            self.master_dir = self.workingDir # self.larsimPathsObject.global_master_dir

        log_filename = self.workingDir / "larsimModelSetUp.log"
        self.log_util.set_log_filename(log_filename=log_filename)

        self.log_util.log_info(f"inputModelDir = {self.inputModelDir}")
        self.log_util.log_info(f"sourceDir = {self.sourceDir}")
        self.log_util.log_info(f"workingDir = {self.workingDir}")
        self.log_util.log_info(f"local_master_dir = {self.master_dir}")

        self.perform_all_setup_steps()

        self.log_util.log_info(f"Model Initial setup is done!")

    def perform_all_setup_steps(self):
        self.copy_master_folder()
        self.configure_master_folder()

        if self.larsimConfObject.boolean_get_measured_discharge:
            # get the path to the files which stores grount-truth/measured data
            if self.larsimConfObject.boolean_read_date_from_saved_data_dir:
                if "ground_truth_file" in kwargs:
                    ground_truth_file = pathlib.Path(kwargs.get('ground_truth_file'))
                else:
                    if self.larsimConfObject.dict_params_for_defining_paths is not None:
                        ground_truth_file = self.larsimConfObject.dict_params_for_defining_paths["ground_truth_file"]
                    else:
                        raise Exception(
                            f"[LarsimModelSetUp ERROR] - error when reading ground truth data from saved file!")
                read_from_prepared_file = True
                if ground_truth_file.is_absolute():
                    read_file_path = ground_truth_file
                else:
                    read_file_path = self.larsimPathsObject.saved_data_dir / ground_truth_file
            else:
                read_from_prepared_file = False
                read_file_path = self.master_dir / self.larsimPathsObject.list_input_lila_files[0]  # TODO
            self.get_measured_discharge(path_to_measured_data=read_file_path,
                                        read_from_prepared_file=read_from_prepared_file,
                                        write_in_file=True)

        if self.larsimConfObject.boolean_get_saved_simulations:
            self.get_saved_simulations(write_in_file=True)
        if self.larsimConfObject.boolean_run_unaltered_sim:
            self.run_unaltered_sim(createNewFolder=False, write_in_file=True)
        if self.larsimConfObject.boolean_get_measured_discharge and self.larsimConfObject.boolean_run_unaltered_sim:
            self._compare_measurements_and_unalteredSim(get_all_possible_stations=True, write_in_file=True)

    def copy_master_folder(self):
        # for safety reasons make a copy of the global_master_dir in the working_dir and continue working with that one
        if self.master_dir.resolve().absolute() != self.larsimPathsObject.global_master_dir.resolve().absolute():
            self.master_dir.mkdir(parents=True, exist_ok=True)
            master_dir_for_copying = str(self.larsimPathsObject.global_master_dir) + "/."
            subprocess.run(['cp', '-a', master_dir_for_copying, self.master_dir])
            if self.larsimConfObject.boolean_copy_and_configure_whms:
                larsimConfigurationSettings._delete_larsim_whm_files(self.master_dir)
            if self.larsimConfObject.boolean_copy_and_configure_lila_input_files:
                larsimConfigurationSettings._delete_larsim_lila_files(self.master_dir)
            self.log_util.log_info(f"Copying to local master folder!")

    def configure_master_folder(self):
        """
        Copy configuration files & do all the configurations needed for proper execution
        & copy initial and input data files to master folder
        :return:
        """
        # # Get the timeframe for running the simulation from the configuration file
        # self.timeframe = larsimTimeUtility.parse_datetime_configuration(self.configurationObject)

        if not self.master_dir.is_dir():
            self.log_util.log_error(f"Please first create the following folder: {self.master_dir}!")
            raise IOError(f'[LarsimModelSetUp Error] Please first create the following folder: '
                          f'{self.master_dir} {IOError.strerror}')

        # Based on time settings change tape10_master file - needed for unaltered run -
        # this will be repeated once again by each process in LarsimModel.run()
        if self.larsimConfObject.boolean_configure_tape10:
            local_tape10_path = self.master_dir / 'tape10'
            global_master_tape10_file = self.larsimPathsObject.master_tape_10_path
            # global_master_tape10_file = local_tape10_path_copied_from_master = self.larsimPathsObject.global_master_dir / global_master_tape10_file.name
            try:
                larsimTimeUtility.tape10_configuration(timeframe=self.larsimConfObject.timeframe,
                                                       master_tape10_file=global_master_tape10_file, \
                                                       new_tape10_file=local_tape10_path,
                                                       warm_up_duration=self.larsimConfObject.warm_up_duration)
            except larsimTimeUtility.ValidDurationError as e:
                self.log_util.log_error(f"Error in configure_master_folder - time settings - tape10_configuration!")
                print("[LarsimModelSetUp ERROR] - Something is  wrong with the time settings \n]" + str(e))
                return None

        # Filter out whm files
        if self.larsimConfObject.boolean_copy_and_configure_whms:
            larsimConfigurationSettings.copy_whm_files(timeframe=self.larsimConfObject.timeframe,
                                                       input_whms_dir=self.larsimPathsObject.whms_dir,
                                                       new_whms_dir=self.master_dir)

        # Parse big lila files and create small ones
        if self.larsimConfObject.boolean_copy_and_configure_lila_input_files:
            list_local_lila_paths = [self.master_dir / single_lila_file \
                                     for single_lila_file in self.larsimPathsObject.list_input_lila_files]
            larsimConfigurationSettings.master_lila_parser_based_on_time_crete_new(timeframe=self.larsimConfObject.timeframe,
                                                                                   master_lila_paths=self.larsimPathsObject.list_input_master_lila_paths,
                                                                                   new_lila_paths=list_local_lila_paths)

            for one_lila_file in list_local_lila_paths:
                paths.check_if_file_exists(one_lila_file, f"[LarsimModelSetUp Error] File {one_lila_file} does not exist!")

        # Parse big kala files and create small ones
        if self.larsimConfObject.boolean_copy_and_configure_kala_input_files:
            raise NotImplementedError("Should have implemented this")

        self.log_util.log_info(f"Initial configuration of local master folder {self.master_dir} is completed!")

    def get_measured_discharge(self, path_to_measured_data, read_from_prepared_file=False,
                               filtered_timesteps_vs_station_values=True, write_in_file=True,
                               write_file_path=None, **kwargs):
        #####################################
        # extract measured (ground truth) discharge values
        # there are multiple ways how one can do that
        # This function is tailored for WHM Regen Larsim Model, some things might have to be changed for other models...
        #####################################
        if not path_to_measured_data.is_file():
            raise Exception(f"[LarsimModelSetUp ERROR] - get_measured_discharge")

        if read_from_prepared_file:
            self.df_measured = larsimDataPostProcessing.read_process_write_discharge(df=path_to_measured_data,
                                                                                     timeframe=self.larsimConfObject.timeframe,
                                                                                     station=self.larsimConfObject.station_for_model_runs,
                                                                                     compression="gzip"
                                                                                     )
        else:
            # example for this branch is when path_to_measured_data is of type  "./q_2003-11-01_2018-01-01_time_and_values_filtered.pkl"
            # however it might be that this file does not exist. In that case one will read and process/filter the whole dataFrame again
            drop_duplicates = kwargs["drop_duplicates"] if "drop_duplicates" in kwargs else True
            fill_missing_timesteps = kwargs["fill_missing_timesteps"] if "fill_missing_timesteps" in kwargs else True
            interpolate_missing_values = kwargs["interpolate_missing_values"] if "interpolate_missing_values" in kwargs else True
            interpolation_method = kwargs["interpolation_method"] if "interpolation_method" in kwargs else 'time'
            self.df_measured = larsimDataPreparation.get_filtered_df(df=path_to_measured_data,
                                                                     stations=self.larsimConfObject.station_for_model_runs,
                                                                     start_date=self.larsimConfObject.timeframe[0],
                                                                     end_date=self.larsimConfObject.timeframe[1],
                                                                     drop_duplicates=drop_duplicates,
                                                                     fill_missing_timesteps=fill_missing_timesteps,
                                                                     interpolate_missing_values=interpolate_missing_values,
                                                                     interpolation_method=interpolation_method,
                                                                     only_time_series_values=filtered_timesteps_vs_station_values,
                                                                     )
        if filtered_timesteps_vs_station_values:
            #TODO check the format of self.df_measured, if not timesteps_vs_station_values throw an error
            pass

        if write_in_file:
            if write_file_path is None:
                write_file_path = self.workingDir / "df_measured.pkl"
            larsimInputOutputUtilities.write_dataFrame_to_file(self.df_measured,
                                                               file_path=write_file_path,
                                                               compression="gzip")

    # TODO - Rewrite this function such that it is more general, and can be used for other Larsim models besides Regen
    def get_saved_simulations(self, filtered_timesteps_vs_station_values=True, write_in_file=True,
                              write_file_path=None, *args, **kwargs):
        """
        This function is tailored for WHM Regen Larsim Model, some things might have to be changed for other models...
        """
        if not self.larsimConfObject.boolean_get_saved_simulations:
            raise Exception(f"[LarsimModelSetUp ERROR] - you have tried to run the get_saved_simulations "
                            f"function, however boolean_get_Larsim_saved_simulations is set to False!")

        list_of_df_per_station = []
        if self.larsimConfObject.station_for_model_runs is None or self.larsimConfObject.station_for_model_runs == "all":
            station_for_model_runs = list(larsimDataPostProcessing.get_Stations(self.df_measured))
        if not isinstance(station_for_model_runs, list):
            station_for_model_runs = [station_for_model_runs, ]
        for station in station_for_model_runs:
            df_station_sim_path = self.larsimPathsObject.saved_data_dir / f"larsim_output_{station}_2005_2017.pkl"
            df_station_sim_filtered_path = self.larsimPathsObject.saved_data_dir / f"larsim_output_{station}_2005_2017_filtered.pkl"
            if df_station_sim_filtered_path.is_file():
                df_sim = larsimInputOutputUtilities.read_dataFrame_from_file(df_station_sim_filtered_path, compression="gzip")
                df_sim = larsimDataPostProcessing.parse_df_based_on_time(df_sim, (self.larsimConfObject.timeframe[0], self.larsimConfObject.timeframe[1]))
            elif df_station_sim_path.is_file():
                drop_duplicates = kwargs["drop_duplicates"] if "drop_duplicates" in kwargs else True
                fill_missing_timesteps = kwargs[
                    "fill_missing_timesteps"] if "fill_missing_timesteps" in kwargs else True
                interpolate_missing_values = kwargs[
                    "interpolate_missing_values"] if "interpolate_missing_values" in kwargs else True
                interpolation_method = kwargs["interpolation_method"] if "interpolation_method" in kwargs else 'time'
                df_sim = larsimDataPreparation.get_filtered_df(df=df_station_sim_path,
                                                               start_date=self.larsimConfObject.timeframe[0],
                                                               end_date=self.larsimConfObject.timeframe[1],
                                                               drop_duplicates=drop_duplicates,
                                                               fill_missing_timesteps=fill_missing_timesteps,
                                                               interpolate_missing_values=interpolate_missing_values,
                                                               interpolation_method=interpolation_method,
                                                               only_time_series_values=filtered_timesteps_vs_station_values,
                                                               )
            else:
                raise Exception(f"[LarsimModelSetUp ERROR] - you have tried to run the get_saved_simulations "
                                f"function, however neither {df_station_sim_path} nor {df_station_sim_filtered_path}"
                                f"are not proper files!")
            if filtered_timesteps_vs_station_values:
                df_sim.rename(columns={"Value": station}, inplace=True)
            list_of_df_per_station.append(df_sim)

        if filtered_timesteps_vs_station_values:
            self.df_past_sim = reduce(lambda x, y: pd.merge(x, y, on="TimeStamp", how='outer'), list_of_df_per_station)
        else:
            self.df_past_sim = pd.concat(list_of_df_per_station, ignore_index=True, sort=False, axis=0)

        if write_in_file:
            if write_file_path is None:
                write_file_path = self.workingDir / "df_past_simulated.pkl"
            larsimInputOutputUtilities.write_dataFrame_to_file(self.df_past_sim,
                                                               file_path=write_file_path,
                                                               compression="gzip")

    # TODO change run_unaltered_sim such that it can as well run in cut_runs mode
    def run_unaltered_sim(self, createNewFolder=False, write_in_file=True, write_file_path=None):
        #####################################
        ### run unaltered simulation
        #####################################
        if createNewFolder:
            temp = self.larsimPathsObject.larsim_model_dir.name + "000"
            dir_unaltered_run = self.workingDir / temp
            dir_unaltered_run.mkdir(parents=True, exist_ok=True)
            master_dir_for_copying = str(self.master_dir) + "/."
            subprocess.run(['cp', '-a', master_dir_for_copying, dir_unaltered_run])
        else:
            dir_unaltered_run = self.master_dir

        os.chdir(dir_unaltered_run)
        larsimConfigurationSettings._delete_larsim_output_files(curr_directory=dir_unaltered_run)
        local_log_file = dir_unaltered_run / "run.log"
        subprocess.run([str(self.larsimPathsObject.larsim_exe)], stdout=open(local_log_file, 'w'))
        os.chdir(self.sourceDir)
        print(f"[LarsimModelSetUp INFO] Unaltered Run is completed, current folder is: {self.sourceDir}")

        result_file_path = dir_unaltered_run / 'ergebnis.lila'
        self.df_unaltered_ergebnis = larsimInputOutputUtilities.ergebnis_parser_toPandas(result_file_path)

        simulation_start_timestamp = self.larsimConfObject.timeframe[0]
        if self.larsimConfObject.warm_up_duration is not None:
            simulation_start_timestamp = self.larsimConfObject.timeframe[0] + datetime.timedelta(hours=self.larsimConfObject.warm_up_duration)
        self.df_unaltered_ergebnis = larsimDataPostProcessing.parse_df_based_on_time(self.df_unaltered_ergebnis, (simulation_start_timestamp, None))

        # filter out results for a concrete station if specified in configuration json file
        if self.larsimConfObject.station_for_model_runs is not None and self.larsimConfObject.station_for_model_runs != "all":
            self.df_unaltered_ergebnis = larsimDataPostProcessing.filterResultForStation(self.df_unaltered_ergebnis,
                                                                                         station=self.larsimConfObject.station_for_model_runs)
        if self.larsimConfObject.type_of_output_of_Interest is not None and self.larsimConfObject.type_of_output_of_Interest != "all":
            self.df_unaltered_ergebnis = larsimDataPostProcessing.filterResultForTypeOfOutpu(self.df_unaltered_ergebnis,
                                                                                             type_of_output=self.larsimConfObject.type_of_output_of_Interest)

        if write_in_file:
            if write_file_path is None:
                write_file_path = self.workingDir / "df_unaltered.pkl"
            larsimInputOutputUtilities.write_dataFrame_to_file(self.df_unaltered_ergebnis,
                                                               file_path=write_file_path,
                                                               compression="gzip")

        # delete ergebnis.lila and all other not necessary files
        if createNewFolder:
            larsimConfigurationSettings.cleanDirectory_completely(curr_directory=dir_unaltered_run)
        else:
            larsimConfigurationSettings._delete_larsim_output_files(curr_directory=dir_unaltered_run)

    def _compare_measurements_and_unalteredSim(self, get_all_possible_stations=True,
                                               write_in_file=True, write_file_path=None):
        if self.df_measured is None or self.df_unaltered_ergebnis is None:
            return None

        stations = larsimDataPostProcessing.get_stations_intersection(self.df_measured, self.df_unaltered_ergebnis)
        if not get_all_possible_stations and (self.larsimConfObject.station_of_Interest != "all"
                                              or self.larsimConfObject.station_of_Interest is not None):
            if not isinstance(self.larsimConfObject.station_of_Interest, list):
                self.larsimConfObject.station_of_Interest = [self.larsimConfObject.station_of_Interest, ]
            stations = list(set(stations).intersection(self.larsimConfObject.station_of_Interest))

        gof_list_over_stations = []
        for station in stations:
            df_sim = larsimDataPostProcessing.filterResultForStationAndTypeOfOutpu(self.df_unaltered_ergebnis,
                                                                                   station=station,
                                                                                   type_of_output=self.larsimConfObject.type_of_output_of_Interest)
            temp_gof_dict = larsimDataPostProcessing.calculateGoodnessofFit_simple(measuredDF=self.df_measured,
                                                                                   predictedDF=df_sim,
                                                                                   gof_list=self.larsimConfObject.objective_function,
                                                                                   measuredDF_column_name=station,
                                                                                   simulatedDF_column_name='Value'
                                                                                   )
            temp_gof_dict["station"] = station
            gof_list_over_stations.append(temp_gof_dict)
        self.df_gof_unaltered_meas = pd.DataFrame(gof_list_over_stations)

        if self.df_past_sim is not None:
            gof_list_over_stations_sim_mes = []
            for station in stations:
                temp_gof_dict = larsimDataPostProcessing.calculateGoodnessofFit_simple(measuredDF=self.df_measured,
                                                                                       predictedDF=self.df_past_sim,
                                                                                       gof_list=self.larsimConfObject.objective_function,
                                                                                       measuredDF_column_name=station,
                                                                                       simulatedDF_column_name=station
                                                                                       )
                temp_gof_dict["station"] = station
                gof_list_over_stations_sim_mes.append(temp_gof_dict)
            self.df_gof_sim_meas = pd.DataFrame(gof_list_over_stations_sim_mes)

        if write_in_file:
            if write_file_path is None:
                write_file_path = self.workingDir / "gof_unaltered_meas.pkl"
            larsimInputOutputUtilities.write_dataFrame_to_file(self.df_gof_unaltered_meas,
                                                               file_path=write_file_path,
                                                               compression="gzip")
            if self.df_gof_sim_meas is not None:
                write_file_path = write_file_path.parents[0] / "gof_past_sim_meas.pkl"
                larsimInputOutputUtilities.write_dataFrame_to_file(self.df_gof_sim_meas,
                                                                   file_path=write_file_path,
                                                                   compression="gzip")


class LarsimModel(Model):

    def __init__(self, configurationObject, inputModelDir, workingDir=None,
                 log_level=utils.log_levels.INFO, print_level=utils.print_levels.INFO,
                 debug=False, *args, **kwargs):
        Model.__init__(self)

        #####################################
        # set-up utility logging
        #####################################

        self.log_util = utils.LogUtility(log_level=log_level, print_level=print_level)
        self.debug = debug
        if self.debug:
            self.log_util.log_debug(f'LarsimModel debug: {self.debug}')
        self.log_util.set_print_prefix('LarsimModel')
        self.log_util.set_log_prefix('LarsimModel')

        #####################################
        # Set-up configuration object
        #####################################

        if isinstance(configurationObject, LarsimConfigurations):
            self.larsimConfObject = configurationObject
        else:
            self.larsimConfObject = LarsimConfigurations(configurationObject, True,
                                                         utils.log_levels.INFO, utils.print_levels.INFO,
                                                         **kwargs)

        #####################################
        # Specification of different directories/paths - some are machine / location dependent,
        # adjust path in larsimPaths module, and calling code accordingly
        #####################################
        self.inputModelDir = pathlib.Path(inputModelDir)

        if self.larsimConfObject.dict_params_for_defining_paths is not None:
            self.larsimPathsObject = paths.LarsimPathsObject(root_data_dir=inputModelDir,
                                                             **self.larsimConfObject.dict_params_for_defining_paths)
        else:
            self.larsimPathsObject = paths.LarsimPathsObject(root_data_dir=inputModelDir)

        self.sourceDir = pathlib.Path(kwargs.get('sourceDir')) if 'sourceDir' in kwargs and pathlib.Path(
            kwargs.get('sourceDir')).is_absolute() \
            else pathlib.Path(__file__).resolve().parents[1]

        self.workingDir = pathlib.Path(workingDir)
        if self.workingDir is None:
            self.workingDir = pathlib.Path(paths.workingDir)

        if self.larsimConfObject.boolean_make_local_copy_of_master_dir:
            self.master_dir = self.workingDir / 'master_configuration'
        else:
            self.master_dir = self.workingDir  # self.larsimPathsObject.global_master_dir

        log_filename = self.workingDir / "larsimModel.log"
        self.log_util.set_log_filename(log_filename=log_filename)

        #####################################
        # Set of configuration variables propagated via UQsim.args
        #####################################
        self.uq_method = kwargs.get('uq_method', None)
        if self.uq_method is not None and self.uq_method == "sc":  # always break when running gPCE simulation
            self.larsimConfObject.raise_exception_on_model_break = True

        self.disable_statistics = kwargs.get('disable_statistics', True)
        self.larsimConfObject.run_and_save_simulations = \
            self.larsimConfObject.run_and_save_simulations and self.disable_statistics

        #####################################
        self.measuredDF = None
        self._is_measuredDF_computed = False
        self.measuredDF_column_name = 'Value'
        #self._set_measured_df()

        self.log_util.log_info(f"INITIALIZATION DONE!")

    def prepare(self):
        #pass
        self._set_measured_df()

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def set_configuration_object(self, configurationObject, **kwargs):
        if isinstance(configurationObject, LarsimConfigurations):
            self.larsimConfObject = configurationObject
        else:
            self.larsimConfObject = LarsimConfigurations(configurationObject, **kwargs)
        # self.configurationObject = larsimConfigurationSettingstings.return_configuration_object(configurationObject)

    def set_timeframe(self, timeframe):
        self.larsimConfObject.timeframe = larsimTimeUtility.timeframe_to_datetime_list(timeframe)
        larsimConfigurationSettings.update_configurationObject_with_datetime_info(
            self.larsimConfObject.configurationObject, timeframe)
        self.larsimConfObject.t = larsimTimeUtility.get_tape10_timesteps(timeframe)

    def set_station_of_Interest(self, station_of_Interest):
        larsimConfigurationSettings.update_config_dict_station_of_interest(
            self.larsimConfObject.configurationObject, station_of_Interest)
        self.larsimConfObject.station_of_Interest = station_of_Interest

    def timesteps(self):
        return self.larsimConfObject.t

    def __set_measuredDF_column_name(self):
        if self.measuredDF is not None and 'Value' not in self.measuredDF.columns:
            filtered_timesteps_vs_station_values = True
            self.measuredDF_column_name = 'station'

    def _set_measured_df(self):
        local_measurement_file = self.workingDir / "df_measured.pkl" #self.local_measurement_file
        if local_measurement_file.exists():
            self.measuredDF = larsimInputOutputUtilities.read_dataFrame_from_file(local_measurement_file,
                                                                              compression="gzip")
        else:
            local_measurement_file = self.master_dir / paths.LILA_FILES[0]
            self.measuredDF = larsimDataPostProcessing.read_process_write_discharge(df=local_measurement_file,
                                                                               timeframe=self.larsimConfObject.timeframe,
                                                                               station=self.larsimConfObject.station_for_model_runs,
                                                                               compression="gzip")
        self._is_measuredDF_computed = True
        self.__set_measuredDF_column_name()

    def _get_measured_df(self):
        if not self._is_measuredDF_computed:
            self._set_measured_df()
        return self.measuredDF

    def setUp(self):
        pass

    def copy_master_folder(self):
        pass

    def configure_master_folder(self):
        pass

    def run_unaltered_run(self, index_of_unaltered_run: int = 1000):
        pass

    def run(self, i_s=[0,], parameters=None, raise_exception_on_model_break=None, *args, **kwargs):  # i_s - index chunk; parameters - parameters chunk

        print(f"[LarsimModel INFO] {i_s} parameter: {parameters}")

        if raise_exception_on_model_break is None:
            raise_exception_on_model_break = self.larsimConfObject.raise_exception_on_model_break
        max_retries = kwargs.get("max_retries", self.larsimConfObject.max_retries)

        take_direct_value = kwargs.get("take_direct_value", False)
        createNewFolder = kwargs.get("createNewFolder", True)
        deleteFolderAfterwards = kwargs.get("deleteFolderAfterwards", True)

        results_array = []
        for ip in range(0, len(i_s)): # for each peace of work
            i = i_s[ip]  # i is unique index run

            if parameters is not None:
                parameter = parameters[ip]
            else:
                parameter = None

            start = time.time()

            # create local directory for this particular run
            if createNewFolder:
                working_folder_name = self.larsimPathsObject.larsim_model_dir.name + "_" + str(i)
                curr_working_dir = self.workingDir / working_folder_name
                curr_working_dir.mkdir(parents=True, exist_ok=True)
            else:
                curr_working_dir = self.workingDir

            # copy all the necessary files to the newly created directory
            if self.master_dir.resolve().absolute() != curr_working_dir.resolve().absolute():
                master_dir_for_copying = str(self.master_dir) + "/."
                subprocess.run(['cp', '-a', master_dir_for_copying, curr_working_dir])
                print("[LarsimModel INFO] Successfully copied all the files")

            # change values
            id_dict = {"index_run": i}
            # if parameter is not None:
            tape35_path = curr_working_dir / self.larsimPathsObject.tape_35_path.name
            lanu_path = curr_working_dir / self.larsimPathsObject.lanu_path.name
            if parameter is not None:
                parameters_dict = larsimConfigurationSettings.params_configurations(parameters=parameter,
                                                                                    tape35_path=tape35_path,
                                                                                    lanu_path=lanu_path,
                                                                                    configurationObject=self.larsimConfObject.configurationObject,
                                                                                    process_id=i,
                                                                                    take_direct_value=take_direct_value)
                parameters_dict = {**id_dict, **parameters_dict}
            else:
                parameters_dict = None
            # else: #TODO add option when parameter is None to read default parameters values and run unaltered run
            #     parameters_dict = {**id_dict,}

            # change working directory
            os.chdir(curr_working_dir)

            # Run Larsim
            if self.larsimConfObject.cut_runs:
                result = self._multiple_short_larsim_runs(timeframe=self.larsimConfObject.timeframe,
                                                          timestep=self.larsimConfObject.timestep,
                                                          curr_working_dir=curr_working_dir,
                                                          larsim_exe=self.larsimPathsObject.larsim_exe, index_run=i,
                                                          warm_up_duration=self.larsimConfObject.warm_up_duration,
                                                          raise_exception_on_model_break=raise_exception_on_model_break,
                                                          max_retries=max_retries)
            else:
                result = _single_larsim_run(timeframe=self.larsimConfObject.timeframe,
                                            curr_working_dir=curr_working_dir,
                                            larsim_exe=self.larsimPathsObject.larsim_exe,
                                            index_run=i,
                                            warm_up_duration=self.larsimConfObject.warm_up_duration,
                                            boolean_configure_tape10=self.larsimConfObject.boolean_configure_tape10,
                                            raise_exception_on_model_break=raise_exception_on_model_break,
                                            max_retries=max_retries)
            if result is None:
                larsimConfigurationSettings.cleanDirectory_completely(curr_directory=curr_working_dir)
                os.chdir(self.sourceDir)
                if raise_exception_on_model_break:
                    raise Exception(f"[LarsimModel ERROR] Process {i}: Larsim run was unsuccessful!")
                else:
                    # TODO write in some log file runs which have returned None
                    end = time.time()
                    runtime = end - start
                    results_array.append((None, runtime))
                    continue

            # filter output time-series in order to disregard warm-up time;
            # if not, then at least disregard these values when calculating statistics and GoF
            # however, take care that is is not done twice!
            simulation_start_timestamp = self.larsimConfObject.timeframe[0]
            if self.larsimConfObject.warm_up_duration is not None:
                simulation_start_timestamp = self.larsimConfObject.timeframe[0] + \
                                             datetime.timedelta(hours=self.larsimConfObject.warm_up_duration)  # pd.Timestamp(result.TimeStamp.min())
            result = larsimDataPostProcessing.parse_df_based_on_time(result, (simulation_start_timestamp, None))

            # filter out results for a concrete station if specified in configuration json file
            if self.larsimConfObject.station_for_model_runs is not None \
                    and self.larsimConfObject.station_for_model_runs != "all":
                result = larsimDataPostProcessing.filterResultForStation(
                    result, station=self.larsimConfObject.station_for_model_runs)
            if self.larsimConfObject.type_of_output_of_Interest is not None \
                    and self.larsimConfObject.type_of_output_of_Interest != "all":
                result = larsimDataPostProcessing.filterResultForTypeOfOutpu(
                    result, type_of_output=self.larsimConfObject.type_of_output_of_Interest)

            end = time.time()
            runtime = end - start
            result_dict = {"run_time": runtime, "parameters_dict": parameters_dict}

            ######################################################################################################

            # compute (some) GoF for the whole time period
            # TODO eventually design the code to disregard the runs with an unsatisfying value of some GoF
            #  when uq_method is mc or saltelli, or to identify those runs that break
            index_parameter_gof_DF = None
            if self.larsimConfObject.calculate_GoF:
                index_parameter_gof_DF = self._calculate_GoF(predictedDF=result,
                                                             parameters_dict=parameters_dict,
                                                             objective_function=self.larsimConfObject.objective_function,
                                                             get_all_possible_stations=self.larsimConfObject.get_all_possible_stations)
                result_dict["gof_df"] = index_parameter_gof_DF

            ######################################################################################################

            # process result DF to compute the final GoI
            processed_result = None
            change_original_result_and_keep_it = True
            if self.larsimConfObject.mode == "sliding_window":
                if self.larsimConfObject.qoi == "GoF":
                    processed_result = self._process_time_series_sliding_window_gof(predictedDF=result,
                                                                                    interval=self.larsimConfObject.interval,
                                                                                    min_periods=self.larsimConfObject.min_periods,
                                                                                    objective_function=self.larsimConfObject.objective_function_qoi,
                                                                                    get_all_possible_stations=self.larsimConfObject.get_all_possible_stations)
                else:
                    processed_result = self._process_time_series_sliding_window_q(predictedDF=result,
                                                                                  interval=self.larsimConfObject.interval,
                                                                                  min_periods=self.larsimConfObject.min_periods,
                                                                                  method=self.larsimConfObject.method)
            # elif self.larsimConfObject.mode == "resampling":
            #     # resampling can happen in Statistics as well
            #     if self.larsimConfObject.qoi == "GoF":
            #         processed_result = self._process_time_series_resampling_gof(predictedDF=result,
            #                                                                     interval=self.larsimConfObject.interval,
            #                                                                     min_periods=self.larsimConfObject.min_periods,
            #                                                                     objective_function=self.larsimConfObject.objective_function_qoi,
            #                                                                     get_all_possible_stations=self.larsimConfObject.get_all_possible_stations)
            #     else:
            #         processed_result = self._process_time_series_resampling_q(predictedDF=result,
            #                                                                   interval=self.larsimConfObject.interval,
            #                                                                   min_periods=self.larsimConfObject.min_periods,
            #                                                                   method=self.larsimConfObject.method)

            # Important, from now on the result is changed
            if processed_result is not None:
                # before one overwrites result - check if the raw model output should be saved as well
                if self.larsimConfObject.run_and_save_simulations and self.larsimConfObject.always_save_original_model_runs:
                    file_path = self.workingDir / f"df_Larsim_raw_run_{i}.pkl"  # TODO
                    result.to_pickle(file_path, compression="gzip")
                result = processed_result

            result_dict["result_time_series"] = result

            ######################################################################################################

            # Postprocessing the timeframe
            # self.larsimConfObject.timeframe[0], self.larsimConfObject.timeframe[1] = result["TimeStamp"].min(), result["TimeStamp"].max()
            # larsimConfigurationSettings.update_configurationObject_with_datetime_info(self.larsimConfObject.configurationObject, self.larsimConfObject.timeframe)
            # self.larsimConfObject.t = larsimTimeUtility.get_tape10_timesteps(self.larsimConfObject.timeframe)

            ######################################################################################################

            # compute gradient of the output/QoI, or gradient of some likelihood measure w.r.t parameters
            # for now f is only evaluation of some GoF, this makes sense only when:
            # self.larsimConfObject.mode=="continuous", self.larsimConfObject.qoi="Q", self.larsimConfObject.calculate_GoF=True
            # self.larsimConfObject.mode="resampling", self.larsimConfObject.qoi="GoF"
            if self.larsimConfObject.compute_gradients:

                # self.larsimConfObject.configurationObject["parameters_settings"]["cut_limits"] = "True"

                # compute different GoFs (those in self.larsimConfObject.objective_function_qoi)
                # only for stations in self.larsimConfObject.station_of_Interest, for main result to use it as a baseline
                # and only when self.uq_method != "sc"
                if index_parameter_gof_DF is None:
                    index_parameter_gof_DF = self._calculate_GoF(predictedDF=result,
                                                                 parameters_dict=parameters_dict,
                                                                 objective_function=self.larsimConfObject.objective_function_qoi,
                                                                 get_all_possible_stations=False)

                if not isinstance(self.larsimConfObject.station_of_Interest, list):
                    list_of_stations = [self.larsimConfObject.station_of_Interest, ]
                else:
                    list_of_stations = self.larsimConfObject.station_of_Interest
                list_of_gof = self.larsimConfObject.objective_function_qoi
                list_of_gof = [single_gof.__name__ if callable(single_gof) else single_gof for single_gof in list_of_gof]

                h_vector = []
                gradient_vectors_dict = defaultdict(list)
                parameter_names = []
                gradient_vectors_param_dict = defaultdict(list)

                # CD = 1 central differences; CD = 0 forward differences
                # Assumption: parameter is a list, not a dictionary
                length_evaluations_gradient = 2 * len(parameter) if self.larsimConfObject.CD else len(parameter)

                for id_param in range(length_evaluations_gradient):
                    # 2.1. For every uncertain parameter, create a new folder where 1 parameter is changed, copy the set-up
                    curr_working_dir_gradient = self._copy_files_for_gradient_computation(curr_working_dir, i, id_param)

                    # 2.2. Adjust configuration files (tape35 and lanu.par)
                    if self.larsimConfObject.CD:
                        eps_val = self.larsimConfObject.eps_val_global \
                            if id_param % 2 == 0 else -self.larsimConfObject.eps_val_global  # used for computing f(x+-h)
                        param_index = int(id_param / 2)
                    else:  # FD
                        eps_val = self.larsimConfObject.eps_val_global  # used for computing f(x+h)
                        param_index = id_param

                    tape35_path = curr_working_dir_gradient / self.larsimPathsObject.tape_35_path.name
                    lanu_path = curr_working_dir_gradient / self.larsimPathsObject.lanu_path.name
                    parameter_dict = larsimConfigurationSettings.params_configurations(parameters=parameter,
                                                                                       tape35_path=tape35_path,
                                                                                       lanu_path=lanu_path,
                                                                                       configurationObject=self.larsimConfObject.configurationObject,
                                                                                       process_id=i,
                                                                                       write_new_values_to_tape35=True,
                                                                                       write_new_values_to_lanu=True,
                                                                                       perturb_single_param_around_nominal=True,
                                                                                       parameter_index_to_perturb=param_index,
                                                                                       eps_val=eps_val)
                    current_param_name = list(parameter_dict.keys())[0]
                    h = parameter_dict[current_param_name][1]

                    if not self.larsimConfObject.CD:
                        h_vector.append(h)  # update vector of h's
                        parameter_names.append(current_param_name)
                    elif self.larsimConfObject.CD and (id_param % 2 == 0):
                        h_vector.append(2 * h)
                        parameter_names.append(current_param_name)

                    # 2.3. Run the simulation
                    # Run Larsim
                    if self.larsimConfObject.cut_runs:
                        result_grd = self._multiple_short_larsim_runs(timeframe=self.larsimConfObject.timeframe,
                                                                      timestep=self.larsimConfObject.timestep,
                                                                      curr_working_dir=curr_working_dir_gradient,
                                                                      larsim_exe=self.larsimPathsObject.larsim_exe,
                                                                      index_run=i,
                                                                      warm_up_duration=self.larsimConfObject.warm_up_duration,
                                                                      raise_exception_on_model_break=raise_exception_on_model_break,
                                                                      max_retries=max_retries)
                    else:
                        result_grd = _single_larsim_run(timeframe=self.larsimConfObject.timeframe,
                                                        curr_working_dir=curr_working_dir_gradient,
                                                        larsim_exe=self.larsimPathsObject.larsim_exe,
                                                        index_run=i,
                                                        warm_up_duration=self.larsimConfObject.warm_up_duration,
                                                        boolean_configure_tape10=self.larsimConfObject.boolean_configure_tape10,
                                                        raise_exception_on_model_break=raise_exception_on_model_break,
                                                        max_retries=max_retries)

                    if result_grd is None:
                        larsimConfigurationSettings.cleanDirectory_completely(curr_directory=curr_working_dir_gradient)
                        os.chdir(curr_working_dir)
                        if raise_exception_on_model_break:
                            raise Exception(f"[LarsimModel ERROR] Process {i} - "
                                            f"computing gradient param_index:{param_index}, eps_val:{eps_val}: "
                                            f"Larsim run was unsuccessful!")
                        else:
                            # TODO write in some log file runs which have returned None
                            gradient_vectors_param_dict = dict()
                            gradient_vectors_dict = dict()
                            break

                    # 2.4. Preparations before computing GoF
                    result_grd = larsimDataPostProcessing.parse_df_based_on_time(result_grd,
                                                                                 (simulation_start_timestamp, None))
                    # filter out results
                    result_grd = larsimDataPostProcessing.filterResultForStation(
                        result_grd, station=self.larsimConfObject.station_of_Interest)
                    result_grd = larsimDataPostProcessing.filterResultForTypeOfOutpu(
                        result_grd, type_of_output=self.larsimConfObject.type_of_output_of_Interest)

                    # 2.5. Compute goodness of fit (GoF) &
                    result_grd_gof_DF = self._calculate_GoF(predictedDF=result_grd,
                                                            objective_function=self.larsimConfObject.objective_function_qoi,
                                                            get_all_possible_stations=False)


                    # 2.6. Extract subset of data for each analysed station and each analysed GoF
                    for single_station in list_of_stations:
                        for single_gof in list_of_gof:
                            grad_estimation = np.nan
                            if self.larsimConfObject.CD:  # Central Difference (CD) computation
                                if id_param % 2 == 0:
                                    f_x_ij_p_h = result_grd_gof_DF.loc[(result_grd_gof_DF["station"] == single_station)][single_gof].values[0]
                                    gradient_vectors_dict[(single_station, single_gof)].append(f_x_ij_p_h)
                                    gradient_vectors_param_dict[(single_station, single_gof, current_param_name)].append(f_x_ij_p_h)
                                else:
                                    f_x_ij_m_h = result_grd_gof_DF.loc[(result_grd_gof_DF["station"] == single_station)][single_gof].values[0]
                                    gradient_vectors_dict[(single_station, single_gof)].append(f_x_ij_m_h)
                                    gradient_vectors_param_dict[(single_station, single_gof, current_param_name)].append(f_x_ij_m_h)
                                    gradient_vectors_param_dict[(single_station, single_gof, current_param_name)].append(2 * h)
                            else:  # Forward Difference (FD) computation
                                f_x_ij_p_h = result_grd_gof_DF.loc[(result_grd_gof_DF["station"] == single_station)][single_gof].values[0]
                                f_x_ij = index_parameter_gof_DF.loc[(index_parameter_gof_DF["station"] == single_station)][single_gof].values[0]
                                gradient_vectors_dict[(single_station, single_gof)].append(f_x_ij_p_h)
                                gradient_vectors_dict[(single_station, single_gof)].append(f_x_ij)
                                gradient_vectors_param_dict[(single_station, single_gof, current_param_name)].append(f_x_ij_p_h)
                                gradient_vectors_param_dict[(single_station, single_gof, current_param_name)].append(f_x_ij)
                                gradient_vectors_param_dict[(single_station, single_gof, current_param_name)].append(h)

                    # Delete everything except .log and .csv files
                    larsimConfigurationSettings.cleanDirectory_completely(curr_directory=curr_working_dir_gradient)

                    # change back to starting directory of all the processes
                    os.chdir(curr_working_dir)

                    # Delete local working folder
                    subprocess.run(["rm", "-r", curr_working_dir_gradient])

                # 3. Process data for generating gradient matrices
                gradient_matrix_dict = dict()
                if gradient_vectors_dict:
                    for single_station in list_of_stations:
                        for single_gof in list_of_gof:
                            if self.larsimConfObject.CD:  # Central Difference (CD) computation
                                f_x_ij_p_h_array = np.array(gradient_vectors_dict[(single_station, single_gof)][0::2], dtype=np.float32)
                                f_x_ij_m_h_array = np.array(gradient_vectors_dict[(single_station, single_gof)][1::2], dtype=np.float32)
                                grad_estimation = (f_x_ij_p_h_array - f_x_ij_m_h_array) / np.array(h_vector)
                            else:  # Forward Difference (FD) computation
                                f_x_ij_p_h_array = np.array(gradient_vectors_dict[(single_station, single_gof)][0::2], dtype=np.float32)
                                f_x_ij_array = np.array(gradient_vectors_dict[(single_station, single_gof)][1::2], dtype=np.float32)
                                grad_estimation = (f_x_ij_p_h_array - f_x_ij_array) / np.array(h_vector)

                            gradient_matrix_dict[(single_station, single_gof)] = np.outer(grad_estimation, grad_estimation)

                # finally, add gradient estimated to index_parameter_gof_DF
                if gradient_vectors_param_dict:
                    for current_param_name in parameter_names:
                        for single_gof in list_of_gof:
                            new_column_name = "d_" + single_gof + "_d_" + current_param_name
                            index_parameter_gof_DF[new_column_name] = \
                                index_parameter_gof_DF["station"].apply(lambda x: \
                                                                            gradient_vectors_param_dict.get((x, single_gof, current_param_name), np.nan))

                if gradient_matrix_dict:
                    result_dict["gradient_matrix_dict"] = gradient_matrix_dict
                else:
                    result_dict["gradient_matrix_dict"] = None
            ######################################################################################################

            # save all the sub-results in case there is no LarsimStatistics run afterward
            if self.larsimConfObject.run_and_save_simulations:
                if parameters_dict is not None:
                    file_path = self.workingDir / f"parameters_Larsim_run_{i}.pkl"  # TODO
                    with open(file_path, 'wb') as f:
                        dill.dump(parameters_dict, f)

                if result is not None:
                    file_path = self.workingDir / f"df_Larsim_run_{i}.pkl"  #TODO
                    result.to_pickle(file_path, compression="gzip")

                if self.larsimConfObject.calculate_GoF:
                    file_path = self.workingDir / f"gof_{i}.pkl"
                    index_parameter_gof_DF.to_pickle(file_path, compression="gzip")

                if self.larsimConfObject.compute_gradients and gradient_matrix_dict:
                    file_path = self.workingDir / f"gradients_matrices_{i}.npy"  # TODO
                    np.save(file_path, gradient_matrix_dict)

            ######################################################################################################
            # Final cleaning and appending the results
            ######################################################################################################
            print(f"[LarsimModel INFO] Process {i} returned / appended it's results")

            # result_dict contains at least the following entries:  "result_time_series", "run_time", "parameters_dict"
            # optionally: "gof_df", "gradient" , etc.
            results_array.append((result_dict, runtime))

            # Delete everything except .log and .csv files
            larsimConfigurationSettings.cleanDirectory_completely(curr_directory=curr_working_dir)

            # change back to starting directory of all the processes
            os.chdir(self.sourceDir)

            # Delete local working folder
            subprocess.run(["rm", "-r", curr_working_dir])

            print(f"[LarsimModel INFO] I am done - solver number {i}")

        return results_array

    def _single_larsim_run(self, timeframe, curr_working_dir, larsim_exe, index_run=0, sub_index_run=0,
                           warm_up_duration=None, boolean_configure_tape10=True, **kwargs):

        if warm_up_duration is None:
            warm_up_duration = 0

        raise_exception_on_model_break = kwargs.get('raise_exception_on_model_break', True)
        max_retries = kwargs.get('max_retries', None)

        # start clean
        larsimConfigurationSettings._delete_larsim_output_files(curr_directory=curr_working_dir)

        # change tape 10 accordingly
        if boolean_configure_tape10:
            local_master_tape10_file = curr_working_dir / self.larsimPathsObject.master_tape_10_path.name
            local_tape10_adjusted_path = curr_working_dir / self.larsimPathsObject.tape_35_path.name
            try:
                larsimTimeUtility.tape10_configuration(timeframe=timeframe, master_tape10_file=local_master_tape10_file,
                                                       new_tape10_file=local_tape10_adjusted_path,
                                                       warm_up_duration=warm_up_duration)
            except larsimTimeUtility.ValidDurationError as e:
                print(f"[LarsimModel ERROR - None is returned after the Larsim run \n]" + str(e))
                return None

        # command = 'chmod 755 ' + local_tape10_adjusted_path
        # subprocess.run(command.split())
        # local_tape12 = curr_working_dir / 'tape12'
        # command = 'chmod 755 ' + local_tape12
        # subprocess.run(command.split())
        # local_tape29 = curr_working_dir / 'tape29'
        # command = 'chmod 755 ' + local_tape29
        # subprocess.run(command.split())
        # local_tape35 = curr_working_dir / 'tape35'
        # command = 'chmod 755 ' + local_tape35
        # subprocess.run(command.split())

        # log file for larsim
        local_log_file = curr_working_dir / f"run_{index_run}_{sub_index_run}.log"
        # run Larsim as external process
        subprocess.run([str(larsim_exe)], stdout=open(local_log_file, 'w'))
        print(f"[LarsimModel INFO] I am done with LARSIM Execution {index_run}")

        # check if larsim.ok exist - Larsim execution was successful
        ok_found = larsimConfigurationSettings.check_larsim_ok_file(curr_working_dir=curr_working_dir,
                                                                    max_retries=max_retries)
        if not ok_found and raise_exception_on_model_break:
            raise Exception(f"[LarsimModel ERROR] Process {index_run}: Larsim run was unsuccessful!")
        elif not ok_found and not raise_exception_on_model_break:
            return None

        result_file_path = curr_working_dir / 'ergebnis.lila'
        try:
            df_single_ergebnis = larsimInputOutputUtilities.ergebnis_parser_toPandas(result_file_path, index_run)
            return df_single_ergebnis
        except paths.FileError as e:
            print(str(e))
            print(f"[LarsimModel ERROR] Process {index_run}: The following Ergebnis file was not found "
                  f"- {result_file_path}. None was returned")
            raise

    def _multiple_short_larsim_runs(self, timeframe, timestep, curr_working_dir, larsim_exe, index_run=0,
                                    warm_up_duration=None, timestep_in_hours=False, **kwargs):

        if warm_up_duration is None:
            warm_up_duration = 0

        raise_exception_on_model_break = kwargs.get('raise_exception_on_model_break', True)
        max_retries = kwargs.get('max_retries', None)

        # if you want to cut execution into shorter runs...
        local_timestep = timestep

        if timestep_in_hours:
            number_of_runs = (timeframe[1] - timeframe[0]).hours // datetime.timedelta(hours=local_timestep).hours
            number_of_runs_mode = (timeframe[1] - timeframe[0]).hours % datetime.timedelta(hours=local_timestep).hours
        else:
            number_of_runs = (timeframe[1] - timeframe[0]).days // datetime.timedelta(days=local_timestep).days
            # TODO eventually, add logic to run for these days as well...
            number_of_runs_mode = (timeframe[1] - timeframe[0]).days % datetime.timedelta(days=local_timestep).days

        local_end_date = timeframe[0]

        # TODO
        result_file_path = curr_working_dir / 'ergebnis.lila'
        larsim_ok_file_path = curr_working_dir / 'larsim.ok'
        tape11_file_path = curr_working_dir / 'tape11'
        karte_path = curr_working_dir / 'karten'  # curr_working_dir / 'karten/*'
        tape10_path = curr_working_dir / 'tape10'

        print(f"[LarsimModel INFO] process {index_run} gonna run {number_of_runs} "
              f"shorter Larsim runs (and number_of_runs_mode {number_of_runs_mode})")

        local_resultDF_list = []
        for i in range(number_of_runs):
            local_warm_up_duration = warm_up_duration

            # remove previous tape10
            subprocess.run(["rm", "-f", tape10_path])

            # calculate times - make sure that outputs are continuous in time
            if i == 0:
                local_start_date = local_end_date
                # local_start_date_warmup = local_start_date - datetime.timedelta(hours=warm_up_duration)
            else:
                # local_start_date_warmup = local_end_date
                # local_start_date = local_start_date_warmup - datetime.timedelta(hours=warm_up_duration)
                local_start_date = local_end_date - datetime.timedelta(hours=warm_up_duration)

            if local_start_date.hour != 0:
                local_warm_up_duration = warm_up_duration + local_start_date.hour
                local_start_date = local_start_date.replace(hour=0, minute=0, second=0)

            local_start_prediction_date = local_start_date + datetime.timedelta(
                hours=local_warm_up_duration)

            if local_start_date > timeframe[1]:
                break

            if timestep_in_hours:
                if i == 0:
                    local_end_date = local_end_date + datetime.timedelta(hours=warm_up_duration) + datetime.timedelta(
                        hours=local_timestep)
                else:
                    local_end_date = local_end_date + datetime.timedelta(hours=local_timestep)
            else:
                if i == 0:
                    local_end_date = local_end_date + datetime.timedelta(hours=warm_up_duration) + datetime.timedelta(days=local_timestep)
                else:
                    local_end_date = local_end_date + datetime.timedelta(days=local_timestep)

            if local_end_date > timeframe[1]:
                local_end_date = timeframe[1]

            print(f"[LarsimModel INFO] Process {index_run}; local_start_date: {local_start_date}; "
                  f"local_start_prediction_date: {local_start_prediction_date}"
                  f"local_end_date: {local_end_date};")
            single_run_timeframe = (local_start_date, local_end_date)

            # run larsim for this shorter period and returned already parsed 'small' ergebnis
            local_resultDF = _single_larsim_run(timeframe=single_run_timeframe,
                                                curr_working_dir=curr_working_dir,
                                                larsim_exe=larsim_exe,
                                                index_run=index_run,
                                                sub_index_run=i,
                                                warm_up_duration=local_warm_up_duration,
                                                boolean_configure_tape10=True,
                                                raise_exception_on_model_break=raise_exception_on_model_break,
                                                max_retries=max_retries)

            if local_resultDF is None:
                local_resultDF_list = []
                break

            # postprocessing of time variables
            local_start_date, local_end_date = local_resultDF["TimeStamp"].min(), local_resultDF["TimeStamp"].max()
            # local_start_date_warmup should be equal to local_start_prediction_date
            local_start_date_warmup = local_start_date + datetime.timedelta(hours=local_warm_up_duration)

            local_resultDF_drop = local_resultDF.drop(local_resultDF[local_resultDF['TimeStamp'] <
                                                                     local_start_date_warmup].index)

            local_resultDF_list.append(local_resultDF_drop)

            # rename ergebnis.lila
            local_result_file_path = curr_working_dir / f'ergebnis_{i}.lila'
            subprocess.run(["mv", result_file_path, local_result_file_path])
            local_larsim_ok_file_path = curr_working_dir / f'larsim_{i}.ok'
            subprocess.run(["mv", larsim_ok_file_path, local_larsim_ok_file_path])

        if local_resultDF_list:
            # concatenate it
            df_simulation_result = pd.concat(local_resultDF_list, ignore_index=True, sort=True, axis=0)

            # sorting by time_calculate_GoF_sliding_window_single_gof_single_station
            df_simulation_result.sort_values("TimeStamp", inplace=True)

            # clean concatenated file - dropping time duplicate values
            df_simulation_result.drop_duplicates(subset=['TimeStamp', 'Stationskennung', 'Type'], keep='first', inplace=True)

            return df_simulation_result
        elif not raise_exception_on_model_break:
            return None
        else:
            raise Exception(f"[LarsimModel ERROR] Process {index_run}: Larsim run was unsuccessful!")

    def _process_time_series_sliding_window_gof(self, predictedDF, objective_function, interval=24, min_periods=1,
                                                center=True, closed="neither", get_all_possible_stations=False):
        """
         if center=False the result is set to the right edge of the window
        """

        # center=True can not work with datetimelike
        # if not isinstance(interval, str) or (isinstance(interval, str) and not interval.endswith(('H', 'h'))):
        #     interval = f"{interval}H"

        if self.measuredDF is None and not self._is_measuredDF_computed:
            self._set_measured_df()
        elif self.measuredDF is None and self._is_measuredDF_computed:
            return None

        if interval == "whole":
            interval = predictedDF.TimeStamp.nunique()
            min_periods = 1

        # TODO it makes sense as well to have here self.larsimConfObject.station_for_model_runs or get_all_possible_stations=True
        stations = LarsimModel.compute_and_get_final_list_of_stations(self.measuredDF, predictedDF,
                                                                      get_all_possible_stations,
                                                                      self.larsimConfObject.station_of_Interest)

        list_of_results_per_station = []
        for single_station in stations:
            predictedDF_station_subset = predictedDF.loc[predictedDF["Stationskennung"] == single_station]
            if predictedDF_station_subset.index.name != "TimeStamp":
                predictedDF_station_subset.set_index("TimeStamp", inplace=True)

            def dataframe_roll(df, single_gof_function):
                def compute_gof_over_window(window_series):
                    window_df = df.loc[(df.index >= window_series.index[0]) & (df.index <= window_series.index[-1])]
                    window_df.reset_index(inplace=True)
                    window_df.rename(columns={window_df.index.name: 'TimeStamp'}, inplace=True)
                    list_over_objective_function = self._calculate_GoF_sliding_window_single_gof_single_station(
                        predictedDF=window_df, station=single_station,
                        objective_function_qoi=single_gof_function, return_dict=True)
                    return list_over_objective_function[single_gof_function.__name__]
                return compute_gof_over_window

            for single_gof_function in objective_function:
                predictedDF_station_subset[single_gof_function.__name__] = predictedDF_station_subset.rolling(
                    window=interval, min_periods=min_periods, center=center, closed=closed, win_type=None).\
                    Value.apply(dataframe_roll(predictedDF_station_subset, single_gof_function), raw=False)

            # disregard first couple of timesteps and last couple of time steps
            start = predictedDF_station_subset.index.min() + datetime.timedelta(hours=math.floor(interval/2)-1)
            end = predictedDF_station_subset.index.max() - datetime.timedelta(hours=math.floor(interval/2)-1)
            predictedDF_station_subset = predictedDF_station_subset.loc[start:end]
            # TODO Think about this, what if predictedDF_station_subset is empty at the end, implement differently
            if predictedDF_station_subset.empty or predictedDF_station_subset.dropna().empty:
                raise Exception(
                    f"[LarsimModel ERROR:] _process_time_series_sliding_window_gof: return empty DF "
                    f"for station:{single_station}!")
            predictedDF_station_subset.reset_index(inplace=True)

            list_of_results_per_station.append(predictedDF_station_subset)

        processed_result = pd.concat(list_of_results_per_station, ignore_index=True, sort=False, axis=0)
        return processed_result

    @staticmethod
    def _process_time_series_sliding_window_q(predictedDF, interval=24, method="avrg", min_periods=1, center=True):
        # if not isinstance(interval, str) or (isinstance(interval, str) and not interval.endswith(('H', 'h'))):
        #     interval = f"{interval}H"
        processed_result = predictedDF.copy(deep=True)
        if processed_result.index.name != "TimeStamp":
            processed_result.set_index("TimeStamp", inplace=True)

        if method == "avrg":
            processed_result = processed_result.groupby(["Stationskennung", "Index_run", "Type"]).\
                rolling(window=interval, min_periods=min_periods, center=center,
                        win_type=None).Value.mean().dropna().reset_index()
        elif method == "min":
            processed_result = processed_result.groupby(["Stationskennung", "Index_run", "Type"]).\
                rolling(window=interval, min_periods=min_periods, center=center,
                        win_type=None).Value.min().dropna().reset_index()
        elif method == "max":
            processed_result = processed_result.groupby(["Stationskennung", "Index_run", "Type"]). \
                rolling(window=interval, min_periods=min_periods, center=center,
                        win_type=None).Value.max().dropna().reset_index()
        else:
            raise Exception(
                f"[LarsimModel ERROR:] Error in _process_time_series_sliding_window_q - no correct method specified!")
        #processed_result.reset_index(inplace=True)
        return processed_result

    def _process_time_series_resampling_gof(self, predictedDF, interval, min_periods,
                                         objective_function, get_all_possible_stations=False):
        processed_result = None
        return processed_result

    def _process_time_series_resampling_q(self, predictedDF, interval, method):
        processed_result = None
        return processed_result

    def _calculate_GoF_sliding_window_single_gof_single_station(self, predictedDF, station,
                                                                objective_function_qoi=None, return_dict=False):
        """
        This function assumes that self.measuredDF is already computed by self._set_measured_df function
        and that self._is_measuredDF_computed is set to True

        Important note: predictedDF structure is part of standard resultDF

        Can work both for single and multiple stations
        """
        # TODO: to speed-up change the functions such that the predictedDF
        # TODO is just a series with TimeStamp column as index column and one extra Value column
        # TODO if we remove self.measuredDF, self._is_measuredDF_computed self._set_measured_df() - make it staticmethod

        # if self.measuredDF is None and not self._is_measuredDF_computed:
        #     self._set_measured_df()
        # elif self.measuredDF is None and self._is_measuredDF_computed:
        #     return None

        # if self.measuredDF.empty or predictedDfF.empty:
        #     print(f"Failed in self.measuredDF.empty or predictedDF.empty")
        # if self.measuredDF_column_name not in self.measuredDF.columns:
        #     print(f"self.measuredDF_column_name not in self.measuredDF.columns")
        # if "Value" not in predictedDF.columns:
        #     print(f"Value not in predictedDF.columns")

        #measuredDF = self.measuredDF
        measuredDF = self.measuredDF.copy(deep=True)
        measuredDF = larsimDataPostProcessing.filterResultForStation(measuredDF, station=station)

        if objective_function_qoi is None:
            objective_function_qoi = self.larsimConfObject.objective_function_qoi
        #objective_function_qoi = larsimDataPostProcessing._gof_list_to_function_names(objective_function_qoi)

        # list_over_objective_function = larsimDataPostProcessing.\
        #     calculateGoodnessofFit_ForSingleStation(measuredDF=measuredDF,
        #                                             predictedDF=predictedDF,
        #                                             station=station,
        #                                             gof_list=objective_function_qoi,
        #                                             measuredDF_column_name=self.measuredDF_column_name,
        #                                             simulatedDF_column_name="Value",
        #                                             filter_station=True,
        #                                             filter_type_of_output=False,
        #                                             return_dict=return_dict
        #                                             )
        list_over_objective_function = larsimDataPostProcessing.\
            calculateGoodnessofFit_simple(measuredDF=measuredDF,
                                          predictedDF=predictedDF,
                                          gof_list=objective_function_qoi,
                                          measuredDF_column_name=self.measuredDF_column_name,
                                          simulatedDF_column_name="Value",
                                          station=station,
                                          return_dict=return_dict)

        return list_over_objective_function

    def _calculate_GoF(self, predictedDF, parameters_dict=None,
                       objective_function=None, get_all_possible_stations=True):
        measuredDF = self._get_measured_df()
        if measuredDF is None:
            return None

        # get the structure of the df_measured
        measuredDF_column_name = self.measuredDF_column_name

        if objective_function is None:
            objective_function = self.larsimConfObject.objective_function
        #objective_function = larsimDataPostProcessing._gof_list_to_function_names(objective_function)

        stations = LarsimModel.compute_and_get_final_list_of_stations(measuredDF, predictedDF,
                                                                      get_all_possible_stations,
                                                                      self.larsimConfObject.station_of_Interest)

        gof_list_over_stations = larsimDataPostProcessing.\
            calculateGoodnessofFit(measuredDF=measuredDF,
                                   predictedDF=predictedDF,
                                   station=stations,
                                   gof_list=objective_function,
                                   measuredDF_column_name=measuredDF_column_name,
                                   simulatedDF_column_name='Value',
                                   type_of_output_of_Interest=self.larsimConfObject.type_of_output_of_Interest,
                                   dailyStatisict=False,
                                   disregard_initila_timesteps=False,
                                   warm_up_duration=self.larsimConfObject.warm_up_duration,
                                   keep_info_on_TimeStampas=False,
                                   filter_station=True,
                                   filter_type_of_output=True,
                                   return_dict=True
                                   )

        index_parameter_gof_list_of_dictionaries = []
        for single_stations_gof in gof_list_over_stations:
            if parameters_dict is not None:
                index_parameter_gof_dict = {**parameters_dict, **single_stations_gof}
            else:
                index_parameter_gof_dict = {**single_stations_gof}
            index_parameter_gof_list_of_dictionaries.append(index_parameter_gof_dict)
        return pd.DataFrame(index_parameter_gof_list_of_dictionaries)

    @staticmethod
    def compute_and_get_final_list_of_stations(measuredDF, predictedDF,
                                               get_all_possible_stations=True, station_of_Interest="all"):
        stations = larsimDataPostProcessing.get_stations_intersection(measuredDF, predictedDF)
        if not get_all_possible_stations and station_of_Interest != "all" and station_of_Interest is not None:
            if not isinstance(station_of_Interest, list):
                station_of_Interest = [station_of_Interest, ]
            stations = list(set(stations).intersection(station_of_Interest))
            if not stations:
                raise Exception(
                    f"[LarsimModel ERROR:] Error in _calculate_GoF - no intersection between "
                    f"LarsimModel.station_of_Interest and stations in LarsimModel.measuredDF, LarsimModel.predictedDF!")
        return stations

    def _copy_files_for_gradient_computation(self, curr_working_dir, i, id_param):
        os.chdir(curr_working_dir)
        # working_folder_name = "compute_gradient_" + str(i) + "_" + str(id_param)
        working_folder_name = f"compute_gradient_{i}_{id_param}"  #TODO
        curr_working_dir_gradient = curr_working_dir / working_folder_name
        curr_working_dir_gradient.mkdir(parents=True, exist_ok=True)

        # copy all the necessary files to the newly created directory
        # master_dir_for_copying = self.larsimPathsObject.global_master_dir + "/."
        # subprocess.run(['cp', '-a', master_dir_for_copying, curr_working_dir_gradient])
        curr_working_dir_for_copying = str(curr_working_dir) + "/."
        master_dir_for_copying = str(self.master_dir) + "/."
        subprocess.run(['cp', '-a', master_dir_for_copying, curr_working_dir_gradient])

        tape35_path = curr_working_dir / self.larsimPathsObject.tape_35_path.name
        lanu_path = curr_working_dir / self.larsimPathsObject.lanu_path.name

        subprocess.run(['cp', tape35_path, curr_working_dir_gradient])
        subprocess.run(['cp', lanu_path, curr_working_dir_gradient])
        print("[LarsimModel INFO] Successfully copied all the files for gradient computation")
        # change working directory
        os.chdir(curr_working_dir_gradient)

        return curr_working_dir_gradient




