import datetime
import dill
from distutils.util import strtobool
from functools import reduce
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


class LarsimModelSetUp():
    def __init__(self, configurationObject, *args, **kwargs):

        if not isinstance(configurationObject, dict):
            self.configurationObject = larsimConfigurationSettings.return_configuration_object(configurationObject)
        else:
            self.configurationObject = configurationObject

        #####################################
        # Specification of different directories - some are machine / location dependent,
        # adjust path in larsimPaths module and configuration file/object accordingly
        #####################################

        self.sourceDir = kwargs.get('sourceDir') if 'sourceDir' in kwargs and osp.isabs(kwargs.get('sourceDir')) \
            else osp.dirname(pathlib.Path(__file__).resolve())

        self.inputModelDir = kwargs.get('inputModelDir') if 'inputModelDir' in kwargs else paths.larsim_data_path

        if "workingDir" in kwargs:
            self.workingDir = kwargs.get('workingDir')
        else:
            try:
                self.workingDir = self.configurationObject["Directories"]["workingDir"]
            except KeyError:
                self.workingDir = paths.workingDir

        self.master_dir = osp.abspath(osp.join(self.workingDir, 'master_configuration'))

        self.global_master_dir = osp.abspath(osp.join(self.inputModelDir,'WHM Regen','master_configuration'))
        self.master_lila_paths = [osp.abspath(osp.join(self.inputModelDir,'WHM Regen', i)) for i in paths.master_lila_files]
        self.lila_configured_paths = [os.path.abspath(os.path.join(self.master_dir, i)) for i in paths.lila_files]
        self.all_whms_path = osp.abspath(osp.join(self.inputModelDir,'WHM Regen','var/WHM Regen WHMS'))

        try:
            self.larsim_exe = self.configurationObject["Directories"]["larsim_exe"]
        except KeyError:
            self.larsim_exe = osp.abspath(osp.join(self.inputModelDir, 'Larsim-exe', 'larsim-linux-intel-1000.exe'))

        self.sourceDir = pathlib.Path(self.sourceDir)
        self.workingDir = pathlib.Path(self.workingDir)
        self.inputModelDir = pathlib.Path(self.inputModelDir)
        self.master_dir = pathlib.Path(self.master_dir)
        self.global_master_dir = pathlib.Path(self.global_master_dir)
        self.all_whms_path = pathlib.Path(self.all_whms_path)
        self.larsim_exe = pathlib.Path(self.larsim_exe)

        for i, file in enumerate(self.master_lila_paths):
            self.master_lila_paths[i] = pathlib.Path(file)
        for i, file in enumerate(self.lila_configured_paths):
            self.lila_configured_paths[i] = pathlib.Path(file)

        self.regen_saved_data_files = self.inputModelDir / 'WHM Regen' / 'data_files'

        print(self.sourceDir)
        print(self.workingDir)
        print(self.inputModelDir)
        print(self.global_master_dir)
        print(self.all_whms_path)
        print(self.larsim_exe)

        #####################################
        # Specification of different variables for setting the model run and purpose of the model run
        #####################################

        try:
            self.station_of_Interest = self.configurationObject["Output"]["station_calibration_postproc"]
        except KeyError:
            self.station_of_Interest = "MARI"

        try:
            self.station_for_model_runs = self.configurationObject["Output"]["station_model_runs"]
        except KeyError:
            self.station_for_model_runs = "all"

        try:
            self.type_of_output_of_Interest = self.configurationObject["Output"]["type_of_output"]
        except KeyError:
            self.type_of_output_of_Interest = "Abfluss Messung + Vorhersage"

        try:
            self.type_of_output_of_Interest_measured = self.configurationObject["Output"]["type_of_output_measured"]
        except KeyError:
            self.type_of_output_of_Interest_measured  = "Ground Truth"

        try:
            self.warm_up_duration = self.configurationObject["Timeframe"]["warm_up_duration"]
        except KeyError:
            self.warm_up_duration = 53

        self.calculate_GoF = strtobool(self.configurationObject["Output"]["calculate_GoF"])\
                       if "calculate_GoF" in self.configurationObject["Output"] else True
        if self.calculate_GoF:
            self.objective_function = self.configurationObject["Output"]["objective_function"] \
                if 'objective_function' in self.configurationObject["Output"] else "all"
            self.objective_function = larsimDataPostProcessing._gof_list_to_function_names(self.objective_function)

        larsimConfigurationSettings.update_configurationObject_with_parameters_info(self.configurationObject)

        # Get the timeframe for running the simulation from the configuration file
        if "Timeframe" in self.configurationObject:
            self.timeframe = larsimTimeUtility.parse_datetime_configuration(self.configurationObject)
        else:
            raise Exception(f"[LarsimModelSetUp ERRO:] Timeframe specification is missing from the configuration object!")

        self.copy_master_folder()
        self.configure_master_folder()

        self.df_simulation = None
        self.df_measured = None
        self.df_unaltered_ergebnis = None
        self.df_gof_unaltered_meas = None
        self.df_gof_sim_meas = None

        self.boolean_get_measured_discharge = kwargs.get('get_measured_discharge') \
            if 'get_measured_discharge' in kwargs else True
        self.boolean_get_Larsim_saved_simulations = kwargs.get('get_Larsim_saved_simulations') \
            if 'get_Larsim_saved_simulations' in kwargs else True
        self.boolean_run_unaltered_sim = kwargs.get('run_unaltered_sim') if 'run_unaltered_sim' in kwargs else True

        if self.boolean_get_measured_discharge:
            self.get_measured_discharge(write_in_file=True)
        if self.boolean_get_Larsim_saved_simulations:
            self.get_Larsim_saved_simulations(write_in_file=True)
        if self.boolean_run_unaltered_sim:
            self.run_unaltered_sim(createNewFolder=False, write_in_file=True)
        if self.boolean_get_measured_discharge and self.boolean_run_unaltered_sim:
            self._compare_measurements_and_unalteredSim(get_all_possible_stations=True, write_in_file=True)

        print("[LarsimModelSetUp INFO] Model Initial setup is done! ")

    def copy_master_folder(self):
        # for safety reasons make a copy of the master_dir in the working_dir and continue working with that one
        self.master_dir.mkdir(parents=True, exist_ok=True)
        master_dir_for_copying = str(self.global_master_dir) + "/."
        subprocess.run(['cp', '-a', master_dir_for_copying, self.master_dir])

    def configure_master_folder(self):
        """
        Copy configuration files & do all the configurations needed for proper execution
        & copy initial and input data files to master folder
        :return:
        """
        # # Get the timeframe for running the simulation from the configuration file
        # self.timeframe = larsimTimeUtility.parse_datetime_configuration(self.configurationObject)

        # if not osp.isdir(self.master_dir): raise IOError('LarsimModelSetUp Error: Please first creat the following folder: %s. %s' % (self.master_dir, IOError.strerror))
        if not self.master_dir.is_dir():
            raise IOError(f'[LarsimModelSetUp Error] Please first create the following folder: '
                          f'{self.master_dir} {IOError.strerror}')

        # Based on time settings change tape10_master file - needed for unaltered run -
        # this will be repeted once again by each process in LarsimModel.run()
        # tape10_adjusted_path = osp.abspath(osp.join(self.master_dir, 'tape10'))
        # master_tape10_file = osp.abspath(osp.join(self.master_dir, 'tape10_master'))
        tape10_adjusted_path = self.master_dir / 'tape10'
        master_tape10_file = self.master_dir / 'tape10_master'

        try:
            larsimTimeUtility.tape10_configuration(timeframe=self.timeframe, master_tape10_file=master_tape10_file, \
                                                   new_path=tape10_adjusted_path, warm_up_duration=self.warm_up_duration)
        except larsimTimeUtility.ValidDurationError as e:
            print("[LarsimModelSetUp ERROR] - Something is  wrong with the time settings \n]"+str(e))
            return None

        # Filter out whm files
        larsimConfigurationSettings.copy_whm_files(timeframe=self.timeframe, all_whms_path=self.all_whms_path,
                                                   new_path=self.master_dir)

        # Parse big lila files and create small ones
        larsimConfigurationSettings.master_lila_parser_based_on_time_crete_new(timeframe=self.timeframe,
                                                                               master_lila_paths=self.master_lila_paths,
                                                                               new_lila_paths=self.lila_configured_paths)

        for one_lila_file in self.lila_configured_paths:
            paths._check_if_file_exists(one_lila_file, f"[LarsimModelSetUp Error] File {one_lila_file} does not exist!")

        print("[LarsimModelSetUp INFO] Initial configuration is done - all the files have been copied to master folder!")

    def get_measured_discharge(self, read_file_path=None, filtered_timesteps_vs_station_values=True, write_in_file=True,
                               write_file_path=None, *args, **kwargs):
        #####################################
        # extract measured (ground truth) discharge values
        # there are multiple ways how one can do that
        #####################################

        if read_file_path is None:
            if filtered_timesteps_vs_station_values:
                read_file_path = self.regen_saved_data_files / 'q_2003-11-01_2018-01-01_time_and_values_filtered.pkl'
            else:
                read_file_path = self.master_dir / paths.lila_files[0]

        if read_file_path.is_file():
            self.df_measured = larsimDataPostProcessing.read_process_write_discharge(df=read_file_path,
                                                                                     timeframe=self.timeframe,
                                                                                     station=self.station_for_model_runs,
                                                                                     compression="gzip"
                                                                                     )
        else:
            # example for this branch is when read_file_path is of type  "./q_2003-11-01_2018-01-01_time_and_values_filtered.pkl"
            # however it might be that this file does not exist. In that case one will read and process/filter the whole dataFrame again
            drop_duplicates = kwargs["drop_duplicates"] if "drop_duplicates" in kwargs else True
            fill_missing_timesteps = kwargs["fill_missing_timesteps"] if "fill_missing_timesteps" in kwargs else True
            interpolate_missing_values = kwargs["interpolate_missing_values"] if "interpolate_missing_values" in kwargs else True
            interpolation_method = kwargs["interpolation_method"] if "interpolation_method" in kwargs else 'time'

            read_file_path = self.master_dir / paths.lila_files[0]
            if read_file_path.is_file():
                self.df_measured = larsimDataPreparation.get_filtered_df(df=read_file_path,
                                                                         stations=self.station_for_model_runs,
                                                                         start_date=self.timeframe[0],
                                                                         end_date=self.timeframe[1],
                                                                         drop_duplicates=drop_duplicates,
                                                                         fill_missing_timesteps=fill_missing_timesteps,
                                                                         interpolate_missing_values=interpolate_missing_values,
                                                                         interpolation_method=interpolation_method,
                                                                         only_time_series_values=filtered_timesteps_vs_station_values,
                                                                         )
        if write_in_file:
            if write_file_path is None:
                write_file_path= self.workingDir / "df_measured.pkl"
            larsimInputOutputUtilities.write_dataFrame_to_file(self.df_measured,
                                                               file_path=write_file_path,
                                                               compression="gzip")

    def get_Larsim_saved_simulations(self, filtered_timesteps_vs_station_values=True, write_in_file=True,
                                     write_file_path=None, *args, **kwargs):
        list_of_df_per_station = []
        if self.station_for_model_runs is None or self.station_for_model_runs == "all":
            station_for_model_runs = list(larsimDataPostProcessing.get_Stations(self.df_measured))
        if not isinstance(station_for_model_runs, list):
            station_for_model_runs = [station_for_model_runs, ]
        for station in station_for_model_runs:
            df_station_sim_path = self.regen_saved_data_files / f"larsim_output_{station}_2005_2017.pkl"
            df_station_sim_filtered_path = self.regen_saved_data_files / f"larsim_output_{station}_2005_2017_filtered.pkl"
            if df_station_sim_filtered_path.is_file():
                df_sim = larsimInputOutputUtilities.read_dataFrame_from_file(df_station_sim_filtered_path, compression="gzip")
                df_sim = larsimDataPostProcessing.parse_df_based_on_time(df_sim, (self.timeframe[0], self.timeframe[1]))
            elif df_station_sim_path.is_file():
                drop_duplicates = kwargs["drop_duplicates"] if "drop_duplicates" in kwargs else True
                fill_missing_timesteps = kwargs[
                    "fill_missing_timesteps"] if "fill_missing_timesteps" in kwargs else True
                interpolate_missing_values = kwargs[
                    "interpolate_missing_values"] if "interpolate_missing_values" in kwargs else True
                interpolation_method = kwargs["interpolation_method"] if "interpolation_method" in kwargs else 'time'
                df_sim = larsimDataPreparation.get_filtered_df(df=df_station_sim_path,
                                                               start_date=self.timeframe[0],
                                                               end_date=self.timeframe[1],
                                                               drop_duplicates=drop_duplicates,
                                                               fill_missing_timesteps=fill_missing_timesteps,
                                                               interpolate_missing_values=interpolate_missing_values,
                                                               interpolation_method=interpolation_method,
                                                               only_time_series_values=filtered_timesteps_vs_station_values,
                                                               )
            if filtered_timesteps_vs_station_values:
                df_sim.rename(columns={"Value": station}, inplace=True)
            list_of_df_per_station.append(df_sim)

        if filtered_timesteps_vs_station_values:
            self.df_simulation = reduce(lambda x, y: pd.merge(x, y, on="TimeStamp", how='outer'), list_of_df_per_station)
        else:
            self.df_simulation = pd.concat(list_of_df_per_station, ignore_index=True, sort=False, axis=0)

        if write_in_file:
            if write_file_path is None:
                write_file_path = self.workingDir / "df_past_simulated.pkl"
            larsimInputOutputUtilities.write_dataFrame_to_file(self.df_simulation,
                                                               file_path=write_file_path,
                                                               compression="gzip")

    # TODO change run_unaltered_sim such that it can as well run in cut_runs mode
    def run_unaltered_sim(self, createNewFolder=False, write_in_file=True, write_file_path=None):
        #####################################
        ### run unaltered simulation
        #####################################
        if createNewFolder:
            dir_unaltered_run = self.workingDir / "WHM Regen 000"
            dir_unaltered_run.mkdir(parents=True, exist_ok=True)
            master_dir_for_copying = str(self.master_dir) + "/."
            subprocess.run(['cp', '-a', master_dir_for_copying, dir_unaltered_run])
        else:
            dir_unaltered_run = self.master_dir

        os.chdir(dir_unaltered_run)
        larsimConfigurationSettings._delete_larsim_output_files(curr_directory=dir_unaltered_run)
        local_log_file = dir_unaltered_run / "run.log"
        subprocess.run([str(self.larsim_exe)], stdout=open(local_log_file, 'w'))
        os.chdir(self.sourceDir)
        print(f"[LarsimModelSetUp INFO:] Unaltered Run is completed, current folder is: {self.sourceDir}")

        result_file_path = dir_unaltered_run / 'ergebnis.lila'
        self.df_unaltered_ergebnis = larsimInputOutputUtilities.ergebnis_parser_toPandas(result_file_path)

        simulation_start_timestamp = self.timeframe[0]
        if self.warm_up_duration is not None:
            simulation_start_timestamp = self.timeframe[0] + datetime.timedelta(hours=self.warm_up_duration)
        self.df_unaltered_ergebnis = larsimDataPostProcessing.parse_df_based_on_time(self.df_unaltered_ergebnis, (simulation_start_timestamp, None))

        # filter out results for a concrete station if specified in configuration json file
        if self.station_for_model_runs is not None and self.station_for_model_runs!="all":
            self.df_unaltered_ergebnis = larsimDataPostProcessing.filterResultForStation(self.df_unaltered_ergebnis, station=self.station_for_model_runs)
        if self.type_of_output_of_Interest is not None and self.type_of_output_of_Interest != "all":
            self.df_unaltered_ergebnis = larsimDataPostProcessing.filterResultForTypeOfOutpu(self.df_unaltered_ergebnis,
                                                                         type_of_output=self.type_of_output_of_Interest)

        if write_in_file:
            if write_file_path is None:
                write_file_path = self.workingDir / "df_unaltered.pkl"
            larsimInputOutputUtilities.write_dataFrame_to_file(self.df_unaltered_ergebnis,
                                                               file_path=write_file_path,
                                                               compression="gzip")

        # delete ergebnis.lila and all other not necessary files
        if createNewFolder:
            larsimConfigurationSettings.cleanDirecory_completely(curr_directory=dir_unaltered_run)
        else:
            larsimConfigurationSettings._delete_larsim_output_files(curr_directory=dir_unaltered_run)

    def _compare_measurements_and_unalteredSim(self, get_all_possible_stations=True,
                                               write_in_file=True, write_file_path=None):
        if self.df_measured is None or self.df_unaltered_ergebnis is None:
            return None

        stations = larsimDataPostProcessing.get_stations_intersection(self.df_measured, self.df_unaltered_ergebnis)
        if not get_all_possible_stations and (self.station_of_Interest != "all" or self.station_of_Interest is not None):
            if not isinstance(self.station_of_Interest, list):
                self.station_of_Interest = [self.station_of_Interest,]
            stations = list(set(stations).intersection(self.station_of_Interest))

        gof_list_over_stations = []
        for station in stations:
            df_sim = larsimDataPostProcessing.filterResultForStationAndTypeOfOutpu(self.df_unaltered_ergebnis,
                                                                                   station=station,
                                                                                   type_of_output=self.type_of_output_of_Interest)
            temp_gof_dict = larsimDataPostProcessing.calculateGoodnessofFit_simple(self.df_measured,
                                                                                   df_sim,
                                                                                   gof_list=self.objective_function,
                                                                                   measuredDF_column_name=station,
                                                                                   simulatedDF_column_name='Value'
                                                                                   )
            temp_gof_dict["station"] = station
            gof_list_over_stations.append(temp_gof_dict)
        self.df_gof_unaltered_meas = pd.DataFrame(gof_list_over_stations)

        if self.df_simulation is not None:
            gof_list_over_stations_sim_mes = []
            for station in stations:
                temp_gof_dict = larsimDataPostProcessing.calculateGoodnessofFit_simple(self.df_measured,
                                                                                       self.df_simulation,
                                                                                       gof_list=self.objective_function,
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

    def __init__(self, configurationObject, *args, **kwargs):
        Model.__init__(self)

        if not isinstance(configurationObject, dict):
            self.configurationObject = larsimConfigurationSettings.return_configuration_object(configurationObject)
        else:
            self.configurationObject = configurationObject

        #####################################
        # Specification of different directories - some are machine / location dependent,
        # adjust path in larsimPaths moduel and configuration file/object accordingly
        #####################################

        self.sourceDir = kwargs.get('sourceDir') if 'sourceDir' in kwargs and osp.isabs(kwargs.get('sourceDir')) \
                            else osp.dirname(pathlib.Path(__file__).resolve())

        self.inputModelDir = kwargs.get('inputModelDir') if 'inputModelDir' in kwargs else paths.larsim_data_path

        try:
            self.larsim_exe = self.configurationObject["Directories"]["larsim_exe"]
        except KeyError:
            self.larsim_exe = osp.abspath(osp.join(self.inputModelDir, 'Larsim-exe', 'larsim-linux-intel-1000.exe'))

        # directory for the larsim runs
        if "workingDir" in kwargs:
            self.workingDir = kwargs.get('workingDir')
        else:
            try:
                self.workingDir = self.configurationObject["Directories"]["workingDir"]
            except KeyError:
                self.workingDir = paths.workingDir

        self.master_dir = osp.abspath(osp.join(self.workingDir, 'master_configuration'))

        self.sourceDir = pathlib.Path(self.sourceDir)
        self.workingDir = pathlib.Path(self.workingDir)
        self.inputModelDir = pathlib.Path(self.inputModelDir)
        self.master_dir = pathlib.Path(self.master_dir)
        self.larsim_exe = pathlib.Path(self.larsim_exe)

        self.local_measurement_file = self.workingDir / "df_measured.pkl"
        if not self.local_measurement_file.exists():
            self.local_measurement_file = self.master_dir / paths.lila_files[0]
        #####################################
        # Specification of different variables for setting the model run and purpose of the model run
        #####################################
        try:
            self.station_of_Interest = self.configurationObject["Output"]["station_calibration_postproc"]
        except KeyError:
            self.station_of_Interest = "MARI"

        try:
            self.station_for_model_runs = self.configurationObject["Output"]["station_model_runs"]
        except KeyError:
            self.station_for_model_runs = "all"

        try:
            self.type_of_output_of_Interest = self.configurationObject["Output"]["type_of_output"]
        except KeyError:
            self.type_of_output_of_Interest = "Abfluss Messung + Vorhersage"

        try:
            self.type_of_output_of_Interest_measured = self.configurationObject["Output"]["type_of_output_measured"]
        except KeyError:
            self.type_of_output_of_Interest_measured  = "Ground Truth"

        self.cut_runs = strtobool(self.configurationObject["Timeframe"]["cut_runs"])\
                       if "cut_runs" in self.configurationObject["Timeframe"] else False

        self.warm_up_duration = self.configurationObject["Timeframe"]["warm_up_duration"] \
                                if "warm_up_duration" in self.configurationObject["Timeframe"] else 53

        self.variable_names = []
        if "tuples_parameters_info" in self.configurationObject:
            for i in self.configurationObject["tuples_parameters_info"]:
                self.variable_names.append(i["name"])
        else:
            for i in self.configurationObject["parameters"]:
                self.variable_names.append(i["name"])
            larsimConfigurationSettings.update_configurationObject_with_parameters_info(self.configurationObject)

        #####################################
        # this variable stands for the purpose of LarsimModel run
        # distinguish between different modes / purposes of LarsimModel runs:
        #               calibration, run_and_save_simulations, gradient_computation, UQ_analysis
        # These modes do not have to be mutually exclusive!
        #####################################
        self.qoi = self.configurationObject["Output"]["QOI"] if "QOI" in self.configurationObject["Output"] else "Q"
        if self.qoi != "Q" and self.qoi != "GoF":
            raise Exception(f"[LarsimModel ERROR:] self.qoi should either be \"Q\" or \"GoF\" ")

        self.mode = self.configurationObject["Output"]["mode"] \
            if "mode" in self.configurationObject["Output"] else "continuous"
        if self.mode != "continuous" and self.mode != "sliding_window" and self.mode != "resampling":
            raise Exception(f"[LarsimModel ERROR:] self.mode should have one of the  following values:"
                            f" \"continuous\" or \"sliding_window\" or \"resampling\"")

        if self.mode == "sliding_window" or self.mode == "resampling":
            self.interval = self.configurationObject["Output"]["interval"] \
                if "interval" in self.configurationObject["Output"] else 24
            if self.qoi == "Q":
                self.method = self.configurationObject["Output"]["method"] \
                    if "method" in self.configurationObject["Output"] else "avg"
                if self.method != "avg" and self.method != "max":
                    raise Exception(f"[LarsimModel ERROR:] self.method should be either \"avg\" or \"max\" ")

        # if calibration is True some likelihood / objective functions / GoF functio should be calculated from model run and propageted further
        self.calculate_GoF = strtobool(self.configurationObject["Output"]["calculate_GoF"]) \
            if "calculate_GoF" in self.configurationObject["Output"] else True

        # TODO Distinguish between two types of objective_function function,
        #  e.g., objective_function vs. objective_function_qoi
        if self.calculate_GoF or self.qoi == "GoF":
            self.objective_function = self.configurationObject["Output"]["objective_function"] \
                if 'objective_function' in self.configurationObject["Output"] else "all"
            self.objective_function = larsimDataPostProcessing._gof_list_to_function_names(self.objective_function)

        # save the output of each simulation just in run function just in case when run_and_save_simulations in json configuration file is True
        # and no statistics calculations will be performed afterwards, otherwise the simulation results will be saved in LarsimStatistics
        self.disable_statistics = kwargs.get('disable_statistics') if 'disable_statistics' in kwargs else False
        self.run_and_save_simulations = strtobool(self.configurationObject["Output"]["run_and_save_simulations"])\
                                        if "run_and_save_simulations" in self.configurationObject["Output"] else False
        self.run_and_save_simulations = self.run_and_save_simulations and self.disable_statistics

        # if we want to compute the gradient (of some likelihood fun or output itself) w.r.t parameters
        self.compute_gradients = strtobool(self.configurationObject["Output"]["compute_gradients"]) \
            if "compute_gradients" in self.configurationObject["Output"] else False
        if self.compute_gradients:
            if self.configurationObject["Output"]["gradients_method"] == "Central Difference":
                self.CD = 1  # flag for using Central Differences (with 2 * num_evaluations)
            elif self.configurationObject["Output"]["gradients_method"] == "Forward Difference":
                self.CD = 0  # flag for using Forward Differences (with num_evaluations)
            else:
                raise Exception(f"[LarsimModel ERROR:] NUMERICAL GRADIENT EVALUATION ERROR: Only \"Central Difference\" "
                                f"and \"Forward Difference\" supported")
            try:
                # difference for gradient computation
                self.eps_val_global = self.configurationObject["Output"]["eps_gradients"]
            except KeyError:
                self.eps_val_global = 1e-4

        #####################################
        # getting the time span for running the model from the json configuration file
        #####################################

        if "Timeframe" in self.configurationObject:
            self.timeframe = larsimTimeUtility.parse_datetime_configuration(self.configurationObject)
            # generate timesteps for plotting based on tape10 settings which are set in LarsimModelSetUp
            self.t = larsimTimeUtility.get_tape10_timesteps(self.timeframe)
        else:
            raise Exception(f"[LarsimModel ERROR:] Timeframe specification is missing from the configuration object!")

        # how long one consecutive run should take - used later on in each Larsim run
        self.timestep = self.configurationObject["Timeframe"]["timestep"] if "timestep" in self.configurationObject["Timeframe"] else 5

        self.measuredDF = None
        self._is_measuredDF_computed = False
        self.measuredDF_column_name = 'Value'
        self._set_measured_df()

        print("[LarsimModel INFO] INITIALIZATION DONE]\n")

    def prepare(self):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def set_configuration_object(self, configurationObject):
        self.configurationObject = larsimConfigurationSettings.return_configuration_object(configurationObject)

    def set_timeframe(self, timeframe):
        self.timeframe = larsimTimeUtility.timeframe_to_datetime_list(timeframe)
        larsimConfigurationSettings.update_configurationObject_with_datetime_info(self.configurationObject, timeframe)
        self.t = larsimTimeUtility.get_tape10_timesteps(timeframe)

    def set_station_of_Interest(self, station_of_Interest):
        larsimConfigurationSettings.update_config_dict_station_of_interest(self.configurationObject, station_of_Interest)
        self.station_of_Interest = station_of_Interest

    def timesteps(self):
        return self.t

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
            local_measurement_file = self.master_dir / paths.lila_files[0]
            self.measuredDF = larsimDataPostProcessing.read_process_write_discharge(df=local_measurement_file,
                                                                               timeframe=self.timeframe,
                                                                               station=self.station_for_model_runs,
                                                                               compression="gzip")
        self._is_measuredDF_computed = True
        self.__set_measuredDF_column_name()

    def _get_measured_df(self):
        if not self._is_measuredDF_computed:
            self._set_measured_df()
        return self.measuredDF

    def run(self, i_s=[0,], parameters=None):  # i_s - index chunk; parameters - parameters chunk

        print(f"[LarsimModel INFO] {i_s} paramater: {parameters}")

        results = []
        for ip in range(0, len(i_s)): # for each peace of work
            i = i_s[ip]  # i is unique index run

            if parameters is not None:
                parameter = parameters[ip]
            else:
                parameter = None

            start = time.time()

            # create local directory for this particular run
            working_folder_name = f"WHM Regen{i}"
            curr_working_dir = self.workingDir / working_folder_name
            curr_working_dir.mkdir(parents=True, exist_ok=True)

            # copy all the necessary files to the newly created directory
            master_dir_for_copying = str(self.master_dir) + "/."
            subprocess.run(['cp', '-a', master_dir_for_copying, curr_working_dir])
            print("[LarsimModel INFO] Successfully copied all the files")

            # change values
            id_dict = {"index_run": i}
            # if parameter is not None:
            tape35_path = curr_working_dir / "tape35"
            lanu_path = curr_working_dir / "lanu.par"
            parameters_dict = larsimConfigurationSettings.params_configurations(parameters=parameter,
                                                                                tape35_path=tape35_path,
                                                                                lanu_path=lanu_path,
                                                                                configurationObject=self.configurationObject,
                                                                                process_id=i)
            parameters_dict = {**id_dict, **parameters_dict}
            # else: #TODO add option when parameter is None to read default parameters values and run unaltered run
            #     parameters_dict = {**id_dict,}

            # change working directory
            os.chdir(curr_working_dir)

            # Run Larsim
            if self.cut_runs:
                result = self._multiple_short_larsim_runs(timeframe=self.timeframe, timestep=self.timestep,
                                                          curr_working_dir=curr_working_dir, index_run=i,
                                                          warm_up_duration=self.warm_up_duration)
            else:
                result = self._single_larsim_run(timeframe=self.timeframe, curr_working_dir=curr_working_dir,
                                                 index_run=i, warm_up_duration=self.warm_up_duration)
            if result is None:
                # end = time.time()
                # runtime = end - start
                # results.append((None, runtime))
                larsimConfigurationSettings.cleanDirecory_completely(curr_directory=curr_working_dir)
                os.chdir(self.sourceDir)
                continue

            # Postprocessing the timeframe
            # self.timeframe[0], self.timeframe[1] = result["TimeStamp"].min(), result["TimeStamp"].max()
            # larsimConfigurationSettings.update_configurationObject_with_datetime_info(self.configurationObject, self.timeframe)
            # self.t = larsimTimeUtility.get_tape10_timesteps(self.timeframe)

            # filter output time-series in order to disregard warm-up time;
            # if not, then at least disregard these values when calculating statistics and GoF
            # however, take care that is is not done twice!
            simulation_start_timestamp = self.timeframe[0]
            if self.warm_up_duration is not None:
                simulation_start_timestamp = self.timeframe[0] + datetime.timedelta(hours=self.warm_up_duration) # pd.Timestamp(result.TimeStamp.min())
            result = larsimDataPostProcessing.parse_df_based_on_time(result, (simulation_start_timestamp, None))

            # filter out results for a concrete station if specified in configuration json file
            if self.station_for_model_runs is not None and self.station_for_model_runs != "all":
                result = larsimDataPostProcessing.filterResultForStation(result, station=self.station_for_model_runs)
            if self.type_of_output_of_Interest is not None and self.type_of_output_of_Interest != "all":
                result = larsimDataPostProcessing.filterResultForTypeOfOutpu(result, type_of_output=self.type_of_output_of_Interest)

            end = time.time()
            runtime = end - start
            result_dict = {"run_time": runtime, "parameters_dict": parameters_dict}

            processed_result = None
            if self.mode == "sliding_window":
                if self.qoi == "GoF":
                    processed_result = self._process_time_series_sliding_window_gof(result,
                                                                                    self.interval,
                                                                                    self.objective_function)
                else:
                    processed_result = self._process_time_series_sliding_window_q(result,
                                                                                  self.interval,
                                                                                  self.method)
            elif self.mode == "resampling":
                if self.qoi == "GoF":
                    processed_result = self._process_time_series_cutting_gof(result,
                                                                             self.interval,
                                                                             self.objective_function)
                else:
                    processed_result = self._process_time_series_cutting_q(result,
                                                                           self.interval,
                                                                           self.method)
            if processed_result is None:
                result_dict["result_time_series"] = result
            else:
                result_dict["result_time_series"] = processed_result
                result_dict["raw_time_series"] = result

            if self.calculate_GoF:
                index_parameter_gof_DF = self._calculate_GoF(predictedDF=result,
                                                             parameters_dict=parameters_dict,
                                                             get_all_possible_stations=True)
                result_dict["gof_df"] = index_parameter_gof_DF

            # compute gradient of the output, or some likelihood measure w.r.t parameters
            if self.compute_gradients:
                # CD = 1 central differences; CD = 0 forward differences
                gradient_matrix = []
                if gradient_matrix:
                    result_dict["gradient"] = gradient_matrix

            if self.run_and_save_simulations:
                file_path = self.workingDir / f"parameters_Larsim_run_{i}.pkl"
                with open(file_path, 'wb') as f:
                    dill.dump(parameters_dict, f)
                file_path = self.workingDir / f"df_Larsim_run_{i}.pkl"
                larsimDataPostProcessing.read_process_write_discharge(result,
                                                                      type_of_output=self.type_of_output_of_Interest,
                                                                      station=self.station_for_model_runs,
                                                                      write_to_file=file_path, compression="gzip")
                if processed_result is not None:
                    file_path = self.workingDir / f"df_Larsim_run_processed_{i}.pkl"
                    processed_result.to_pickle(file_path, compression="gzip")

                if self.calculate_GoF:
                    file_path = self.workingDir / f"gof_{i}.pkl"
                    index_parameter_gof_DF.to_pickle(file_path, compression="gzip")

            #####################################
            # Final cleaning and appending the results
            #####################################
            print(f"[LarsimModel INFO] Process {i} returned / appended it's results")

            # result_dict contains at least the following entries:  "result_time_series", "run_time", "parameters_dict"
            # optionally: "gof_df", "gradient" , etc.
            results.append((result_dict, runtime))

            # Delete everything except .log and .csv files
            larsimConfigurationSettings.cleanDirecory_completely(curr_directory=curr_working_dir)

            # change back to starting directory of all the processes
            os.chdir(self.sourceDir)

            # Delete local working folder
            subprocess.run(["rm", "-r", curr_working_dir])

            print(f"[LarsimModel INFO] I am done - solver number {i}")

        return results

    def _single_larsim_run(self, timeframe, curr_working_dir, index_run=0, sub_index_run=0, warm_up_duration=None):

        if warm_up_duration is None: warm_up_duration = self.warm_up_duration
        # start clean
        larsimConfigurationSettings._delete_larsim_output_files(curr_directory=curr_working_dir)

        # change tape 10 accordingly
        local_master_tape10_file = curr_working_dir / 'tape10_master'
        local_tape10_adjusted_path = curr_working_dir / 'tape10'
        try:
            larsimTimeUtility.tape10_configuration(timeframe=timeframe, master_tape10_file=local_master_tape10_file,
                                                   new_path=local_tape10_adjusted_path, warm_up_duration=warm_up_duration)
        except larsimTimeUtility.ValidDurationError as e:
            print(f"[LarsimModel ERROR - None is returned after the Larsim run \n]"+str(e))
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
        subprocess.run([str(self.larsim_exe)], stdout=open(local_log_file, 'w'))
        print(f"[LarsimModel INFO] I am done with LARSIM Execution {index_run}")

        # check if larsim.ok exist - Larsim execution was successful
        larsimConfigurationSettings.check_larsim_ok_file(curr_working_dir=curr_working_dir)
        result_file_path = curr_working_dir / 'ergebnis.lila'
        try:
            df_single_ergebnis = larsimInputOutputUtilities.ergebnis_parser_toPandas(result_file_path, index_run)
            return df_single_ergebnis
        except paths.FileError as e:
            print(str(e))
            print(f"[LarsimModel ERROR] Process {index_run}: The following Ergebnis file was not found - {result_file_path}")
            raise

    # TODO Change _multiple_short_larsim_runs such that local_timestep/timestep are set in hours
    def _multiple_short_larsim_runs(self, timeframe, timestep, curr_working_dir, index_run=0, warm_up_duration=None):

        if warm_up_duration is None: warm_up_duration = self.warm_up_duration

        # if you want to cut execution into shorter runs...
        local_timestep = timestep

        number_of_runs = (timeframe[1] - timeframe[0]).days // datetime.timedelta(days=local_timestep).days
        number_of_runs_mode = (timeframe[1] - timeframe[0]).days % datetime.timedelta(days=local_timestep).days

        local_end_date = timeframe[0]

        result_file_path = curr_working_dir / 'ergebnis.lila'
        larsim_ok_file_path = curr_working_dir / 'larsim.ok'
        tape11_file_path = curr_working_dir / 'tape11'
        karte_path = curr_working_dir / 'karten'  # curr_working_dir / 'karten/*'
        tape10_path = curr_working_dir / 'tape10'

        print(f"[LarsimModel INFO] process {index_run} gonna run {number_of_runs} "
              f"shorter Larsim runs (and number_of_runs_mode {number_of_runs_mode})")

        local_resultDF_list = []
        for i in range(number_of_runs):

            # remove previous tape10
            subprocess.run(["rm", "-f", tape10_path])

            # calculate times - make sure that outputs are continuous in time
            if i == 0:
                local_start_date = local_end_date
                local_start_date_warmup = local_start_date - datetime.timedelta(hours=warm_up_duration)
            else:
                local_start_date_warmup = local_end_date
                local_start_date = local_start_date_warmup - datetime.timedelta(hours=warm_up_duration)

            if local_start_date > timeframe[1]:
                break

            if i == 0:
                local_end_date = local_end_date + datetime.timedelta(hours=warm_up_duration) + datetime.timedelta(days=local_timestep)
            else:
                local_end_date = local_end_date + datetime.timedelta(days=local_timestep)

            if local_end_date > timeframe[1]:
                local_end_date = timeframe[1]

            print(f"[LarsimModel INFO] Process {index_run}; local_start_date: {local_start_date}; local_end_date: {local_end_date}")
            single_run_timeframe = (local_start_date, local_end_date)

            # run larsim for this shorter period and returned already parsed 'small' ergebnis
            local_resultDF = self._single_larsim_run(timeframe=single_run_timeframe, curr_working_dir=curr_working_dir,
                                                     index_run=index_run, sub_index_run=i, warm_up_duration=warm_up_duration)

            if local_resultDF is None:
                break

            # postprocessing of time variables
            local_start_date, local_end_date = local_resultDF["TimeStamp"].min(), local_resultDF["TimeStamp"].max()
            local_start_date_warmup = local_start_date + datetime.timedelta(hours=warm_up_duration)

            local_resultDF_drop = local_resultDF.drop(local_resultDF[local_resultDF['TimeStamp'] < local_start_date_warmup].index)

            local_resultDF_list.append(local_resultDF_drop)

            # rename ergebnis.lila
            local_result_file_path = curr_working_dir / f'ergebnis_{i}.lila'
            subprocess.run(["mv", result_file_path, local_result_file_path])
            local_larsim_ok_file_path = curr_working_dir / f'larsim_{i}.ok'
            subprocess.run(["mv", larsim_ok_file_path, local_larsim_ok_file_path])

        if local_resultDF_list:
            # concatenate it
            df_simulation_result = pd.concat(local_resultDF_list, ignore_index=True, sort=True, axis=0)

            # sorting by time
            df_simulation_result.sort_values("TimeStamp", inplace=True)

            # clean concatenated file - dropping time duplicate values
            df_simulation_result.drop_duplicates(subset=['TimeStamp', 'Stationskennung', 'Type'], keep='first', inplace=True)

            return df_simulation_result
        else:
            return None

    def _calculate_GoF_sliding_window_single_gof_single_station(self, predictedDF, station,
                                                                objective_function=None, return_dict=False):
        """
        This function assumes that self.measuredDF is already computed by self._set_measured_df function
        and that self._is_measuredDF_computed is set to True

        Important note: predictedDF structure is part of standard resultDF

        Can work both for single and multiple stations
        """
        # TODO: to speed-up change the functions such that the predictedDF
        # TODO is just a series with TimeStamp column as index column and one extra Value column
        # TODO if we remove self.measuredDF, self._is_measuredDF_computed self._set_measured_df() - make it staticmethod

        if self.measuredDF is None and not self._is_measuredDF_computed:
            self._set_measured_df()
        elif self.measuredDF is None and self._is_measuredDF_computed:
            return None

        # if self.measuredDF.empty or predictedDF.empty:
        #     print(f"Failed in self.measuredDF.empty or predictedDF.empty")
        # if self.measuredDF_column_name not in self.measuredDF.columns:
        #     print(f"self.measuredDF_column_name not in self.measuredDF.columns")
        # if "Value" not in predictedDF.columns:
        #     print(f"Value not in predictedDF.columns")

        measuredDF_temp = larsimDataPostProcessing.filterResultForStation(self.measuredDF, station=station)

        if objective_function is None:
            objective_function = self.objective_function

        list_over_objective_function = larsimDataPostProcessing.\
            calculateGoodnessofFit_ForSingleStation(measuredDF=self.measuredDF,
                                                    predictedDF=predictedDF,
                                                    station = station,
                                                    gof_list=objective_function,
                                                    measuredDF_column_name=self.measuredDF_column_name,
                                                    simulatedDF_column_name="Value",
                                                    filter_station=True,
                                                    filter_type_of_output=False,
                                                    return_dict = return_dict
                                                    )
        # calculateGoodnessofFit_simple(measuredDF=measuredDF_temp,
        #                               predictedDF=predictedDF,
        #                               gof_list=objective_function,
        #                               measuredDF_column_name=self.measuredDF_column_name,
        #                               simulatedDF_column_name="Value",
        #                               station=station,
        #                               return_dict=return_dict)

        return list_over_objective_function

    def _calculate_GoF(self, predictedDF, parameters_dict=None, get_all_possible_stations=True):
        measuredDF = self._get_measured_df()
        if measuredDF is None:
            return None

        # get the structure of the  df_measured
        measuredDF_column_name = self.measuredDF_column_name

        stations = larsimDataPostProcessing.get_stations_intersection(measuredDF, predictedDF)
        if not get_all_possible_stations and (self.station_of_Interest != "all" or self.station_of_Interest is not None):
            if not isinstance(self.station_of_Interest, list):
                self.station_of_Interest = [self.station_of_Interest,]
            stations = list(set(stations).intersection(self.station_of_Interest))

        gof_list_over_stations = larsimDataPostProcessing.\
            calculateGoodnessofFit(measuredDF=measuredDF,
                                   predictedDF=predictedDF,
                                   station=stations,
                                   gof_list=self.objective_function,
                                   measuredDF_column_name = measuredDF_column_name,
                                   simulatedDF_column_name='Value',
                                   type_of_output_of_Interest=self.type_of_output_of_Interest,
                                   dailyStatisict=False,
                                   disregard_initila_timesteps=False,
                                   warm_up_duration=self.warm_up_duration,
                                   keep_info_on_TimeStampas=False,
                                   filter_station=True,
                                   filter_type_of_output=True
                                   )

        index_parameter_gof_list_of_dictionaries = []
        for single_stations_gof in gof_list_over_stations:
            if parameters_dict is not None:
                index_parameter_gof_dict = {**parameters_dict, **single_stations_gof}
            else:
                index_parameter_gof_dict = {**single_stations_gof}
            index_parameter_gof_list_of_dictionaries.append(index_parameter_gof_dict)
        return pd.DataFrame(index_parameter_gof_list_of_dictionaries)

    def _process_time_series_sliding_window_gof(self, result, interval, objective_function):
        processed_result = None
        # for each station for each GoF function
        return processed_result

    def _process_time_series_sliding_window_q(self, result, interval, method):
        processed_result = None
        return processed_result

    def _process_time_series_cutting_gof(self, result, interval, objective_function):
        processed_result = None
        return processed_result

    def _process_time_series_cutting_q(self, result, interval, method):
        processed_result = None
        return processed_result

