import datetime
import dill
from distutils.util import strtobool
from functools import reduce
import inspect
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

        self.configurationObject = configurationObject

        #####################################
        # Sepcification of different directories - some are machine / location dependent,
        # adjust path in larsimPaths moduel and configuration file/object accordingly
        #####################################

        self.current_dir = kwargs.get('sourceDir') if 'sourceDir' in kwargs and osp.isabs(kwargs.get('sourceDir')) \
                            else osp.dirname(pathlib.Path(__file__).resolve())

        self.inputModelDir = kwargs.get('inputModelDir') if 'inputModelDir' in kwargs else paths.larsim_data_path

        if "working_dir" in kwargs:
            self.working_dir = kwargs.get('working_dir')
        else:
            try:
                self.working_dir = self.configurationObject["Directories"]["working_dir"]
            except KeyError:
                self.working_dir = paths.working_dir

        self.master_dir = osp.abspath(osp.join(self.working_dir, 'master_configuration'))

        self.global_master_dir = osp.abspath(osp.join(self.inputModelDir,'WHM Regen','master_configuration'))
        self.master_lila_paths = [osp.abspath(osp.join(self.inputModelDir,'WHM Regen', i)) for i in paths.master_lila_files]
        self.lila_configured_paths = [os.path.abspath(os.path.join(self.master_dir, i)) for i in paths.lila_files]
        self.all_whms_path = osp.abspath(osp.join(self.inputModelDir,'WHM Regen','var/WHM Regen WHMS'))

        self.larsim_exe = osp.abspath(osp.join(self.inputModelDir, 'Larsim-exe', 'larsim-linux-intel-1000.exe'))

        self.current_dir = pathlib.Path(self.current_dir)
        self.working_dir = pathlib.Path(self.working_dir)
        self.inputModelDir = pathlib.Path(self.inputModelDir)
        self.master_dir = pathlib.Path(self.master_dir)
        self.global_master_dir = pathlib.Path(self.global_master_dir)
        self.master_lila_paths = pathlib.Path(self.master_lila_paths)
        self.lila_configured_paths = pathlib.Path(self.lila_configured_paths)
        self.all_whms_path = pathlib.Path(self.all_whms_path)
        self.larsim_exe = pathlib.Path(self.larsim_exe)

        #####################################
        # Sepcification of different variables for setting the model run and purpose of the model run
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
            self.type_of_output_of_Interest = "Abfluss Messung"

        try:
            self.type_of_output_of_Interest_measured = self.configurationObject["Output"]["type_of_output_measured"]
        except KeyError:
            self.type_of_output_of_Interest_measured  = "Ground Truth"

        try:
            self.warm_up_duration = self.configurationObject["Timeframe"]["warm_up_duration"]
        except KeyError:
            self.warm_up_duration = 53

        larsimConfigurationSettings.update_configurationObject_with_parameters_info(configurationObject)

        self.copy_master_folder()
        self.configure_master_folder()

        self.get_measured_discharge()
        self.run_unaltered_sim(createNewFolder=False, write_in_file=True)
        self._compare_measurements_and_unalteredSim()

        print("[LarsimModelSetUp INFO:] Model Initial setup is done! ")

    def copy_master_folder(self):
        # for safety reasons make a copy of the master_dir in the working_dir and continue working with that one
        # if not osp.isdir(self.master_dir): subprocess.run(["mkdir", self.master_dir])
        # master_dir_for_copying = self.global_master_dir + "/."
        self.master_dir.mkdir(parents=True, exist_ok=True)
        master_dir_for_copying = self.global_master_dir / "./"
        subprocess.run(['cp', '-a', master_dir_for_copying, self.master_dir])

    def configure_master_folder(self):
        """
        Copy configuration files & do all the configurations needed for proper execution
        & copy initial and input data files to master folder
        :return:
        """
        # Get the timeframe for running the simulation from the configuration file
        self.timeframe = larsimTimeUtility.parse_datetime_configuration(
            self.configurationObject)  # tuple with EREIGNISBEGINN EREIGNISENDE

        #if not osp.isdir(self.master_dir): raise IOError('LarsimModelSetUp Error: Please first creat the following folder: %s. %s' % (self.master_dir, IOError.strerror))
        if not self.master_dir.is_dir():
            raise IOError('[LarsimModelSetUp Error:] Please first create the following folder: %s. %s' % (self.master_dir, IOError.strerror))


        # Based on time settings change tape10_master file - needed for unaltered run - this will be repeted once again by each process in LarsimModel.run()
        # tape10_adjusted_path = osp.abspath(osp.join(self.master_dir, 'tape10'))
        # master_tape10_file = osp.abspath(osp.join(self.master_dir, 'tape10_master'))
        tape10_adjusted_path = self.master_dir / 'tape10'
        master_tape10_file = self.master_dir / 'tape10_master'


        larsimTimeUtility.tape10_configuration(timeframe=self.timeframe, master_tape10_file=master_tape10_file, \
                                               new_path=tape10_adjusted_path, warm_up_duration=self.warm_up_duration)

        # Filter out whm files
        larsimConfigurationSettings.copy_whm_files(timeframe=self.timeframe, all_whms_path=self.all_whms_path, new_path=self.master_dir)

        # Parse big lila files and create small ones
        larsimConfigurationSettings.master_lila_parser_based_on_time_crete_new(timeframe=self.timeframe,
                                                                               master_lila_paths=self.master_lila_paths,
                                                                               new_lila_paths=self.lila_configured_paths)

        for one_lila_file in self.lila_configured_paths:
            paths._check_file(one_lila_file, f"[LarsimModelSetUp Error:] File {one_lila_file} does not exist!")


        print("[LarsimModelSetUp INFO:] Initial configuration is done - all the files have been copied to master folder! ")

    def get_measured_discharge(self, read_file_path=None, filtered_timesteps_vs_station_values=True, write_in_file=True,
                               write_file_path=None, *args, **kwargs):
        #####################################
        ### extract measured (ground truth) discharge values
        ### there are multiple ways how one can do that
        #####################################

        if read_file_path is None:
            if filtered_timesteps_vs_station_values:
                read_file_path = paths.regen_saved_data_files / 'q_2003-11-01_2018-01-01_time_and_values_filtered.pkl'
            else:
                read_file_path = self.master_dir / paths.lila_files[0]

        if read_file_path.is_file():
            self.df_measured = larsimDataPostProcessing.read_process_write_discharge(df=read_file_path,
                                                                                     timeframe=self.timeframe,
                                                                                     station=self.station_for_model_runs,
                                                                                     )
        else:
            # example for this branch is when read_file_path is of type  "./q_2003-11-01_2018-01-01_time_and_values_filtered.pkl"
            # however it might be that this file does not exists. In that case one will read and process/filter the whole dataFrame again
            read_file_path = self.master_dir / paths.lila_files[0]
            if read_file_path.is_file():
                self.df_measured = larsimDataPreparation.get_filtered_df(df=read_file_path,
                                                                         stations=self.station_for_model_runs,
                                                                         start_date=self.timeframe[0],
                                                                         end_date=self.timeframe[1],
                                                                         drop_duplicates=kwargs["drop_duplicates"],
                                                                         fill_missing_timesteps=kwargs["fill_missing_timesteps"],
                                                                         interpolate_missing_values=kwargs["interpolate_missing_values"],
                                                                         interpolation_method=kwargs["interpolation_method"],
                                                                         only_time_series_values=True,
                                                                         )
        if write_in_file:
            if write_file_path is None:
                write_file_path= self.working_dir / "df_measured.pkl"
            larsimInputOutputUtilities.write_dataFrame_to_file(self.df_measured,
                                                               write_to_file= write_file_path,
                                                               compression="gzip")

    def get_Larsim_saved_simulations(self, filtered_timesteps_vs_station_values=True, write_in_file=True,
                                     write_file_path=None, *args, **kwargs):
        list_of_df_per_station = []
        if self.station_for_model_runs is None or self.station_for_model_runs=="all":
            station_for_model_runs = list(larsimDataPostProcessing.get_Stations(self.df_measured))
        if not isinstance(station_for_model_runs, list):
            station_for_model_runs = [station_for_model_runs, ]
        for station in station_for_model_runs:
            df_station_sim_path = paths.regen_saved_data_files / f"larsim_output_{station}_2005_2017.pkl"
            df_station_sim_filtered_path = paths.regen_saved_data_files / f"larsim_output_{station}_2005_2017_filtered.pkl"
            if df_station_sim_filtered_path.is_file():
                df_sim = larsimInputOutputUtilities.read_dataFrame_from_file(df_station_sim_filtered_path, compression="gzip")
            elif df_station_sim_path.is_file():
                df_sim = larsimDataPreparation.get_filtered_df(df=df_station_sim_path,
                                                               start_date=self.timeframe[0],
                                                               end_date=self.timeframe[1],
                                                               drop_duplicates=kwargs["drop_duplicates"],
                                                               fill_missing_timesteps=kwargs["fill_missing_timesteps"],
                                                               interpolate_missing_values=kwargs["interpolate_missing_values"],
                                                               interpolation_method=kwargs["interpolation_method"],
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
                write_file_path = self.working_dir / "df_simulated.pkl"
            larsimInputOutputUtilities.write_dataFrame_to_file(self.df_simulation,
                                                               write_to_file=write_file_path,
                                                               compression="gzip")

    def run_unaltered_sim(self, createNewFolder=False, write_in_file=True, write_file_path=None):
        #####################################
        ### run unaltered simulation
        #####################################
        if createNewFolder:
            dir_unaltered_run = self.working_dir / "WHM Regen 000"
            dir_unaltered_run.mkdir(parents=True, exist_ok=True)
            master_dir_for_copying = self.master_dir / "/."
            subprocess.run(['cp', '-a', master_dir_for_copying, dir_unaltered_run])
        else:
            dir_unaltered_run = self.master_dir

        os.chdir(dir_unaltered_run)
        larsimConfigurationSettings._delete_larsim_output_files(curr_directory=dir_unaltered_run)
        local_log_file = dir_unaltered_run / "run.log"
        subprocess.run([self.larsim_exe], stdout=open(local_log_file, 'w'))
        os.chdir(self.current_dir)
        print("[LarsimModelSetUp INFO:] Unaltered Run is completed, current folder is:{}".format(self.current_dir))

        result_file_path = dir_unaltered_run / 'ergebnis.lila'
        self.df_unaltered_ergebnis = larsimInputOutputUtilities.ergebnis_parser_toPandas(result_file_path)

        # filter output time-series in order to disregard warm-up time; important that these values are not take into account while computing GoF
        # However, take care that is is not done twice!
        #simulation_start_timestamp = self.timeframe[0] + datetime.timedelta(hours=self.warm_up_duration) # pd.Timestamp(result.TimeStamp.min())
        #self.df_unaltered_ergebnis = larsimDataPostProcessing.parse_df_based_on_time(self.df_unaltered_ergebnis, (simulation_start_timestamp, None))

        # filter out results for a concrete station if specified in configuration json file
        if self.station_for_model_runs is not None and self.station_for_model_runs!="all":
            self.df_unaltered_ergebnis = larsimDataPostProcessing.filterResultForStation(self.df_unaltered_ergebnis, station=self.station_for_model_runs)

        if write_in_file:
            if write_file_path is None:
                write_file_path = osp.abspath(osp.join(self.working_dir, "df_unaltered_ergebnis.pkl"))
            larsimInputOutputUtilities.write_dataFrame_to_file(self.df_unaltered_ergebnis,
                                                               write_to_file=write_file_path,
                                                               compression="gzip")

        # delte ergebnis.lila and all other not necessary files
        if createNewFolder:
            larsimConfigurationSettings.cleanDirecory_completely(curr_directory=dir_unaltered_run)
        else:
            larsimConfigurationSettings._delete_larsim_output_files(curr_directory=dir_unaltered_run)


    def _compare_measurements_and_unalteredSim(self):
        #####################################
        ### compare ground truth measurements and unaltered run for this simulation (compute RMSE | BIAS | NSE | logNSE)
        #####################################
        #hourly
        goodnessofFit_list_of_dictionaries = larsimDataPostProcessing.calculateGoodnessofFit(measuredDF=self.df_measured, predictedDF=self.df_unaltered_ergebnis,\
                                                                       station=self.station_of_Interest,\
                                                                       type_of_output_of_Interest_measured=self.type_of_output_of_Interest_measured,\
                                                                       type_of_output_of_Interest=self.type_of_output_of_Interest,\
                                                                       dailyStatisict=False, gof_list="all",\
                                                                       disregard_initila_timesteps=True, warm_up_duration=self.warm_up_duration)
        # write in a file GOF values of the unaltered model prediction
        self.index_parameter_gof_DF = pd.DataFrame(goodnessofFit_list_of_dictionaries)
        gof_file_path = osp.abspath(osp.join(self.working_dir, "GOF_Measured_vs_Unaltered.pkl"))
        self.index_parameter_gof_DF.to_pickle(gof_file_path, compression="gzip")


class LarsimModel(Model):

    def __init__(self, configurationObject, *args, **kwargs):
        Model.__init__(self)

        self.configurationObject = configurationObject

        #####################################
        # Sepcification of different directories - some are machine / location dependent,
        # adjust path in larsimPaths moduel and configuration file/object accordingly
        #####################################

        self.current_dir = kwargs.get('sourceDir') if 'sourceDir' in kwargs and osp.isabs(kwargs.get('sourceDir')) \
                            else osp.dirname(pathlib.Path(__file__).resolve())

        self.inputModelDir = kwargs.get('inputModelDir') if 'inputModelDir' in kwargs else paths.larsim_data_path

        self.larsim_exe = osp.abspath(osp.join(self.inputModelDir, 'Larsim-exe', 'larsim-linux-intel-1000.exe'))

        # directoy for the larsim runs
        if "working_dir" in kwargs:
            self.working_dir = kwargs.get('working_dir')
        else:
            try:
                self.working_dir = self.configurationObject["Directories"]["working_dir"]
            except KeyError:
                self.working_dir = paths.working_dir

        self.master_dir = osp.abspath(osp.join(self.working_dir, 'master_configuration'))

        #####################################
        # Sepcification of different variables for setting the model run and purpose of the model run
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
            self.type_of_output_of_Interest = "Abfluss Messung"

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
                self.variable_names.append(i["parameter_name"])
        else:
            for i in self.configurationObject["parameters"]:
                self.variable_names.append(i["name"])

        # this variable stands for the purpose of LarsimModel run
        # distinguis between different modes / purposes of LarsimModel runs:
        #               calibration, run_and_save_simulations, gradient_computation, UQ_analysis
        # These modes do not have to be mutually exclusive!

        # if calibration is True some likelihood / objective functions / GoF functio should be calculated from model run and propageted further
        self.calculate_GoF = strtobool(self.configurationObject["Output"]["calculate_GoF"])\
                       if "calculate_GoF" in self.configurationObject["Output"] else False
        if self.calculate_GoF:
            # TODO-Ivana deal with situation when self.station_for_model_runs is a list/dictionary
            self.objective_function = self.configurationObject["Output"]["objective_function"]
            self.objective_function = larsimDataPostProcessing._gof_list_to_function_names(self.objective_function)

        # save the output of each simulation just in run function just in case when run_and_save_simulations in json configuration file is True
        # and no statistics calculations will be performed afterwards, otherwise the simulation results will be saved in LarsimStatistics
        self.disable_statistics = kwargs.get('disable_statistics') if 'disable_statistics' in kwargs else False
        self.run_and_save_simulations = strtobool(self.configurationObject["Output"]["run_and_save_simulations"])\
                                        if "run_and_save_simulations" in self.configurationObject["Output"] else False
        #self.run_and_save_simulations = self.run_and_save_simulations and self.disable_statistics


        # if we want to compute the gradient (of some likelihood fun or output itself) w.r.t parameters
        self.compute_gradients = strtobool(self.configurationObject["Output"]["compute_gradients"])\
                       if "compute_gradients" in self.configurationObject["Output"] else False

        #####################################
        # getting the time span for running the model from the json configuration file
        #####################################

        self.timeframe = larsimTimeUtility.parse_datetime_configuration(
            self.configurationObject)
        self.timestep = self.configurationObject["Timeframe"]["timestep"]\
                       if "timestep" in self.configurationObject["Timeframe"] else 5  # how long one consecutive run should take - used later on in each Larsim run
        # generate timesteps for plotting based on tape10 settings which are set in LarsimModelSetUp
        self.t = larsimTimeUtility.get_tape10_timesteps(self.timeframe)

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

    def run(self, i_s, parameters): #i_s - index chunk; parameters - parameters chunk

        print("LarsimModel INFO {}: paramater: {}".format(i_s, parameters))

        results = []
        for ip in range(0, len(i_s)): # for each peace of work
            i = i_s[ip]# i is unique index run

            if parameters is not None:
                parameter = parameters[ip]
            else:
                parameter = None

            start = time.time()

            # create local directory for this particular run
            working_folder_name = "WHM Regen" + str(i)
            curr_working_dir = osp.abspath(osp.join(self.working_dir,working_folder_name))

            if not osp.isdir(curr_working_dir):
                subprocess.run(["mkdir", curr_working_dir])

            # copy all the necessary files to the newly created directoy
            master_dir_for_copying = self.master_dir + "/."
            subprocess.run(['cp', '-a', master_dir_for_copying, curr_working_dir])  # TODO IVANA Check if copy succeed
            print("LarsimModel INFO: Successfully copied all the files")

            # change values
            if parameter is not None:
                #config.tape35_configurations(parameters=parameter, curr_working_dir=curr_working_dir, configurationObject=self.configurationObject)
                tape35_path = curr_working_dir + "/tape35"
                lanu_path = curr_working_dir + "/lanu.par"
                parameters_dict = larsimConfigurationSettings.params_configurations(parameters = parameter, tape35_path = tape35_path,
                                                                  lanu_path = lanu_path,
                                                                  configurationObject = self.configurationObject,
                                                                  process_id = i)

            id_dict = {"index_run": i}
            parameters_dict = {**id_dict, **parameters_dict}

            # change working directory
            os.chdir(curr_working_dir)

            # Run Larsim
            if self.cut_runs:
                result = self._multiple_short_larsim_runs(timeframe=self.timeframe, timestep = self.timestep, curr_working_dir=curr_working_dir,\
                                                          index_run=i, warm_up_duration=self.warm_up_duration)
            else:
                result = self._single_larsim_run(timeframe=self.timeframe, curr_working_dir=curr_working_dir, index_run=i)

            # filter output time-series in order to disregard warm-up time; if not then at least disregard these values when calculating statistics and GoF
            # however, take care that is is not done twice!
            simulation_start_timestamp = self.timeframe[0] + datetime.timedelta(hours=self.warm_up_duration) # pd.Timestamp(result.TimeStamp.min())
            result = larsimDataPostProcessing.parse_df_based_on_time(result, (simulation_start_timestamp, self.timeframe[1]))

            #filter out results for a concret station if specified in configuration json file
            # TODO-Ivana deal with situation when self.station_for_model_runs is a list
            if self.station_for_model_runs!="all":
                result = larsimDataPostProcessing.filterResultForStation(result, station=self.station_for_model_runs)

            end = time.time()
            runtime = end - start

            result_dict = {"result_time_series": result, "run_time":runtime, "parameters_dict":parameters_dict}

            # What comes fron this point onwards is to determin what is the purpose of the LarsimModel simulation run
            # e.g. is to just run multiple simulations and store their outputs, and/or to propaget results for calibration,
            # and/or to propaget results for UQ analysis, and/or to calculate gradients, etc.
            # the behsviour is determined based on a couple of configuration variables, mostly coming from json configuration file such as:
            #               calibration, run_and_save_simulations, UQ_analysis, gradient_computation
            # These modes do not have to be mutually exclusive!

            #TODO-Ivana distinguis between doing only calibration and doing UQ_analysis with evaluating some GoF
            #####################################
            ### compare model predictions of this simulation with measured (ground truth) data
            ### this can be moved to Statistics - positioned here due to parallelisation
            #####################################
            # if calibration is True some likelihood / objective functions / GoF functio should be calculated from model run and propageted further and'or saved to file
            if self.calculate_GoF:
                # get the DataFrame storing measurments / ground truth discharge
                local_measurment_file = osp.abspath(osp.join(self.working_dir, "df_measured.pkl"))
                if os.path.exists(local_measurment_file):
                    gt_dataFrame = larsimInputOutputUtilities.read_dataFrame_from_file(local_measurment_file, compression="gzip")
                else:
                    #gt_dataFrame=None #this will wokr as well because when calculationg GoF groundTruth DF will be read anyway
                    gt_dataFrame = larsimConfigurationSettings.extract_measured_discharge(self.timeframe[0], self.timeframe[1], index_run=0)
                    #gt_dataFrame = larsimConfigurationSettings.extract_measured_discharge(simulation_start_timestamp, self.timeframe[1], index_run=0)

                # Make sure that burn-in time is disregard in result or will be disregard while computing GoF: disregard_initila_timesteps=False or disregard_initila_timesteps=True
                # Check for which stations GoF should be calculated: station=self.station_of_Interest or station=self.station_for_model_runs
                # Chech weather you want daily or hourly based computation of GoF functions: dailyStatisict=False or dailyStatisict=True
                goodnessofFit_list_of_dictionaries = larsimDataPostProcessing.calculateGoodnessofFit(measuredDF=gt_dataFrame, predictedDF=result,\
                                                                               station=self.station_of_Interest,\
                                                                               type_of_output_of_Interest_measured=self.type_of_output_of_Interest_measured,\
                                                                               type_of_output_of_Interest=self.type_of_output_of_Interest,\
                                                                               dailyStatisict=False, gof_list=self.objective_function,\
                                                                               disregard_initila_timesteps=False)
                index_parameter_gof_list_of_dictionaries = []
                for single_stations_gof in goodnessofFit_list_of_dictionaries:
                    index_parameter_gof_dict = {**parameters_dict, **single_stations_gof}
                    index_parameter_gof_list_of_dictionaries.append(index_parameter_gof_dict)
                #index_parameter_gof_DF = pd.DataFrame(goodnessofFit_list_of_dictionaries)
                index_parameter_gof_DF = pd.DataFrame(index_parameter_gof_list_of_dictionaries)
                result_dict["gof_df"] = index_parameter_gof_DF

                # save index_run, parameter values and GoF as pd.DataFrame in a file
                # optionally glue it to the results and propagate everything further for postprocessing
                #index_parameter_gof_DF.to_pickle(osp.abspath(osp.join(self.working_dir, "goodness_of_fit_" + str(i) +  ".pkl")), compression="gzip")

            # compute gradient of the output, or some likelihood measure w.r.t parameters
            if self.compute_gradients:
                gradient_matrix = []
                if gradient_matrix:
                    result_dict["gradient"] = gradient_matrix

            #TODO-Ivana distinguis between only saving reults and saving and propagating further
            # save the output of each simulation here just in case the purpose of the simulation is to run multiple Larsim runs
            # for different parameters and to save these simulations runs and when no statistics calculations will be performed afterwards,
            # otherwise the simulation results will be saved in LarsimStatistics
            if self.run_and_save_simulations:
                file_path =  osp.abspath(osp.join(self.working_dir, "parameters_Larsim_run_" + str(i) +  ".pkl"))
                with open(file_path, 'wb') as f:
                    #pickle.dump(parameters_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                    dill.dump(parameters_dict, f)
                #if self.disable_statistics:
                file_path =  osp.abspath(osp.join(self.working_dir, "df_Larsim_run_" + str(i) +  ".pkl"))
                larsimDataPostProcessing.read_process_write_discharge(result, type_of_output=self.type_of_output_of_Interest,\
                                             station=self.station_for_model_runs, write_to_file=file_path, compression="gzip")
                if self.calculate_GoF:
                    # save index_run, parameter values and GoF as pd.DataFrame in a file
                    # optionally glue it to the results and propagate everything further for postprocessing
                    index_parameter_gof_DF.to_pickle(osp.abspath(osp.join(self.working_dir, "goodness_of_fit_" + str(i) +  ".pkl")), compression="gzip")

            #Debugging  - TODO Delete afterwards
            print("LarsimModel INFO: Process {} returned / appended it's results".format(i))
            #assert len(result['TimeStamp'].unique()) == len(self.t), "Assesrtion Failed: Something went wrong with time resolution of the result"
            #assert isinstance(self.variable_names, list), "Assertion Failed - variable names not a list"
            #assert len(self.variable_names) == len(parameter), "Assertion Failed parametr not of the same length as variable names"

            #if self.calculate_GoF:
            #    # Extend resulted ergebnis pd.DataFrame with pd.DataFrame storing index_run, parameter values and GoF
            #    result = (result, index_parameter_gof_DF)
            #else:
            #    # Extend reults with dict storing index_run and parameter values
            #    result = (result, parameters_dict)

            # propagate further results for post-processingin LarsimStatistics
            #results.append((result, runtime))

            # result_dict contains at least the following entries:  "result_time_series", "run_time", "parameters_dict"
            # optionally: "gof_df", "gradient" , etc.
            results.append((result_dict, runtime))

            # Delete everything except .log and .csv files
            larsimConfigurationSettings.cleanDirecory_completely(curr_directory=curr_working_dir)

            # change back to starting directory of all the processes
            os.chdir(self.current_dir)

            # Delete local working folde
            subprocess.run(["rm", "-r", curr_working_dir])

            print("LarsimModel INFO: I am done - solver number {}".format(i))

        return results


    def timesteps(self):
        return self.t


    def _single_larsim_run(self, timeframe, curr_working_dir, index_run=0, sub_index_run=0):

        # start clean
        larsimConfigurationSettings._delete_larsim_output_files(curr_directory=curr_working_dir)

        # change tape 10 accordingly
        local_master_tape10_file = osp.abspath(osp.join(curr_working_dir, 'tape10_master'))
        local_adjusted_path = osp.abspath(osp.join(curr_working_dir, 'tape10'))
        larsimTimeUtility.tape10_configuration(timeframe=timeframe, master_tape10_file=local_master_tape10_file, new_path=local_adjusted_path, warm_up_duration=self.warm_up_duration)

        # log file for larsim
        local_log_file = osp.abspath(
            osp.join(curr_working_dir, "run" + str(index_run) + "_" + str(sub_index_run) + ".log"))
        # print("LARSIM MODEL INFO: This is where I'm gonna write my log - {}".format(local_log_file))

        # run Larsim as external process
        subprocess.run([self.larsim_exe], stdout=open(local_log_file, 'w'))
        print("LarsimModel INFO: I am done with LARSIM Execution {}".format(index_run))

        # check if larsim.ok exist - Larsim execution was successful
        larsimConfigurationSettings.check_larsim_ok_file(curr_working_dir=curr_working_dir)

        result_file_path = osp.abspath(osp.join(curr_working_dir, 'ergebnis.lila'))

        if osp.isfile(result_file_path):
            df_single_ergebnis = larsimInputOutputUtilities.ergebnis_parser_toPandas(result_file_path, index_run)
            return df_single_ergebnis
        else:
            return None #TODO Handle this more elegantly


    def _multiple_short_larsim_runs(self, timeframe, timestep, curr_working_dir, index_run=0, warm_up_duration=53):
        # if you want to cut execution into shorter runs...
        local_timestep = timestep

        #number_of_runs = datetime.timedelta(days=(timeframe[1] - timeframe[0]).days).days // datetime.timedelta(days=local_timestep).days
        number_of_runs = (timeframe[1] - timeframe[0]).days // datetime.timedelta(days=local_timestep).days
        number_of_runs_mode = (timeframe[1] - timeframe[0]).days % datetime.timedelta(days=local_timestep).days

        local_end_date = timeframe[0]

        result_file_path = osp.abspath(osp.join(curr_working_dir, 'ergebnis.lila'))
        larsim_ok_file_path = osp.abspath(osp.join(curr_working_dir, 'larsim.ok'))
        tape11_file_path = osp.abspath(osp.join(curr_working_dir, 'tape11'))
        karte_path = osp.abspath(osp.join(curr_working_dir, 'karten'))  # curr_working_dir + 'karten/*'
        tape10_path = osp.abspath(osp.join(curr_working_dir, 'tape10'))

        print("LarsimModel INFO: process {} gonna run {} shorter Larsim runs (and number_of_runs_mode {})".format(index_run, number_of_runs, number_of_runs_mode))

        local_resultDF_list = []
        for i in range(number_of_runs+1):

            # remove previous tape10
            subprocess.run(["rm", "-f", tape10_path])

            # calulcate times - make sure that outputs are continuous in time
            if i == 0:
                local_start_date = local_end_date
                local_start_date_p_53 = local_start_date - datetime.timedelta(hours=warm_up_duration)
            else:
                local_start_date_p_53 = local_end_date
                local_start_date = local_start_date_p_53 - datetime.timedelta(hours=warm_up_duration)

            if local_start_date > timeframe[1]:
                break

            #TODO This brings some confusion, try without this!
            #local_start_date = local_start_date.replace(hour=0, minute=0, second=0)

            if i == 0:
                local_end_date = local_end_date + datetime.timedelta(hours=warm_up_duration) + datetime.timedelta(days=local_timestep)
            else:
                local_end_date = local_end_date + datetime.timedelta(days=local_timestep)


            if local_end_date > timeframe[1]:
                local_end_date = timeframe[1]

            print("LarsimModel INFO: Process {}; local_start_date: {}; local_end_date: {}".format(index_run, local_start_date, local_end_date))
            single_run_timeframe = (local_start_date, local_end_date)

            # run larsim for this shorter period and returned already parsed 'small' ergebnis
            local_resultDF = self._single_larsim_run(timeframe=single_run_timeframe, curr_working_dir=curr_working_dir, index_run=index_run, sub_index_run=i)

            #TODO Handle this more elegantly
            if local_resultDF is None:
                raise ValueError("LarsimModel ERROR: Process {}: The following Ergebnis file was not found - {}".format(index_run, result_file_path))

            #TODO Take maybe an average over duplicated timestamps instead of droping
            local_resultDF_drop = local_resultDF.drop(local_resultDF[local_resultDF['TimeStamp'] < local_start_date_p_53].index)

            #TODO Check if some interpolation is needed...
            local_resultDF_list.append(local_resultDF_drop)

            # rename ergebnis.lila
            local_result_file_path = osp.abspath(osp.join(curr_working_dir, 'ergebnis' + '_' + str(i) + '.lila'))
            subprocess.run(["mv", result_file_path, local_result_file_path])
            local_larsim_ok_file_path = osp.abspath(osp.join(curr_working_dir, 'larsim' + '_' + str(i) + '.ok'))
            subprocess.run(["mv", larsim_ok_file_path, local_larsim_ok_file_path])

        # concatinate it
        df_simulation_result = pd.concat(local_resultDF_list, ignore_index=True, sort=True, axis=0)

        # sorting by time
        df_simulation_result.sort_values("TimeStamp", inplace=True)

        # clean concatanated file - dropping time duplicate values
        df_simulation_result.drop_duplicates(subset=['TimeStamp', 'Stationskennung', 'Type'], keep='last', inplace=True)

        #print("DEBUGGING LARSIM INFO: process {} - After Droping -  MARI and Messung (Hourly):\n".format(i))
        #print(len(df_simulation_result.TimeStamp.unique()))
        #print("\n")
        #print(len((df_simulation_result.loc[(df_simulation_result['Stationskennung'] == "MARI") & (df_simulation_result['Type'] == "Abfluss Messung")]).TimeStamp.unique()))

        return df_simulation_result
