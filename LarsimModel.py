import datetime
from distutils.util import strtobool
from decimal import Decimal
import os
import glob
import os.path as osp
import pandas as pd
import numpy as np
import subprocess
import time
import linecache

from uqef.model import Model

import paths
import LARSIM_configs as config


class LarsimModelSetUp():
    def __init__(self, configurationObject):
        self.configurationObject = configurationObject

        self.current_dir = paths.current_dir  # base dircetory of the code
        self.larsim_exe_dir = paths.larsim_exe_dir
        self.larsim_exe = os.path.abspath(os.path.join(self.larsim_exe_dir, 'larsim-linux-intel-1000.exe'))

        try:
            self.working_dir = configurationObject["Directories"]["working_dir"]
        except KeyError:
            self.working_dir = paths.working_dir  # directoy for all the larsim runs

        #self.master_dir = paths.master_dir  # directoy containing all the base files for Larsim execution
        # for safty reasons make a copy of the master_dir in the working_dir and continue working with that one
        self.master_dir = os.path.abspath(os.path.join(self.working_dir, 'master_configuration'))
        if not os.path.isdir(self.master_dir): subprocess.run(["mkdir", self.master_dir])
        master_dir_for_copying = paths.master_dir + "/."
        subprocess.run(['cp', '-a', master_dir_for_copying, self.master_dir])  # TODO Check if copy succeed

        timeframe = config.datetime_parse(self.configurationObject)  # tuple with EREIGNISBEGINN EREIGNISENDE
        #timestep = self.configurationObject["Timeframe"]["timestep"]  # how long one consecutive run should take - used later on in each Larsim run
        #if timestep == 0:
        #    pass
        #else:
        #    timeframe[1] = timeframe[0] + datetime.timedelta(days=timestep)

        #####################################
        ### copy configuration files & do all the configurations needed for proper execution
        #####################################

        # Based on (big) time settings change tape10_master file - needed for unaltered run - this will be repeted once again by each process in LarsimModel.run()
        tape10_adjusted_path = self.master_dir + '/tape10' #os.path.abspath(os.path.join(self.master_dir, 'tape10'))
        master_tape10_file = os.path.abspath(os.path.join(self.master_dir, 'tape10_master'))
        #config.tape10_configurations(timeframe=timeframe, master_tape10_file=paths.master_tape10_file, new_path=tape10_adjusted_path)
        config.tape10_configurations(timeframe=timeframe, master_tape10_file=master_tape10_file, new_path=tape10_adjusted_path)

        # Filter out whm files
        config.copy_whm_files(timeframe=timeframe, all_whms_path=paths.all_whms_path, new_path=self.master_dir)

        # Parse big lila files and create small ones
        lila_configured_paths = [os.path.abspath(os.path.join(self.master_dir, i)) for i in paths.lila_files]
        config.master_lila_parser_on_time_crete_new(timeframe=timeframe, master_lila_paths=paths.master_lila_paths,
                                                    new_lila_paths=lila_configured_paths)

        for one_lila_file in lila_configured_paths:
            if not osp.exists(one_lila_file):
                raise IOError('LARSIM Error: File does not exist: %s. %s' % (one_lila_file, IOError.strerror))

        print("LARSIM INFO: Model has been prepared - all the files have been copied to master folder! ")

        #####################################
        ### extract measured (ground truth) discharge values
        #####################################
        # station_wq.lila file containing ground truth (measured) discharges to lila file
        local_wq_file = lila_configured_paths[0] #os.path.abspath(os.path.join(self.master_dir, paths.lila_files[0]))
        # print("File containing measured discharges - {}".format(local_wq_file))
        self.df_measured = config.lila_parser_toPandas(local_wq_file, index_run=0)
        self.df_measured['Value'] = self.df_measured['Value'].astype(float)
        #print("Data Frame with Measured Discharges dtypes : {}".format(self.df_measured.dtypes))
        self.df_measured.to_csv(path_or_buf=os.path.abspath(os.path.join(self.working_dir, "df_measured.csv")), index=True)

        #####################################
        ### run unaltered simulation
        #####################################
        dir_unaltered_run = os.path.abspath(os.path.join(self.working_dir,"WHM Regen 00"))
        if not os.path.isdir(dir_unaltered_run):
            subprocess.run(["mkdir", dir_unaltered_run])
        master_dir_for_copying = self.master_dir + "/."
        subprocess.run(['cp', '-a', master_dir_for_copying, dir_unaltered_run])
        os.chdir(dir_unaltered_run)
        #config.delete_larsim_output_files(curr_directory=dir_unaltered_run) #TODO This will provide error - make it so that files are deleted if they exist
        local_log_file = os.path.abspath(os.path.join(dir_unaltered_run, "run.log"))
        subprocess.run([self.larsim_exe], stdout=open(local_log_file, 'w'))
        os.chdir(self.current_dir)
        result_file_path = os.path.abspath(os.path.join(dir_unaltered_run, 'ergebnis.lila'))
        self.df_unaltered_ergebnis = config.result_parser_toPandas(result_file_path, index_run=0)
        self.df_unaltered_ergebnis['Value'] = self.df_unaltered_ergebnis['Value'].astype(float)
        #print("Data Frame with Unaltered Simulation Discharges dtypes : {}".format(self.df_unaltered_ergebnis.dtypes))
        # delte ergebnis.lila
        #subprocess.run(["rm", result_file_path])
        #subprocess.run(["rm", os.path.abspath(os.path.join(dir_unaltered_run, 'larsim.ok'))])
        self.df_unaltered_ergebnis.to_csv(path_or_buf=os.path.abspath(os.path.join(self.working_dir, "df_unaltered_ergebnis.csv")), index=True)


        #####################################
        ### compare ground truth measurements and unaltered run for this simulation (compute RMSE | BIAS | NSE | logNSE)
        #####################################
        station_of_Interest = self.configurationObject["Output"]["station"]
        allStations = self.df_measured["Stationskennung"].unique()
        type_of_output_of_Interest = self.configurationObject["Output"]["type_of_output"]

        goodnessofFit_tuple = config.calculateGoodnessofFit(measuredDF=self.df_measured, predictedDF=self.df_unaltered_ergebnis, station=station_of_Interest, type_of_output_of_Interest=type_of_output_of_Interest, dailyStatisict=False)
        (rmse, bias, nse, logNse) = goodnessofFit_tuple[station_of_Interest]
        goodnessofFit_DailyBasis_tuple = config.calculateGoodnessofFit(measuredDF=self.df_measured, predictedDF=self.df_unaltered_ergebnis, station=station_of_Interest, type_of_output_of_Interest=type_of_output_of_Interest, dailyStatisict=True)
        (rmse_DailyBasis, bias_DailyBasis, nse_DailyBasis, logNSE_DailyBasis) = goodnessofFit_DailyBasis_tuple[station_of_Interest]

        # write in a file GOF values of the unaltered model prediction
        column_labels = ("RMSE", "BIAS", "NSE", "LogNSE")
        gof_file_path = os.path.abspath(os.path.join(self.working_dir, "GOF_Measured_vs_Unaltered.txt"))
        with open(gof_file_path, 'w') as f:
            line = ' '.join(str(x) for x in column_labels)
            f.write(line + "\n")
            #f.write('{} {} {} {} \n'.format(column_labels))
            f.write('{:.4f} {:.4f} {:.4f} {:.4f} \n'.format(rmse, bias, nse, logNse))
            f.write('{:.4f} {:.4f} {:.4f} {:.4f} \n'.format(rmse_DailyBasis, bias_DailyBasis, nse_DailyBasis, logNSE_DailyBasis))
            #f.write('%f %f %f %f \n' % (rmse_DailyBasis, bias_DailyBasis, nse_DailyBasis, logNSE_DailyBasis))
        f.close()

        print("LARSIM INFO: Model Initial setup is done! ")


class LarsimModel(Model):

    def __init__(self, configurationObject):
        Model.__init__(self)

        self.configurationObject = configurationObject
        self.current_dir = paths.current_dir #base dircetory of the code
        self.larsim_exe_dir = paths.larsim_exe_dir
        self.larsim_exe = os.path.abspath(os.path.join(self.larsim_exe_dir, 'larsim-linux-intel-1000.exe'))

        try:
            self.working_dir = self.configurationObject["Directories"]["working_dir"]
        except KeyError:
            self.working_dir = paths.working_dir  # directoy for all the larsim runs

        #self.master_dir = paths.master_dir #directoy containing all the base files for Larsim execution
        self.master_dir = os.path.abspath(os.path.join(self.working_dir, 'master_configuration'))

        self.timeframe = config.datetime_parse(self.configurationObject)
        self.timestep = self.configurationObject["Timeframe"]["timestep"]  # how long one consecutive run should take - used later on in each Larsim run

        # generate timesteps for plotting based on tape10 settings which are set in LarsimModelSetUp
        tape10_adjusted_path = self.master_dir + '/tape10'
        #self.t = config.tape10_timesteps(tape10_adjusted_path)
        self.t = config.timeArray_of_tape10_timesteps(self.timeframe)

        self.cut_runs = strtobool(self.configurationObject["Timeframe"]["cut_runs"])

        self.variable_names = []
        for i in self.configurationObject["Variables"]:
            self.variable_names.append(i["name"])


    def prepare(self):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def run(self, i_s, parameters): #i_s - index chunk; parameters - parameters chunk

        print("LARSIM MODEL INFO {}: paramater: {}".format(i_s, parameters))

        results = []
        for ip in range(0, len(i_s)): # for each peace of work
            i = i_s[ip]# i is unique index run
            parameter = parameters[ip]
            start = time.time()

            # create local directory for this particular run
            working_folder_name = "WHM Regen" + str(i)
            curr_working_dir = os.path.abspath(os.path.join(self.working_dir,working_folder_name))

            if not os.path.isdir(curr_working_dir):
                subprocess.run(["mkdir", curr_working_dir])
            # try:
            #    os.mkdir(curr_working_dir)
            # except FileExistsError:
            #    pass

            # copy all the necessary files to the newly created directoy
            master_dir_for_copying = self.master_dir + "/."
            subprocess.run(['cp', '-a', master_dir_for_copying, curr_working_dir])  # TODO IVANA Check if copy succeed
            print("LARSIM MODEL INFO: Successfully copied all the files")

            # change values inside tape35
            if parameter is not None:
                config.tape35_configurations(parameters=parameter, curr_working_dir=curr_working_dir,
                                             configurationObject=self.configurationObject)
                print("LARSIM MODEL INFO: Process {} successfully changed its tape35".format(i))

            # change working directory
            os.chdir(curr_working_dir)

            # Run Larsim
            if self.cut_runs:
                result = self._multiple_short_larsim_runs(timeframe=self.timeframe, timestep = self.timestep, curr_working_dir=curr_working_dir,
                                                parameters=parameter, index_run=i)
            else:
                result = self._single_larsim_run(timeframe=self.timeframe, curr_working_dir=curr_working_dir,
                                            parameters=parameter, index_run=i)

            #assert len(result['TimeStamp'].unique()) == len(self.t), "Assesrtion Failed: Something went wrong with time resolution of the result"

            end = time.time()
            runtime = end - start

            results.append((result, runtime))

            #Debugging TODO Delete afterwards
            print("LARSIM MODEL INFO: Process {} returned / appended it's results".format(i))

            #assert isinstance(self.variable_names, list), "Assertion Failed - variable names not a list"
            #assert len(self.variable_names) == len(parameter), "Assertion Failed parametr not of the same length as variable names"

            result.to_csv(
                path_or_buf=os.path.abspath(os.path.join(curr_working_dir, "ergebnis_df_" + str(i) + ".csv")),
                index=True)


            #####################################
            ### compare model predictions of this simulation with measured (ground truth) data (compute RMSE | BIAS | NSE | logNSE)
            ### this can be moved to Statistics - positioned here due to parallelisation
            #####################################
            gt_dataFrame = pd.read_csv(os.path.abspath(os.path.join(self.working_dir, "df_measured.csv")))
            gt_dataFrame['TimeStamp'] = gt_dataFrame['TimeStamp'].astype('datetime64[ns]')
            station_of_Interest = self.configurationObject["Output"]["station"]
            type_of_output_of_Interest = self.configurationObject["Output"]["type_of_output"]
            #allStations = result["Stationskennung"].unique()

            goodnessofFit_tuple = config.calculateGoodnessofFit(measuredDF=gt_dataFrame, predictedDF=result, station=station_of_Interest, type_of_output_of_Interest=type_of_output_of_Interest, dailyStatisict=False)
            #(rmse, bias, nse, logNse) = goodnessofFit_tuple[station_of_Interest]
            goodnessofFit_DailyBasis_tuple = config.calculateGoodnessofFit(measuredDF=gt_dataFrame, predictedDF=result, station=station_of_Interest, type_of_output_of_Interest=type_of_output_of_Interest, dailyStatisict=True)
            #(rmse_DailyBasis, bias_DailyBasis, nse_DailyBasis, logNSE_DailyBasis) = goodnessofFit_DailyBasis_tuple[station_of_Interest]

            header_array = ["Index_run",]
            for variable_name in self.variable_names:
                header_array.append(variable_name)
            for gof_name in ["RMSE", "BIAS", "NSE", "LogNSE"]:
                header_array.append(gof_name)
            index_parameter_gof_array = [int(i),]
            for single_param in parameter:
                index_parameter_gof_array.append(round(Decimal(single_param), 4))
            for single_gof in goodnessofFit_tuple[station_of_Interest]:
                index_parameter_gof_array.append(round(Decimal(single_gof), 4))
            index_parameter_gof_DF = pd.DataFrame([index_parameter_gof_array], columns=header_array)
            index_parameter_gof_DF.to_csv(
                path_or_buf= os.path.abspath(os.path.join(curr_working_dir, "goodness_of_fit_" + str(i) +  ".csv")),
                index=True)

            # at the end you might delete everything except ergebnis files and saved dataFrame
            # If you want you can delete some of the local run data / or the whole folder
            #subprocess.run(["rm", result_file_path])
            #subprocess.run(["rm", os.path.abspath(os.path.join(curr_working_dir, 'larsim.ok'))])

            #all_in_curr_working_dir = curr_working_dir + "/*"
            #subprocess.run(["rm", all_in_curr_working_dir])

            #Debugging TODO Delete afterwards
            #print("LARSIM INFO DEBUGGING: process {} - Number of Unique TimeStamps in result (Hourly): {}".format(i, len(result.TimeStamp.unique())))
            #result_temp = result.loc[(result['Stationskennung'] == "MARI") & (result['Type'] == "Abfluss Messung")]
            #print("LARSIM INFO DEBUGGING: process {} - Number of Unique TimeStamps in result MARI and Messung (Hourly): {}".format(i, len(result_temp.TimeStamp.unique())))

            #Delete everything except .log and .csv files
            #list_of_files_to_be_deleted = [f for f in glob.glob(curr_working_dir)]
            for single_file in glob.glob(curr_working_dir + "/*"):
                if single_file.endswith(".csv") or single_file.endswith(".log"):
                    pass
                else:
                    #os.remove(single_file)
                    subprocess.run(["rm", "-r", single_file])


            # change back to starting directory of all the processes
            os.chdir(self.current_dir)

            print("LARSIM MODEL INFO: I am done - solver number {}".format(i))


        return results

    def timesteps(self):

        return self.t

    def _single_larsim_run(self, timeframe, curr_working_dir, parameters=None, index_run=0, sub_index_run=0):

        # start clean
        config.delete_larsim_output_files(curr_directory=curr_working_dir) #TODO This will provide warnings - make it so that files are deleted if they exist

        # change tape 10 accordingly
        local_master_tape10_file = os.path.abspath(os.path.join(curr_working_dir, 'tape10_master'))
        local_adjusted_path = os.path.abspath(os.path.join(curr_working_dir, 'tape10'))
        config.tape10_configurations(timeframe=timeframe, master_tape10_file=local_master_tape10_file,
                                     new_path=local_adjusted_path)

        # log file for larsim
        local_log_file = os.path.abspath(
            os.path.join(curr_working_dir, "run" + str(index_run) + "_" + str(sub_index_run) + ".log"))
        # print("LARSIM MODEL INFO: This is where I'm gonna write my log - {}".format(local_log_file))

        # run Larsim as external process
        subprocess.run([self.larsim_exe], stdout=open(local_log_file, 'w'))
        print("LARSIM MODEL INFO: I am done with LARSIM Execution {}".format(index_run))

        # check if larsim.ok exist - Larsim execution was successful
        self._check_larsim_ok_file(curr_working_dir, index_run)

        result_file_path = os.path.abspath(os.path.join(curr_working_dir, 'ergebnis.lila'))

        if os.path.isfile(result_file_path):
            # if you want to transfer path to the resulted file
            # results.append((result_file_path, runtime))
            # if you want to the transfer already read and processed result
            df_single_ergebnis = config.result_parser_toPandas(result_file_path, index_run)
            df_single_ergebnis['Value'] = df_single_ergebnis['Value'].astype(float)
            return df_single_ergebnis
        else:
            return None #TODO Handle this more elegantly


    #TODO Remove this - not used
    def divtd(td1, td2):
        divtdi = datetime.timedelta.__div__
        if isinstance(td2, (int, long)):
            return divtdi(td1, td2)
        us1 = td1.microseconds + 1000000 * (td1.seconds + 86400 * td1.days)
        us2 = td2.microseconds + 1000000 * (td2.seconds + 86400 * td2.days)
        return us1 / us2  # this does integer division, use float(us1) / us2 for fp division


    def _multiple_short_larsim_runs(self, timeframe, timestep, curr_working_dir, parameters=None, index_run=0):
        # if you want to cut execution into shorter runs...
        local_timestep = timestep

        #number_of_runs = datetime.timedelta(days=(timeframe[1] - timeframe[0]).days).days // datetime.timedelta(days=local_timestep).days
        number_of_runs = (timeframe[1] - timeframe[0]).days // datetime.timedelta(days=local_timestep).days
        number_of_runs_mode = (timeframe[1] - timeframe[0]).days % datetime.timedelta(days=local_timestep).days

        local_end_date = timeframe[0]

        result_file_path = os.path.abspath(os.path.join(curr_working_dir, 'ergebnis.lila'))
        larsim_ok_file_path = os.path.abspath(os.path.join(curr_working_dir, 'larsim.ok'))
        tape11_file_path = os.path.abspath(os.path.join(curr_working_dir, 'tape11'))
        karte_path = os.path.abspath(os.path.join(curr_working_dir, 'karten'))  # curr_working_dir + 'karten/*'
        tape10_path = os.path.abspath(os.path.join(curr_working_dir, 'tape10'))

        print("LARSIM MODEL INFO: process {} gonna run {} shorter Larsim runs (and number_of_runs_mode {})".format(index_run, number_of_runs, number_of_runs_mode))

        local_resultDF_list = []
        for i in range(number_of_runs+1):

            # remove previous tape10
            subprocess.run(["rm", tape10_path])


            # calulcate times - make sure that outputs are continuous in time
            if i == 0:
                local_start_date = local_end_date
                local_start_date_p_53 = local_start_date - datetime.timedelta(hours=53)
            else:
                local_start_date_p_53 = local_end_date
                local_start_date = local_start_date_p_53 - datetime.timedelta(hours=53)

            if local_start_date > timeframe[1]:
                break

            local_start_date = local_start_date.replace(hour=0, minute=0, second=0)

            local_end_date = local_start_date + datetime.timedelta(days=local_timestep)

            if local_end_date > timeframe[1]:
                local_end_date = timeframe[1]

            print("LARSIM MODEL INFO: Process {}; local_start_date: {}; local_end_date: {}".format(index_run, local_start_date, local_end_date))
            single_run_timeframe = (local_start_date, local_end_date)

            # run larsim for this shorter period and returned already parsed 'small' ergebnis
            local_resultDF = self._single_larsim_run(timeframe=single_run_timeframe, curr_working_dir=curr_working_dir, parameters=parameters, index_run=index_run, sub_index_run=i)

            #TODO Handle this more elegantly
            if local_resultDF is None:
                raise ValueError("LARSIM MODEL INFO: ERROR - Process {}: The following Ergebnis file was not found - {}".format(index_run, result_file_path))

            local_resultDF_drop = local_resultDF.drop(local_resultDF[local_resultDF['TimeStamp'] < local_start_date_p_53].index)

            local_resultDF_list.append(local_resultDF_drop)

            # rename ergebnis.lila
            local_result_file_path = os.path.abspath(os.path.join(curr_working_dir, 'ergebnis' + '_' + str(i) + '.lila'))
            subprocess.run(["mv", result_file_path, local_result_file_path])
            local_larsim_ok_file_path = os.path.abspath(os.path.join(curr_working_dir, 'larsim' + '_' + str(i) + '.ok'))
            subprocess.run(["mv", larsim_ok_file_path, local_larsim_ok_file_path])

        # concatinate it
        df_simulation_result = pd.concat(local_resultDF_list, ignore_index=True, sort=True, axis=0)

        # sorting by time
        df_simulation_result.sort_values("TimeStamp", inplace=True)

        # clean concatanated file - dropping time duplicate values
        df_simulation_result.drop_duplicates(subset=['TimeStamp', 'Stationskennung', 'Type'], keep='first', inplace=True)

        #print("DEBUGGING LARSIM INFO: process {} - After Droping -  MARI and Messung (Hourly):\n".format(i))
        #print(len(df_simulation_result.TimeStamp.unique()))
        #print("\n")
        #print(len((df_simulation_result.loc[(df_simulation_result['Stationskennung'] == "MARI") & (df_simulation_result['Type'] == "Abfluss Messung")]).TimeStamp.unique()))

        return df_simulation_result


    def _check_larsim_ok_file(self, curr_working_dir, index_run):
        # check for existence of larsim.ok and whether the file is readable
        # while not os.path.exists(curr_working_dir+"/larsim.ok"):
        #    time.sleep(5)
        larsim_ok = False
        larsim_ok_file = curr_working_dir + "/larsim.ok"
        while larsim_ok is False:
            lines = linecache.getlines(larsim_ok_file)

            for l in lines:
                if "ok" in l:
                    larsim_ok = True

            linecache.clearcache()
            time.sleep(0.1)
            print("LARSIM INFO: rank {} retries reading larsim_ok".format(index_run))
