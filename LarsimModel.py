import datetime
from distutils.util import strtobool
import os
import os.path as osp
import pandas as pd
import subprocess
import time
import linecache

from uqef.model import Model

import paths
import LARSIM_configs as config


def delete_larsim_output_files(curr_directory):

    result_file_path = os.path.abspath(os.path.join(curr_directory, 'ergebnis.lila'))
    larsim_ok_file_path = os.path.abspath(os.path.join(curr_directory, 'larsim.ok'))
    tape11_file_path = os.path.abspath(os.path.join(curr_directory, 'tape11'))
    karte_path = os.path.abspath(os.path.join(curr_directory, 'karten'))  # curr_working_dir + 'karten/*'
    tape10_path = os.path.abspath(os.path.join(curr_directory, 'tape10'))

    subprocess.run(["rm", result_file_path])
    subprocess.run(["rm", larsim_ok_file_path])
    subprocess.run(["rm", tape11_file_path])

    # subprocess.run(["rm", "-R", karte_path])
    # if not os.path.isdir(karte_path):
    #    subprocess.run(["mkdir", karte_path])


def delete_larsim_lila_whm_files():

    subprocess.run(["rm", "*.whm"])
    subprocess.run(["rm", "*.lila"])



class LarsimModelSetUp():
    def __init__(self, configurationObject):
        self.configurationObject = configurationObject

        self.master_dir = paths.master_dir  # directoy containing all the base files for Larsim execution
        self.current_dir = paths.current_dir  # base dircetory of the code
        self.larsim_exe_dir = paths.larsim_exe_dir
        self.larsim_exe = os.path.abspath(os.path.join(self.larsim_exe_dir, 'larsim-linux-intel-1000.exe'))

        try:
            self.working_dir = configurationObject["Directories"]["working_dir"]
        except KeyError:
            self.working_dir = paths.working_dir  # directoy for all the larsim runs

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
        tape10_adjusted_path = self.master_dir + '/tape10'
        config.tape10_configurations(timeframe=timeframe, master_tape10_file=paths.master_tape10_file, new_path=tape10_adjusted_path)

        # Filter out whm files
        config.copy_whm_files(timeframe=timeframe, all_whms_path=paths.all_whms_path, new_path=self.master_dir)

        # Parse big lila files and create small ones
        config.master_lila_parser_on_time_crete_new(timeframe=timeframe, master_lila_paths=paths.master_lila_paths,
                                                    new_lila_files=paths.lila_files, new_path=self.master_dir)

        for one_lila_file in paths.lila_configured_paths:
            if not osp.exists(one_lila_file):
                raise IOError('LARSIM Error: File does not exist: %s. %s' % (one_lila_file, IOError.strerror))

        print("LARSIM INFO: Model has been prepared - all the files have been copied to master folder! ")

        #####################################
        ### extract ground truth discharge values
        #####################################
        # station_wq.lila file containing ground truth (measured) discharges to lila file
        local_wq_file = os.path.abspath(os.path.join(self.master_dir, paths.lila_files[0]))
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
        master_dir = self.master_dir + "/."
        subprocess.run(['cp', '-a', master_dir, dir_unaltered_run])
        os.chdir(dir_unaltered_run)
        delete_larsim_output_files(curr_directory=dir_unaltered_run)
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

        print("LARSIM INFO: Model Initial setup is done! ")

class LarsimModel(Model):

    def __init__(self, configurationObject):
        Model.__init__(self)

        self.configurationObject = configurationObject

        self.master_dir = paths.master_dir #directoy containing all the base files for Larsim execution
        self.current_dir = paths.current_dir #base dircetory of the code
        self.larsim_exe_dir = paths.larsim_exe_dir
        self.larsim_exe = os.path.abspath(os.path.join(self.larsim_exe_dir, 'larsim-linux-intel-1000.exe'))

        try:
            self.working_dir = self.configurationObject["Directories"]["working_dir"]
        except KeyError:
            self.working_dir = paths.working_dir  # directoy for all the larsim runs

        # generate timesteps for plotting based on tape10 settings which are set in LarsimModelSetUp
        tape10_adjusted_path = self.master_dir + '/tape10'
        self.t = config.tape10_timesteps(tape10_adjusted_path)

        self.cut_runs = strtobool(self.configurationObject["Timeframe"]["cut_runs"])


    def prepare(self):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def run(self, i_s, parameters): #i_s - index chunk; parameters - parameters chunk

        print("LARSIM INFO {}: paramater: {}".format(i_s, parameters))

        results = []
        for ip in range(0, len(i_s)): # for each peace of work
            i = i_s[ip]# i is unique index run
            parameter = parameters[ip]
            start = time.time()

            self.timeframe = config.datetime_parse(self.configurationObject)
            self.timestep = self.configurationObject["Timeframe"]["timestep"]  # how long one consecutive run should take - used later on in each Larsim run

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
            master_dir = self.master_dir + "/."
            subprocess.run(['cp', '-a', master_dir, curr_working_dir])  # TODO IVANA Check if copy succeed
            print("LARSIM INFO: Successfully copied all the files")

            # change values inside tape35
            if parameter is not None:
                config.tape35_configurations(parameters=parameter, curr_working_dir=curr_working_dir,
                                             configurationObject=self.configurationObject)
                print("LARSIM INFO: Process {} successfully changed its tape35".format(i))

            # change working directory
            os.chdir(curr_working_dir)

            # Run Larsim
            if self.cut_runs:
                result = self._multiple_short_larsim_runs(timeframe=self.timeframe, timestep = self.timestep, curr_working_dir=curr_working_dir,
                                                parameters=parameter, index_run=i)
            else:
                result = self._single_larsim_run(timeframe=self.timeframe, curr_working_dir=curr_working_dir,
                                            parameters=parameter, index_run=i)

            end = time.time()
            runtime = end - start

            results.append((result, runtime))

            # at the end you might delete everything except ergebnis files and saved dataFrame
            # If you want you can delete some of the local run data / or the whole folder
            #subprocess.run(["rm", result_file_path])
            #subprocess.run(["rm", os.path.abspath(os.path.join(curr_working_dir, 'larsim.ok'))])

            #all_in_curr_working_dir = curr_working_dir + "/*"
            #subprocess.run(["rm", all_in_curr_working_dir])

            result.to_csv(
                path_or_buf=os.path.abspath(os.path.join(curr_working_dir, "ergebnis_df_" + str(i) + ".csv")),
                index=True)

            # change back to starting directory of all the processes
            os.chdir(self.current_dir)

            print("LARSIM INFO: I am done - solver number {}".format(i))


        return results

    def timesteps(self):

        return self.t

    def _single_larsim_run(self, timeframe, curr_working_dir, parameters=None, index_run=0, sub_index_run=0):

        # start clean
        delete_larsim_output_files(curr_directory=curr_working_dir)

        # change tape 10 accordingly
        local_master_tape10_file = os.path.abspath(os.path.join(curr_working_dir, 'tape10_master'))
        local_adjusted_path = os.path.abspath(os.path.join(curr_working_dir, 'tape10'))
        config.tape10_configurations(timeframe=timeframe, master_tape10_file=local_master_tape10_file,
                                     new_path=local_adjusted_path)

        # log file for larsim
        local_log_file = os.path.abspath(
            os.path.join(curr_working_dir, "run" + str(index_run) + "_" + str(sub_index_run) + ".log"))
        # print("LARSIM INFO: This is where I'm gonna write my log - {}".format(local_log_file))

        # run Larsim as external process
        subprocess.run([self.larsim_exe], stdout=open(local_log_file, 'w'))
        print("LARSIM INFO: I am done with LARSIM Execution {}".format(index_run))

        # check if larsim.ok exist - Larsim execution was successful
        self._check_larsim_ok_file(curr_working_dir, index_run)

        result_file_path = os.path.abspath(os.path.join(curr_working_dir, 'ergebnis.lila'))

        if os.path.isfile(result_file_path):
            # if you want to transfer path to the resulted file
            # results.append((result_file_path, runtime))
            # if you want to the transfer already read and processed result
            df_single_ergebnis = config.result_parser_toPandas(result_file_path, index_run)
            return df_single_ergebnis
        else:
            return None #TODO Handle this more elegantly


    #TODO This is not needed at all
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

        print("LARSIM INFO: process {} gonna run {} shorter Lars runs (and number_of_runs_mode {})".format(index_run, number_of_runs, number_of_runs_mode))

        local_resultDF_list = []
        for i in range(number_of_runs+1):

            # remove previous tape10
            subprocess.run(["rm", tape10_path])


            # calulcate times - make sure that outputs are continuous in time
            if i == 0:
                local_start_date = local_end_date
                local_start_date_p_53 = local_start_date + datetime.timedelta(hours=53)
            else:
                local_start_date_p_53 = local_end_date
                local_start_date = local_start_date_p_53 - datetime.timedelta(hours=53)

            if local_start_date > timeframe[1]:
                break

            local_start_date = local_start_date.replace(hour=0, minute=0, second=0)

            local_end_date = local_start_date + datetime.timedelta(days=local_timestep)

            if local_end_date > timeframe[1]:
                local_end_date = timeframe[1]

            print("Process: {}; local_start_date: {}; local_end_date: {}".format(index_run, local_start_date, local_end_date))
            single_run_timeframe = (local_start_date, local_end_date)



            # run larsim for this shorter period and returned already parsed 'small' ergebnis
            local_resultDF = self._single_larsim_run(timeframe=single_run_timeframe, curr_working_dir=curr_working_dir, parameters=parameters, index_run=index_run, sub_index_run=i)

            # disregarde first 53 from each ergebnis
            #local_start_date_p_53_pd = pd.to_datetime(local_start_date_p_53)
            #local_start_date_pd = pd.to_datetime(local_start_date)
            #local_end_date_pd = pd.to_datetime(local_end_date)
            # local_resultDF['TimeStamp'] = local_resultDF['TimeStamp'].apply(lambda x: pd.Timestamp(x))
            #local_resultDF = local_resultDF.between_time(local_start_date_p_53, local_end_date)

            #TODO Handle this more elegantly
            if local_resultDF is None:
                raise ValueError("Process {}: The following Ergebnis file was not found - {}".format(index_run, result_file_path))

            local_resultDF = local_resultDF.drop(local_resultDF[local_resultDF['TimeStamp'] < local_start_date_p_53].index)

            #
            local_resultDF_list.append(local_resultDF)

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
        df_simulation_result.drop_duplicates(subset="TimeStamp", keep='first', inplace=True)


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



