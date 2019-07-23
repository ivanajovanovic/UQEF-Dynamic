import datetime
import os
import os.path as osp
import subprocess
import time
import linecache

from uqef.model import Model

import paths
import LARSIM_configs as config

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
        timestep = self.configurationObject["Timeframe"]["timestep"]  # how long one consecutive run should take
        if timestep == 0:
            pass
        else:
            timeframe[1] = timeframe[0] + datetime.timedelta(days=timestep)

        #####################################
        ### copy configuration files & do all the configurations needed for proper execution
        #####################################

        # Based on time settings change tape10_master file
        tape10_adjusted_path = self.master_dir + '/tape10'
        config.tape10_configurations(timeframe=timeframe, master_tape10_file=paths.master_tape10_file,
                                     new_path=tape10_adjusted_path)

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
            self.working_dir = configurationObject["Directories"]["working_dir"]
        except KeyError:
            self.working_dir = paths.working_dir  # directoy for all the larsim runs

        # generate timesteps for plotting based on tape10 settings
        tape10_adjusted_path = self.master_dir + '/tape10'
        self.t = config.tape10_timesteps(tape10_adjusted_path)


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

            # create local directory for this particular run
            working_folder_name = "WHM Regen" + str(i)
            curr_working_dir = os.path.abspath(os.path.join(self.working_dir,working_folder_name))

            self.single_larsim_run(master_dir=self.master_dir, curr_working_dir=curr_working_dir, parameters=parameter, index_run=i)

            end = time.time()
            runtime = end - start

            result_file_path = os.path.abspath(os.path.join(curr_working_dir, 'ergebnis.lila'))


            if os.path.isfile(result_file_path):
                # if you want to transfer path to the resulted file
                #results.append((result_file_path, runtime))
                # if you want to the transfer already read and processed result
                df_single_ergebnis = config.result_parser_toPandas(result_file_path, i)
                results.append((df_single_ergebnis, runtime))
            else:
                results.append((None, runtime))


            # If you want you can delete some of the local run data / or the whole folder
            #subprocess.run(["rm", result_file_path])
            #subprocess.run(["rm", os.path.abspath(os.path.join(curr_working_dir, 'larsim.ok'))])
            #subprocess.run(["rm","-R",curr_working_dir], shell=True)


            print("LARSIM INFO: I am done - solver number {}".format(i))


        return results

    def timesteps(self):

        return self.t

    def single_larsim_run(self, master_dir, curr_working_dir, parameters=None, index_run=0):

        if not os.path.isdir(curr_working_dir):
            subprocess.run(["mkdir", curr_working_dir])
        # try:
        #    os.mkdir(curr_working_dir)
        # except FileExistsError:
        #    pass

        # copy all the necessary files to the newly created directoy
        master_dir = master_dir + "/."
        subprocess.run(['cp', '-a', master_dir, curr_working_dir]) #TODO IVANA Check if copy succeed
        # print("LARSIM INFO: Successfully copied all the files")

        # change working directory
        os.chdir(curr_working_dir)

        # change values inside tape35
        if parameters is not None:
            config.tape35_configurations(parameters=parameters, curr_working_dir=curr_working_dir,
                                     configurationObject=self.configurationObject)
            #print("LARSIM INFO: Process {} successfully changed its tape35".format(i))

        local_log_file = os.path.abspath(os.path.join(curr_working_dir, "run" + str(index_run) + ".log"))
        # print("LARSIM INFO: This is where I'm gonna write my log - {}".format(local_log_file))

        # Run Larsim as external process
        subprocess.run([self.larsim_exe], stdout=open(local_log_file, 'w'))
        print("LARSIM INFO: I am done with LARSIM Execution {}".format(index_run))

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

        # change back to starting directory of all the processes
        os.chdir(self.current_dir)



# Things that should be done to continue simulation ntimewise
# start new simulation
# generate nodes for the first time and save them
#for loop
# inside for loop - change tape10
# inside for loop - call solver.init()
# inside for loop - call simulation.prepareSolver()
# solver.solve()
# inside for loop - inside of the model.run mkdir, copy all the files - tape10 just once
# inside for loop - inside of the model.run delete larsim.ok, tape11, ergebnis.lila and karte
# inside for loop - inside of the model.run copy tape10
# inside for loop - inside of the model.run copy new ergebnis to safe place

#or

#inside model.run - constanlty change tape10
#inside model.run - constanlty delete larsim.ok, tape11, ergebnis.lila and karte
#inside model.run - constanlty run larsim.exe
#inside model.run - constanlty concatinate ergebnis and save to some final ergebnis file
