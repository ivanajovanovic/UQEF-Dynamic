import os
import os.path
import subprocess
#from subprocess import call
import time

from uqef.model import Model

import paths
import LARSIM_configs as config

#TODO Ivana 0 think of sleeping times and time needed for copying/deleting the folders....


class LarsimModel(Model):

    def __init__(self):
        Model.__init__(self)

        self.master_dir = paths.master_dir
        self.current_dir = paths.current_dir
        self.working_dir = paths.working_dir
        self.larsim_exe_dir = paths.larsim_exe_dir

        #generate timesteps for plotting based on tape10 settings
        #tape10_file = os.path.abspath(os.path.join(self.master_dir,'tape10'))
        self.t = config.tape10_timesteps(paths.master_tape10_file)

    def prepare(self):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def run(self, i_s, parameters): #i_s - index chunk; parameters - parameters chunk
        results = []
        for ip in range(0, len(i_s)): # for each peace of work
            i = i_s[ip]
            parameter = parameters[ip]
            start = time.time()

            working_folder_name = "WHM Regen" + str(i)
            curr_working_dir = os.path.abspath(os.path.join(self.working_dir,working_folder_name))
            print(curr_working_dir)
            if not os.path.isdir(curr_working_dir):
                subprocess.run(["mkdir", curr_working_dir])

            #try:
            #    os.mkdir(curr_working_dir)
            #except FileExistsError:
            #    pass



            master_dir = paths.master_dir + "/."
            print(master_dir)
            print(curr_working_dir)
            subprocess.run(['cp','-a',master_dir,curr_working_dir])

            print("Done cp -a command")

            #TODO IVANA Check if copy succeed
            print("Successfully copied all the files")
            #Change values
            #time.sleep(5)
            config.tape35_configurations(parameters=parameter, curr_working_dir=curr_working_dir)

            print("Process {} successfully changed its tape35".format(i))


            #time.sleep(5)
            # Change working directory and execute LARSIM

            #subprocess.run(['cd', curr_working_dir], stdin=None, input=None, stdout=None,
            #               stderr=None, capture_output=False, shell=True,
            #               cwd=None, timeout=None, check=True, encoding=None, errors=None, text=None, env=None,
            #               universal_newlines=None)
            #subprocess.run(['cd', curr_working_dir], capture_output=False, shell=True)
            os.chdir(curr_working_dir)



            larsim_exe = os.path.abspath(os.path.join(self.larsim_exe_dir, 'larsim-linux-intel-1000.exe'))
            #larsim_exe = "/import/home/ga45met/Repositories/Larsim-data/Larsim-exe/larsim-linux-intel-1000.exe"
            print("INFO: This is what I'm gonna execute: {}".format(larsim_exe))
            #subprocess.run([larsim_exe,'>', '/dev/null'], stdin=None, input=None, stdout=None,
            #               stderr=None, capture_output=True, shell=False,
            #               cwd=None, timeout=None, check=True, encoding=None, errors=None, text=None, env=None,
            #               universal_newlines=None)

            local_log_file = os.path.abspath(os.path.join(curr_working_dir,"run"+str(i)+".log"))
            print("INFO: This is where I'm gonna write my log - {}".format(local_log_file))
            #subprocess.run([larsim_exe, '>', local_log_file])
            #subprocess.run([larsim_exe, '>', '/dev/null'])
            subprocess.run([larsim_exe])
            print("INFO: I am done with LARSIM Execution {}".format(i))
            #collect results
            #time.sleep(45)
            #value_of_interest = config.Result_parser.result_parser(self.WHM_regen+str(i) + "/ergebnis.lila")
            #time.sleep(5)
            end = time.time()
            runtime = end-start

            os.chdir(self.current_dir)

            #TODO check for existence of larsim.ok
            while not os.path.exists(curr_working_dir+"/larsim.ok"):
                time.sleep(5)

            result_file_path = os.path.abspath(os.path.join(curr_working_dir, 'ergebnis.lila'))

            if os.path.isfile(result_file_path):
                results.append((result_file_path, runtime))
            else:
                results.append((None, runtime))

            #time.sleep(5)
            #subprocess.run(["rm","-R",curr_working_dir], shell=True)
            print("INFO: I am done - solver number {}".format(i))


        return results

    def timesteps(self):

        return self.t
