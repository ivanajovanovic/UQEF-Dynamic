"""
Usage of the UQEF with a Larsim model.
@author: Florian Kuenzner and Ivana Jovanovic
"""

# plotting
import matplotlib
#matplotlib.use('Agg')

# numerical stuff
import uqef

import LarsimModel
import LarsimStatistics

import LinearDampedOscillatorModel
import LinearDampedOscillatorStatistics

import IshigamiModel
import IshigamiStatistics

import ProductFunctionModel
import ProductFunctionStatistics

import os
import subprocess
import datetime
import socket

#added by IVANA
import glob
import pandas as pd
import pickle
import numpy as np
import time
from tabulate import tabulate

#import paths
import larsimPaths as paths

# instantiate UQsim
uqsim = uqef.UQsim()

#####################################
#####################################
# change args locally for testing and debugging
local_debugging = True
if local_debugging:
    uqsim.args.model = "larsim"
    uqsim.args.uq_method = "saltelli"
    uqsim.args.uncertain = "all"
    uqsim.args.chunksize = 1
    uqsim.args.mc_numevaluations = 2 #50
    #uqsim.args.outputResultDir = os.path.abspath(os.path.join(paths.scratch_dir, 'Larsim_runs/trial_run'))
    uqsim.args.outputResultDir = os.path.abspath(os.path.join(paths.scratch_dir, 'trial_run_lai_updated'))
    uqsim.args.outputModelDir = uqsim.args.outputResultDir
    uqsim.args.inputModelDir = paths.larsim_data_path
    uqsim.args.sourceDir = paths.sourceDir
    #uqsim.args.config_file = "/dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/configurations_Larsim/configuration_larsim_uqsim_cm2_v4.json" #"configuration_larsim_uqsim.json"
    uqsim.args.config_file = "/home/ga45met/Repositories/Larsim/Larsim-UQ/configurations_Larsim/configuration_larsim_updated_local_test_run.json"
    uqsim.args.disable_statistics = False
    uqsim.args.transformToStandardDist = True
    uqsim.args.mpi = True
    uqsim.args.mpi_method = "MpiPoolSolver"
    uqsim.args.sampling_rule = "S"# | "sobol" | "latin_hypercube" | "halton"  | "hammersley"
    uqsim.args.uqsim_store_to_file=False

    uqsim.setup_configuration_object()

#####################################
### additional path settings:
#####################################

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.args.outputResultDir): subprocess.run(["mkdir", "-p", uqsim.args.outputResultDir])
    print("outputResultDir: {}".format(uqsim.args.outputResultDir))

#Set the working folder where all the model runs related output and files will be written
try:
    uqsim.configuration_object["Directories"]["working_dir"] = os.path.abspath(os.path.join(uqsim.args.outputResultDir,uqsim.configuration_object["Directories"]["working_dir"]))
except KeyError:
    try:
        uqsim.configuration_object["Directories"]["working_dir"] = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "model_runs"))
    except KeyError:
        uqsim.configuration_object["Directories"] = {}
        uqsim.configuration_object["Directories"]["working_dir"] = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "model_runs"))

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.configuration_object["Directories"]["working_dir"]):
        subprocess.run(["mkdir", uqsim.configuration_object["Directories"]["working_dir"]])

#####################################
#####################################

# register model
uqsim.models.update({"larsim"         : (lambda: LarsimModel.LarsimModel(uqsim.configuration_object, inputModelDir=uqsim.args.inputModelDir, sourceDir=uqsim.args.sourceDir, disable_statistics=uqsim.args.disable_statistics))})
uqsim.models.update({"oscillator"     : (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModel(uqsim.configuration_object))})
uqsim.models.update({"ishigami"       : (lambda: IshigamiModel.IshigamiModel(uqsim.configuration_object))})
uqsim.models.update({"productFunction": (lambda: ProductFunctionModel.ProductFunctionModel(uqsim.configuration_object))})

# register statistics
uqsim.statistics.update({"larsim"         : (lambda: LarsimStatistics.LarsimStatistics(uqsim.configuration_object))})
uqsim.statistics.update({"oscillator"     : (lambda: LinearDampedOscillatorStatistics.LinearDampedOscillatorStatistics())})
uqsim.statistics.update({"ishigami"       : (lambda: IshigamiStatistics.IshigamiStatistics(uqsim.configuration_object))})
uqsim.statistics.update({"productFunction": (lambda: ProductFunctionStatistics.ProductFunctionStatistics(uqsim.configuration_object))})

# setup
uqsim.setup()

#####################################
### one time initial model setup
#####################################
# put here if there is something specifically related to the model that should be done only once
if uqsim.is_master() and not uqsim.is_restored():
    def initialModelSetUp():
        models = {
            "larsim"         : (lambda: LarsimModel.LarsimModelSetUp(uqsim.configuration_object, inputModelDir=uqsim.args.inputModelDir, sourceDir=uqsim.args.sourceDir))
           ,"oscillator"     : (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModelSetUp(uqsim.configuration_object))
           ,"ishigami"       : (lambda: IshigamiModel.IshigamiModelSetUp(uqsim.configuration_object))
           ,"productFunction": (lambda: ProductFunctionModel.ProductFunctionModelSetUp(uqsim.configuration_object))
        }
        models[uqsim.args.model]()
    initialModelSetUp()
    #experiment by Ivana - remove
    print(uqsim.configuration_object["tuples_parameters_info"])

simulationNodes_save_file = "nodes"
uqsim.save_simulationNodes(fileName=simulationNodes_save_file)

#####################################
#####################################

# start the simulation
uqsim.simulate()

# statistics:
uqsim.calc_statistics()
uqsim.plot_statistics(display=False)
uqsim.save_statistics()

#save the dictionary with the arguments
argsFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "uqsim_args.pkl"))
with open(argsFileName, 'wb') as handle:
    pickle.dump(uqsim.args, handle, protocol=pickle.HIGHEST_PROTOCOL)

# tear down
#just for trying, to check what is saved
uqsim.args.uqsim_file=os.path.abspath(os.path.join(uqsim.args.outputResultDir, "uqsim.saved"))
#uqsim.store_to_file()
uqsim.tear_down()
