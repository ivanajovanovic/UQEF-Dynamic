"""
A simple example for the usage of the UQEF with a test model.
@author: Florian Kuenzner
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

import paths

# instantiate UQsim
uqsim = uqef.UQsim()

# change args locally for testing and debugging
local_debugging = True
if local_debugging:
    uqsim.args.model = "larsim"
    uqsim.args.uq_method = "saltelli"
    uqsim.args.mc_numevaluations = 50#2
    uqsim.args.outputResultDir =  "/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/Repositories/larsim_runs" # paths.scratch_dir| "./larsim_runs/"
    uqsim.args.outputModelDir = uqsim.args.outputResultDir
    uqsim.args.inputModelDir = uqsim.args.outputResultDir
    uqsim.args.config_file = "/dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/configuration_larsim_uqsim_cm2.json" #"configuration_larsim_uqsim.json"
    uqsim.args.disable_statistics = False
    uqsim.args.transformToStandardDist = True
    uqsim.args.mpi = True
    uqsim.args.sampling_rule = "S"# | "sobol"  | "latin_hypercube" | "halton"  | "hammersley"
    #
    uqsim.args.uqsim_store_to_file=True


    uqsim.setup_configuration_object()

#####################################
### additional path settings:
#####################################
if socket.gethostname().startswith("cm2"):
    #outputResultDir = uqsim.args.outputResultDir
    outputResultDir = os.path.abspath(os.path.join(uqsim.args.outputResultDir, datetime.datetime.now().strftime("%Y-%m-%d:%H:%M")))
    #outputResultDir = os.path.abspath(os.path.join(paths.working_dir, datetime.datetime.now().strftime("%Y-%m-%d:%H:%M")))
    uqsim.args.outputResultDir = outputResultDir
else:
    outputResultDir = os.path.abspath(os.path.join(uqsim.args.outputResultDir, datetime.datetime.now().strftime("%Y-%m-%d:%H:%M")))
    uqsim.args.outputResultDir = outputResultDir

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(outputResultDir): subprocess.run(["mkdir", outputResultDir])
    print("outputResultDir: {}".format(outputResultDir))

#Set the working folder where all the model runs related output and files will be written
try:
    uqsim.configuration_object["Directories"]["working_dir"] = os.path.abspath(os.path.join(outputResultDir,
                                                                                            uqsim.configuration_object["Directories"]["working_dir"]))
except KeyError:
    try:
        uqsim.configuration_object["Directories"]["working_dir"] = os.path.abspath(os.path.join(outputResultDir, "model_runs"))
    except KeyError:
        uqsim.configuration_object["Directories"] = {}
        uqsim.configuration_object["Directories"]["working_dir"] = os.path.abspath(os.path.join(outputResultDir, "model_runs"))

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.configuration_object["Directories"]["working_dir"]):
        subprocess.run(["mkdir", uqsim.configuration_object["Directories"]["working_dir"]])

#####################################
#####################################

# register model
uqsim.models.update({"larsim"         : (lambda: LarsimModel.LarsimModel(uqsim.configuration_object))})
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
            "larsim"         : (lambda: LarsimModel.LarsimModelSetUp(uqsim.configuration_object))
           ,"oscillator"     : (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModelSetUp(uqsim.configuration_object))
           ,"ishigami"       : (lambda: IshigamiModel.IshigamiModelSetUp(uqsim.configuration_object))
           ,"productFunction": (lambda: ProductFunctionModel.ProductFunctionModelSetUp(uqsim.configuration_object))
        }
        models[uqsim.args.model]()
    initialModelSetUp()

simulationNodes_save_file = "/nodes"
uqsim.save_simulationNodes(fileName=simulationNodes_save_file)

#just for trying, to check what is saved
uqsim.args.uqsim_file=os.path.abspath(os.path.join(uqsim.configuration_object["Directories"]["working_dir"], "uqsim.saved"))
uqsim.store_to_file()

# start the simulation
uqsim.simulate()

# save simulation results
#samples = LarsimStatistics.LarsimSamples(uqsim.solver.results, station=uqsim.configuration_object["Output"]["station"],
#                          type_of_output=uqsim.configuration_object["Output"]["type_of_output"],
#                          pathsDataFormat=uqsim.configuration_object["Output"]["pathsDataFormat"],
#                          dailyOutput=uqsim.configuration_object["Output"]["dailyOutput"])
#samples.save_samples_to_file(uqsim.configuration_object["Directories"]["working_dir"])

# statistics:
uqsim.calc_statistics()
#TODO: probably do not need it...
#uqsim.print_statistics()
#TODO: customize this; to be the same as statistics.plotResults(simulationNodes, display=True)
uqsim.plot_statistics(display=False)
#uqsim.statistic.plotResults(simulationNodes, display=True)
#TODO: think how much space these saved files require...
uqsim.save_statistics()

# tear down
#uqsim.tear_down()
