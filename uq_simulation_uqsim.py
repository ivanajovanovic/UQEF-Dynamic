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

import sys
import os
import subprocess
import datetime
import socket

# time measure
import time
import datetime

# for message passing
from mpi4py import MPI
import mpi4py

# instantiate UQsim
uqsim = uqef.UQsim()

# change args locally for testing and debugging
local_debugging = False
if local_debugging:
    uqsim.args.model = "larsim"
    uqsim.args.uq_method = "mc"
    uqsim.args.mc_numevaluations = 5
    uqsim.args.outputResultDir = "./larsim_runs/"
    uqsim.args.config_file = "configuration_larsim_snow1.json"
    uqsim.args.disable_statistics = True

    uqsim.setup_configuration_object()

#####################################
### path settings:
#####################################
# each mpi processor has it's own outputResultDir
if uqsim.is_master():
    print("path settings...")

if uqsim.args.outputResultDir:
    rootDir = uqsim.args.outputResultDir
else:
    rootDir = os.getcwd()

if socket.gethostname().startswith("mpp2"):
    outputResultDir=rootDir
else:
    outputResultDir = os.path.abspath(os.path.join(rootDir, datetime.datetime.now().strftime("%Y-%m-%d:%H:%M")))
#args.outputResultDir = outputResultDir

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(outputResultDir): subprocess.run(["mkdir", outputResultDir])
    print("outputResultDir: {}".format(outputResultDir))

#####################################
### read configuration setting and additional path settings:
#####################################

if uqsim.is_master():
    print(uqsim.configuration_object)

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
### one time initial model setup
#####################################
# put here is there is something specifically related to the model that should be done only once
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

# start the simulation
uqsim.simulate()
if uqsim.is_master():
    print("simulation done: now start stats")
    sys.stdout.flush()
uqsim.store_to_file()

# statistics:
uqsim.calc_statistics()
uqsim.print_statistics()
uqsim.plot_statistics(display=False)
uqsim.save_statistics()

# tear down
uqsim.tear_down()

rank = MPI.COMM_WORLD.Get_rank()
print("rank {}: endtime: {}".format(rank, datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')))
