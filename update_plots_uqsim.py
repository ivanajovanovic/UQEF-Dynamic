"""
A simple example for the usage of the UQEF with a test model and to see the store and restore functionality.

@author: Florian Kuenzner
"""

# plotting
import matplotlib
matplotlib.use('Agg')

#parsing args
import argparse

# numerical stuff
import chaospy as cp
import uqef

#file system
import os
import glob
import sys

#regex
import re

import LarsimModel
import LarsimStatistics

import LinearDampedOscillatorModel
import LinearDampedOscillatorStatistics

import IshigamiModel
import IshigamiStatistics

import ProductFunctionModel
import ProductFunctionStatistics

import multiprocessing

#####################################
### parsing args:
#####################################
parser = argparse.ArgumentParser(description='Uncertainty Quantification update plots.')
parser.add_argument('-d','--dir')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--mpi', action='store_true')
parser.add_argument('--sc_p_order', type=int, default=1)
args = parser.parse_args()

#####################################
### path settings:
#####################################

if args.dir:
    directory = args.dir
    directory = directory.rstrip("/")

    # We have to remove the --dir, because the arg parse uf UQsim doesn't know this.
    idx = sys.argv.index("--dir")
    del sys.argv[idx + 1]
    del sys.argv[idx]

else:
    #directory = "/data/repos/repos_tum/promotion.git/Dissertation/DissertationRuns/LARSIM/cluster_results1"
    directory = "/data/repos/repos_tum/promotion.git/Dissertation/DissertationRuns/LARSIM/cluster_results1/0121_sim"
print("dir: {}".format(directory))

dirs = [
    directory
]

def updatePlot(uqsim_file):
    # instantiate UQsim
    uqsim = uqef.UQsim()

    # parsing args:
    uqsim.parse_args()

    #outputResultDir = "/data/repos/repos_tum/promotion.git/Dissertation/DissertationRuns/VADERE/scenario3_cosm/vadere.v1.0.linux/cluster_result1/uq_sc_mpp2.0022"
    #uqsim_file      = outputResultDir + "/" + "uqsim.saved"

    outputResultDir = os.path.dirname(uqsim_file)

    uqsim.args.outputResultDir = outputResultDir
    uqsim.args.uqsim_file      = uqsim_file

    # change args locally
    uqsim.args.uqsim_restore_from_file = True

    # setup
    uqsim.setup()

    uqsim.args.outputResultDir = outputResultDir
    uqsim.args.uqsim_file = uqsim_file

    uqsim.configuration_object["Directories"]["working_dir"] = outputResultDir + "/model_runs"
    uqsim.configuration_object["Output"]["dailyOutput"] = "True"
    uqsim.configuration_object["Output"]["station"] = "MARI"

    # re register statistics
    uqsim.statistics.update({"larsim"         : (lambda: LarsimStatistics.LarsimStatistics(uqsim.configuration_object))})
    uqsim.statistics.update({"oscillator"     : (lambda: LinearDampedOscillatorStatistics.LinearDampedOscillatorStatistics())})
    uqsim.statistics.update({"ishigami"       : (lambda: IshigamiStatistics.IshigamiStatistics(uqsim.configuration_object))})
    uqsim.statistics.update({"productFunction": (lambda: ProductFunctionStatistics.ProductFunctionStatistics(uqsim.configuration_object))})

    print("uqsim.args.disable_statistics: {}".format(uqsim.args.disable_statistics))

    force_recalc = True
    # turn statistics off here, after locally restored the UQsim instance
    # only calc stats if it isn't already calculated...
    if uqsim.args.disable_statistics is True or force_recalc is True:
        uqsim.args.analyse_runtime = True
        uqsim.args.uqsim_store_to_file = True
        uqsim.args.disable_statistics = False
        uqsim.args.disable_recalc_statistics = False
        uqsim.args.outputResultDir = outputResultDir
        uqsim.args.uqsim_file = uqsim_file
    else:
        uqsim.args.analyse_runtime = True
        uqsim.args.uqsim_store_to_file = False
        uqsim.args.disable_statistics = False
        uqsim.args.disable_recalc_statistics = True
        uqsim.args.outputResultDir = outputResultDir
        uqsim.args.uqsim_file      = uqsim_file

    # start the simulation
    uqsim.simulate()

    if uqsim.statistic:
        uqsim.statistic.working_dir = uqsim.configuration_object["Directories"]["working_dir"]

    # statistics:
    uqsim.calc_statistics()
    uqsim.print_statistics()
    #uqsim.plot_nodes()
    uqsim.plot_statistics()
    uqsim.save_statistics()

    # tear down
    uqsim.tear_down()

for directory in dirs:
    # directory = dirs[0]
    print("dir: {}".format(directory))

    #####################################
    ### update plots:
    #####################################

    type = "uqsim"
    #type = "stat"

    fileNames = []
    if type == "uqsim":
        # find *.stat files
        fileNames.extend(glob.glob(directory + "/" + "uqsim.saved"))
        fileNames.extend(glob.glob(directory + "/**/" + "uqsim.saved"))
        fileNames.sort()

        fileNames = [f for f in fileNames if not "_runtime" in f]

        # fileNames = ["/data/repos/repos_tum/promotion.git/Dissertation/DissertationRuns/VADERE/scenario2_family/results_cluster/uq_sc_mpp2.0001.1/sc.stat"]
        print(fileNames)

        #pool = multiprocessing.Pool(multiprocessing.cpu_count())
        #samples = pool.map(updatePlot, fileNames)

        for f in fileNames:
           updatePlot(f)
