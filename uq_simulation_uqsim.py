"""
Usage of the UQEF with a (mainly) Larsim model.
@author: Florian Kuenzner and Ivana Jovanovic
"""

import os
import subprocess
import sys
import pickle
import dill
from distutils.util import strtobool

import uqef

from larsim import LarsimModel
from larsim import LarsimStatistics

from linearDampedOscillator import LinearDampedOscillatorModel
from linearDampedOscillator import LinearDampedOscillatorStatistics

from ishigami import IshigamiModel
from ishigami import IshigamiStatistics

from productFunction import ProductFunctionModel
from productFunction import ProductFunctionStatistics

import LarsimUtilityFunctions.larsimPaths as paths

# additionally added for the debugging of the nodes
import chaospy as cp
import os.path as osp
import pandas as pd
import pathlib
from LarsimUtilityFunctions import larsimConfigurationSettings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

sys.path.insert(0, os.getcwd())

# instantiate UQsim
uqsim = uqef.UQsim()

#####################################
#####################################
# change args locally for testing and debugging
local_debugging = False
if local_debugging:
    save_solver_results = False

    uqsim.args.model = "larsim"

    uqsim.args.uncertain = "all"
    uqsim.args.chunksize = 1

    uqsim.args.uq_method = "sc"  # "saltelli" | "mc"
    uqsim.args.mc_numevaluations = 1000
    uqsim.args.sampling_rule = "latin_hypercube" # | "sobol" | "latin_hypercube" | "halton"  | "hammersley"
    uqsim.args.sc_q_order = 10 #7 #10 3
    uqsim.args.sc_p_order = 6 #6 #8 6
    uqsim.args.sc_poly_rule = "three_terms_recurrence" # "gram_schmidt" | "three_terms_recurrence" | "cholesky"
    uqsim.args.sc_poly_normed = True
    uqsim.args.sc_sparse_quadrature = False  # True
    uqsim.args.regression = False

    uqsim.args.outputResultDir = os.path.abspath(os.path.join("/gpfs/scratch/pr63so/ga45met2", "Larsim_runs", 'larsim_run_24_05_parallel'))
    uqsim.args.inputModelDir = os.path.abspath(os.path.join('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2','Larsim-data'))
    uqsim.args.sourceDir = os.path.abspath(os.path.join('/dss/dsshome1/lxc0C/ga45met2', 'Repositories', 'Larsim-UQ'))
    uqsim.args.outputModelDir = uqsim.args.outputResultDir
    uqsim.args.config_file = '/dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/configurations_Larsim/configurations_larsim_high_flow.json'

    uqsim.args.transformToStandardDist = True
    uqsim.args.mpi = True
    uqsim.args.mpi_method = "MpiPoolSolver" #"LinearSolver"
    uqsim.args.uqsim_store_to_file = False

    uqsim.args.disable_statistics = False
    uqsim.args.parallel_statistics = True  # False
    uqsim.args.compute_Sobol_t = True
    uqsim.args.compute_Sobol_m = True

    uqsim.args.num_cores = 1

    uqsim.setup_configuration_object()


#####################################
# additional path settings:
#####################################

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.args.outputResultDir): subprocess.run(["mkdir", "-p", uqsim.args.outputResultDir])
    print("outputResultDir: {}".format(uqsim.args.outputResultDir))

# Set the working folder where all the model runs related output and files will be written
try:
    uqsim.args.workingDir = os.path.abspath(os.path.join(uqsim.args.outputResultDir,
                                                         uqsim.configuration_object["Directories"]["workingDir"]))
except KeyError:
    uqsim.args.workingDir = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "model_runs"))

try:
    uqsim.configuration_object["Directories"]["workingDir"] = uqsim.args.workingDir
except KeyError:
    uqsim.configuration_object["Directories"] = {}
    uqsim.configuration_object["Directories"]["workingDir"] = uqsim.args.workingDir

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.configuration_object["Directories"]["workingDir"]):
        subprocess.run(["mkdir", uqsim.configuration_object["Directories"]["workingDir"]])

#####################################
# one time initial model setup
#####################################
# put here if there is something specifically related to the model that should be done only once
if uqsim.is_master() and not uqsim.is_restored():
    def initialModelSetUp():
        models = {
            "larsim"         : (lambda: LarsimModel.LarsimModelSetUp(configurationObject=uqsim.configuration_object,
                                                                     inputModelDir=uqsim.args.inputModelDir,
                                                                     sourceDir=uqsim.args.sourceDir,
                                                                     workingDir=uqsim.args.workingDir,
                                                                     ))
           ,"oscillator"     : (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModelSetUp(uqsim.configuration_object))
           ,"ishigami"       : (lambda: IshigamiModel.IshigamiModelSetUp(uqsim.configuration_object))
           ,"productFunction": (lambda: ProductFunctionModel.ProductFunctionModelSetUp(uqsim.configuration_object))
        }
        models[uqsim.args.model]()
    initialModelSetUp()

#####################################
#####################################
# register model
uqsim.models.update({"larsim"         : (lambda: LarsimModel.LarsimModel(configurationObject=uqsim.configuration_object,
                                                                         inputModelDir=uqsim.args.inputModelDir,
                                                                         sourceDir=uqsim.args.sourceDir,
                                                                         workingDir=uqsim.args.workingDir,
                                                                         disable_statistics=uqsim.args.disable_statistics,
                                                                         raise_exception_on_model_break=False,
                                                                         uq_method=uqsim.args.uq_method,
                                                                         max_retries=10
                                                                         ))})
uqsim.models.update({"oscillator"     : (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModel(uqsim.configuration_object))})
uqsim.models.update({"ishigami"       : (lambda: IshigamiModel.IshigamiModel(uqsim.configuration_object))})
uqsim.models.update({"productFunction": (lambda: ProductFunctionModel.ProductFunctionModel(uqsim.configuration_object))})

# register statistics
uqsim.statistics.update({"larsim"         : (lambda: LarsimStatistics.LarsimStatistics(uqsim.configuration_object,
                                                                                       parallel_statistics=uqsim.args.parallel_statistics,
                                                                                       mpi_chunksize=uqsim.args.mpi_chunksize,
                                                                                       unordered=False,
                                                                                       uq_method=uqsim.args.uq_method,
                                                                                       compute_Sobol_t=uqsim.args.compute_Sobol_t,
                                                                                       compute_Sobol_m=uqsim.args.compute_Sobol_m,
                                                                                       store_qoi_data_in_stat_dict=False
                                                                                       ))})
uqsim.statistics.update({"oscillator"     : (lambda: LinearDampedOscillatorStatistics.LinearDampedOscillatorStatistics())})
uqsim.statistics.update({"ishigami"       : (lambda: IshigamiStatistics.IshigamiStatistics(uqsim.configuration_object))})
uqsim.statistics.update({"productFunction": (lambda: ProductFunctionStatistics.ProductFunctionStatistics(uqsim.configuration_object))})

#####################################
#####################################
# setup
uqsim.setup()

# save simulation nodes
simulationNodes_save_file = "nodes"
uqsim.save_simulationNodes(fileName=simulationNodes_save_file)

# save the dictionary with the arguments - once before the simulation
if uqsim.is_master():
    argsFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "uqsim_args.pkl"))
    with open(argsFileName, 'wb') as handle:
        pickle.dump(uqsim.args, handle, protocol=pickle.HIGHEST_PROTOCOL)

#####################################
#####################################
# start the simulation
uqsim.simulate()

#####################################
#####################################
# check-up
if uqsim.is_master():
    if uqsim.args.disable_statistics:
        processed_sample_results = LarsimStatistics.LarsimSamples(uqsim.solver.results,
                                                                  configurationObject=uqsim.configuration_object)
        processed_sample_results.save_samples_to_file(uqsim.args.outputResultDir)
        processed_sample_results.save_index_parameter_values(uqsim.args.outputResultDir)
        processed_sample_results.save_index_parameter_gof_values(uqsim.args.outputResultDir)
        if strtobool(uqsim.configuration_object["Output"]["compute_gradients"]) :
            processed_sample_results.save_dict_of_approx_matrix_c(uqsim.args.outputResultDir)
            processed_sample_results.save_dict_of_matrix_c_eigen_decomposition(uqsim.args.outputResultDir)


#####################################
#####################################
# save uqsim.configuration_object - problem: would this work? nodes do not work...
if uqsim.is_master():
    fileName = pathlib.Path(uqsim.args.outputResultDir) / "configuration_object"
    with open(fileName, 'wb') as f:
        dill.dump(uqsim.configuration_object, f)

#####################################
#####################################
# statistics:
uqsim.calc_statistics()
uqsim.save_statistics()
uqsim.plot_statistics(display=False, plot_measured_timeseries=True, plot_unalteres_timeseries=True)

# uqsim.args.uqsim_file = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "uqsim.saved"))
# #uqsim.store_to_file()

# tear down
uqsim.tear_down()
