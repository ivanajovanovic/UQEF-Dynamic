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

# additionally added for the debugging of the nodes
import chaospy as cp
import os.path as osp
import pandas as pd
import pathlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

from larsim import LarsimModel
from larsim import LarsimStatistics

from linearDampedOscillator import LinearDampedOscillatorModel
from linearDampedOscillator import LinearDampedOscillatorStatistics

from ishigami import IshigamiModel
from ishigami import IshigamiStatistics

from productFunction import ProductFunctionModel
from productFunction import ProductFunctionStatistics

from LarsimUtilityFunctions import larsimModel
from LarsimUtilityFunctions import larsimConfigurationSettings

sys.path.insert(0, os.getcwd())

# instantiate UQsim
uqsim = uqef.UQsim()

#####################################
#####################################
# change args locally for testing and debugging
local_debugging = True
if local_debugging:
    local_debugging_nodes = False  # True
    exit_after_debugging_nodes = False
    save_solver_results = False  # True

    uqsim.args.model = "ishigami"  # "larsim"

    uqsim.args.uncertain = "all"
    uqsim.args.chunksize = 1

    uqsim.args.uq_method = "sc"  # "sc" | "saltelli" | "mc" | "ensemble"
    uqsim.args.mc_numevaluations = 1000
    uqsim.args.sampling_rule = "random"  # | "sobol" | "latin_hypercube" | "halton"  | "hammersley"
    uqsim.args.sc_q_order = 8  # 11 7 #10 3
    uqsim.args.sc_p_order = 7  # 8 6 #8 6
    uqsim.args.sc_quadrature_rule = "p"  # "clenshaw_curtis", "patterson", "G"

    uqsim.args.read_nodes_from_file = True
    l = 8
    path_to_file = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights")
    uqsim.args.parameters_file = path_to_file / f"GQU_d3_l{l}.asc" # f"KPU_d3_l{l}.asc"

    uqsim.args.sc_poly_rule = "three_terms_recurrence"  # "gram_schmidt" | "three_terms_recurrence" | "cholesky"
    uqsim.args.sc_poly_normed = True  # True
    uqsim.args.sc_sparse_quadrature = True  # True
    uqsim.args.regression = False

    uqsim.args.inputModelDir = os.path.abspath(os.path.join('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2','Larsim-data'))
    uqsim.args.sourceDir = os.path.abspath(os.path.join('/dss/dsshome1/lxc0C/ga45met2', 'Repositories', 'Larsim-UQ'))
    # uqsim.args.outputResultDir = os.path.abspath(os.path.join("/gpfs/scratch/pr63so/ga45met2", "Larsim_runs", 'larsim_run_ensemble_2'))
    uqsim.args.outputResultDir = os.path.abspath(os.path.join("/gpfs/scratch/pr63so/ga45met2", "ishigami_runs", 'ishigami_run_sc_sg_gqu_p7_q8'))
    uqsim.args.outputModelDir = uqsim.args.outputResultDir
    #uqsim.args.config_file = "/dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/configurations_Larsim/configuration_larsim_uqsim_cm2_v4.json" #"configuration_larsim_uqsim.json"
    #uqsim.args.config_file = '/dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/configurations_Larsim/configurations_larsim_master_lai_small.json'
    # uqsim.args.config_file = '/dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/configurations_Larsim/configurations_larsim_boundery_values_mls.json'
    uqsim.args.config_file = '/dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/configurations/configuration_ishigami.json'

    uqsim.args.sampleFromStandardDist = True  # True
    uqsim.args.transformToStandardDist = True  # True

    uqsim.args.mpi = True
    uqsim.args.mpi_method = "MpiPoolSolver"  # "LinearSolver"

    uqsim.args.uqsim_store_to_file = False

    uqsim.args.disable_statistics = False
    uqsim.args.parallel_statistics = False  # True
    uqsim.args.compute_Sobol_t = True  # True
    uqsim.args.compute_Sobol_m = True  # True

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
                                                         uqsim.configuration_object["model_paths"]["workingDir"]))
except KeyError:
    uqsim.args.workingDir = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "model_runs"))

try:
    uqsim.configuration_object["model_paths"]["workingDir"] = uqsim.args.workingDir
except KeyError:
    uqsim.configuration_object["model_paths"] = {}
    uqsim.configuration_object["model_paths"]["workingDir"] = uqsim.args.workingDir

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.configuration_object["model_paths"]["workingDir"]):
        subprocess.run(["mkdir", uqsim.configuration_object["model_paths"]["workingDir"]])

#####################################
# register model
#####################################

uqsim.models.update({"larsim"         : (lambda: LarsimModel.LarsimModelUQ(
    configurationObject=uqsim.configuration_object,
    inputModelDir=uqsim.args.inputModelDir,
    workingDir=uqsim.args.workingDir,
    sourceDir=uqsim.args.sourceDir,
    disable_statistics=uqsim.args.disable_statistics,
    uq_method=uqsim.args.uq_method))})
uqsim.models.update({"oscillator"     : (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModel(uqsim.configuration_object))})
uqsim.models.update({"ishigami"       : (lambda: IshigamiModel.IshigamiModel(
    configurationObject=uqsim.configuration_object))})
uqsim.models.update({"productFunction": (lambda: ProductFunctionModel.ProductFunctionModel(uqsim.configuration_object))})

#####################################
# register statistics
#####################################

uqsim.statistics.update({"larsim"         : (lambda: LarsimStatistics.LarsimStatistics(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.workingDir,
    sampleFromStandardDist=uqsim.args.sampleFromStandardDist,
    store_qoi_data_in_stat_dict=False,
    parallel_statistics=uqsim.args.parallel_statistics,
    mpi_chunksize=uqsim.args.mpi_chunksize,
    unordered=False,
    uq_method=uqsim.args.uq_method,
    compute_Sobol_t=uqsim.args.compute_Sobol_t,
    compute_Sobol_m=uqsim.args.compute_Sobol_m
))})
uqsim.statistics.update({"oscillator"     : (lambda: LinearDampedOscillatorStatistics.LinearDampedOscillatorStatistics())})
uqsim.statistics.update({"ishigami"       : (lambda: IshigamiStatistics.IshigamiStatistics(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.workingDir,
    sampleFromStandardDist=uqsim.args.sampleFromStandardDist,
    uq_method=uqsim.args.uq_method,
    compute_Sobol_t=uqsim.args.compute_Sobol_t,
    compute_Sobol_m=uqsim.args.compute_Sobol_m
))})
uqsim.statistics.update({"productFunction": (lambda: ProductFunctionStatistics.ProductFunctionStatistics(uqsim.configuration_object))})

#####################################
# setup
#####################################

uqsim.setup()

# save simulation nodes
simulationNodes_save_file = "nodes"
uqsim.save_simulationNodes(fileName=simulationNodes_save_file)

# print the dictionary with the arguments
if uqsim.is_master():
    uqsim_args_temp_dict = vars(uqsim.args)
    print(f"UQSIM.ARGS")
    for key, value in uqsim_args_temp_dict.items():
        print(f"{key}: {value}")

# save the dictionary with the arguments - once before the simulation
if uqsim.is_master():
    argsFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "uqsim_args.pkl"))
    with open(argsFileName, 'wb') as handle:
        pickle.dump(uqsim.args, handle, protocol=pickle.HIGHEST_PROTOCOL)

#####################################
# check-up (for Larsim model)
#####################################

if uqsim.is_master():
    if uqsim.args.model == "larsim" and local_debugging_nodes:
        # print(uqsim.configuration_object["tuples_parameters_info"])

        # play with different sampling rules...
        larsimConfigurationsObject = larsimModel.LarsimConfigurations(configurationObject=uqsim.configuration_object)

        # plot position of the nodes
        local_nodes = uqsim.simulationNodes.nodes.T
        print(f"Shape of simulationNodes.nodes.T is: {local_nodes.shape}")
        local_parameters = uqsim.simulationNodes.parameters.T
        print(f"Shape of simulationNodes.parameters.T is: {local_parameters.shape}")
        local_simulation_parameters = uqsim.simulation.parameters
        print(f"Shape of simulation.parameters is: {local_simulation_parameters.shape}")
        local_dist = uqsim.simulationNodes.joinedDists

        if uqsim.simulationNodes.distNodes:
            local_distNodes = uqsim.simulationNodes.distNodes.T
            print(f"Shape of simulationNodes.distNodes.T is: {local_distNodes.shape}")
        if uqsim.simulationNodes.weights:
            local_weights = uqsim.simulationNodes.weights
            print(f"Shape of simulationNodes.weights is: {local_weights.shape}")

        # Problem with Saltelli&MC is that uqsim.simulationNodes.parameters
        # are different from uqsim.simulation.parameters
        # plot position of the final parameters
        if "tuples_parameters_info" not in uqsim.configuration_object:
            print(f"uqsim.configuration_object was not updated together with model.configurationObject")
            larsimConfigurationSettings.update_configurationObject_with_parameters_info(uqsim.configuration_object)
        list_of_parameters_dict = []
        local_master_dir = pathlib.Path(osp.abspath(osp.join(uqsim.args.workingDir, 'master_configuration')))
        tape35_path = local_master_dir / "tape35"
        lanu_path = local_master_dir / "lanu.par"
        for parameter in local_parameters: # local_simulation_parameters
            ordered_dict_of_all_params, _ = larsimConfigurationSettings.params_configurations(parameter, tape35_path,
                                                                                              lanu_path,
                                                                                              uqsim.configuration_object,
                                                                                              process_id=0,
                                                                                              reference_value_from_TGB=3085,
                                                                                              take_direct_value=False,
                                                                                              write_new_values_to_tape35=False,
                                                                                              write_new_values_to_lanu=False)
            list_of_parameters_dict.append(ordered_dict_of_all_params)
        df_with_final_parameters = pd.DataFrame(list_of_parameters_dict)
        temp_file_path = pathlib.Path(uqsim.args.outputResultDir) / "parameters.pkl"
        df_with_final_parameters.to_pickle(temp_file_path, compression="gzip")

        # # Plot polynomials
        # # polynomial_expansion = cp.orth_ttr(order, dist)
        if uqsim.args.uq_method == "sc":
            polynomial_expansion = cp.generate_expansion(uqsim.args.sc_q_order, local_dist,
                                                         rule=uqsim.args.sc_poly_rule,
                                                         normed=uqsim.args.sc_poly_normed)
        # plotting simulation nodes
        if uqsim.simulationNodes.dists:
            uqsim.simulationNodes.plotDists(fileName=uqsim.args.outputResultDir + "/dists", fileNameIdentIsFullName=True)
            uqsim.simulationNodes.plotDistsSetup(fileName=uqsim.args.outputResultDir + "/distsSetup.pdf",
                                                 numCollocationPointsPerDim=10)
        # uqsim.plot_nodes()
        if exit_after_debugging_nodes:
            sys.exit()

#####################################
# start the simulation
#####################################

uqsim.simulate()

#####################################
# check-up (for Larsim model)
#####################################

if uqsim.is_master():
    if uqsim.args.model == "larsim" and save_solver_results and uqsim.args.disable_statistics:
        processed_sample_results = LarsimStatistics.LarsimSamples(uqsim.solver.results,
                                                                  configurationObject=uqsim.configuration_object)
        processed_sample_results.save_samples_to_file(uqsim.args.outputResultDir)
        processed_sample_results.save_index_parameter_values(uqsim.args.outputResultDir)
        processed_sample_results.save_index_parameter_gof_values(uqsim.args.outputResultDir)
        if strtobool(uqsim.configuration_object["model_run_settings"]["compute_gradients"]) :
            processed_sample_results.save_dict_of_approx_matrix_c(uqsim.args.outputResultDir)
            processed_sample_results.save_dict_of_matrix_c_eigen_decomposition(uqsim.args.outputResultDir)

#####################################
# save uqsim.configuration_object
#####################################

if uqsim.is_master():
    fileName = pathlib.Path(uqsim.args.outputResultDir) / "configuration_object"
    with open(fileName, 'wb') as f:
        dill.dump(uqsim.configuration_object, f)

#####################################
# statistics
#####################################

uqsim.calc_statistics()
uqsim.save_statistics()
if uqsim.args.model == "larsim":
    uqsim.plot_statistics(display=False, plot_measured_timeseries=True, plot_unalteres_timeseries=False)
else:
    uqsim.plot_statistics(display=False)

# uqsim.args.uqsim_file = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "uqsim.saved"))
# #uqsim.store_to_file()

#####################################
# tear down
#####################################

uqsim.tear_down()
